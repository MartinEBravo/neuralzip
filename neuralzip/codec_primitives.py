"""Dtype, byte-layout, clustering, and mode-selection primitives."""

import math

import numpy as np
import torch

DTYPE_BY_NAME = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

BYTE_GROUP_DTYPE_CONFIG = {
    torch.float32: {"element_size": 4, "bit_reorder": 1},
    torch.bfloat16: {"element_size": 2, "bit_reorder": 1},
    torch.float16: {"element_size": 2, "bit_reorder": 0},
}

_FLOAT_EXPONENT_LAYOUTS = {
    torch.float16: (np.dtype("<u2"), 10, 0x1F),
    torch.bfloat16: (np.dtype("<u2"), 7, 0xFF),
    torch.float32: (np.dtype("<u4"), 23, 0xFF),
}

def dtype_name(dtype):
    """Return the stable, unqualified name used in NeuralZip payloads."""

    return str(dtype).replace("torch.", "")

def _tensor_raw_bytes(tensor):
    tensor = tensor.detach().cpu().contiguous()
    return tensor.reshape(-1).view(torch.uint8).numpy().tobytes()

def _tensor_from_raw_bytes(raw_bytes, dtype, shape):
    # Native decoders return an owning bytearray that PyTorch can wrap
    # directly. Preserve the defensive copy for immutable/archive inputs.
    owner = raw_bytes if isinstance(raw_bytes, bytearray) else bytearray(raw_bytes)
    shape = tuple(int(dimension) for dimension in shape)
    expected_bytes = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
    if len(owner) != expected_bytes:
        raise ValueError(f"Decoded tensor byte length mismatch: {len(owner)} != {expected_bytes}.")
    if not owner:
        return torch.empty(shape, dtype=dtype)
    return torch.frombuffer(owner, dtype=torch.uint8).view(dtype).reshape(shape)

def _byte_group_dtype_config(dtype):
    return BYTE_GROUP_DTYPE_CONFIG[dtype]

def _transform_exponent_layout(raw_bytes, dtype, *, restore):
    storage_dtype, mantissa_bits, exponent_mask = _FLOAT_EXPONENT_LAYOUTS[dtype]
    unsigned = storage_dtype.type
    exponent_bits = int(exponent_mask).bit_length()
    values = np.frombuffer(raw_bytes, dtype=storage_dtype).copy()
    mantissa = values & unsigned((1 << mantissa_bits) - 1)
    if restore:
        sign = (values << unsigned(exponent_bits)) & unsigned(1 << (mantissa_bits + exponent_bits))
        exponent = (values >> unsigned(1)) & unsigned(exponent_mask << mantissa_bits)
    else:
        sign = (values >> unsigned(exponent_bits)) & unsigned(1 << mantissa_bits)
        exponent = (values << unsigned(1)) & unsigned(exponent_mask << (mantissa_bits + 1))
    return (sign | exponent | mantissa).astype(storage_dtype, copy=False).tobytes()

def _byte_group_reordered_bytes(tensor, config):
    raw_bytes = _tensor_raw_bytes(tensor)
    return _transform_exponent_layout(raw_bytes, tensor.dtype, restore=False) if config["bit_reorder"] else raw_bytes

def exponent_distribution(tensor):
    """Return the normalized exponent histogram of a floating tensor."""

    storage_dtype, mantissa_bits, exponent_mask = _FLOAT_EXPONENT_LAYOUTS[tensor.dtype]
    bits = np.frombuffer(_tensor_raw_bytes(tensor), dtype=storage_dtype)
    exponent = (bits >> mantissa_bits) & exponent_mask
    counts = np.bincount(exponent, minlength=256).astype(np.float64)
    total = counts.sum()
    return counts / total if total else counts

def jensen_shannon_distance(left, right):
    """Return the base-2 Jensen--Shannon distance between distributions."""

    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    middle = 0.5 * (left + right)
    left_mask = left > 0
    right_mask = right > 0
    divergence = 0.5 * np.sum(left[left_mask] * np.log2(left[left_mask] / middle[left_mask]))
    divergence += 0.5 * np.sum(right[right_mask] * np.log2(right[right_mask] / middle[right_mask]))
    return float(np.sqrt(max(float(divergence), 0.0)))

_KIND_PRIORITY = {
    "shared_huff0": 0,
    "packed_huff0": 1,
    "raw": 2,
}

def _priority(mode):
    kind = str(mode.get("kind", ""))
    return _KIND_PRIORITY.get(kind, len(_KIND_PRIORITY)), kind

def estimate_mode(mode, scale):
    fixed = float(mode.get("fixed_cost", 0))
    size = mode.get("full_cost")
    if size is None:
        size = fixed + max(0.0, float(mode["sample_cost"]) - fixed) * scale
    return mode, float(size), float(mode["sample_seconds"]) * scale

def select_fastest_within_ratio(scored, *, ratio_tolerance=0.0):
    """Select the fastest mode within a size tolerance of the smallest."""

    tolerance = float(ratio_tolerance)
    scored = tuple(scored)
    best_bytes = min(
        scored,
        key=lambda item: (item[1], *_priority(item[0])),
    )[1]
    byte_limit = best_bytes * (1.0 + tolerance)
    return min(
        (item for item in scored if item[1] <= byte_limit),
        key=lambda item: (
            item[2],
            item[1],
            *_priority(item[0]),
        ),
    )

def select_packed_time_budget(evaluations, *, packed_time_budget_ratio):
    """Choose packed opportunities within the global setup-time budget."""

    rows = tuple(evaluations)
    budget_seconds = float(packed_time_budget_ratio) * sum(float(row["base_seconds"]) for row in rows)

    opportunities = []
    for row in rows:
        opportunity = row["opportunity"]
        if not opportunity or opportunity["saved_bytes"] <= 0:
            continue
        saved = float(opportunity["saved_bytes"])
        extra = max(0.0, float(opportunity["extra_seconds"]))
        speed = math.inf if extra == 0 else saved / extra
        opportunities.append((row["key"], extra, speed))

    opportunities.sort(key=lambda item: (-item[2], repr(item[0])))
    selected_keys = set()
    used_seconds = 0.0
    if packed_time_budget_ratio > 0:
        for key, extra_seconds, _ in opportunities:
            if extra_seconds > 0 and used_seconds + extra_seconds > budget_seconds:
                continue
            selected_keys.add(key)
            used_seconds += extra_seconds
    return frozenset(selected_keys)

ESCAPE_SYMBOL = 255

def _validate_ngram(n):
    n = int(n)
    if not 2 <= n <= 8:
        raise ValueError("Packed exponent n-grams require n in [2, 8].")
    return n

def _symbol_dtype(n):
    return np.dtype("<u4" if n <= 4 else "<u8")

def _pack_ngram_symbols(data, n):
    n = _validate_ngram(n)
    dtype = _symbol_dtype(n)
    data = bytes(data)
    complete_length = (len(data) // n) * n
    if complete_length == 0:
        return np.empty(0, dtype=dtype), data

    values = np.frombuffer(data[:complete_length], dtype=np.uint8).reshape(-1, n)
    packed = np.zeros(values.shape[0], dtype=dtype)
    for offset in range(n):
        packed |= values[:, offset].astype(dtype) << (8 * offset)
    return packed, data[complete_length:]

def topk_vocab(data, n, top_k=254):
    """Return a sorted vocabulary of the most frequent n-grams."""

    n = _validate_ngram(n)
    top_k = int(top_k)
    if not 1 <= top_k <= 254:
        raise ValueError("top_k must be in [1, 254].")
    packed, _ = _pack_ngram_symbols(data, n)
    if packed.size == 0:
        return b""
    symbols, counts = np.unique(packed, return_counts=True)
    order = np.argsort(counts, kind="stable")[::-1]
    selected = np.sort(symbols[order[:top_k]]).astype(_symbol_dtype(n), copy=False)
    return selected.tobytes()

def tokenize_ngrams(data, n, vocab):
    """Tokenize n-grams with byte IDs and a lossless escape stream."""

    n = _validate_ngram(n)
    packed, tail = _pack_ngram_symbols(data, n)
    vocab_values = np.frombuffer(bytes(vocab), dtype=_symbol_dtype(n))
    if packed.size == 0:
        return b"", b"", tail
    if vocab_values.size == 0:
        complete = bytes(data)[: packed.size * n]
        return bytes([ESCAPE_SYMBOL]) * packed.size, complete, tail

    positions = np.searchsorted(vocab_values, packed)
    safe_positions = np.minimum(positions, vocab_values.size - 1)
    hits = (positions < vocab_values.size) & (vocab_values[safe_positions] == packed)
    tokens = np.full(packed.size, ESCAPE_SYMBOL, dtype=np.uint8)
    tokens[hits] = positions[hits].astype(np.uint8)
    source = np.frombuffer(bytes(data)[: packed.size * n], dtype=np.uint8).reshape(-1, n)
    escapes = source[~hits].reshape(-1).tobytes() if np.any(~hits) else b""
    return tokens.tobytes(), escapes, tail
