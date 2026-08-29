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
    torch.float32: {
        "element_size": 4,
        "bit_reorder": 1,
    },
    torch.bfloat16: {
        "element_size": 2,
        "bit_reorder": 1,
    },
    torch.float16: {
        "element_size": 2,
        "bit_reorder": 0,
    },
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
        raise ValueError(
            f"Decoded tensor byte length mismatch: {len(owner)} != {expected_bytes}."
        )
    if not owner:
        return torch.empty(shape, dtype=dtype)
    return torch.frombuffer(owner, dtype=torch.uint8).view(dtype).reshape(shape)


def _byte_group_dtype_config(dtype):
    config = BYTE_GROUP_DTYPE_CONFIG.get(dtype)
    if config is None:
        raise TypeError(
            "NeuralZip supports torch.float32, torch.bfloat16, and torch.float16 "
            f"tensors; got {dtype}."
        )
    return config


def _reorder_float32(raw_bytes):
    values = np.frombuffer(raw_bytes, dtype=np.dtype("<u4")).copy()
    sign = (values >> np.uint32(8)) & np.uint32(0x00800000)
    exponent = (values << np.uint32(1)) & np.uint32(0xFF000000)
    mantissa = values & np.uint32(0x007FFFFF)
    return (exponent | sign | mantissa).astype(np.dtype("<u4"), copy=False).tobytes()


def _restore_float32(raw_bytes):
    values = np.frombuffer(raw_bytes, dtype=np.dtype("<u4")).copy()
    sign = (values << np.uint32(8)) & np.uint32(0x80000000)
    exponent = (values >> np.uint32(1)) & np.uint32(0x7F800000)
    mantissa = values & np.uint32(0x007FFFFF)
    return (sign | exponent | mantissa).astype(np.dtype("<u4"), copy=False).tobytes()


def _reorder_bfloat16(raw_bytes):
    values = np.frombuffer(raw_bytes, dtype=np.dtype("<u2")).copy()
    sign = (values >> np.uint16(8)) & np.uint16(0x0080)
    exponent = (values << np.uint16(1)) & np.uint16(0xFF00)
    mantissa = values & np.uint16(0x007F)
    return (exponent | sign | mantissa).astype(np.dtype("<u2"), copy=False).tobytes()


def _restore_bfloat16(raw_bytes):
    values = np.frombuffer(raw_bytes, dtype=np.dtype("<u2")).copy()
    sign = (values << np.uint16(8)) & np.uint16(0x8000)
    exponent = (values >> np.uint16(1)) & np.uint16(0x7F80)
    mantissa = values & np.uint16(0x007F)
    return (sign | exponent | mantissa).astype(np.dtype("<u2"), copy=False).tobytes()


def _byte_group_reordered_bytes(tensor, config):
    raw_bytes = _tensor_raw_bytes(tensor)
    if not config["bit_reorder"]:
        return raw_bytes
    if tensor.dtype == torch.float32:
        return _reorder_float32(raw_bytes)
    if tensor.dtype == torch.bfloat16:
        return _reorder_bfloat16(raw_bytes)
    return raw_bytes


def _byte_group_restored_bytes(raw_bytes, dtype, config):
    if not config["bit_reorder"]:
        return raw_bytes
    if dtype == torch.float32:
        return _restore_float32(raw_bytes)
    if dtype == torch.bfloat16:
        return _restore_bfloat16(raw_bytes)
    return raw_bytes


def exponent_distribution(tensor):
    """Return the normalized exponent histogram of a floating tensor."""

    if not torch.is_tensor(tensor) or not tensor.is_floating_point():
        raise TypeError("exponent_distribution expects a floating-point tensor.")
    layout = _FLOAT_EXPONENT_LAYOUTS.get(tensor.dtype)
    if layout is None:
        raise TypeError(f"Unsupported floating dtype: {tensor.dtype}.")
    storage_dtype, mantissa_bits, exponent_mask = layout
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
    divergence = 0.5 * np.sum(
        left[left_mask] * np.log2(left[left_mask] / middle[left_mask])
    )
    divergence += 0.5 * np.sum(
        right[right_mask] * np.log2(right[right_mask] / middle[right_mask])
    )
    return float(np.sqrt(max(float(divergence), 0.0)))


_KIND_PRIORITY = {
    "shared_huff0": 0,
    "packed_huff0": 1,
    "raw": 2,
}


def _finite_nonnegative(value, label):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{label} must be finite and non-negative.")
    return number


def _kind(mode):
    return str(mode.get("kind", ""))


def select_fastest_within_ratio(
    candidates,
    *,
    estimated_bytes,
    estimated_seconds,
    ratio_tolerance=0.0,
):
    """Select the fastest mode within a size tolerance of the smallest."""

    tolerance = _finite_nonnegative(ratio_tolerance, "ratio_tolerance")
    modes = tuple(candidates)
    if not modes:
        raise ValueError("candidates must not be empty.")

    scored = []
    for mode in modes:
        byte_count = _finite_nonnegative(
            estimated_bytes(mode), f"estimated bytes for {_kind(mode)!r}"
        )
        seconds = _finite_nonnegative(
            estimated_seconds(mode), f"estimated seconds for {_kind(mode)!r}"
        )
        scored.append((mode, byte_count, seconds))

    best_bytes = min(
        scored,
        key=lambda item: (
            item[1],
            _KIND_PRIORITY.get(_kind(item[0]), len(_KIND_PRIORITY)),
            _kind(item[0]),
        ),
    )[1]
    byte_limit = best_bytes * (1.0 + tolerance)
    return min(
        (item for item in scored if item[1] <= byte_limit),
        key=lambda item: (
            item[2],
            item[1],
            _KIND_PRIORITY.get(_kind(item[0]), len(_KIND_PRIORITY)),
            _kind(item[0]),
        ),
    )[0]


def select_packed_time_budget(evaluations, *, packed_time_budget_ratio):
    """Choose packed opportunities within the global setup-time budget."""

    budget_ratio = _finite_nonnegative(
        packed_time_budget_ratio, "packed_time_budget_ratio"
    )
    rows = tuple(evaluations)
    budget_seconds = budget_ratio * sum(
        _finite_nonnegative(row.get("base_seconds", 0.0), "base_seconds")
        for row in rows
    )

    opportunities = []
    for row in rows:
        opportunity = row.get("opportunity")
        if opportunity is None:
            continue
        saved_bytes = _finite_nonnegative(
            opportunity.get("saved_bytes", 0.0), "packed saved_bytes"
        )
        raw_extra_seconds = float(opportunity.get("extra_seconds", 0.0))
        if not math.isfinite(raw_extra_seconds):
            raise ValueError("packed extra_seconds must be finite.")
        extra_seconds = max(0.0, raw_extra_seconds)
        if saved_bytes <= 0:
            continue
        savings_per_second = (
            math.inf if extra_seconds == 0 else saved_bytes / extra_seconds
        )
        opportunities.append((row.get("key"), extra_seconds, savings_per_second))

    opportunities.sort(key=lambda item: (-item[2], repr(item[0])))
    selected_keys = set()
    used_seconds = 0.0
    if budget_ratio > 0:
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
    source = np.frombuffer(bytes(data)[: packed.size * n], dtype=np.uint8).reshape(
        -1, n
    )
    escapes = source[~hits].reshape(-1).tobytes() if np.any(~hits) else b""
    return tokens.tobytes(), escapes, tail
