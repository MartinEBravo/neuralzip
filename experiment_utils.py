import os
import re
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm

from bitpacking import bits_required, packed_byte_length
from float_utils import extract_float_parts, get_float_format


MM_REPAIR_RATIO_RE = re.compile(r"Compression ratio:\s+([0-9.]+)%")
MM_REPAIR_SIZE_RE = re.compile(r"Estimated output size \(bytes\):\s+([0-9]+)")
MM_REPAIR_RULES_RE = re.compile(r"Number of rules:\s+([0-9]+)")
MM_REPAIR_FINAL_RE = re.compile(r"Final sequence length:\s+([0-9]+)")


def default_models(args):
    if args.models is not None:
        return args.models
    if args.mode == "full":
        return None
    return ["Gemma 4 E2B"]


def log_bits(name, bits, original_bits):
    mbits = bits / 1e6
    mbytes = bits / 8e6
    ratio = 100 * bits / original_bits if original_bits else 0.0
    print(f"[{name}] {mbits:8.2f} Mbits | {mbytes:8.2f} MB | {ratio:5.2f}%")


def iter_model_field_values(model, field):
    for _, param in model.model.named_parameters():
        sign, exponent, mantissa = extract_float_parts(param)
        yield {
            "sign": sign,
            "exponent": exponent,
            "mantissa": mantissa,
        }[field]


def collect_exponents(model):
    return [
        values.astype(np.uint8, copy=False)
        for values in iter_model_field_values(model, "exponent")
    ]


def collect_field(model, field, max_symbols):
    chunks = []
    seen = 0
    for values in iter_model_field_values(model, field):
        remaining = None if max_symbols <= 0 else max_symbols - seen
        if remaining is not None and remaining <= 0:
            break
        if remaining is not None:
            values = values[:remaining]
        chunks.append(values.astype(np.uint32, copy=False))
        seen += values.size

    if not chunks:
        return np.array([], dtype=np.uint32)
    return np.concatenate(chunks).astype(np.uint32, copy=False)


def official_zipnn_bits(model, method):
    try:
        from zipnn import ZipNN
    except ImportError as exc:
        raise ImportError(
            "The official zipnn package is required. Install it with `pip install zipnn`."
        ) from exc

    total_bits = 0
    for _, param in model.named_parameters():
        compressor = ZipNN(method=method, input_format="torch")
        tensor = param.detach().cpu().contiguous().clone()
        compressed = compressor.compress(tensor)
        total_bits += len(compressed) * 8
    return total_bits


def zipnn_exponent_bits(exponents, method):
    try:
        from zipnn import ZipNN
    except ImportError as exc:
        raise ImportError(
            "The official zipnn package is required. Install it with `pip install zipnn`."
        ) from exc

    compressor = ZipNN(
        method=method,
        input_format="byte",
        bytearray_dtype="bfloat16",
        is_streaming=False,
    )
    original = exponents.tobytes()
    compressed = compressor.compress(bytearray(original))
    decoded = compressor.decompress(compressed)
    if bytes(decoded) != original:
        raise ValueError("ZipNN exponent byte round-trip reconstruction failed.")
    return len(compressed) * 8


def zipnn_exponent_bits_by_tensor(layer_exponents, method):
    return sum(
        zipnn_exponent_bits(exponents, method)
        for exponents in tqdm(layer_exponents, desc="ZipNN exponent tensors")
    )


def should_validate_huffman(exponents, args):
    return args.validate_huffman and exponents.size <= args.validate_huffman_limit


def huffman_exponent_bits(model, exponents, validate=False):
    payload = model.encode_huffman_payload(
        exponents.astype(np.uint32, copy=False),
        validate=validate,
    )
    return model.cluster_payload_byte_length(payload) * 8


def global_huffman_exponent_bits(model, layer_exponents, validate=False):
    return huffman_exponent_bits(
        model,
        np.concatenate(layer_exponents),
        validate=validate,
    )


def zipnn_huffman_exponent_bits(model, exponents):
    payload = model.encode_zipnn_huffman_payload(
        exponents.astype(np.uint32, copy=False),
        validate=True,
    )
    return model.cluster_payload_byte_length(payload) * 8


def trigram_huffman_exponent_bits(model, exponents, validate=False):
    packed, tail = model.pack_ngrams_uint_with_tail(exponents, n=3)
    payload = model.encode_huffman_payload(
        packed.astype(np.uint32, copy=False),
        validate=validate,
    )
    return (model.cluster_payload_byte_length(payload) + tail.size) * 8


def ngram_huffman_exponent_bits(model, exponents, n, validate=False):
    usable_length = (exponents.size // n) * n
    tail_size = exponents.size - usable_length
    if usable_length == 0:
        return tail_size * 8

    if n <= 4:
        symbols = model.pack_ngrams_uint(exponents[:usable_length], n=n)
        unique_symbols, inverse = np.unique(symbols, return_inverse=True)
        unique_count = unique_symbols.size
        dictionary_size = unique_symbols.size * n
    else:
        ngrams = np.ascontiguousarray(exponents[:usable_length].reshape(-1, n))
        row_dtype = np.dtype((np.void, ngrams.dtype.itemsize * n))
        rows = ngrams.view(row_dtype).reshape(-1)
        unique_rows, inverse = np.unique(rows, return_inverse=True)
        unique_count = unique_rows.size
        dictionary_size = unique_rows.size * n

    ids = inverse.astype(np.uint32, copy=False)
    payload = model.encode_huffman_payload(ids, validate=validate)
    encoded_size = getattr(payload, "encoded_byte_length", len(payload.encoded))

    header_size = 4 + 8 + 1
    code_lengths_size = unique_count
    total_bytes = header_size + dictionary_size + code_lengths_size + encoded_size
    return (total_bytes + tail_size) * 8


def zipnn_trigram_huffman_exponent_bits(model, exponents):
    packed, tail = model.pack_ngrams_uint_with_tail(exponents, n=3)
    payload = model.encode_zipnn_huffman_payload(
        packed.astype(np.uint32, copy=False),
        validate=True,
    )
    return (model.cluster_payload_byte_length(payload) + tail.size) * 8


def ctx_huffman_exponent_bits(model, exponents, order):
    payload = model.encode_context_huffman_payload(
        exponents.astype(np.uint32, copy=False),
        order=order,
    )
    return model.cluster_payload_byte_length(payload) * 8


def per_tensor_bits(layer_exponents, estimator):
    total_bits = 0
    for exponents in tqdm(layer_exponents, desc="NeuralZip exponent tensors"):
        if exponents.size:
            total_bits += estimator(exponents)
    return total_bits


def per_tensor_ngram_bits(model, layer_exponents, n, args):
    return per_tensor_bits(
        layer_exponents,
        lambda exponents: ngram_huffman_exponent_bits(
            model,
            exponents,
            n,
            validate=should_validate_huffman(exponents, args),
        ),
    )


def per_tensor_best_bits(model, layer_exponents, ctx_order):
    total_bytes = 16
    for exponents in tqdm(layer_exponents, desc="NeuralZip exponent tensor-best"):
        if exponents.size == 0:
            continue
        options = [
            huffman_exponent_bits(model, exponents),
            trigram_huffman_exponent_bits(model, exponents),
            ctx_huffman_exponent_bits(model, exponents, ctx_order),
        ]
        total_bytes += 1 + int(np.ceil(min(options) / 8))
    return total_bytes * 8


def log_per_tensor_metric(name, layer_exponents, original_bits, estimator):
    log_bits(name, per_tensor_bits(layer_exponents, estimator), original_bits)


def trigram_estimator(model, args):
    return lambda exponents: trigram_huffman_exponent_bits(
        model,
        exponents,
        validate=should_validate_huffman(exponents, args),
    )


def clustered_best_exponent_bits(model, layer_exponents, ctx_order):
    layer_arrays = [param for _, param in model.model.named_parameters()]
    clusters = model.cluster_layers(layer_arrays)
    total_bytes = 16
    cluster_id_bits = bits_required(len(clusters))
    if cluster_id_bits:
        total_bytes += packed_byte_length(len(layer_exponents), cluster_id_bits)

    for cluster_ids in clusters:
        cluster_exps = np.concatenate([layer_exponents[i] for i in cluster_ids])
        options = [
            huffman_exponent_bits(model, cluster_exps),
            trigram_huffman_exponent_bits(model, cluster_exps),
            ctx_huffman_exponent_bits(model, cluster_exps, ctx_order),
        ]
        total_bytes += 1 + int(np.ceil(min(options) / 8))
    return total_bytes * 8


def entropy_from_counts(counts):
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts[counts > 0] / total
    return float(-np.sum(probs * np.log2(probs)))


def empirical_entropy_order0(symbols):
    _, counts = np.unique(symbols, return_counts=True)
    return entropy_from_counts(counts)


def empirical_conditional_entropy(symbols, order):
    if order == 0:
        return empirical_entropy_order0(symbols)
    if symbols.size <= order:
        return 0.0

    context_counts = Counter()
    joint_counts = Counter()
    data = symbols.tolist()
    for idx in range(order, len(data)):
        context = tuple(data[idx - order : idx])
        symbol = data[idx]
        context_counts[context] += 1
        joint_counts[(context, symbol)] += 1

    total = sum(context_counts.values())
    entropy = 0.0
    for (context, _), joint_count in joint_counts.items():
        p_joint = joint_count / total
        p_symbol_given_context = joint_count / context_counts[context]
        entropy -= p_joint * np.log2(p_symbol_given_context)
    return float(entropy)


def pack_non_overlapping_ngrams(symbols, n):
    usable = (symbols.size // n) * n
    if usable == 0:
        return np.array([], dtype=np.uint64)
    groups = symbols[:usable].reshape(-1, n).astype(np.uint64)
    packed = np.zeros(groups.shape[0], dtype=np.uint64)
    width = max(1, int(symbols.max()).bit_length())
    for idx in range(n):
        packed = (packed << width) | groups[:, idx]
    return packed


def ensure_irepair(mm_repair_dir, build=False):
    irepair = mm_repair_dir / "brepair" / "irepair"
    if irepair.exists():
        return irepair
    if not build:
        raise FileNotFoundError(
            f"Missing {irepair}. Build it with `make -C {irepair.parent}` "
            "or rerun with --build."
        )
    subprocess.run(["make", "-C", str(irepair.parent), "irepair"], check=True)
    if not irepair.exists():
        raise FileNotFoundError(f"Build finished but {irepair} is still missing.")
    return irepair


def parse_irepair_stderr(stderr):
    size_match = MM_REPAIR_SIZE_RE.search(stderr)
    ratio_match = MM_REPAIR_RATIO_RE.search(stderr)
    rules_match = MM_REPAIR_RULES_RE.search(stderr)
    final_match = MM_REPAIR_FINAL_RE.search(stderr)
    if not size_match:
        raise ValueError(f"Could not parse irepair output:\n{stderr}")
    return {
        "estimated_bytes": int(size_match.group(1)),
        "reported_ratio": float(ratio_match.group(1)) if ratio_match else None,
        "rules": int(rules_match.group(1)) if rules_match else None,
        "final_length": int(final_match.group(1)) if final_match else None,
    }


def run_irepair(irepair, symbols, memory_mb):
    with tempfile.TemporaryDirectory(prefix="mm-repair-probe-") as tmpdir:
        stream_path = Path(tmpdir) / "exponents.i32"
        symbols.astype(np.int32, copy=False).tofile(stream_path)
        result = subprocess.run(
            [str(irepair), str(stream_path), str(memory_mb)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        parsed = parse_irepair_stderr(result.stderr)
        parsed["raw_bytes"] = stream_path.stat().st_size
        parsed["rule_file_bytes"] = os.path.getsize(f"{stream_path}.R")
        parsed["sequence_file_bytes"] = os.path.getsize(f"{stream_path}.C")
        return parsed


def exponent_budget_bits(model, symbol_count):
    first_dtype = next(model.model.parameters()).dtype
    return symbol_count * get_float_format(first_dtype)["exponent_bits"]
