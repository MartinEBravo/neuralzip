import os
import re
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .bitpacking import bits_required, packed_byte_length


MM_REPAIR_RATIO_RE = re.compile(r"Compression ratio:\s+([0-9.]+)%")
MM_REPAIR_SIZE_RE = re.compile(r"Estimated output size \(bytes\):\s+([0-9]+)")
MM_REPAIR_RULES_RE = re.compile(r"Number of rules:\s+([0-9]+)")
MM_REPAIR_FINAL_RE = re.compile(r"Final sequence length:\s+([0-9]+)")


def log_bits(name, bits, original_bits):
    mbits = bits / 1e6
    mbytes = bits / 8e6
    ratio = 100 * bits / original_bits if original_bits else 0.0
    print(f"[{name}] {mbits:8.2f} Mbits | {mbytes:8.2f} MB | {ratio:5.2f}%")


def iter_model_field_values(model, field):
    yield from model.iter_field_values(field)


def collect_exponents(model):
    return model.collect_exponents()


def collect_named_exponents(model):
    return model.named_exponents()


def collect_field(model, field, max_symbols):
    return model.collect_field(field, max_symbols=max_symbols)


def official_zipnn_bits(model, method):
    try:
        from zipnn import ZipNN
    except ImportError as exc:
        raise ImportError(
            "The official zipnn package is required. Install it with `pip install zipnn`."
        ) from exc

    total_bits = 0
    named_tensors = (
        model.iter_tensors()
        if hasattr(model, "iter_tensors")
        else model.named_parameters()
    )
    for _, param in named_tensors:
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
    layer_arrays = [param for _, param in model.iter_tensors()]
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


def clustered_trigram_huffman_exponent_bits(model, layer_exponents):
    layer_arrays = [param for _, param in model.iter_tensors()]
    clusters = model.cluster_layers(layer_arrays)
    total_bytes = 16
    cluster_id_bits = bits_required(len(clusters))
    if cluster_id_bits:
        total_bytes += packed_byte_length(len(layer_exponents), cluster_id_bits)

    for cluster_ids in clusters:
        packed_parts = []
        tail_size = 0
        for layer_id in cluster_ids:
            packed, tail = model.pack_ngrams_uint_with_tail(
                layer_exponents[layer_id],
                n=3,
            )
            if packed.size:
                packed_parts.append(packed)
            tail_size += tail.size
        packed_stream = (
            np.concatenate(packed_parts).astype(np.uint32, copy=False)
            if packed_parts
            else np.array([], dtype=np.uint32)
        )
        payload = model.encode_huffman_payload(packed_stream)
        total_bytes += model.cluster_payload_byte_length(payload) + tail_size
    return total_bytes * 8


def architecture_component(name):
    parts = name.lower().split(".")
    if "components" in parts:
        idx = parts.index("components")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    if parts and parts[0] in {"unet", "vae", "text_encoder", "text_encoder_2"}:
        return parts[0]
    return "model"


def architecture_role(name):
    name = name.lower()
    parts = name.split(".")
    leaf = parts[-1] if parts else name
    parent = parts[-2] if len(parts) > 1 else ""
    joined = "." + name + "."

    if leaf == "bias" or name.endswith(".bias"):
        return "bias"
    if "norm" in name or "ln_" in name or "layer_norm" in name:
        return "norm"
    if "embed" in name or "embedding" in name:
        return "embed"
    if "lm_head" in name:
        return "lm_head"
    if "conv" in parent or "conv" in leaf or ".conv" in joined:
        return "conv"
    if (
        "q_proj" in name
        or "to_q" in name
        or ".q." in joined
        or "query" in name
    ):
        return "attention_q"
    if (
        "k_proj" in name
        or "to_k" in name
        or ".k." in joined
        or "key" in name
    ):
        return "attention_k"
    if (
        "v_proj" in name
        or "to_v" in name
        or ".v." in joined
        or "value" in name
    ):
        return "attention_v"
    if (
        "o_proj" in name
        or "out_proj" in name
        or "to_out" in name
        or "proj_out" in name
        or "dense" in name
    ):
        return "attention_o"
    if "gate_proj" in name or "gate" in name:
        return "mlp_gate"
    if "up_proj" in name or "ff.net.0" in name or parent in {"fc1", "wi", "w1"}:
        return "mlp_up"
    if "down_proj" in name or "ff.net.2" in name or parent in {"fc2", "wo", "w2"}:
        return "mlp_down"
    if "mlp" in name or "feed_forward" in name or "ffn" in name or ".ff." in joined:
        return "mlp_other"
    return "other"


def architecture_group(name):
    return f"{architecture_component(name)}:{architecture_role(name)}"


def ngram_huffman_payload_bytes(model, exponents, n):
    if exponents.size == 0:
        return 0
    if n == 1:
        payload = model.encode_huffman_payload(exponents.astype(np.uint32, copy=False))
        return model.cluster_payload_byte_length(payload)

    packed, tail = model.pack_ngrams_uint_with_tail(exponents, n=n)
    payload = model.encode_huffman_payload(packed.astype(np.uint32, copy=False))
    return model.cluster_payload_byte_length(payload) + tail.size


def best_ngram_huffman_payload_bytes(model, exponents, n_values=(1, 2, 3, 4)):
    best_n = None
    best_size = None
    for n in n_values:
        size = ngram_huffman_payload_bytes(model, exponents, n)
        if best_size is None or size < best_size:
            best_n = n
            best_size = size
    return best_size or 0, best_n or 1


def predictive_transform(exponents, predictor):
    values = exponents.astype(np.uint8, copy=False)
    if predictor == "raw" or values.size < 2:
        return values

    predicted = np.empty_like(values)
    predicted[0] = values[0]
    if predictor == "delta":
        deltas = values[1:].astype(np.int16) - values[:-1].astype(np.int16)
        predicted[1:] = (deltas & 0xFF).astype(np.uint8, copy=False)
        return predicted
    if predictor == "xor":
        predicted[1:] = np.bitwise_xor(values[1:], values[:-1])
        return predicted
    raise ValueError(f"Unsupported predictor: {predictor}")


def neuralzip_predictor_exponent_bits(
    model,
    named_exponents,
    n_values=(1, 2, 3, 4),
    predictors=("raw", "delta", "xor"),
    max_merge_passes=2,
):
    exponents = [entry[1] for entry in named_exponents]
    families = {}
    for idx, (name, _) in enumerate(named_exponents):
        families.setdefault(architecture_group(name), []).append(idx)

    cost_cache = {}

    def cluster_stream(cluster, predictor):
        parts = [
            predictive_transform(exponents[idx], predictor)
            for idx in cluster
            if exponents[idx].size
        ]
        if not parts:
            return np.array([], dtype=np.uint8)
        return np.concatenate(parts).astype(np.uint8, copy=False)

    def cluster_cost(cluster):
        key = tuple(cluster)
        if key not in cost_cache:
            best = None
            for predictor in predictors:
                payload_bytes, best_n = best_ngram_huffman_payload_bytes(
                    model,
                    cluster_stream(cluster, predictor),
                    n_values=n_values,
                )
                # Store predictor and n-gram selectors for each cluster.
                candidate = payload_bytes + 2, predictor, best_n
                if best is None or candidate[0] < best[0]:
                    best = candidate
            cost_cache[key] = best
        return cost_cache[key]

    final_clusters = []
    for family_indices in families.values():
        clusters = [[idx] for idx in family_indices]
        for _ in range(max_merge_passes):
            changed = False
            merged_clusters = []
            idx = 0
            while idx < len(clusters):
                if idx + 1 >= len(clusters):
                    merged_clusters.append(clusters[idx])
                    idx += 1
                    continue

                left = clusters[idx]
                right = clusters[idx + 1]
                left_cost, _, _ = cluster_cost(left)
                right_cost, _, _ = cluster_cost(right)
                merged = left + right
                merged_cost, _, _ = cluster_cost(merged)
                if merged_cost < left_cost + right_cost:
                    merged_clusters.append(merged)
                    changed = True
                    idx += 2
                else:
                    merged_clusters.append(left)
                    idx += 1

            clusters = merged_clusters
            if not changed:
                break
        final_clusters.extend(clusters)

    total_bytes = 16
    cluster_id_bits = bits_required(len(final_clusters))
    if cluster_id_bits:
        total_bytes += packed_byte_length(len(named_exponents), cluster_id_bits)

    chosen_orders = {}
    chosen_predictors = {}
    for cluster in final_clusters:
        payload_bytes, predictor, best_n = cluster_cost(cluster)
        total_bytes += payload_bytes
        chosen_orders[best_n] = chosen_orders.get(best_n, 0) + 1
        chosen_predictors[predictor] = chosen_predictors.get(predictor, 0) + 1

    return total_bytes * 8, {
        "clusters": len(final_clusters),
        "families": len(families),
        "orders": chosen_orders,
        "predictors": chosen_predictors,
    }


def neuralzip_v2_exponent_bits(
    model,
    named_exponents,
    n_values=(1, 2, 3, 4),
    max_merge_passes=2,
):
    exponents = [entry[1] for entry in named_exponents]
    families = {}
    for idx, (name, _) in enumerate(named_exponents):
        families.setdefault(architecture_group(name), []).append(idx)

    cost_cache = {}

    def cluster_stream(cluster):
        parts = [exponents[idx] for idx in cluster if exponents[idx].size]
        if not parts:
            return np.array([], dtype=np.uint8)
        return np.concatenate(parts).astype(np.uint8, copy=False)

    def cluster_cost(cluster):
        key = tuple(cluster)
        if key not in cost_cache:
            payload_bytes, best_n = best_ngram_huffman_payload_bytes(
                model,
                cluster_stream(cluster),
                n_values=n_values,
            )
            # Store one selector byte so the decoder knows which n-gram order was used.
            cost_cache[key] = payload_bytes + 1, best_n
        return cost_cache[key]

    final_clusters = []
    for family_indices in families.values():
        clusters = [[idx] for idx in family_indices]
        for _ in range(max_merge_passes):
            changed = False
            merged_clusters = []
            idx = 0
            while idx < len(clusters):
                if idx + 1 >= len(clusters):
                    merged_clusters.append(clusters[idx])
                    idx += 1
                    continue

                left = clusters[idx]
                right = clusters[idx + 1]
                left_cost, _ = cluster_cost(left)
                right_cost, _ = cluster_cost(right)
                merged = left + right
                merged_cost, _ = cluster_cost(merged)
                if merged_cost < left_cost + right_cost:
                    merged_clusters.append(merged)
                    changed = True
                    idx += 2
                else:
                    merged_clusters.append(left)
                    idx += 1

            clusters = merged_clusters
            if not changed:
                break
        final_clusters.extend(clusters)

    total_bytes = 16
    cluster_id_bits = bits_required(len(final_clusters))
    if cluster_id_bits:
        total_bytes += packed_byte_length(len(named_exponents), cluster_id_bits)

    chosen_orders = {}
    for cluster in final_clusters:
        payload_bytes, best_n = cluster_cost(cluster)
        total_bytes += payload_bytes
        chosen_orders[best_n] = chosen_orders.get(best_n, 0) + 1

    return total_bytes * 8, {
        "clusters": len(final_clusters),
        "families": len(families),
        "orders": chosen_orders,
    }


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
    return symbol_count * model.field_bits_per_value("exponent")
