import math
import time
import zlib
import lzma
import heapq

import numpy as np
import pandas as pd
import torch

from . import huffman, turborc
from .metrics import (
    jensen_shannon_distance as jensen_shannon,
    shannon_entropy_from_counts as _entropy_from_counts,
)
from .neuralzip_codec import NeuralZipCodec
from .packed_exponents import transpose_2d_stream
from .payloads import RAW_CHUNK_METADATA_BYTES
from .utils import split_fields
from .zipnn_codec import ZipNNCodec, _zipnn_dtype_config, _zipnn_reordered_bytes


def named_floating_tensors(model):
    module = model.model if hasattr(model, "model") else model
    if isinstance(module, dict):
        for name, tensor in module.items():
            if torch.is_tensor(tensor) and tensor.is_floating_point():
                yield name, tensor
        return

    for name, tensor in module.named_parameters():
        if torch.is_tensor(tensor) and tensor.is_floating_point():
            yield name, tensor


def layer_signature(name):
    # ModuleList/Sequential indices identify repeated blocks. Digits embedded in
    # semantic names (for example ResNet's layer2 or conv1) identify the stage or
    # operation and must remain part of the architecture label.
    return ".".join(
        "#" if part.isdigit() else part
        for part in str(name).split(".")
    )


def exponent_stream(tensor):
    _, exponent, _ = split_fields(tensor.detach().cpu().contiguous())
    return exponent.reshape(-1).to(torch.uint8).numpy()


def zipnn_exponent_stream(tensor, *, layout="original"):
    tensor = tensor.detach().cpu().contiguous()
    config = _zipnn_dtype_config(tensor.dtype)
    reordered = _zipnn_reordered_bytes(tensor, config)
    element_size = int(config["element_size"])
    stream = reordered[element_size - 1 :: element_size]
    if layout == "transpose2d":
        stream, _ = transpose_2d_stream(
            stream,
            tuple(int(dimension) for dimension in tensor.shape),
        )
        return stream
    if layout != "original":
        raise ValueError("layout must be 'original' or 'transpose2d'.")
    return stream


def zero_order_entropy(values):
    if len(values) == 0:
        return 0.0
    _, counts = np.unique(values, return_counts=True)
    probabilities = counts.astype(np.float64) / float(counts.sum())
    return float(-(probabilities * np.log2(probabilities)).sum())


def _first_order_entropy_from_counts(transition_counts):
    transition_counts = np.asarray(transition_counts, dtype=np.float64)
    total = float(transition_counts.sum())
    if total == 0.0:
        return 0.0

    entropy = 0.0
    for row in transition_counts:
        row_total = float(row.sum())
        if row_total == 0.0:
            continue
        probabilities = row[row > 0] / row_total
        conditional_entropy = float(
            -(probabilities * np.log2(probabilities)).sum()
        )
        entropy += (row_total / total) * conditional_entropy
    return entropy


def first_order_entropy(values):
    values = np.asarray(values, dtype=np.uint8).reshape(-1)
    if values.size < 2:
        return 0.0
    pair_ids = (
        (values[:-1].astype(np.uint16) << 8)
        | values[1:].astype(np.uint16)
    )
    transition_counts = np.bincount(
        pair_ids.astype(np.int64), minlength=256 * 256
    ).reshape(256, 256)
    return _first_order_entropy_from_counts(transition_counts)


def exponent_order_entropies(model):
    symbol_counts = np.zeros(256, dtype=np.int64)
    transition_counts = np.zeros((256, 256), dtype=np.int64)
    tensor_count = 0

    for _, tensor in named_floating_tensors(model):
        values = np.asarray(exponent_stream(tensor), dtype=np.uint8).reshape(-1)
        if values.size == 0:
            continue
        tensor_count += 1
        symbol_counts += np.bincount(values, minlength=256)
        if values.size < 2:
            continue
        pair_ids = (
            (values[:-1].astype(np.uint16) << 8)
            | values[1:].astype(np.uint16)
        )
        transition_counts += np.bincount(
            pair_ids.astype(np.int64), minlength=256 * 256
        ).reshape(256, 256)

    probabilities = symbol_counts[symbol_counts > 0].astype(np.float64)
    probabilities /= float(symbol_counts.sum()) if symbol_counts.sum() else 1.0
    h0 = float(-(probabilities * np.log2(probabilities)).sum())
    h1 = _first_order_entropy_from_counts(transition_counts)
    return {
        "tensors": tensor_count,
        "exponent_symbols": int(symbol_counts.sum()),
        "transitions": int(transition_counts.sum()),
        "h0_bits_per_exponent": h0,
        "h1_bits_per_exponent": h1,
        "entropy_reduction_bits_per_exponent": h0 - h1,
        "entropy_reduction_pct": 100.0 * (h0 - h1) / h0 if h0 else 0.0,
    }


def _module_type_lookup(model):
    module = model.model if hasattr(model, "model") else model
    if isinstance(module, dict) or not hasattr(module, "named_modules"):
        return {}
    return {
        str(name): child.__class__.__name__
        for name, child in module.named_modules()
    }


def _numeric_module_groups(module_types):
    groups = {}
    for module_path, module_type in module_types.items():
        parent_path, _, child_name = module_path.rpartition(".")
        if child_name.isdigit():
            groups.setdefault(parent_path, []).append(
                (int(child_name), module_path, module_type)
            )
    return groups


def _architecture_signature(name, repeated_numeric_parents):
    parts = str(name).split(".")
    normalized = []
    for index, part in enumerate(parts):
        parent_path = ".".join(parts[:index])
        if part.isdigit() and parent_path in repeated_numeric_parents:
            normalized.append("#")
        else:
            normalized.append(part)
    return ".".join(normalized)


def _partition_alignment_metrics(assignments):
    contingency = pd.crosstab(
        assignments["architecture_signature"],
        assignments["js_cluster"],
    )
    counts = contingency.to_numpy(dtype=np.float64)
    total = float(counts.sum())
    if total == 0:
        return {
            "adjusted_rand_index": 0.0,
            "normalized_mutual_information": 0.0,
            "js_cluster_purity": 0.0,
            "signature_concentration": 0.0,
        }

    row_totals = counts.sum(axis=1)
    column_totals = counts.sum(axis=0)
    expected = np.outer(row_totals, column_totals) / total
    nonzero = counts > 0
    mutual_information = float(
        np.sum((counts[nonzero] / total) * np.log2(counts[nonzero] / expected[nonzero]))
    )

    def entropy(values):
        probabilities = values[values > 0] / total
        return float(-(probabilities * np.log2(probabilities)).sum())

    row_entropy = entropy(row_totals)
    column_entropy = entropy(column_totals)
    nmi_denominator = math.sqrt(row_entropy * column_entropy)
    if nmi_denominator:
        normalized_mutual_information = mutual_information / nmi_denominator
    else:
        normalized_mutual_information = float(
            row_entropy == 0.0 and column_entropy == 0.0
        )

    def choose_two(values):
        return values * (values - 1.0) / 2.0

    total_pairs = float(choose_two(np.array([total]))[0])
    observed_index = float(choose_two(counts).sum())
    row_index = float(choose_two(row_totals).sum())
    column_index = float(choose_two(column_totals).sum())
    if total_pairs:
        expected_index = row_index * column_index / total_pairs
        maximum_index = 0.5 * (row_index + column_index)
        denominator = maximum_index - expected_index
        adjusted_rand_index = (
            (observed_index - expected_index) / denominator
            if denominator
            else 1.0
        )
    else:
        adjusted_rand_index = 1.0

    return {
        "adjusted_rand_index": adjusted_rand_index,
        "normalized_mutual_information": normalized_mutual_information,
        "js_cluster_purity": float(counts.max(axis=0).sum() / total),
        "signature_concentration": float(counts.max(axis=1).sum() / total),
    }


def _weighted_partition_purity(assignments):
    weighted = assignments.pivot_table(
        index="architecture_signature",
        columns="js_cluster",
        values="numel",
        aggfunc="sum",
        fill_value=0,
    ).to_numpy(dtype=np.float64)
    total = float(weighted.sum())
    if total == 0:
        return 0.0, 0.0
    return (
        float(weighted.max(axis=0).sum() / total),
        float(weighted.max(axis=1).sum() / total),
    )


def architecture_js_alignment(model, *, cluster_js_threshold=0.05, max_tensors=None):
    """Compare JS exponent clusters with architecture-derived parameter groups.

    Hugging Face does not infer clusters. It exposes the module hierarchy through
    ``named_modules`` and tensor paths through ``named_parameters``. The
    architecture signature used here removes numeric layer indices from those
    paths, so corresponding parameters in repeated blocks share one label.
    """

    named_tensors = sorted(named_floating_tensors(model), key=lambda item: item[0])
    if max_tensors is not None:
        named_tensors = named_tensors[: int(max_tensors)]

    cluster_started = time.perf_counter()
    codec = NeuralZipCodec(
        cluster_js_threshold=cluster_js_threshold,
        ngram_values=(),
        packed_source_budget_ratio=0.0,
    )
    codec._prepare_clusters(named_tensors)
    cluster_seconds = time.perf_counter() - cluster_started

    module_types = _module_type_lookup(model)
    numeric_module_groups = _numeric_module_groups(module_types)
    repeated_numeric_parents = {
        parent_path
        for parent_path, children in numeric_module_groups.items()
        if len(children) > 1 and len({child[2] for child in children}) == 1
    }
    repeated_block_summary = pd.DataFrame(
        [
            {
                "module_path": parent_path,
                "repeated_blocks": len(children),
                "block_type": children[0][2],
                "child_modules": " | ".join(
                    child[1] for child in sorted(children)
                ),
            }
            for parent_path, children in sorted(numeric_module_groups.items())
            if parent_path in repeated_numeric_parents
        ]
    )
    rows = []
    histograms = []
    for tensor_index, (name, tensor) in enumerate(named_tensors):
        values = np.asarray(exponent_stream(tensor), dtype=np.uint8).reshape(-1)
        if values.size == 0:
            continue
        counts = np.bincount(values, minlength=256).astype(np.float64)
        module_path, separator, parameter_role = str(name).rpartition(".")
        if not separator:
            module_path = ""
            parameter_role = str(name)
        module_type = module_types.get(module_path, "MappingTensor")
        rows.append(
            {
                "tensor_index": tensor_index,
                "name": str(name),
                "module_path": module_path,
                "module_type": module_type,
                "parameter_role": parameter_role,
                "module_type_group": f"{module_type}.{parameter_role}",
                "architecture_signature": _architecture_signature(
                    name, repeated_numeric_parents
                ),
                "js_cluster": codec.cluster_id(name),
                "numel": int(tensor.numel()),
                "raw_bytes": int(tensor.numel() * tensor.element_size()),
                "entropy_bits": zero_order_entropy(values),
                "dominant_exponent": int(np.argmax(counts)),
                "dominant_exponent_mass": float(counts.max() / counts.sum()),
            }
        )
        histograms.append(counts / counts.sum())

    assignments = pd.DataFrame(rows)
    if assignments.empty:
        empty = pd.DataFrame()
        return {
            "assignments": assignments,
            "summary": empty,
            "pair_summary": empty,
            "signature_summary": empty,
            "cluster_summary": empty,
            "repeated_block_summary": repeated_block_summary,
            "cluster_seconds": cluster_seconds,
        }

    pair_rows = []
    for left_index in range(len(assignments)):
        for right_index in range(left_index + 1, len(assignments)):
            left = assignments.iloc[left_index]
            right = assignments.iloc[right_index]
            pair_rows.append(
                {
                    "same_signature": bool(
                        left["architecture_signature"]
                        == right["architecture_signature"]
                    ),
                    "same_module_type_group": bool(
                        left["module_type_group"] == right["module_type_group"]
                    ),
                    "same_js_cluster": bool(
                        left["js_cluster"] == right["js_cluster"]
                    ),
                    "js_distance": jensen_shannon(
                        histograms[left_index], histograms[right_index]
                    ),
                }
            )
    pairs = pd.DataFrame(pair_rows)
    if pairs.empty:
        pair_summary = pd.DataFrame()
    else:
        pair_summary = (
            pairs.groupby("same_signature", as_index=False)
            .agg(
                pairs=("js_distance", "size"),
                mean_js=("js_distance", "mean"),
                median_js=("js_distance", "median"),
                same_js_cluster_fraction=("same_js_cluster", "mean"),
            )
            .sort_values("same_signature", ascending=False)
        )

    signature_rows = []
    for signature, group in assignments.groupby("architecture_signature", sort=True):
        cluster_counts = group["js_cluster"].value_counts()
        modal_cluster = str(cluster_counts.index[0])
        signature_rows.append(
            {
                "architecture_signature": signature,
                "module_type": ",".join(sorted(group["module_type"].unique())),
                "parameter_role": ",".join(
                    sorted(group["parameter_role"].unique())
                ),
                "tensors": int(len(group)),
                "numel": int(group["numel"].sum()),
                "js_clusters": int(group["js_cluster"].nunique()),
                "modal_js_cluster": modal_cluster,
                "modal_cluster_fraction": float(cluster_counts.iloc[0] / len(group)),
                "mean_entropy_bits": float(group["entropy_bits"].mean()),
                "example_tensor": str(group.iloc[0]["name"]),
            }
        )
    signature_summary = pd.DataFrame(signature_rows).sort_values(
        ["numel", "tensors"], ascending=False
    )

    cluster_rows = []
    for cluster_id, group in assignments.groupby("js_cluster", sort=True):
        signature_counts = group["architecture_signature"].value_counts()
        cluster_rows.append(
            {
                "js_cluster": cluster_id,
                "tensors": int(len(group)),
                "numel": int(group["numel"].sum()),
                "architecture_signatures": int(
                    group["architecture_signature"].nunique()
                ),
                "modal_architecture_signature": str(signature_counts.index[0]),
                "signature_purity": float(signature_counts.iloc[0] / len(group)),
                "example_tensors": " | ".join(group["name"].head(3)),
            }
        )
    cluster_summary = pd.DataFrame(cluster_rows).sort_values(
        ["numel", "tensors"], ascending=False
    )

    metrics = _partition_alignment_metrics(assignments)
    weighted_purity, weighted_concentration = _weighted_partition_purity(assignments)
    if pair_summary.empty:
        same_signature = pair_summary
        different_signature = pair_summary
    else:
        same_signature = pair_summary[pair_summary["same_signature"]]
        different_signature = pair_summary[~pair_summary["same_signature"]]

    def pair_value(frame, column):
        return float(frame.iloc[0][column]) if not frame.empty else 0.0

    summary = pd.DataFrame(
        [
            {
                "tensors": int(len(assignments)),
                "architecture_signatures": int(
                    assignments["architecture_signature"].nunique()
                ),
                "module_type_groups": int(
                    assignments["module_type_group"].nunique()
                ),
                "repeated_block_groups": int(len(repeated_block_summary)),
                "js_clusters": int(assignments["js_cluster"].nunique()),
                "js_threshold": float(cluster_js_threshold),
                "js_clustering_seconds": cluster_seconds,
                **metrics,
                "element_weighted_js_cluster_purity": weighted_purity,
                "element_weighted_signature_concentration": weighted_concentration,
                "same_signature_mean_js": pair_value(same_signature, "mean_js"),
                "different_signature_mean_js": pair_value(
                    different_signature, "mean_js"
                ),
                "same_signature_same_cluster_fraction": pair_value(
                    same_signature, "same_js_cluster_fraction"
                ),
                "different_signature_same_cluster_fraction": pair_value(
                    different_signature, "same_js_cluster_fraction"
                ),
            }
        ]
    )
    return {
        "assignments": assignments,
        "summary": summary,
        "pair_summary": pair_summary,
        "signature_summary": signature_summary,
        "cluster_summary": cluster_summary,
        "repeated_block_summary": repeated_block_summary,
        "cluster_seconds": cluster_seconds,
    }


def _matched_random_cluster_map(assignments, seed):
    labels = assignments["js_cluster"].to_numpy(copy=True)
    generator = np.random.default_rng(int(seed))
    generator.shuffle(labels)
    return dict(zip(assignments["name"], labels, strict=True))


def _exponent_field_cost(result):
    stream_results = tuple(result.get("stream_results", ()))
    if not stream_results:
        return 0, 0
    exponent_stream_id = max(int(row["stream_id"]) for row in stream_results)
    stream = next(
        row for row in stream_results if int(row["stream_id"]) == exponent_stream_id
    )
    codebook_bytes = sum(
        codebook.metadata_bytes
        for codebook in result.get("codebooks", ())
        if int(codebook.stream_id) == exponent_stream_id
    )
    return int(stream["raw_bytes"]), int(stream["compressed_bytes"] + codebook_bytes)


def cluster_structure_ablation(
    model,
    *,
    cluster_js_threshold=0.05,
    repeats=5,
    random_seed=2026,
    chunk_size=256 << 10,
    compression_threshold=0.95,
    max_codebook_sample_bytes=8 << 20,
):
    """Compare architecture-derived, random, global, and JS cluster maps."""

    if int(repeats) < 1:
        raise ValueError("repeats must be at least 1.")

    from .compressor import Compressor

    alignment = architecture_js_alignment(
        model,
        cluster_js_threshold=cluster_js_threshold,
    )
    assignments = alignment["assignments"]
    names = assignments["name"].tolist()
    cluster_maps = {
        "Global shared": {name: "global" for name in names},
        "Module type": dict(
            zip(assignments["name"], assignments["module_type_group"], strict=True)
        ),
        "Architecture signature": dict(
            zip(
                assignments["name"],
                assignments["architecture_signature"],
                strict=True,
            )
        ),
        "Random matched to JS": _matched_random_cluster_map(
            assignments, random_seed
        ),
        "JS distance": dict(
            zip(assignments["name"], assignments["js_cluster"], strict=True)
        ),
    }

    run_rows = []
    configurations = [("ZipNN local", None), *cluster_maps.items()]
    for repeat in range(1, int(repeats) + 1):
        for strategy, cluster_map in configurations:
            if cluster_map is None:
                compressor = Compressor(
                    "zipnn-local",
                    verify=False,
                    chunk_size=chunk_size,
                    compression_threshold=compression_threshold,
                )
            else:
                compressor = Compressor(
                    "neuralzip",
                    verify=False,
                    cluster_map=cluster_map,
                    chunk_size=chunk_size,
                    compression_threshold=compression_threshold,
                    max_codebook_sample_bytes=max_codebook_sample_bytes,
                    ngram_values=(),
                    packed_source_budget_ratio=0.0,
                )
            result = compressor.compress(model)
            exponent_raw, exponent_compressed = _exponent_field_cost(result)
            run_rows.append(
                {
                    "strategy": strategy,
                    "repeat": repeat,
                    "clusters": (
                        0 if cluster_map is None else len(set(cluster_map.values()))
                    ),
                    "raw_bytes": int(result["raw_bytes"]),
                    "compressed_bytes": int(result["compressed_bytes"]),
                    "full_model_ratio_pct": 100.0
                    * float(result["compression_ratio"]),
                    "exponent_raw_bytes": exponent_raw,
                    "exponent_compressed_bytes": exponent_compressed,
                    "exponent_ratio_pct": (
                        100.0 * exponent_compressed / exponent_raw
                        if exponent_raw
                        else 0.0
                    ),
                    "precompute_seconds": float(
                        result.get("precompute_seconds") or 0.0
                    ),
                    "codebook_setup_seconds": float(
                        result.get("codebook_setup_seconds") or 0.0
                    ),
                    "encode_seconds": float(result["encode_seconds"]),
                    "compression_seconds": float(result["compression_seconds"]),
                    "total_compression_seconds": float(
                        result["total_compression_seconds"]
                    ),
                    "codebook_bytes": int(result.get("codebook_metadata_bytes") or 0),
                }
            )

    verification = {}
    for strategy, cluster_map in configurations:
        if cluster_map is None:
            compressor = Compressor(
                "zipnn-local",
                verify=True,
                chunk_size=chunk_size,
                compression_threshold=compression_threshold,
            )
        else:
            compressor = Compressor(
                "neuralzip",
                verify=True,
                cluster_map=cluster_map,
                chunk_size=chunk_size,
                compression_threshold=compression_threshold,
                max_codebook_sample_bytes=max_codebook_sample_bytes,
                ngram_values=(),
                packed_source_budget_ratio=0.0,
            )
        verification[strategy] = bool(compressor.compress(model)["roundtrip_ok"])

    runs = pd.DataFrame(run_rows)
    summaries = []
    for strategy, group in runs.groupby("strategy", sort=False):
        summaries.append(
            {
                "strategy": strategy,
                "repeats": int(len(group)),
                "clusters": int(group["clusters"].iloc[0]),
                "full_model_ratio_pct": float(group["full_model_ratio_pct"].median()),
                "exponent_ratio_pct": float(group["exponent_ratio_pct"].median()),
                "precompute_seconds": float(
                    group["precompute_seconds"].median()
                ),
                "codebook_setup_seconds": float(
                    group["codebook_setup_seconds"].median()
                ),
                "encode_seconds": float(group["encode_seconds"].median()),
                "compression_seconds": float(
                    group["compression_seconds"].median()
                ),
                "total_compression_seconds": float(
                    group["total_compression_seconds"].median()
                ),
                "encode_iqr_seconds": float(
                    group["encode_seconds"].quantile(0.75)
                    - group["encode_seconds"].quantile(0.25)
                ),
                "codebook_bytes": int(group["codebook_bytes"].median()),
                "roundtrip_ok": verification[strategy],
            }
        )
    compression_summary = pd.DataFrame(summaries)
    alignment["compression_runs"] = runs
    alignment["compression_summary"] = compression_summary
    return alignment


def _iter_chunks(data, chunk_size):
    for start in range(0, len(data), int(chunk_size)):
        yield data[start : start + int(chunk_size)]


def _huffman_lengths_from_counts(counts):
    counts = {
        int(symbol): int(count)
        for symbol, count in enumerate(np.asarray(counts, dtype=np.int64))
        if int(count) > 0
    }
    if not counts:
        return {}
    if len(counts) == 1:
        symbol = next(iter(counts))
        return {symbol: 1}

    order = 0
    heap = []
    for symbol, count in counts.items():
        heap.append((count, order, {symbol: 0}))
        order += 1
    heapq.heapify(heap)

    while len(heap) > 1:
        left_count, _, left_depths = heapq.heappop(heap)
        right_count, _, right_depths = heapq.heappop(heap)
        merged = {}
        for symbol, depth in left_depths.items():
            merged[symbol] = depth + 1
        for symbol, depth in right_depths.items():
            merged[symbol] = depth + 1
        heapq.heappush(heap, (left_count + right_count, order, merged))
        order += 1
    return heap[0][2]


def _huffman_bits_from_counts(counts):
    counts = np.asarray(counts, dtype=np.int64)
    lengths = _huffman_lengths_from_counts(counts)
    return int(sum(int(counts[symbol]) * length for symbol, length in lengths.items()))


def _delta_stream(stream):
    values = np.frombuffer(bytes(stream), dtype=np.uint8)
    if values.size == 0:
        return b""
    delta = np.empty_like(values)
    delta[0] = values[0]
    if values.size > 1:
        delta[1:] = (values[1:].astype(np.int16) - values[:-1].astype(np.int16)) & 0xFF
    return delta.tobytes()


def delta_exponent_diagnostics(
    model,
    *,
    layout="original",
    chunk_size=256 << 10,
):
    exponent_counts = np.zeros(256, dtype=np.int64)
    delta_counts = np.zeros(256, dtype=np.int64)
    raw_bytes = 0
    transitions = 0
    repeated_transitions = 0
    tensors = 0

    for _, tensor in named_floating_tensors(model):
        stream = zipnn_exponent_stream(tensor, layout=layout)
        values = np.frombuffer(bytes(stream), dtype=np.uint8)
        if values.size == 0:
            continue

        tensors += 1
        raw_bytes += int(values.size)
        exponent_counts += np.bincount(values, minlength=256)

        delta = np.frombuffer(_delta_stream(stream), dtype=np.uint8)
        delta_counts += np.bincount(delta, minlength=256)

        if values.size > 1:
            repeated_transitions += int(np.count_nonzero(values[1:] == values[:-1]))
            transitions += int(values.size - 1)

    zipnn = zipnn_exponent_huffman_cost(
        model,
        layout=layout,
        chunk_size=chunk_size,
    ).iloc[0]
    delta = delta_exponent_huffman_cost(
        model,
        layout=layout,
        chunk_size=chunk_size,
    ).iloc[0]

    exponent_h0 = _entropy_from_counts(exponent_counts)
    delta_h0 = _entropy_from_counts(delta_counts)
    same_pct = (
        100.0 * repeated_transitions / transitions
        if transitions
        else 0.0
    )

    return pd.DataFrame(
        [
            {
                "method": "Delta exponent diagnostic",
                "layout": layout,
                "raw_bytes": raw_bytes,
                "tensors": tensors,
                "transitions": transitions,
                "same_exponent_transition_pct": same_pct,
                "exponent_h0_bits": exponent_h0,
                "delta_h0_bits": delta_h0,
                "delta_h0_minus_exponent_h0_bits": delta_h0 - exponent_h0,
                "ideal_exponent_pct": 100.0 * exponent_h0 / 8.0,
                "ideal_delta_pct": 100.0 * delta_h0 / 8.0,
                "zipnn_compressed_pct": float(zipnn["compressed_pct"]),
                "delta_compressed_pct": float(delta["compressed_pct"]),
                "delta_gain_vs_zipnn_pp": (
                    float(zipnn["compressed_pct"]) - float(delta["compressed_pct"])
                ),
                "zipnn_compressed_bytes": int(zipnn["compressed_bytes"]),
                "delta_compressed_bytes": int(delta["compressed_bytes"]),
                "delta_metadata_bytes": int(delta["metadata_bytes"]),
                "chunks": int(delta["chunks"]),
                "delta_huffman_chunks": int(delta["huffman_chunks"]),
                "delta_raw_chunks": int(delta["raw_chunks"]),
            }
        ]
    )


def delta_exponent_huffman_cost(
    model,
    *,
    layout="original",
    chunk_size=256 << 10,
):
    raw_bytes = 0
    compressed_bytes = 0
    payload_bytes = 0
    chunks = 0
    huffman_chunks = 0
    raw_chunks = 0
    codec = ZipNNCodec(
        chunk_size=chunk_size,
        compression_threshold=1.0,
    )

    stream_specs = []
    for _, tensor in named_floating_tensors(model):
        stream = zipnn_exponent_stream(tensor, layout=layout)
        raw_bytes += len(stream)
        delta = _delta_stream(stream)
        stream_specs.append((delta, "delta", chunk_size))

    for group in codec._compress_streams(stream_specs):
        compressed_bytes += sum(chunk.compressed_bytes for chunk in group.chunks)
        payload_bytes += group.payload_bytes
        chunks += len(group.chunks)
        huffman_chunks += group.local_huff0_chunks
        raw_chunks += group.raw_chunks

    return pd.DataFrame(
        [
            {
                "method": "Delta exponent Huffman",
                "layout": layout,
                "raw_bytes": raw_bytes,
                "payload_bytes": payload_bytes,
                "metadata_bytes": compressed_bytes - payload_bytes,
                "compressed_bytes": compressed_bytes,
                "compressed_pct": 100 * compressed_bytes / raw_bytes if raw_bytes else 0.0,
                "chunks": chunks,
                "huffman_chunks": huffman_chunks,
                "raw_chunks": raw_chunks,
            }
        ]
    )


def zipnn_exponent_huffman_cost(
    model,
    *,
    layout="original",
    chunk_size=256 << 10,
    compression_threshold=None,
):
    precompute_start = time.perf_counter()
    codec = ZipNNCodec(
        chunk_size=chunk_size,
        compression_threshold=(
            1.0 if compression_threshold is None else compression_threshold
        ),
    )
    rows = []
    raw_bytes = 0
    compressed_bytes = 0
    chunks = 0
    local_huff0_chunks = 0
    raw_chunks = 0
    stream_specs = []
    for _, tensor in named_floating_tensors(model):
        stream = zipnn_exponent_stream(tensor, layout=layout)
        raw_bytes += len(stream)
        stream_specs.append(
            (
                stream,
                "exponent",
                codec._stream_chunk_size(tensor.element_size()),
            )
        )

    # Submit every exponent block through the same native Huff0 worker pool.
    precompute_seconds = time.perf_counter() - precompute_start
    encode_start = time.perf_counter()
    for group in codec._compress_streams(stream_specs):
        compressed_bytes += sum(chunk.compressed_bytes for chunk in group.chunks)
        chunks += len(group.chunks)
        local_huff0_chunks += group.local_huff0_chunks
        raw_chunks += group.raw_chunks

    encode_seconds = time.perf_counter() - encode_start

    rows.append(
        {
            "method": "ZipNN local Huffman",
            "layout": layout,
            "raw_bytes": raw_bytes,
            "compressed_bytes": compressed_bytes,
            "compressed_pct": 100 * compressed_bytes / raw_bytes if raw_bytes else 0.0,
            "chunks": chunks,
            "local_huff0_chunks": local_huff0_chunks,
            "raw_chunks": raw_chunks,
            "precompute_seconds": precompute_seconds,
            "encode_seconds": encode_seconds,
        }
    )
    return pd.DataFrame(rows)


def _clustered_zipnn_exponent_streams(model, *, cluster_js_threshold, layout):
    tensors = list(named_floating_tensors(model))
    codec = NeuralZipCodec(
        clustering="js-distance",
        cluster_js_threshold=float(cluster_js_threshold),
    )
    codec._prepare_clusters(tensors)
    streams_by_cluster = {}
    tensor_streams = []
    raw_bytes = 0
    for name, tensor in tensors:
        stream = zipnn_exponent_stream(tensor, layout=layout)
        cluster_id = codec.cluster_id(name)
        streams_by_cluster.setdefault(cluster_id, []).append(stream)
        tensor_streams.append((cluster_id, stream))
        raw_bytes += len(stream)
    return streams_by_cluster, tensor_streams, raw_bytes


def order1_huffman_cluster_cost(
    model,
    *,
    cluster_js_threshold=0.05,
    layout="original",
):
    streams_by_cluster, _, raw_bytes = _clustered_zipnn_exponent_streams(
        model,
        cluster_js_threshold=cluster_js_threshold,
        layout=layout,
    )
    payload_bits = 0
    first_symbol_bytes = 0
    codebook_bytes = 0
    stream_headers = 0
    nonempty_contexts = 0

    for streams in streams_by_cluster.values():
        counts = np.zeros((256, 256), dtype=np.int64)
        for stream in streams:
            values = np.frombuffer(bytes(stream), dtype=np.uint8)
            if values.size == 0:
                continue
            first_symbol_bytes += 1
            stream_headers += RAW_CHUNK_METADATA_BYTES
            if values.size > 1:
                np.add.at(counts, (values[:-1], values[1:]), 1)

        for context in range(256):
            row = counts[context]
            if int(row.sum()) == 0:
                continue
            nonempty_contexts += 1
            payload_bits += _huffman_bits_from_counts(row)
            codebook_bytes += 1 + 256

    codec_header_bytes = 8 + len("order1-huffman".encode("utf-8")) + 8 + len(layout)
    cluster_header_bytes = len(streams_by_cluster) * (8 + 8)
    payload_bytes = math.ceil(payload_bits / 8) + first_symbol_bytes
    metadata_bytes = codec_header_bytes + cluster_header_bytes + stream_headers + codebook_bytes
    compressed_bytes = payload_bytes + metadata_bytes

    return pd.DataFrame(
        [
            {
                "method": "Order-1 clustered Huffman",
                "cluster_js_threshold": float(cluster_js_threshold),
                "layout": layout,
                "clusters": len(streams_by_cluster),
                "raw_bytes": raw_bytes,
                "payload_bytes": payload_bytes,
                "metadata_bytes": metadata_bytes,
                "codebook_bytes": codebook_bytes,
                "compressed_bytes": compressed_bytes,
                "compressed_pct": 100 * compressed_bytes / raw_bytes if raw_bytes else 0.0,
                "nonempty_contexts": nonempty_contexts,
            }
        ]
    )


def _ngram_counts_and_codebook(data, n):
    n = int(n)
    complete_length = (len(data) // n) * n
    if complete_length == 0:
        return 0, 0, 0.0, b"", len(data)
    values = np.frombuffer(data[:complete_length], dtype=np.dtype(f"V{n}"))
    symbols, counts = np.unique(values, return_counts=True)
    token_count = int(counts.sum())
    probabilities = counts.astype(np.float64) / token_count
    entropy = float(-(probabilities * np.log2(probabilities)).sum())
    codebook = bytearray()
    codebook.append(n)
    codebook.extend(int(len(symbols)).to_bytes(4, "little"))
    codebook.extend(symbols.tobytes())
    codebook.extend(counts.astype("<u4", copy=False).tobytes())
    return int(len(symbols)), token_count, entropy, bytes(codebook), len(data) - complete_length

def _compressed_codebook_size(codebook, compressor):
    if compressor == "raw":
        return len(codebook)
    if compressor == "zlib":
        return len(zlib.compress(codebook, 9))
    if compressor == "lzma":
        return len(lzma.compress(codebook, preset=9))
    raise ValueError("compressor must be 'raw', 'zlib', or 'lzma'.")

def _aligned_ngram_training_data(streams, n):
    n = int(n)
    parts = []
    tail_bytes = 0
    for stream in streams:
        complete_length = (len(stream) // n) * n
        if complete_length:
            parts.append(stream[:complete_length])
        tail_bytes += len(stream) - complete_length
    return b"".join(parts), tail_bytes

def ngram_codebook_boundary(
    model,
    *,
    cluster_js_threshold=0.2,
    n_values=(4, 5, 6, 8),
    layout="transpose2d",
    codebook_compressors=("raw", "zlib", "lzma"),
):
    streams_by_cluster, _, raw_bytes = _clustered_zipnn_exponent_streams(
        model,
        cluster_js_threshold=cluster_js_threshold,
        layout=layout,
    )
    rows = []
    for n in n_values:
        cluster_stats = []
        payload_bits = 0.0
        tail_bytes = 0
        total_unique = 0
        codebooks = []
        for cluster_id, streams in streams_by_cluster.items():
            data, stream_tail = _aligned_ngram_training_data(streams, n)
            if not data:
                tail_bytes += stream_tail
                continue
            (
                unique,
                token_count,
                entropy,
                codebook,
                aligned_tail,
            ) = _ngram_counts_and_codebook(data, n)
            payload_bits += token_count * entropy
            tail_bytes += stream_tail + aligned_tail
            total_unique += unique
            codebooks.append(codebook)
            cluster_stats.append((cluster_id, unique, token_count))

        chunk_headers = len(streams_by_cluster) * (1 + 8 + 8 + 8 + 8 + 8)
        payload_bytes = math.ceil(payload_bits / 8) + tail_bytes + chunk_headers
        for compressor in codebook_compressors:
            codebook_bytes = sum(
                _compressed_codebook_size(codebook, compressor)
                for codebook in codebooks
            )
            compressed_bytes = payload_bytes + codebook_bytes
            rows.append(
                {
                    "method": "clustered n-gram arithmetic",
                    "cluster_js_threshold": float(cluster_js_threshold),
                    "layout": layout,
                    "n": int(n),
                    "codebook_compressor": compressor,
                    "clusters": len(streams_by_cluster),
                    "unique_ngrams": total_unique,
                    "raw_bytes": raw_bytes,
                    "payload_bytes": payload_bytes,
                    "codebook_bytes": codebook_bytes,
                    "compressed_bytes": compressed_bytes,
                    "payload_pct": 100 * payload_bytes / raw_bytes,
                    "codebook_pct": 100 * codebook_bytes / raw_bytes,
                    "compressed_pct": 100 * compressed_bytes / raw_bytes,
                }
            )
    return pd.DataFrame(rows)

def selected_ngram_codebook_cost(
    model,
    *,
    cluster_js_thresholds=(0.2, 0.1, 0.05, 0.03, 0.02),
    n_values=(2, 3, 4),
    layout="transpose2d",
):
    rows = []
    for threshold in cluster_js_thresholds:
        streams_by_cluster, _, raw_bytes = _clustered_zipnn_exponent_streams(
            model,
            cluster_js_threshold=threshold,
            layout=layout,
        )
        total_bytes = 0
        codebook_bytes = 0
        selected = {}
        encoded_tensors = 0
        for cluster_id, streams in streams_by_cluster.items():
            candidates = [
                ("zipnn-local", _zipnn_local_stream_cost(streams), 0, 0)
            ]
            for n in n_values:
                data, _ = _aligned_ngram_training_data(streams, n)
                if not data:
                    continue
                try:
                    codebook = huffman.build_sparse_ngram_arithmetic_codebook(
                        data,
                        n=int(n),
                        max_total=1 << 26,
                    )
                except ValueError:
                    continue
                payload_cost = 0
                encoded_count = 0
                for stream in streams:
                    encoded = huffman.sparse_ngram_arithmetic_encode_with_codebook(
                        stream,
                        codebook,
                        min_saving_bytes=0,
                    )
                    if encoded is None:
                        payload_cost += RAW_CHUNK_METADATA_BYTES + len(stream)
                        continue
                    payload_cost += (
                        1
                        + 8
                        + 8
                        + 8
                        + 8
                        + 8
                        + len(encoded.payload)
                        + len(encoded.tail)
                    )
                    encoded_count += 1
                header = (
                    8
                    + len(str(cluster_id).encode("utf-8"))
                    + 8
                    + len("bfloat16")
                    + 1
                    + 1
                    + 8
                )
                candidates.append(
                    (
                        f"n={int(n)}",
                        payload_cost + len(codebook.payload) + header,
                        len(codebook.payload),
                        encoded_count,
                    )
                )
            mode, cost, mode_codebook_bytes, mode_encoded = min(
                candidates,
                key=lambda candidate: candidate[1],
            )
            total_bytes += cost
            codebook_bytes += mode_codebook_bytes
            encoded_tensors += mode_encoded
            selected[mode] = selected.get(mode, 0) + 1

        rows.append(
            {
                "cluster_js_threshold": float(threshold),
                "layout": layout,
                "clusters": len(streams_by_cluster),
                "raw_bytes": raw_bytes,
                "compressed_bytes": total_bytes,
                "compressed_pct": 100 * total_bytes / raw_bytes,
                "codebook_bytes": codebook_bytes,
                "encoded_tensors": encoded_tensors,
                "selected_modes": selected,
            }
        )
    return pd.DataFrame(rows)

def _zipnn_local_stream_cost(streams):
    codec = ZipNNCodec(compression_threshold=1.0)
    groups = codec._compress_streams(
        (stream, "exponent", None) for stream in streams
    )
    return sum(group.compressed_bytes for group in groups)


def neuralzip_exponent_cost(
    model,
    *,
    enable_shared_codebooks=True,
    enable_packed_exponents=True,
    cluster_js_threshold=0.05,
    n_values=(2, 3, 4),
    layout="original",
    chunk_size=256 << 10,
    top_k=254,
    max_sample_bytes=8 << 20,
    compression_threshold=0.95,
    packed_source_budget_ratio=0.30,
):
    named_tensors = list(named_floating_tensors(model))
    cluster_map = None
    if not enable_shared_codebooks:
        cluster_map = {
            name: f"tensor_{index:06d}"
            for index, (name, _) in enumerate(named_tensors)
        }
    codec = NeuralZipCodec(
        chunk_size=chunk_size,
        compression_threshold=compression_threshold,
        clustering="js-distance",
        cluster_map=cluster_map,
        cluster_js_threshold=cluster_js_threshold,
        max_codebook_sample_bytes=max_sample_bytes,
        ngram_values=tuple(n_values) if enable_packed_exponents else (),
        top_k=top_k,
        exponent_layout=layout,
        packed_source_budget_ratio=(
            packed_source_budget_ratio if enable_packed_exponents else 0.0
        ),
        enable_shared_codebooks=enable_shared_codebooks,
    )

    precompute_start = time.perf_counter()
    codec.prepare_clusters(named_tensors)
    precompute_seconds = time.perf_counter() - precompute_start

    codebook_start = time.perf_counter()
    codec.build_codebooks_and_select_modes(named_tensors)
    codebook_setup_seconds = time.perf_counter() - codebook_start

    raw_bytes = 0
    exponent_keys = set()
    groups = []
    encode_start = time.perf_counter()
    for name, tensor in named_tensors:
        original = tensor.detach().cpu().contiguous()
        config = _zipnn_dtype_config(original.dtype)
        if not config["bit_reorder"]:
            continue
        reordered = _zipnn_reordered_bytes(original, config)
        element_size = int(config["element_size"])
        stream_id = element_size - 1
        stream = reordered[stream_id::element_size]
        stream, _ = codec._apply_exponent_layout(original, stream, True)
        raw_bytes += len(stream)
        exponent_keys.add(codec._codebook_key(name, original, stream_id))
        groups.append(
            codec._compress_cluster_stream(
                name,
                original,
                stream,
                "exponent",
                stream_id,
            )
        )
    encode_seconds = time.perf_counter() - encode_start

    codebooks = tuple(
        payload
        for payload in codec.export_codebooks()
        if (payload.cluster_id, payload.dtype, int(payload.stream_id))
        in exponent_keys
    )
    codebook_bytes = sum(payload.metadata_bytes for payload in codebooks)
    compressed_bytes = codebook_bytes + sum(
        chunk.compressed_bytes for group in groups for chunk in group.chunks
    )
    selected_modes = {}
    for key in exponent_keys:
        mode = codec._mode_by_key.get(key)
        if mode is None:
            continue
        kind = str(mode["kind"])
        if kind == "packed_huff0":
            kind = f"packed_huff0_n{int(mode['ngram_n'])}"
        selected_modes[kind] = selected_modes.get(kind, 0) + 1

    if enable_shared_codebooks and enable_packed_exponents:
        method = "NeuralZip"
    elif enable_shared_codebooks:
        method = "Sharing only"
    elif enable_packed_exponents:
        method = "Packing only"
    else:
        method = "Local Huffman"

    return pd.DataFrame(
        [
            {
                "method": method,
                "layout": layout,
                "clusters": codec.cluster_count,
                "raw_bytes": raw_bytes,
                "compressed_bytes": compressed_bytes,
                "compressed_pct": (
                    100.0 * compressed_bytes / raw_bytes if raw_bytes else 0.0
                ),
                "codebook_bytes": codebook_bytes,
                "codebooks": len(codebooks),
                "selected_modes": selected_modes,
                "packed_ngram_chunks": sum(
                    group.packed_ngram_chunks for group in groups
                ),
                "shared_huff0_chunks": sum(
                    group.shared_huff0_chunks for group in groups
                ),
                "local_huff0_chunks": sum(
                    group.local_huff0_chunks for group in groups
                ),
                "raw_chunks": sum(group.raw_chunks for group in groups),
                "precompute_seconds": precompute_seconds,
                "codebook_setup_seconds": codebook_setup_seconds,
                "encode_seconds": encode_seconds,
                "compression_seconds": codebook_setup_seconds + encode_seconds,
                "total_compression_seconds": (
                    precompute_seconds + codebook_setup_seconds + encode_seconds
                ),
            }
        ]
    )


def neuralzip_component_ablation(
    model,
    *,
    repeats=5,
    cluster_js_threshold=0.05,
    n_values=(2, 3, 4),
    layout="original",
    chunk_size=256 << 10,
    top_k=254,
    max_sample_bytes=8 << 20,
    compression_threshold=0.95,
    packed_source_budget_ratio=0.30,
):
    if int(repeats) < 1:
        raise ValueError("repeats must be at least 1.")

    runs = []
    for repeat in range(1, int(repeats) + 1):
        zipnn = zipnn_exponent_huffman_cost(
            model,
            layout=layout,
            chunk_size=chunk_size,
            compression_threshold=compression_threshold,
        ).iloc[0].to_dict()
        sharing_only = neuralzip_exponent_cost(
            model,
            enable_shared_codebooks=True,
            enable_packed_exponents=False,
            cluster_js_threshold=cluster_js_threshold,
            n_values=n_values,
            layout=layout,
            chunk_size=chunk_size,
            top_k=top_k,
            max_sample_bytes=max_sample_bytes,
            compression_threshold=compression_threshold,
            packed_source_budget_ratio=packed_source_budget_ratio,
        ).iloc[0].to_dict()
        packing_only = neuralzip_exponent_cost(
            model,
            enable_shared_codebooks=False,
            enable_packed_exponents=True,
            cluster_js_threshold=cluster_js_threshold,
            n_values=n_values,
            layout=layout,
            chunk_size=chunk_size,
            top_k=top_k,
            max_sample_bytes=max_sample_bytes,
            compression_threshold=compression_threshold,
            packed_source_budget_ratio=packed_source_budget_ratio,
        ).iloc[0].to_dict()
        neuralzip = neuralzip_exponent_cost(
            model,
            enable_shared_codebooks=True,
            enable_packed_exponents=True,
            cluster_js_threshold=cluster_js_threshold,
            n_values=n_values,
            layout=layout,
            chunk_size=chunk_size,
            top_k=top_k,
            max_sample_bytes=max_sample_bytes,
            compression_threshold=compression_threshold,
            packed_source_budget_ratio=packed_source_budget_ratio,
        ).iloc[0].to_dict()

        for variant, row in (
            ("ZipNN", zipnn),
            ("Sharing only", sharing_only),
            ("Packing only", packing_only),
            ("NeuralZip", neuralzip),
        ):
            runs.append(
                {
                    "variant": variant,
                    "repeat": repeat,
                    "raw_bytes": int(row["raw_bytes"]),
                    "compressed_bytes": int(row["compressed_bytes"]),
                    "compression_pct": float(row["compressed_pct"]),
                    "compression_seconds": float(
                        row.get("compression_seconds", row["encode_seconds"])
                    ),
                    "precompute_seconds": float(
                        row.get("precompute_seconds", 0.0)
                    ),
                    "codebook_setup_seconds": float(
                        row.get("codebook_setup_seconds", 0.0)
                    ),
                    "encode_seconds": float(row["encode_seconds"]),
                    "clusters": int(row.get("clusters", 0)),
                    "codebook_bytes": int(row.get("codebook_bytes", 0)),
                    "selected_modes": row.get("selected_modes", {}),
                    "packed_ngram_chunks": int(
                        row.get("packed_ngram_chunks", 0)
                    ),
                    "local_huff0_chunks": int(
                        row.get("local_huff0_chunks", 0)
                    ),
                }
            )

    runs = pd.DataFrame(runs)
    summaries = []
    for variant in ("ZipNN", "Sharing only", "Packing only", "NeuralZip"):
        group = runs[runs["variant"] == variant]
        first = group.iloc[0]
        summaries.append(
            {
                "variant": variant,
                "repeats": len(group),
                "raw_bytes": int(first["raw_bytes"]),
                "compressed_bytes": int(first["compressed_bytes"]),
                "compression_pct": float(group["compression_pct"].median()),
                "compression_seconds": float(
                    group["compression_seconds"].median()
                ),
                "precompute_seconds": float(
                    group["precompute_seconds"].median()
                ),
                "codebook_setup_seconds": float(
                    group["codebook_setup_seconds"].median()
                ),
                "encode_seconds": float(group["encode_seconds"].median()),
                "clusters": int(first["clusters"]),
                "codebook_bytes": int(first["codebook_bytes"]),
                "packed_ngram_chunks": int(first["packed_ngram_chunks"]),
                "local_huff0_chunks": int(first["local_huff0_chunks"]),
                "selected_modes": first["selected_modes"],
            }
        )
    return runs, pd.DataFrame(summaries)


def selected_turborc_range_cost(
    model,
    *,
    cluster_js_thresholds=(0.2, 0.1, 0.05, 0.03, 0.02),
    layout="transpose2d",
    chunk_size=256 << 10,
    method="rccs",
):
    rows = []
    method = turborc._normalize_method(method)
    method_label = f"turborc-{method}"
    chunk_header_bytes = RAW_CHUNK_METADATA_BYTES
    codec_header_bytes = 8 + len(method_label.encode("utf-8")) + 8 + len(layout)

    for threshold in cluster_js_thresholds:
        streams_by_cluster, _, raw_bytes = _clustered_zipnn_exponent_streams(
            model,
            cluster_js_threshold=threshold,
            layout=layout,
        )
        start = time.perf_counter()
        total_bytes = codec_header_bytes
        payload_bytes = 0
        chunks = 0
        compressed_chunks = 0
        raw_chunks = 0

        for streams in streams_by_cluster.values():
            for stream in streams:
                for chunk in _iter_chunks(stream, chunk_size):
                    chunks += 1
                    raw_cost = chunk_header_bytes + len(chunk)
                    encoded = turborc.compress(
                        chunk,
                        method=method,
                        min_saving_bytes=0,
                    )
                    if encoded is None:
                        total_bytes += raw_cost
                        payload_bytes += len(chunk)
                        raw_chunks += 1
                        continue

                    encoded_cost = chunk_header_bytes + len(encoded.payload)
                    if encoded_cost >= raw_cost:
                        total_bytes += raw_cost
                        payload_bytes += len(chunk)
                        raw_chunks += 1
                        continue

                    total_bytes += encoded_cost
                    payload_bytes += len(encoded.payload)
                    compressed_chunks += 1

        encode_seconds = time.perf_counter() - start
        metadata_bytes = total_bytes - payload_bytes
        rows.append(
            {
                "method": method_label,
                "cluster_js_threshold": float(threshold),
                "layout": layout,
                "clusters": len(streams_by_cluster),
                "raw_bytes": raw_bytes,
                "payload_bytes": payload_bytes,
                "metadata_bytes": metadata_bytes,
                "codebook_bytes": codec_header_bytes,
                "compressed_bytes": total_bytes,
                "compression_ratio": total_bytes / raw_bytes if raw_bytes else 0.0,
                "compressed_pct": 100 * total_bytes / raw_bytes if raw_bytes else 0.0,
                "chunks": chunks,
                "compressed_chunks": compressed_chunks,
                "raw_chunks": raw_chunks,
                "encoded_tensors": compressed_chunks,
                "selected_modes": {
                    method_label: compressed_chunks,
                    "raw": raw_chunks,
                },
                "encode_seconds": encode_seconds,
            }
        )
    return pd.DataFrame(rows)
