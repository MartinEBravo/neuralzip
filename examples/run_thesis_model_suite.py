from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    resnet18,
    resnet34,
)
from transformers import AutoModel

from neuralzip import Compressor
from neuralzip.ablations import (
    delta_exponent_huffman_cost,
    order1_huffman_cluster_cost,
    selected_turborc_range_cost,
    zipnn_exponent_stream,
)
from neuralzip.codec_protocol import measure_compressor_exponents
from neuralzip.metrics import (
    conditional_entropy_from_counts as first_order_entropy_from_counts,
)
from neuralzip.metrics import (
    jensen_shannon_distance as jensen_shannon,
)
from neuralzip.metrics import (
    shannon_entropy_from_counts as entropy_from_counts,
)
from neuralzip.zipnn_source import parse_zipnn_exponent_stream

try:
    from .model_catalog import SPECS_BY_KEY, ModelSpec
except ImportError:  # Direct execution from the examples directory.
    from model_catalog import SPECS_BY_KEY, ModelSpec


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "results" / "model-cache"
OUTPUT_DIR = ROOT / "results" / "thesis-model-suite"
FIGURE_DIR = OUTPUT_DIR / "figures"
NGRAM_SIZES = (1, 2, 4, 8)
NGRAM_SAMPLE_EXPONENTS = 8 << 20
MAX_HEATMAP_TENSORS = 48
JS_THRESHOLD = 0.05
FIELD_CHUNK_ELEMENTS = 4 << 20


PHENOMENON_KEYS = ("gpt2", "bert-base", "resnet18", "resnet34")
MODEL_SPECS = tuple(SPECS_BY_KEY[key] for key in PHENOMENON_KEYS)
SPECS_BY_SLUG = {spec.key: spec for spec in MODEL_SPECS}
LOCAL_MODEL_DIRECTORIES = {"bert-base": "bert-base-uncased"}


class ModelView:
    def __init__(self, spec: ModelSpec, module: torch.nn.Module):
        self.spec = spec
        self.model_name = spec.source
        self.model = module


def load_model(
    spec: ModelSpec,
    *,
    model_root: Path = MODEL_ROOT,
    allow_download: bool = False,
) -> ModelView:
    if spec.backend == "hf":
        local_path = model_root / LOCAL_MODEL_DIRECTORIES.get(spec.key, spec.key)
        source = local_path if (local_path / "config.json").is_file() else spec.source
        module = AutoModel.from_pretrained(
            source,
            dtype=torch.bfloat16,
            cache_dir=model_root,
            local_files_only=not allow_download,
        )
    elif spec.key == "resnet18":
        module = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(
            dtype=torch.bfloat16
        )
    elif spec.key == "resnet34":
        module = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1).to(
            dtype=torch.bfloat16
        )
    else:
        raise ValueError(f"Unsupported phenomenon model: {spec.key}")

    module.eval()
    module.requires_grad_(False)
    dtypes = {parameter.dtype for parameter in module.parameters()}
    if dtypes != {torch.bfloat16}:
        raise RuntimeError(f"{spec.label} is not uniformly BF16: {dtypes}")
    return ModelView(spec, module)


def floating_parameters(view: ModelView):
    return [
        (name, parameter)
        for name, parameter in view.model.named_parameters()
        if parameter.is_floating_point()
    ]


def bf16_fields(parameter: torch.Tensor, start: int, stop: int):
    raw = parameter.detach().cpu().contiguous().view(torch.uint8).reshape(-1).numpy()
    low = raw[2 * start : 2 * stop : 2]
    high = raw[2 * start + 1 : 2 * stop : 2]
    sign = high >> np.uint8(7)
    exponent = (low >> np.uint8(7)) | ((high & np.uint8(0x7F)) << np.uint8(1))
    mantissa = low & np.uint8(0x7F)
    return sign, exponent, mantissa


def sample_allocations(parameters, budget):
    sizes = np.asarray(
        [int(parameter.numel()) for _, parameter in parameters], dtype=np.int64
    )
    total = int(sizes.sum())
    target = min(int(budget), total)
    if target == 0:
        return np.zeros_like(sizes)
    exact = sizes.astype(np.float64) * (float(target) / float(total))
    allocations = np.floor(exact).astype(np.int64)
    allocations = np.minimum(allocations, sizes)
    remaining = target - int(allocations.sum())
    if remaining:
        order = np.argsort(-(exact - allocations), kind="stable")
        for index in order:
            if remaining == 0:
                break
            if allocations[index] < sizes[index]:
                allocations[index] += 1
                remaining -= 1
    return allocations


def sampled_exponent_stream(name, parameter, allocation):
    allocation = int(allocation)
    size = int(parameter.numel())
    if allocation <= 0 or size == 0:
        return np.empty(0, dtype=np.uint8)
    if allocation >= size:
        start = 0
    else:
        digest = hashlib.sha256(name.encode("utf-8")).digest()
        start = int.from_bytes(digest[:8], "little") % (size - allocation + 1)
    _, exponent, _ = bf16_fields(parameter, start, start + allocation)
    return exponent.copy()


def pairwise_js_matrix(histograms):
    matrix = np.zeros((len(histograms), len(histograms)), dtype=np.float64)
    for left in range(len(histograms)):
        for right in range(left):
            distance = jensen_shannon(histograms[left], histograms[right])
            matrix[left, right] = distance
            matrix[right, left] = distance
    return matrix


def greedy_cluster_order(histograms, threshold):
    clusters = []
    centroids = []
    for index, histogram in enumerate(histograms):
        assigned = False
        for cluster_index, centroid in enumerate(centroids):
            if jensen_shannon(histogram, centroid) <= threshold:
                clusters[cluster_index].append(index)
                centroids[cluster_index] = np.mean(
                    [histograms[item] for item in clusters[cluster_index]], axis=0
                )
                assigned = True
                break
        if not assigned:
            clusters.append([index])
            centroids.append(histogram.copy())
    return [index for cluster in clusters for index in cluster], clusters


def run_diagnostics(view: ModelView):
    parameters = floating_parameters(view)
    allocations = sample_allocations(parameters, NGRAM_SAMPLE_EXPONENTS)
    sign_counts = np.zeros(2, dtype=np.int64)
    exponent_counts = np.zeros(256, dtype=np.int64)
    mantissa_counts = np.zeros(128, dtype=np.int64)
    transition_counts = np.zeros((256, 256), dtype=np.int64)
    tensor_rows = []
    histogram_counts = []
    sampled_streams = []

    start_time = time.perf_counter()
    for tensor_index, ((name, parameter), allocation) in enumerate(
        zip(parameters, allocations), start=1
    ):
        size = int(parameter.numel())
        local_counts = np.zeros(256, dtype=np.int64)
        previous = None
        for start in range(0, size, FIELD_CHUNK_ELEMENTS):
            stop = min(size, start + FIELD_CHUNK_ELEMENTS)
            sign, exponent, mantissa = bf16_fields(parameter, start, stop)
            sign_counts += np.bincount(sign, minlength=2)
            local = np.bincount(exponent, minlength=256)
            exponent_counts += local
            local_counts += local
            mantissa_counts += np.bincount(mantissa, minlength=128)

            if exponent.size:
                if previous is not None:
                    transition_counts[int(previous), int(exponent[0])] += 1
                if exponent.size > 1:
                    pair_ids = (
                        exponent[:-1].astype(np.uint16) << np.uint16(8)
                    ) | exponent[1:].astype(np.uint16)
                    transition_counts += np.bincount(
                        pair_ids, minlength=256 * 256
                    ).reshape(256, 256)
                previous = int(exponent[-1])

        distribution = local_counts.astype(np.float64)
        distribution /= distribution.sum() if distribution.sum() else 1.0
        histogram_counts.append(local_counts)
        tensor_rows.append(
            {
                "tensor_index": tensor_index - 1,
                "name": name,
                "numel": size,
                "exponent_entropy_bits": entropy_from_counts(local_counts),
                "dominant_exponent": int(np.argmax(local_counts)),
                "dominant_exponent_mass": float(distribution.max(initial=0.0)),
            }
        )
        sample = sampled_exponent_stream(name, parameter, allocation)
        if sample.size:
            sampled_streams.append(sample)
        if (
            tensor_index == 1
            or tensor_index % 25 == 0
            or tensor_index == len(parameters)
        ):
            print(
                f"[{view.spec.key}] diagnostics {tensor_index}/{len(parameters)} "
                f"{name} ({size:,})",
                flush=True,
            )

    ngram_rows = []
    sampled_exponents = sum(int(stream.size) for stream in sampled_streams)
    for n in NGRAM_SIZES:
        token_arrays = []
        tails = 0
        for stream in sampled_streams:
            complete = (stream.size // n) * n
            tails += int(stream.size - complete)
            if complete:
                token_arrays.append(
                    np.ascontiguousarray(stream[:complete]).view(np.dtype(f"V{n}"))
                )
        if token_arrays:
            tokens = np.concatenate(token_arrays)
            _, counts = np.unique(tokens, return_counts=True)
        else:
            counts = np.empty(0, dtype=np.int64)
        entropy_per_ngram = entropy_from_counts(counts)
        token_count = int(counts.sum())
        ideal_bits = token_count * entropy_per_ngram + tails * 8
        ngram_rows.append(
            {
                "n": n,
                "entropy_per_ngram": entropy_per_ngram,
                "bits_per_exponent": (
                    ideal_bits / sampled_exponents if sampled_exponents else 0.0
                ),
                "sampled_exponents": sampled_exponents,
                "tokens": token_count,
                "tails": tails,
                "unique_ngrams": len(counts),
            }
        )
        del token_arrays
        gc.collect()

    h0 = entropy_from_counts(exponent_counts)
    h1 = first_order_entropy_from_counts(transition_counts)
    metadata = pd.DataFrame(tensor_rows)
    selected = sorted(
        metadata.nlargest(MAX_HEATMAP_TENSORS, "numel")["tensor_index"].tolist()
    )
    selected_names = metadata.set_index("tensor_index").loc[selected, "name"].tolist()
    selected_histograms = np.asarray(
        [histogram_counts[index] / histogram_counts[index].sum() for index in selected]
    )
    order, clusters = greedy_cluster_order(selected_histograms, JS_THRESHOLD)
    matrix = pairwise_js_matrix(selected_histograms)
    matrix = matrix[np.ix_(order, order)]
    selected_names = [selected_names[index] for index in order]

    model_dir = OUTPUT_DIR / view.spec.key
    model_dir.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(model_dir / "tensor_exponent_summary.csv", index=False)
    pd.DataFrame(ngram_rows).to_csv(model_dir / "ngram_entropy.csv", index=False)
    np.savez_compressed(
        model_dir / "pairwise_js_heatmap.npz",
        matrix=matrix,
        names=np.asarray(selected_names, dtype=str),
    )
    result = {
        "model": view.spec.label,
        "source": view.spec.source,
        "dtype": "bfloat16",
        "parameters": int(sum(parameter.numel() for _, parameter in parameters)),
        "tensors": len(parameters),
        "field_entropy": {
            "sign": entropy_from_counts(sign_counts),
            "exponent": h0,
            "mantissa": entropy_from_counts(mantissa_counts),
        },
        "order_entropy": {
            "h0_bits_per_exponent": h0,
            "h1_bits_per_exponent": h1,
            "entropy_reduction_bits_per_exponent": h0 - h1,
            "entropy_reduction_pct": 100.0 * (h0 - h1) / h0 if h0 else 0.0,
            "transitions": int(transition_counts.sum()),
        },
        "ngram_sample_exponents": sampled_exponents,
        "ngram_sizes": list(NGRAM_SIZES),
        "heatmap_tensors": len(selected),
        "heatmap_clusters": len(clusters),
        "heatmap_js_threshold": JS_THRESHOLD,
        "seconds": time.perf_counter() - start_time,
    }
    write_json(model_dir / "diagnostics.json", result)
    return result


def component_exponent_metrics(result):
    report = measure_compressor_exponents(result, dtypes={"bfloat16"})
    return {
        "raw_bytes": report.raw_bytes,
        "compressed_bytes": report.compressed_bytes,
        "codebook_bytes": report.codebook_bytes,
        "exponent_ratio_pct": report.ratio_pct,
    }


def production_exponent_measure(
    view: ModelView,
    method: str,
    threads: int,
    **options,
):
    warmup = Compressor(
        method,
        verify=False,
        keep_payloads=False,
        threads=threads,
        compression_threshold=0.95,
        **options,
    ).compress(view)
    del warmup
    gc.collect()

    result = Compressor(
        method,
        verify=True,
        keep_payloads=True,
        threads=threads,
        compression_threshold=0.95,
        **options,
    ).compress(view)
    if result["roundtrip_ok"] is not True:
        raise RuntimeError(f"Round-trip failed for {view.spec.label} {method}.")
    if method == "zipnn-src":
        raw_bytes, compressed_bytes = zipnn_exponent_metrics(
            result,
            method,
        )
        exponent_metrics = {
            "raw_bytes": raw_bytes,
            "compressed_bytes": compressed_bytes,
            "codebook_bytes": 0,
            "exponent_ratio_pct": 100.0 * compressed_bytes / raw_bytes,
        }
    else:
        exponent_metrics = component_exponent_metrics(result)
    measured = {
        **exponent_metrics,
        "encode_seconds": float(result["encode_seconds"]),
        "setup_seconds": float(result.get("setup_seconds") or 0.0),
        "clusters": int(result.get("clusters") or 0),
        "selected_modes": result.get("selected_modes") or {},
        "roundtrip_ok": bool(result["roundtrip_ok"]),
    }
    del result
    gc.collect()
    return measured


def run_component(view: ModelView, threads: int):
    start = time.perf_counter()
    unique_clusters = {
        name: f"tensor_{index:06d}"
        for index, (name, _) in enumerate(floating_parameters(view))
    }
    variants = (
        ("ZipNN", "zipnn-src", {"zipnn_method": "HUFFMAN"}),
        (
            "Sharing Only",
            "neuralzip",
            {"ngram_values": (), "packed_source_budget_ratio": 0.0},
        ),
        (
            "Packing Only",
            "neuralzip",
            {
                "cluster_map": unique_clusters,
                "enable_shared_codebooks": False,
            },
        ),
        ("NeuralZip", "neuralzip", {}),
    )
    rows = []
    for variant, method, options in variants:
        metrics = production_exponent_measure(view, method, threads, **options)
        rows.append(
            {
                "model": view.spec.label,
                "variant": variant,
                "repeats": 1,
                **metrics,
            }
        )

    summary = pd.DataFrame(rows)
    model_dir = OUTPUT_DIR / view.spec.key
    model_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(model_dir / "component_summary.csv", index=False)
    write_json(
        model_dir / "component_metadata.json",
        {
            "repetitions": 1,
            "warmup_runs": 1,
            "threads": int(threads),
            "backend": "official ZipNN source and NeuralZip production encoder",
            "zipnn_version": importlib.metadata.version("zipnn"),
            "zipnn_defensive_copy_included": True,
            "ratio_scope": "exponent payload plus compressor-native metadata",
            "seconds": time.perf_counter() - start,
        },
    )
    return summary


def timed_cost(function):
    start = time.perf_counter()
    result = function()
    seconds = time.perf_counter() - start
    return result, seconds


def select_mmrepair_pair(view: ModelView):
    groups = {}
    for name, parameter in floating_parameters(view):
        if parameter.ndim < 2:
            continue
        shape = tuple(int(dim) for dim in parameter.shape)
        groups.setdefault(shape, []).append((name, parameter))
    candidates = [items for items in groups.values() if len(items) >= 2]
    if not candidates:
        raise RuntimeError(f"No repeated two-dimensional shape in {view.spec.label}.")
    items = max(candidates, key=lambda group: int(group[0][1].numel()))
    return items[:2]


def run_mmrepair(view: ModelView, mmrepair_dir: Path):
    pair = select_mmrepair_pair(view)
    matrices = []
    shapes = []
    for name, parameter in pair:
        tensor = parameter.detach().cpu().contiguous()
        stream = np.frombuffer(
            zipnn_exponent_stream(tensor, layout="original"), dtype=np.uint8
        ).copy()
        matrices.append(stream.reshape(int(tensor.shape[0]), -1))
        shapes.append(list(tensor.shape))
    matrix = np.concatenate(matrices, axis=0)

    output_dir = OUTPUT_DIR / view.spec.key / "mmrepair"
    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / f"{view.spec.key}.exp.i32"
    matrix.astype("<i4").tofile(base)
    command = [
        sys.executable,
        str(mmrepair_dir / "matrepair"),
        "-r",
        "--i32",
        "--drv",
        str(base),
        str(matrix.shape[0]),
        str(matrix.shape[1]),
    ]
    completed = subprocess.run(
        command,
        cwd=mmrepair_dir,
        check=True,
        capture_output=True,
        text=True,
    )
    log = completed.stdout + completed.stderr
    (output_dir / "matrepair.log").write_text(log)
    matches = re.findall(r"^Elapsed time:\s+([0-9.]+)$", completed.stdout, re.MULTILINE)
    if not matches:
        raise RuntimeError("Could not parse MM-RePair elapsed time.")
    paths = (
        Path(str(base) + ".ivald"),
        Path(str(base) + ".dv.R.iv"),
        Path(str(base) + ".dv.C.ansf.1"),
    )
    compressed_bytes = sum(path.stat().st_size for path in paths)
    raw_bytes = int(matrix.size)
    return {
        "method": "MM-RePair",
        "raw_bytes": raw_bytes,
        "compressed_bytes": compressed_bytes,
        "exponent_ratio_pct": 100.0 * compressed_bytes / raw_bytes,
        "encode_seconds": float(matches[-1]),
        "tensor_names": [name for name, _ in pair],
        "tensor_shapes": shapes,
        "rows": int(matrix.shape[0]),
        "cols": int(matrix.shape[1]),
    }


def run_deferred(view: ModelView, mmrepair_dir: Path, threads: int):
    rows = []
    zipnn = production_exponent_measure(view, "zipnn-local", threads)
    rows.append(
        {
            "method": "ZipNN",
            "encode_seconds": float(zipnn["encode_seconds"]),
            "exponent_ratio_pct": float(zipnn["exponent_ratio_pct"]),
            "raw_bytes": int(zipnn["raw_bytes"]),
            "compressed_bytes": int(zipnn["compressed_bytes"]),
        }
    )

    neuralzip = production_exponent_measure(view, "neuralzip", threads)
    rows.append(
        {
            "method": "NeuralZip",
            "encode_seconds": float(neuralzip["encode_seconds"]),
            "exponent_ratio_pct": float(neuralzip["exponent_ratio_pct"]),
            "raw_bytes": int(neuralzip["raw_bytes"]),
            "compressed_bytes": int(neuralzip["compressed_bytes"]),
        }
    )

    rows.append(run_mmrepair(view, mmrepair_dir))

    delta, delta_seconds = timed_cost(
        lambda: delta_exponent_huffman_cost(view, layout="original")
    )
    rows.append(
        {
            "method": "Delta exponent Huffman",
            "encode_seconds": delta_seconds,
            "exponent_ratio_pct": float(delta["exponent_ratio_pct"]),
            "raw_bytes": int(delta["raw_bytes"]),
            "compressed_bytes": int(delta["compressed_bytes"]),
        }
    )

    order1, order1_seconds = timed_cost(
        lambda: order1_huffman_cluster_cost(
            view, cluster_js_threshold=JS_THRESHOLD, layout="original"
        )
    )
    rows.append(
        {
            "method": "Order-1 clustered Huffman",
            "encode_seconds": order1_seconds,
            "exponent_ratio_pct": float(order1["exponent_ratio_pct"]),
            "raw_bytes": int(order1["raw_bytes"]),
            "compressed_bytes": int(order1["compressed_bytes"]),
        }
    )

    arithmetic = selected_turborc_range_cost(
        view,
        cluster_js_threshold=JS_THRESHOLD,
        layout="transpose2d",
        method="rccs",
    )
    rows.append(
        {
            "method": "Arithmetic coding",
            "encode_seconds": float(arithmetic["encode_seconds"]),
            "exponent_ratio_pct": float(arithmetic["exponent_ratio_pct"]),
            "raw_bytes": int(arithmetic["raw_bytes"]),
            "compressed_bytes": int(arithmetic["compressed_bytes"]),
        }
    )

    frame = pd.DataFrame(rows)
    frame.insert(0, "model", view.spec.label)
    model_dir = OUTPUT_DIR / view.spec.key
    model_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(model_dir / "deferred.csv", index=False)
    return frame


def compact_compression_result(result):
    excluded = {"tensor_results", "codebooks"}
    return {key: value for key, value in result.items() if key not in excluded}


def zipnn_exponent_metrics(result, method: str):
    """Return the complete BF16 exponent representation for local/src ZipNN."""

    raw_bytes = 0
    compressed_bytes = 0
    if method == "zipnn-local":
        for tensor_result in result["tensor_results"]:
            payload = tensor_result["payload"]
            exponent_group = payload.groups[int(payload.element_size) - 1]
            raw_bytes += int(exponent_group.symbol_count)
            compressed_bytes += int(exponent_group.compressed_bytes)
        return raw_bytes, compressed_bytes

    if method != "zipnn-src":
        raise ValueError(method)

    for tensor_result in result["tensor_results"]:
        payload = tensor_result["payload"]
        _, exponent = parse_zipnn_exponent_stream(
            payload,
            element_size=2,
            expected_raw_bytes=tensor_result["raw_bytes"],
            expected_byte_reorder=10,
            expected_bit_reorder=1,
            expected_chunk_size=256 << 10,
        )
        raw_bytes += exponent.raw_bytes
        compressed_bytes += exponent.compressed_bytes
    return raw_bytes, compressed_bytes


def run_zipnn_implementation(view: ModelView, threads: int):
    rows = []
    detailed = {}
    for method in ("zipnn-local", "zipnn-src"):
        warmup = Compressor(
            method,
            verify=False,
            zipnn_method="HUFFMAN",
            compression_threshold=0.95,
            threads=threads,
        ).compress(view)
        del warmup
        gc.collect()

        result = Compressor(
            method,
            verify=True,
            keep_payloads=True,
            zipnn_method="HUFFMAN",
            compression_threshold=0.95,
            threads=threads,
        ).compress(view)
        if result["roundtrip_ok"] is not True:
            raise RuntimeError(f"Round-trip failed for {view.spec.label} {method}.")
        exponent_raw_bytes, exponent_compressed_bytes = zipnn_exponent_metrics(
            result,
            method,
        )
        rows.append(
            {
                "model": view.spec.label,
                "method": method,
                "encode_seconds": float(result["encode_seconds"]),
                "codec_encode_seconds": float(
                    result.get("codec_encode_seconds", result["encode_seconds"])
                ),
                "input_copy_seconds": float(result.get("input_copy_seconds", 0.0)),
                "raw_bytes": exponent_raw_bytes,
                "compressed_bytes": exponent_compressed_bytes,
                "exponent_ratio_pct": 100.0
                * exponent_compressed_bytes
                / exponent_raw_bytes,
                "full_raw_bytes": int(result["raw_bytes"]),
                "full_compressed_bytes": int(result["compressed_bytes"]),
                "roundtrip_ok": bool(result["roundtrip_ok"]),
            }
        )
        detailed[method] = {
            **compact_compression_result(result),
            "exponent_raw_bytes": exponent_raw_bytes,
            "exponent_compressed_bytes": exponent_compressed_bytes,
            "exponent_ratio_pct": (
                100.0 * exponent_compressed_bytes / exponent_raw_bytes
            ),
        }
        print(
            f"[{view.spec.key}] {method}: total={result['encode_seconds']:.6f}s, "
            f"codec={result.get('codec_encode_seconds', result['encode_seconds']):.6f}s, "
            f"copy={result.get('input_copy_seconds', 0.0):.6f}s, "
            f"exponent={100.0 * exponent_compressed_bytes / exponent_raw_bytes:.3f}%",
            flush=True,
        )
    frame = pd.DataFrame(rows)
    model_dir = OUTPUT_DIR / view.spec.key
    model_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(model_dir / "zipnn_implementation.csv", index=False)
    write_json(
        model_dir / "zipnn_implementation.json",
        {
            "setup": {
                "repetitions": 1,
                "warmup_runs": 1,
                "dtype": "bfloat16",
                "ratio_scope": "exponent symbols + raw fallback + group metadata",
                "zipnn_method": "HUFFMAN",
                "compression_threshold": 0.95,
                "threads": int(threads),
                "verification": "bitwise exact round trip",
            },
            "results": detailed,
        },
    )
    return frame


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=json_default))


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def aggregate_csv(filename):
    frames = []
    for spec in MODEL_SPECS:
        path = OUTPUT_DIR / spec.key / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True)


def render_figures():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    diagnostics = []
    ngrams = []
    heatmaps = []
    for spec in MODEL_SPECS:
        model_dir = OUTPUT_DIR / spec.key
        diagnostics.append(json.loads((model_dir / "diagnostics.json").read_text()))
        ngram = pd.read_csv(model_dir / "ngram_entropy.csv")
        ngram.insert(0, "model", spec.label)
        ngrams.append(ngram)
        heatmap = np.load(model_dir / "pairwise_js_heatmap.npz")
        heatmaps.append((spec.label, heatmap["matrix"]))

    entropy_rows = []
    order_rows = []
    for row in diagnostics:
        entropy_rows.append({"model": row["model"], **row["field_entropy"]})
        order_rows.append({"model": row["model"], **row["order_entropy"]})
    entropy = pd.DataFrame(entropy_rows)
    order = pd.DataFrame(order_rows)
    ngram = pd.concat(ngrams, ignore_index=True)
    entropy.to_csv(FIGURE_DIR / "zero_order_entropy.csv", index=False)
    order.to_csv(FIGURE_DIR / "order_0_vs_order_1_entropy.csv", index=False)
    ngram.to_csv(FIGURE_DIR / "ngram_exponent_entropy.csv", index=False)

    colors = ("#c44e52", "#55a868", "#4c72b0")
    fields = (
        ("sign", "Sign Entropy", 1),
        ("exponent", "Exponent Entropy", 8),
        ("mantissa", "Mantissa Entropy", 7),
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.8))
    for axis, (field, title, width), color in zip(axes, fields, colors):
        axis.bar(entropy["model"], entropy[field], color=color)
        axis.axhline(width, color="black", linestyle="--", linewidth=1)
        axis.set_title(title)
        axis.set_ylabel("Entropy (bits)")
        axis.tick_params(axis="x", rotation=22)
        axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "zero_order_entropy.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.4, 5.2))
    for spec in MODEL_SPECS:
        rows = ngram[ngram["model"] == spec.label]
        axis.plot(rows["n"], rows["bits_per_exponent"], marker="o", label=spec.label)
    axis.set_xticks(NGRAM_SIZES)
    axis.set_xlabel("Exponent n-gram size")
    axis.set_ylabel("Ideal bits per original exponent")
    axis.set_title("Packed Exponent n-gram Entropy")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "ngram_exponent_entropy.pdf", bbox_inches="tight")
    plt.close(fig)

    max_distance = max(
        float(matrix[np.tril_indices_from(matrix, k=-1)].max(initial=0.0))
        for _, matrix in heatmaps
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 10.2), squeeze=False)
    image = None
    for axis, (label, matrix) in zip(axes.ravel(), heatmaps):
        image = axis.imshow(matrix, cmap="magma", vmin=0, vmax=max_distance)
        axis.set_title(label)
        step = max(1, matrix.shape[0] // 6)
        ticks = np.arange(0, matrix.shape[0], step)
        axis.set_xticks(ticks)
        axis.set_yticks(ticks)
        axis.set_xlabel("Tensor block index")
        axis.set_ylabel("Tensor block index")
    fig.colorbar(
        image,
        ax=axes.ravel().tolist(),
        fraction=0.025,
        pad=0.03,
        label="Jensen-Shannon distance",
    )
    fig.savefig(FIGURE_DIR / "blockwise_js_heatmaps.pdf", bbox_inches="tight")
    plt.close(fig)

    x = np.arange(len(order))
    width = 0.36
    fig, axis = plt.subplots(figsize=(9, 5.2))
    axis.bar(x - width / 2, order["h0_bits_per_exponent"], width, label="Order 0")
    axis.bar(x + width / 2, order["h1_bits_per_exponent"], width, label="Order 1")
    axis.set_xticks(x, order["model"], rotation=18)
    axis.set_ylabel("Entropy (bits per exponent)")
    axis.set_title("Exponent Entropy by Context Order")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "order_0_vs_order_1_entropy.pdf", bbox_inches="tight")
    plt.close(fig)


def aggregate_tables():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    deferred = aggregate_csv("deferred.csv")
    component = aggregate_csv("component_summary.csv")
    zipnn = aggregate_csv("zipnn_implementation.csv")
    deferred.to_csv(OUTPUT_DIR / "deferred_all_models.csv", index=False)
    component.to_csv(OUTPUT_DIR / "component_all_models.csv", index=False)
    zipnn.to_csv(OUTPUT_DIR / "zipnn_implementation_all_models.csv", index=False)


def write_run_metadata(
    threads: int,
    *,
    model_root: Path,
    allow_download: bool,
):
    write_json(
        OUTPUT_DIR / "run_metadata.json",
        {
            "models": [spec.__dict__ for spec in MODEL_SPECS],
            "dtype": "bfloat16",
            "repetitions": 1,
            "threads": int(threads),
            "model_root": str(model_root.resolve()),
            "allow_download": bool(allow_download),
            "ngram_sample_exponents": NGRAM_SAMPLE_EXPONENTS,
            "max_heatmap_tensors": MAX_HEATMAP_TENSORS,
            "js_threshold": JS_THRESHOLD,
            "host": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(SPECS_BY_SLUG))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument(
        "--phase",
        choices=("diagnostics", "component", "deferred", "zipnn", "all", "render"),
        default="all",
    )
    parser.add_argument("--mmrepair-dir", type=Path, default=Path("/tmp/mm-repair"))
    return parser.parse_args()


def main():
    global OUTPUT_DIR, FIGURE_DIR
    args = parse_args()
    OUTPUT_DIR = args.output_dir
    FIGURE_DIR = OUTPUT_DIR / "figures"
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_run_metadata(
        args.threads,
        model_root=args.model_root,
        allow_download=args.allow_download,
    )
    if args.phase == "render":
        render_figures()
        aggregate_tables()
        return
    if not args.model:
        raise SystemExit("--model is required unless --phase=render")

    spec = SPECS_BY_SLUG[args.model]
    view = load_model(
        spec,
        model_root=args.model_root,
        allow_download=args.allow_download,
    )
    print(
        f"loaded={spec.label} parameters="
        f"{sum(parameter.numel() for parameter in view.model.parameters()):,}",
        flush=True,
    )
    phases = (
        ("diagnostics", run_diagnostics),
        ("component", lambda value: run_component(value, args.threads)),
        (
            "deferred",
            lambda value: run_deferred(value, args.mmrepair_dir, args.threads),
        ),
        ("zipnn", lambda value: run_zipnn_implementation(value, args.threads)),
    )
    for phase, function in phases:
        if args.phase not in (phase, "all"):
            continue
        start = time.perf_counter()
        print(f"phase={phase} status=start", flush=True)
        function(view)
        print(
            f"phase={phase} status=complete seconds={time.perf_counter() - start:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
