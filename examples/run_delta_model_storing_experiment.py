import argparse
import csv
import gc
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from config import MODEL_SUITE, VISION_DIFFUSION_MODEL_SUITE
from neuralzip.ablations import _delta_stream, zipnn_exponent_stream
from neuralzip.experiment_scenarios import _load_vision_diffusion_mapping, _normalise_dtype
from neuralzip.zipnn_codec import ZipNNCodec


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "deferred"
DEFAULT_MODEL_STORING_CSV = (
    Path(__file__).resolve().parent
    / "outputs"
    / "model_hubs"
    / "godel_model_storing_20260617T183015Z"
    / "model_storing_results.csv"
)
DEFAULT_VISION_DIFFUSION_CSV = (
    Path(__file__).resolve().parent
    / "outputs"
    / "vision_diffusion"
    / "vision_diffusion_results.csv"
)
TARGET_LANGUAGE_MODELS = {
    "Gemma 4 31B",
    "Phi-4 14B",
    "Qwen3.5 27B",
    "LatamGPT-70B",
}


def _iter_chunks(data, chunk_size):
    for start in range(0, len(data), int(chunk_size)):
        yield data[start : start + int(chunk_size)]


def _entropy_from_counts(counts):
    counts = np.asarray(counts, dtype=np.float64)
    total = float(counts.sum())
    if total == 0.0:
        return 0.0
    probabilities = counts[counts > 0] / total
    return float(-(probabilities * np.log2(probabilities)).sum())


def _stream_metrics_from_tensors(named_tensors, *, dtype, chunk_size, layout):
    target_dtype = _normalise_dtype(dtype)
    if target_dtype == "auto":
        target_dtype = None

    exponent_counts = np.zeros(256, dtype=np.int64)
    delta_counts = np.zeros(256, dtype=np.int64)
    raw_bytes = 0
    transitions = 0
    repeated_transitions = 0
    tensor_count = 0
    zipnn_compressed_bytes = 0
    zipnn_payload_bytes = 0
    delta_compressed_bytes = 0
    delta_payload_bytes = 0
    chunks = 0
    zipnn_huffman_chunks = 0
    zipnn_raw_chunks = 0
    delta_huffman_chunks = 0
    delta_raw_chunks = 0
    codec = ZipNNCodec(
        chunk_size=chunk_size,
        compression_threshold=1.0,
    )

    for _, tensor in named_tensors:
        if not torch.is_tensor(tensor) or not tensor.is_floating_point():
            continue
        value = tensor.detach().cpu().contiguous()
        if target_dtype is not None:
            value = value.to(dtype=target_dtype)
        stream = zipnn_exponent_stream(value, layout=layout)
        values = np.frombuffer(bytes(stream), dtype=np.uint8)
        if values.size == 0:
            continue

        tensor_count += 1
        raw_bytes += int(values.size)
        exponent_counts += np.bincount(values, minlength=256)
        if values.size > 1:
            repeated_transitions += int(np.count_nonzero(values[1:] == values[:-1]))
            transitions += int(values.size - 1)

        delta = _delta_stream(stream)
        delta_values = np.frombuffer(delta, dtype=np.uint8)
        delta_counts += np.bincount(delta_values, minlength=256)

        zipnn_group, delta_group = codec._compress_streams(
            (
                (stream, "exponent", chunk_size),
                (delta, "delta", chunk_size),
            )
        )
        chunks += len(zipnn_group.chunks)
        zipnn_compressed_bytes += sum(
            chunk.compressed_bytes for chunk in zipnn_group.chunks
        )
        zipnn_payload_bytes += zipnn_group.payload_bytes
        zipnn_huffman_chunks += zipnn_group.local_huff0_chunks
        zipnn_raw_chunks += zipnn_group.raw_chunks
        delta_compressed_bytes += sum(
            chunk.compressed_bytes for chunk in delta_group.chunks
        )
        delta_payload_bytes += delta_group.payload_bytes
        delta_huffman_chunks += delta_group.local_huff0_chunks
        delta_raw_chunks += delta_group.raw_chunks

        del value

    exponent_h0 = _entropy_from_counts(exponent_counts)
    delta_h0 = _entropy_from_counts(delta_counts)
    same_pct = 100.0 * repeated_transitions / transitions if transitions else 0.0
    return {
        "raw_bytes": raw_bytes,
        "tensor_count": tensor_count,
        "transitions": transitions,
        "same_exponent_transition_pct": same_pct,
        "exponent_h0_bits": exponent_h0,
        "delta_h0_bits": delta_h0,
        "delta_h0_minus_exponent_h0_bits": delta_h0 - exponent_h0,
        "ideal_exponent_pct": 100.0 * exponent_h0 / 8.0,
        "ideal_delta_pct": 100.0 * delta_h0 / 8.0,
        "zipnn_compressed_bytes": zipnn_compressed_bytes,
        "zipnn_payload_bytes": zipnn_payload_bytes,
        "zipnn_metadata_bytes": zipnn_compressed_bytes - zipnn_payload_bytes,
        "zipnn_compressed_pct": (
            100.0 * zipnn_compressed_bytes / raw_bytes if raw_bytes else None
        ),
        "delta_compressed_bytes": delta_compressed_bytes,
        "delta_payload_bytes": delta_payload_bytes,
        "delta_metadata_bytes": delta_compressed_bytes - delta_payload_bytes,
        "delta_compressed_pct": (
            100.0 * delta_compressed_bytes / raw_bytes if raw_bytes else None
        ),
        "chunks": chunks,
        "zipnn_huffman_chunks": zipnn_huffman_chunks,
        "zipnn_raw_chunks": zipnn_raw_chunks,
        "delta_huffman_chunks": delta_huffman_chunks,
        "delta_raw_chunks": delta_raw_chunks,
        "_exponent_counts": exponent_counts,
        "_delta_counts": delta_counts,
        "_repeated_transitions": repeated_transitions,
        "_transitions": transitions,
    }


def _language_model_metrics(spec, *, dtype, chunk_size, layout, local_files_only):
    snapshot_path = Path(
        snapshot_download(
            spec["model_id"],
            local_files_only=bool(local_files_only),
            allow_patterns=("*.safetensors", "*.safetensors.index.json"),
        )
    )
    files = sorted(snapshot_path.glob("*.safetensors"))
    if not files:
        raise ValueError(f"No safetensors files found for {spec['model_id']}.")

    totals = None
    start = time.perf_counter()
    for file_path in files:
        state = load_file(str(file_path), device="cpu")
        metrics = _stream_metrics_from_tensors(
            state.items(),
            dtype=dtype,
            chunk_size=chunk_size,
            layout=layout,
        )
        if totals is None:
            totals = metrics
        else:
            for key, value in metrics.items():
                if key in {
                    "exponent_h0_bits",
                    "delta_h0_bits",
                    "delta_h0_minus_exponent_h0_bits",
                    "ideal_exponent_pct",
                    "ideal_delta_pct",
                    "same_exponent_transition_pct",
                    "zipnn_compressed_pct",
                    "delta_compressed_pct",
                }:
                    continue
                if isinstance(value, np.ndarray):
                    totals[key] += value
                else:
                    totals[key] += value
        del state
        gc.collect()

    # Recompute global rates from accumulated counts by rerunning through files
    # is unnecessarily expensive for large checkpoints. The accumulated byte
    # costs are exact; entropy diagnostics are kept approximate only when a
    # checkpoint is sharded. Most interpretation relies on the exact ratios.
    elapsed = time.perf_counter() - start
    if totals is None:
        totals = {}
    raw_bytes = int(totals.get("raw_bytes", 0))
    exponent_h0 = _entropy_from_counts(totals["_exponent_counts"])
    delta_h0 = _entropy_from_counts(totals["_delta_counts"])
    transitions = int(totals["_transitions"])
    repeated = int(totals["_repeated_transitions"])
    totals["exponent_h0_bits"] = exponent_h0
    totals["delta_h0_bits"] = delta_h0
    totals["delta_h0_minus_exponent_h0_bits"] = delta_h0 - exponent_h0
    totals["ideal_exponent_pct"] = 100.0 * exponent_h0 / 8.0
    totals["ideal_delta_pct"] = 100.0 * delta_h0 / 8.0
    totals["same_exponent_transition_pct"] = (
        100.0 * repeated / transitions if transitions else 0.0
    )
    totals["zipnn_compressed_pct"] = (
        100.0 * totals["zipnn_compressed_bytes"] / raw_bytes if raw_bytes else None
    )
    totals["delta_compressed_pct"] = (
        100.0 * totals["delta_compressed_bytes"] / raw_bytes if raw_bytes else None
    )
    totals["delta_seconds"] = elapsed
    totals["load_backend"] = "safetensors-streaming"
    totals["source"] = str(snapshot_path)
    totals["shards"] = len(files)
    del totals["_exponent_counts"]
    del totals["_delta_counts"]
    del totals["_repeated_transitions"]
    del totals["_transitions"]
    return totals


def _vision_model_metrics(spec, *, dtype, chunk_size, layout, local_files_only):
    start = time.perf_counter()
    tensors, metadata = _load_vision_diffusion_mapping(
        spec,
        dtype=dtype,
        local_files_only=local_files_only,
    )
    load_seconds = time.perf_counter() - start
    start = time.perf_counter()
    metrics = _stream_metrics_from_tensors(
        tensors.items(),
        dtype=dtype,
        chunk_size=chunk_size,
        layout=layout,
    )
    metrics["delta_seconds"] = time.perf_counter() - start
    metrics["load_seconds"] = load_seconds
    metrics.update(metadata)
    del tensors
    gc.collect()
    return metrics


def _read_baseline(paths):
    baseline = {}
    for path in paths:
        path = Path(path)
        if not path.exists():
            continue
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row.get("status") != "ok":
                    continue
                model = row.get("model_name") or row.get("model")
                compressor = row.get("compressor")
                if not model or not compressor:
                    continue
                pct = row.get("exponent_compressed_pct") or row.get("compressed_pct")
                seconds = row.get("encode_seconds")
                if pct in (None, ""):
                    continue
                baseline[(model, compressor)] = {
                    "pct": float(pct),
                    "seconds": float(seconds) if seconds not in (None, "") else None,
                }
    return baseline


def _row_for_model(spec, metrics, baseline):
    model = spec["name"]
    zipnn_baseline = baseline.get((model, "ZipNN local"), {})
    neuralzip_baseline = baseline.get((model, "NeuralZip"), {})
    zipnn_pct = metrics.get("zipnn_compressed_pct")
    zipnn_reference_pct = zipnn_pct
    neuralzip_pct = neuralzip_baseline.get("pct")
    delta_pct = metrics.get("delta_compressed_pct")
    return {
        "model": model,
        "family": spec.get("family", "language"),
        "model_id": spec.get("model_id"),
        "dtype": spec.get("default_dtype"),
        "raw_bytes": metrics.get("raw_bytes"),
        "tensor_count": metrics.get("tensor_count"),
        "exponent_h0_bits": metrics.get("exponent_h0_bits"),
        "delta_h0_bits": metrics.get("delta_h0_bits"),
        "delta_h0_minus_exponent_h0_bits": metrics.get(
            "delta_h0_minus_exponent_h0_bits"
        ),
        "same_exponent_transition_pct": metrics.get("same_exponent_transition_pct"),
        "zipnn_compressed_pct": zipnn_reference_pct,
        "zipnn_measured_compressed_pct": zipnn_pct,
        "delta_compressed_pct": delta_pct,
        "neuralzip_compressed_pct": neuralzip_pct,
        "delta_gain_vs_zipnn_pp": (
            zipnn_reference_pct - delta_pct
            if zipnn_reference_pct is not None and delta_pct is not None
            else None
        ),
        "delta_gain_vs_neuralzip_pp": (
            neuralzip_pct - delta_pct
            if neuralzip_pct is not None and delta_pct is not None
            else None
        ),
        "zipnn_seconds": zipnn_baseline.get("seconds"),
        "delta_seconds": metrics.get("delta_seconds"),
        "neuralzip_seconds": neuralzip_baseline.get("seconds"),
        "chunks": metrics.get("chunks"),
        "zipnn_huffman_chunks": metrics.get("zipnn_huffman_chunks"),
        "zipnn_raw_chunks": metrics.get("zipnn_raw_chunks"),
        "delta_huffman_chunks": metrics.get("delta_huffman_chunks"),
        "delta_raw_chunks": metrics.get("delta_raw_chunks"),
        "load_backend": metrics.get("load_backend"),
        "source": metrics.get("source"),
        "status": "ok",
        "error": None,
    }


def run(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline = _read_baseline([args.model_storing_csv, args.vision_diffusion_csv])
    rows = []
    output_csv = args.output_dir / "delta_model_storing_results.csv"
    output_json = args.output_dir / "delta_model_storing_results.json"

    language_specs = [
        spec for spec in MODEL_SUITE if spec["name"] in TARGET_LANGUAGE_MODELS
    ]
    vision_specs = list(VISION_DIFFUSION_MODEL_SUITE)
    if args.models:
        selected = set(args.models)
        language_specs = [spec for spec in language_specs if spec["name"] in selected]
        vision_specs = [spec for spec in vision_specs if spec["name"] in selected]

    for spec in [*language_specs, *vision_specs]:
        try:
            dtype = spec.get("default_dtype", args.dtype)
            if spec in language_specs:
                metrics = _language_model_metrics(
                    spec,
                    dtype=dtype,
                    chunk_size=args.chunk_size,
                    layout=args.layout,
                    local_files_only=args.local_files_only,
                )
            else:
                metrics = _vision_model_metrics(
                    spec,
                    dtype=dtype,
                    chunk_size=args.chunk_size,
                    layout=args.layout,
                    local_files_only=args.local_files_only,
                )
            row = _row_for_model(spec, metrics, baseline)
        except Exception as error:
            row = {
                "model": spec["name"],
                "family": spec.get("family", "language"),
                "model_id": spec.get("model_id"),
                "dtype": spec.get("default_dtype"),
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            }
        rows.append(row)
        pd.DataFrame(rows).to_csv(output_csv, index=False)
        output_json.write_text(json.dumps(rows, indent=2, default=str))
        print(pd.DataFrame([row]).to_string(index=False))

    table = pd.DataFrame(rows)
    table.to_csv(output_csv, index=False)
    output_json.write_text(json.dumps(rows, indent=2, default=str))
    print(f"saved_csv={output_csv}")
    print(f"saved_json={output_json}")
    return table


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-storing-csv", type=Path, default=DEFAULT_MODEL_STORING_CSV)
    parser.add_argument("--vision-diffusion-csv", type=Path, default=DEFAULT_VISION_DIFFUSION_CSV)
    parser.add_argument("--layout", choices=("original", "transpose2d"), default="original")
    parser.add_argument("--chunk-size", type=int, default=256 << 10)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--models", nargs="*")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
