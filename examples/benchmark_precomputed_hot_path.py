from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
from torchvision.models import ResNet18_Weights, resnet18

from neuralzip import Compressor


def exponent_metrics(result):
    raw_bytes = 0
    compressed_bytes = 0
    for row in result["tensor_results"]:
        payload = row.get("payload")
        if payload is None:
            continue
        group = payload.groups[payload.element_size - 1]
        raw_bytes += int(group.symbol_count)
        compressed_bytes += int(group.compressed_bytes)
    codebook_bytes = sum(
        int(codebook.metadata_bytes)
        for codebook in result.get("codebooks") or ()
        if int(codebook.stream_id) == 1 and codebook.dtype in {"bfloat16", "float16"}
    )
    compressed_bytes += codebook_bytes
    return {
        "raw_bytes": raw_bytes,
        "compressed_bytes": compressed_bytes,
        "codebook_metadata_bytes": codebook_bytes,
        "ratio_pct": 100.0 * compressed_bytes / raw_bytes,
    }


def summarize(label, method, tensors, repetitions, threads):
    common = {
        "verify": True,
        "keep_payloads": True,
        "threads": threads,
    }
    warmup = Compressor(method, **common).compress(tensors)
    rows = []
    for _ in range(repetitions):
        result = Compressor(
            method,
            verify=False,
            keep_payloads=True,
            threads=threads,
        ).compress(tensors)
        rows.append(result)

    encode_times = [float(row["encode_seconds"]) for row in rows]
    precompute_times = [
        float(row.get("precompute_total_seconds") or 0.0) for row in rows
    ]
    compressed_sizes = [int(row["compressed_bytes"]) for row in rows]
    selected_mode_signatures = [
        json.dumps(row.get("selected_modes") or {}, sort_keys=True)
        for row in rows
    ]
    selected_modes_per_repetition = [
        row.get("selected_modes") or {} for row in rows
    ]
    representative = rows[len(rows) // 2]
    return {
        "compressor": label,
        "repetitions": repetitions,
        "encode_seconds": encode_times,
        "encode_seconds_median": statistics.median(encode_times),
        "encode_seconds_min": min(encode_times),
        "encode_seconds_max": max(encode_times),
        "precompute_total_seconds": precompute_times,
        "precompute_total_seconds_median": statistics.median(precompute_times),
        "compressed_bytes": compressed_sizes,
        "compressed_size_stable": len(set(compressed_sizes)) == 1,
        "selected_modes_stable": len(set(selected_mode_signatures)) == 1,
        "selected_modes_per_repetition": selected_modes_per_repetition,
        "clustering_seconds": float(representative.get("clustering_seconds") or 0.0),
        "codebook_setup_seconds": float(
            representative.get("codebook_setup_seconds") or 0.0
        ),
        "prepack_seconds": float(representative.get("prepack_seconds") or 0.0),
        "precompute_memory_bytes": int(
            representative.get("precompute_memory_bytes") or 0
        ),
        "prepacked_cache_bytes": int(
            representative.get("prepacked_cache_bytes") or 0
        ),
        "precomputed_codebook_memory_bytes": int(
            representative.get("precomputed_codebook_memory_bytes") or 0
        ),
        "full_ratio_pct": 100.0 * float(representative["compression_ratio"]),
        "exponent": exponent_metrics(representative),
        "selected_modes": representative.get("selected_modes") or {},
        "selection_stats": representative.get("selection_stats") or {},
        "roundtrip_ok": bool(warmup["roundtrip_ok"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent
        / "outputs"
        / "resnet18_precomputed_hot_benchmark.json",
    )
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval()
    tensors = {
        name: tensor.detach().cpu().contiguous().to(torch.bfloat16)
        for name, tensor in model.named_parameters()
        if tensor.is_floating_point()
    }

    zipnn = summarize(
        "ZipNN local",
        "zipnn-local",
        tensors,
        args.repetitions,
        args.threads,
    )
    neuralzip = summarize(
        "NeuralZip precomputed",
        "neuralzip",
        tensors,
        args.repetitions,
        args.threads,
    )
    output = {
        "model": "ResNet-18 ImageNet-1K BF16",
        "threads": args.threads,
        "zipnn": zipnn,
        "neuralzip": neuralzip,
        "hot_speedup": (
            zipnn["encode_seconds_median"]
            / neuralzip["encode_seconds_median"]
        ),
        "exponent_ratio_gain_pp": (
            zipnn["exponent"]["ratio_pct"]
            - neuralzip["exponent"]["ratio_pct"]
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
