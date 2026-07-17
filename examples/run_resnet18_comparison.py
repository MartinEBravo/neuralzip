from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import time
from pathlib import Path

import torch
import torchvision
from torchvision.models import ResNet18_Weights, resnet18

from neuralzip import Compressor


METHODS = (
    ("NeuralZip", "neuralzip"),
)


def load_resnet18_bf16() -> dict[str, torch.Tensor]:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval()
    return {
        name: tensor.detach().cpu().contiguous().to(torch.bfloat16)
        for name, tensor in model.named_parameters()
    }


def run_once(
    tensors: dict[str, torch.Tensor],
    method: str,
    *,
    verify: bool,
) -> dict[str, object]:
    compressor = Compressor(
        method,
        verify=verify,
        compression_threshold=0.95,
    )
    started = time.perf_counter()
    result = compressor.compress(tensors)
    wall_seconds = time.perf_counter() - started
    return {
        "raw_bytes": int(result["raw_bytes"]),
        "compressed_bytes": int(result["compressed_bytes"]),
        "compression_ratio": float(result["compression_ratio"]),
        "compression_pct": 100.0 * float(result["compression_ratio"]),
        "wall_seconds": wall_seconds,
        "precompute_seconds": float(result.get("precompute_seconds") or 0.0),
        "codebook_setup_seconds": float(
            result.get("codebook_setup_seconds") or 0.0
        ),
        "encode_seconds": float(result["encode_seconds"]),
        "compression_seconds": float(result["compression_seconds"]),
        "total_compression_seconds": float(result["total_compression_seconds"]),
        "roundtrip_ok": result.get("roundtrip_ok"),
        "clusters": int(result.get("clusters") or 0),
        "codebooks": len(result.get("codebooks") or ()),
        "packed_ngram_chunks": int(result.get("packed_ngram_chunks") or 0),
        "shared_huff0_chunks": int(result.get("shared_huff0_chunks") or 0),
        "local_huff0_chunks": int(result.get("local_huff0_chunks") or 0),
        "raw_chunks": int(result.get("raw_chunks") or 0),
        "selected_modes": result.get("selected_modes") or {},
        "selection_stats": result.get("selection_stats") or {},
    }


def median_row(label: str, method: str, rows: list[dict[str, object]]) -> dict[str, object]:
    first = rows[0]
    return {
        "compressor": label,
        "method": method,
        "repeats": len(rows),
        "raw_bytes": first["raw_bytes"],
        "compressed_bytes": first["compressed_bytes"],
        "compression_pct": first["compression_pct"],
        "wall_seconds_median": statistics.median(row["wall_seconds"] for row in rows),
        "precompute_seconds_median": statistics.median(
            row["precompute_seconds"] for row in rows
        ),
        "codebook_setup_seconds_median": statistics.median(
            row["codebook_setup_seconds"] for row in rows
        ),
        "encode_seconds_median": statistics.median(
            row["encode_seconds"] for row in rows
        ),
        "compression_seconds_median": statistics.median(
            row["compression_seconds"] for row in rows
        ),
        "total_compression_seconds_median": statistics.median(
            row["total_compression_seconds"] for row in rows
        ),
        "clusters": first["clusters"],
        "codebooks": first["codebooks"],
        "packed_ngram_chunks": first["packed_ngram_chunks"],
        "shared_huff0_chunks": first["shared_huff0_chunks"],
        "local_huff0_chunks": first["local_huff0_chunks"],
        "raw_chunks": first["raw_chunks"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "outputs"
        / "resnet18_comparison",
    )
    args = parser.parse_args()

    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    tensors = load_resnet18_bf16()
    raw_bytes = sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())
    metadata = {
        "model": "torchvision.models.resnet18",
        "weights": "ResNet18_Weights.IMAGENET1K_V1",
        "dtype": "bfloat16",
        "floating_tensors": len(tensors),
        "raw_bytes": raw_bytes,
        "host": platform.node(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "compression_threshold": 0.95,
        "chunk_size_bytes": 256 << 10,
    }

    runs: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    verification: dict[str, bool | None] = {}
    for label, method in METHODS:
        verified = run_once(tensors, method, verify=True)
        verification[method] = verified["roundtrip_ok"]

        method_rows = []
        for repeat in range(1, args.repeats + 1):
            row = run_once(tensors, method, verify=False)
            row.update(compressor=label, method=method, repeat=repeat)
            runs.append(row)
            method_rows.append(row)
            print(
                f"{label} repeat {repeat}/{args.repeats}: "
                f"wall={row['wall_seconds']:.6f}s "
                f"cold={row['cold_seconds']:.6f}s "
                f"hot={row['hot_seconds']:.6f}s "
                f"ratio={row['compression_pct']:.6f}%",
                flush=True,
            )
        summaries.append(median_row(label, method, method_rows))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata,
        "verification": verification,
        "summary": summaries,
        "runs": runs,
    }
    json_path = args.output_dir / "resnet18_zipnn_vs_neuralzip.json"
    csv_path = args.output_dir / "resnet18_zipnn_vs_neuralzip.csv"
    json_path.write_text(json.dumps(payload, indent=2))
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    print(json.dumps(payload["summary"], indent=2), flush=True)
    print(f"verification={json.dumps(verification, sort_keys=True)}", flush=True)
    print(f"saved_json={json_path}", flush=True)
    print(f"saved_csv={csv_path}", flush=True)


if __name__ == "__main__":
    main()
