from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
from pathlib import Path
import subprocess
import time

import torch

from neuralzip.neuralzip_codec import NeuralZipCodec
from run_precompute_family_transfer import (
    PAIR_SPECS,
    floating_parameters,
    load_hf_model,
)


MODEL_SPECS = {
    "gemma3-1b": PAIR_SPECS["gemma"],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run clustering and packed-mode selection without encoding."
    )
    parser.add_argument("--model", choices=MODEL_SPECS, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--ngram-values", type=int, nargs="+", default=(2, 3, 4))
    parser.add_argument("--top-k", type=int, default=254)
    parser.add_argument(
        "--packed-granularity",
        choices=("js-cluster", "tensor"),
        default="js-cluster",
    )
    parser.add_argument("--packed-source-budget-ratio", type=float, default=0.30)
    parser.add_argument("--packed-time-budget-ratio", type=float, default=1.0)
    parser.add_argument("--speed-ratio-tolerance", type=float, default=0.03)
    parser.add_argument("--revision")
    return parser.parse_args()


def current_revision():
    try:
        return subprocess.check_output(
            ("git", "rev-parse", "HEAD"),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    args = parse_args()
    if args.threads < 1:
        raise ValueError("--threads must be positive.")

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    spec = MODEL_SPECS[args.model]
    model_path = args.model_root / str(spec.target_dir)
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(
            f"The local model snapshot is incomplete: {model_path}"
        )

    print(f"[{args.model}] loading {model_path}", flush=True)
    model = load_hf_model(spec.kind, model_path)
    named_tensors = floating_parameters(model)
    parameters = sum(int(tensor.numel()) for _, tensor in named_tensors)
    cluster_map = None
    enable_shared_codebooks = True
    if args.packed_granularity == "tensor":
        cluster_map = {
            name: f"tensor_{index:06d}"
            for index, (name, _) in enumerate(named_tensors)
        }
        enable_shared_codebooks = False

    codec = NeuralZipCodec(
        threads=args.threads,
        compression_threshold=0.95,
        ngram_values=tuple(args.ngram_values),
        top_k=args.top_k,
        packed_source_budget_ratio=args.packed_source_budget_ratio,
        packed_time_budget_ratio=args.packed_time_budget_ratio,
        speed_ratio_tolerance=args.speed_ratio_tolerance,
        cluster_map=cluster_map,
        enable_shared_codebooks=enable_shared_codebooks,
        enable_prepacked_cache=False,
    )

    print(f"[{args.model}] clustering", flush=True)
    started = time.perf_counter()
    codec.prepare_clusters(named_tensors)
    clustering_seconds = time.perf_counter() - started

    print(f"[{args.model}] selecting modes", flush=True)
    started = time.perf_counter()
    codec.build_codebooks_and_select_modes(named_tensors)
    selection_seconds = time.perf_counter() - started

    stats = codec.selection_stats
    diagnostics = codec.selection_diagnostics
    exponent_source_bytes = int(stats.get("exponent_source_bytes") or 0)
    packed_source_bytes = int(stats.get("actual_packed_source_bytes") or 0)
    decision_counts = Counter(row["decision"] for row in diagnostics)
    candidate_counts_by_n = Counter()
    improving_counts_by_n = Counter()
    sample_hits_by_n = Counter()
    sample_misses_by_n = Counter()
    sample_escape_bytes_by_n = Counter()
    for row in diagnostics:
        for candidate in row["candidates"]:
            n = int(candidate["ngram_n"])
            candidate_counts_by_n[n] += 1
            if candidate["improves_selected_base"]:
                improving_counts_by_n[n] += 1
            sample_hits_by_n[n] += int(candidate["sample_ngram_hits"])
            sample_misses_by_n[n] += int(candidate["sample_ngram_misses"])
            sample_escape_bytes_by_n[n] += int(
                candidate["sample_escape_bytes"]
            )

    output = {
        "probe": "packed-precompute-only",
        "encode_performed": False,
        "revision": args.revision or current_revision(),
        "model": args.model,
        "model_id": spec.target_id,
        "model_path": str(model_path),
        "dtype": "bfloat16",
        "threads": args.threads,
        "parameters": parameters,
        "tensors": len(named_tensors),
        "config": {
            "ngram_values": list(codec.ngram_values),
            "top_k": codec.top_k,
            "max_codebook_sample_bytes": codec.max_codebook_sample_bytes,
            "exponent_layout": codec.exponent_layout,
            "packed_source_budget_ratio": codec.packed_source_budget_ratio,
            "packed_time_budget_ratio": codec.packed_time_budget_ratio,
            "speed_ratio_tolerance": codec.speed_ratio_tolerance,
            "compression_threshold": codec.compression_threshold,
            "packed_granularity": args.packed_granularity,
            "enable_shared_codebooks": enable_shared_codebooks,
        },
        "clustering_seconds": clustering_seconds,
        "selection_seconds": selection_seconds,
        "precompute_seconds": clustering_seconds + selection_seconds,
        "clusters": int(codec.cluster_count),
        "codebooks": len(codec.export_codebooks()),
        "codebook_metadata_bytes": int(codec.codebook_metadata_bytes()),
        "selected_modes": codec.selected_modes,
        "planned_packed_coverage_pct": (
            100.0 * packed_source_bytes / exponent_source_bytes
            if exponent_source_bytes
            else 0.0
        ),
        "decision_counts": dict(sorted(decision_counts.items())),
        "candidate_counts_by_n": {
            str(key): value for key, value in sorted(candidate_counts_by_n.items())
        },
        "improving_counts_by_n": {
            str(key): value for key, value in sorted(improving_counts_by_n.items())
        },
        "sample_hit_rate_by_n": {
            str(n): (
                sample_hits_by_n[n]
                / (sample_hits_by_n[n] + sample_misses_by_n[n])
                if sample_hits_by_n[n] + sample_misses_by_n[n]
                else 0.0
            )
            for n in sorted(candidate_counts_by_n)
        },
        "sample_escape_bytes_by_n": {
            str(key): value
            for key, value in sorted(sample_escape_bytes_by_n.items())
        },
        "selection_stats": stats,
        "selection_diagnostics": diagnostics,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "precompute_seconds": output["precompute_seconds"],
                "planned_packed_coverage_pct": output[
                    "planned_packed_coverage_pct"
                ],
                "decision_counts": output["decision_counts"],
                "selection_stats": stats,
            },
            indent=2,
        ),
        flush=True,
    )

    del model, named_tensors, codec
    gc.collect()


if __name__ == "__main__":
    main()
