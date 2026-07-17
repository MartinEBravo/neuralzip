from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import json
from pathlib import Path

import torch

from neuralzip import Compressor
from run_thesis_model_suite import MODEL_SPECS, load_model


def exponent_metrics(result):
    raw_bytes = 0
    compressed_bytes = 0
    for row in result["tensor_results"]:
        payload = row["payload"]
        group = payload.groups[payload.element_size - 1]
        raw_bytes += int(group.symbol_count)
        compressed_bytes += int(group.compressed_bytes)
    codebook_bytes = sum(
        int(codebook.metadata_bytes)
        for codebook in result.get("codebooks") or ()
        if codebook.dtype == "bfloat16" and int(codebook.stream_id) == 1
    )
    compressed_bytes += codebook_bytes
    return {
        "raw_bytes": raw_bytes,
        "compressed_bytes": compressed_bytes,
        "codebook_metadata_bytes": codebook_bytes,
        "ratio_pct": 100.0 * compressed_bytes / raw_bytes,
    }


def measured_compression(method, view, threads):
    # One complete warm-up pass, followed by the single reported pass.
    warmup = Compressor(
        method,
        verify=False,
        keep_payloads=False,
        threads=threads,
    ).compress(view)
    del warmup
    gc.collect()

    result = Compressor(
        method,
        verify=True,
        keep_payloads=True,
        threads=threads,
    ).compress(view)
    row = {
        "encode_seconds": float(result["encode_seconds"]),
        "hot_encode_seconds": float(result["hot_encode_seconds"]),
        "clustering_seconds": float(result.get("clustering_seconds") or 0.0),
        "codebook_setup_seconds": float(
            result.get("codebook_setup_seconds") or 0.0
        ),
        "prepack_seconds": float(result.get("prepack_seconds") or 0.0),
        "precompute_total_seconds": float(
            result.get("precompute_total_seconds") or 0.0
        ),
        "precompute_memory_bytes": int(
            result.get("precompute_memory_bytes") or 0
        ),
        "prepacked_cache_bytes": int(result.get("prepacked_cache_bytes") or 0),
        "full_ratio_pct": 100.0 * float(result["compression_ratio"]),
        "exponent": exponent_metrics(result),
        "selected_modes": result.get("selected_modes") or {},
        "selection_stats": result.get("selection_stats") or {},
        "selection_diagnostics": result.get("selection_diagnostics") or (),
        "roundtrip_ok": bool(result["roundtrip_ok"]),
    }
    del result
    gc.collect()
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--models", nargs="*", default=[])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent
        / "outputs"
        / "precomputed_small_suite.json",
    )
    args = parser.parse_args()
    selected = set(args.models)

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    rows = []
    for original_spec in MODEL_SPECS:
        if selected and original_spec.slug not in selected:
            continue
        spec = replace(
            original_spec,
            local_path=args.model_root / original_spec.local_path.name,
        )
        print(f"[{spec.slug}] loading", flush=True)
        view = load_model(spec)
        print(f"[{spec.slug}] ZipNN", flush=True)
        zipnn = measured_compression("zipnn-local", view, args.threads)
        print(f"[{spec.slug}] NeuralZip", flush=True)
        neuralzip = measured_compression("neuralzip", view, args.threads)
        rows.append(
            {
                "model": spec.label,
                "slug": spec.slug,
                "zipnn": zipnn,
                "neuralzip": neuralzip,
                "hot_speedup": (
                    zipnn["encode_seconds"] / neuralzip["encode_seconds"]
                ),
                "exponent_ratio_gain_pp": (
                    zipnn["exponent"]["ratio_pct"]
                    - neuralzip["exponent"]["ratio_pct"]
                ),
            }
        )
        print(
            f"[{spec.slug}] speedup={rows[-1]['hot_speedup']:.3f}x "
            f"gain={rows[-1]['exponent_ratio_gain_pp']:.4f}pp",
            flush=True,
        )
        del view
        gc.collect()

    output = {
        "threads": args.threads,
        "repetitions": 1,
        "warmup_passes": 1,
        "models": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
