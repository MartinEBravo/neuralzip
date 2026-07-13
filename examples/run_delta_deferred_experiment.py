import argparse
import json
import time
from pathlib import Path

import torch

from neuralzip.ablations import delta_exponent_diagnostics, selected_turborc_range_cost
from neuralzip.turborc import TurboRangeUnavailable
from run_deferred_experiments import DEFAULT_CHECKPOINT, load_checkpoint_tensors


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "deferred"


def run(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tensors, metadata = load_checkpoint_tensors(args.checkpoint, dtype=args.dtype)

    start = time.perf_counter()
    table = delta_exponent_diagnostics(
        tensors,
        layout=args.layout,
        chunk_size=args.chunk_size,
    )
    table["seconds"] = time.perf_counter() - start

    try:
        arithmetic_start = time.perf_counter()
        arithmetic = selected_turborc_range_cost(
            tensors,
            cluster_js_thresholds=(args.cluster_js_threshold,),
            layout=args.arithmetic_layout,
            chunk_size=args.chunk_size,
            method=args.arithmetic_method,
        ).iloc[0]
        arithmetic_seconds = time.perf_counter() - arithmetic_start
        arithmetic_pct = float(arithmetic["compressed_pct"])
        table["arithmetic_status"] = "ok"
        table["arithmetic_layout"] = args.arithmetic_layout
        table["arithmetic_compressed_pct"] = arithmetic_pct
        table["arithmetic_compressed_bytes"] = int(arithmetic["compressed_bytes"])
        table["arithmetic_seconds"] = arithmetic_seconds
        table["arithmetic_gain_vs_zipnn_pp"] = (
            table["zipnn_compressed_pct"] - arithmetic_pct
        )
        table["delta_gain_vs_arithmetic_pp"] = (
            arithmetic_pct - table["delta_compressed_pct"]
        )
    except (ImportError, OSError, TurboRangeUnavailable) as error:
        table["arithmetic_status"] = f"skipped: {type(error).__name__}: {error}"
        table["arithmetic_layout"] = args.arithmetic_layout
        table["arithmetic_compressed_pct"] = None
        table["arithmetic_compressed_bytes"] = None
        table["arithmetic_seconds"] = None
        table["arithmetic_gain_vs_zipnn_pp"] = None
        table["delta_gain_vs_arithmetic_pp"] = None

    for key, value in metadata.items():
        table[key] = value

    csv_path = args.output_dir / "delta_exponent_experiment.csv"
    json_path = args.output_dir / "delta_exponent_experiment.json"
    table.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(table.to_dict(orient="records"), indent=2, default=str)
    )

    display_columns = [
        "raw_bytes",
        "same_exponent_transition_pct",
        "exponent_h0_bits",
        "delta_h0_bits",
        "zipnn_compressed_pct",
        "delta_compressed_pct",
        "delta_gain_vs_zipnn_pp",
        "arithmetic_status",
        "arithmetic_compressed_pct",
        "delta_gain_vs_arithmetic_pp",
        "seconds",
        "arithmetic_seconds",
    ]
    print(table[display_columns].to_string(index=False))
    print(f"saved_csv={csv_path}")
    print(f"saved_json={json_path}")
    return table


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--layout", choices=("original", "transpose2d"), default="original")
    parser.add_argument("--arithmetic-layout", choices=("original", "transpose2d"), default="transpose2d")
    parser.add_argument("--arithmetic-method", default="rccs")
    parser.add_argument("--cluster-js-threshold", type=float, default=0.05)
    parser.add_argument("--chunk-size", type=int, default=256 << 10)
    parser.add_argument("--dtype", default=torch.bfloat16)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
