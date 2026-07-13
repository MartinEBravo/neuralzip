import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch

from neuralzip.ablations import (
    delta_exponent_huffman_cost,
    order1_huffman_cluster_cost,
    selected_turborc_range_cost,
    zipnn_exponent_huffman_cost,
)
from neuralzip.turborc import TurboRangeUnavailable


DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parent
    / "outputs"
    / "checkpoints"
    / "resnet18_imagenet1k_cifar10_finetune20_relela05"
    / "checkpoints"
    / "resnet18_cifar10_finetune_epoch_020.pt"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "deferred"


def load_checkpoint_tensors(path: Path, *, dtype=torch.bfloat16):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state", checkpoint)
    tensors = {
        name: tensor.detach().cpu().contiguous().to(dtype=dtype)
        for name, tensor in state.items()
        if torch.is_tensor(tensor) and tensor.is_floating_point()
    }
    metadata = {
        "checkpoint": str(path),
        "architecture": checkpoint.get("architecture"),
        "dataset": checkpoint.get("dataset"),
        "finetune_epoch": checkpoint.get("finetune_epoch"),
        "test_accuracy": checkpoint.get("test_accuracy"),
        "dtype": str(dtype).replace("torch.", ""),
        "tensor_count": len(tensors),
    }
    return tensors, metadata


def timed(label, fn):
    start = time.perf_counter()
    table = fn()
    table = table.copy()
    table["seconds"] = time.perf_counter() - start
    table["experiment"] = label
    return table


def run(args):
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    tensors, metadata = load_checkpoint_tensors(args.checkpoint)
    full_tables = [
        timed(
            "zipnn-exponent-huffman",
            lambda: zipnn_exponent_huffman_cost(tensors, layout="original"),
        ),
        timed(
            "delta-exponent-huffman",
            lambda: delta_exponent_huffman_cost(tensors, layout="original"),
        ),
        timed(
            "order1-clustered-huffman",
            lambda: order1_huffman_cluster_cost(
                tensors,
                cluster_js_threshold=args.cluster_js_threshold,
                layout="original",
            ),
        ),
    ]

    try:
        full_tables.append(
            timed(
                "turborc-order1-arithmetic",
                lambda: selected_turborc_range_cost(
                    tensors,
                    cluster_js_thresholds=(args.cluster_js_threshold,),
                    layout="transpose2d",
                    method="rccs",
                ),
            )
        )
    except (ImportError, OSError, TurboRangeUnavailable) as error:
        full_tables.append(
            pd.DataFrame(
                [
                    {
                        "method": "TurboRC order-1 arithmetic",
                        "status": "skipped",
                        "error": f"{type(error).__name__}: {error}",
                    }
                ]
            )
        )

    full = pd.concat(full_tables, ignore_index=True, sort=False)
    full = full.drop(columns=["tensor_results", "selected_modes"], errors="ignore")
    for key, value in metadata.items():
        full[key] = value
    full_path = output_dir / "deferred_full_model_results.csv"
    full.to_csv(full_path, index=False)

    metadata_path = output_dir / "deferred_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str))

    display_columns = [
        "method",
        "layout",
        "compressed_pct",
        "raw_bytes",
        "compressed_bytes",
        "metadata_bytes",
        "seconds",
    ]
    available = [column for column in display_columns if column in full.columns]
    print(full[available].to_string(index=False))
    print(f"saved_full={full_path}")
    return full


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cluster-js-threshold", type=float, default=0.05)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
