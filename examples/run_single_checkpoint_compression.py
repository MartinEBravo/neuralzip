from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neuralzip.experiment_scenarios import _compress_mapping  # noqa: E402


def main() -> None:
    default_checkpoint = (
        ROOT
        / "examples"
        / "outputs"
        / "checkpoints"
        / "resnet18_cifar10_pretrained_epoch200_relela01"
        / "checkpoint_epoch_200.pt"
    )
    checkpoint_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_checkpoint
    output_dir = checkpoint_path.parent

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state", checkpoint)
    tensors = {
        name: tensor.detach().cpu().contiguous().to(dtype=torch.bfloat16)
        for name, tensor in state.items()
        if torch.is_tensor(tensor) and tensor.is_floating_point()
    }

    methods = (
        ("ZipNN local", "zipnn-local"),
        ("NeuralZip-Ratio", "neuralzip-ratio"),
    )
    df = pd.DataFrame(_compress_mapping(checkpoint_path.stem, tensors, methods=methods))
    df["checkpoint"] = checkpoint_path.name
    df["epoch"] = int(checkpoint.get("epoch", -1)) if isinstance(checkpoint, dict) else -1
    df["dtype"] = "bf16"
    df["tensor_count"] = len(tensors)

    output_csv = output_dir / "ratio_vs_zipnn_checkpoint_epoch_200.csv"
    output_json = output_dir / "ratio_vs_zipnn_checkpoint_epoch_200.json"
    df.to_csv(output_csv, index=False)
    output_json.write_text(json.dumps(df.to_dict(orient="records"), indent=2))

    display_cols = [
        "compressor",
        "scope",
        "status",
        "raw_bytes",
        "compressed_bytes",
        "compressed_pct",
        "exponent_compressed_pct",
        "cold_encode_seconds",
        "clusters",
    ]
    print(df[display_cols].to_string(index=False))
    print(f"saved_csv={output_csv}")
    print(f"saved_json={output_json}")


if __name__ == "__main__":
    main()
