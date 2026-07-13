import json
from pathlib import Path

import timm
import torch
from torchvision.models import ResNet18_Weights, resnet18

from neuralzip import Compressor
from neuralzip.ablations import neuralzip_component_ablation


def codec_result(model, method, *, budget=0.10):
    result = Compressor(
        method,
        verify=False,
        compression_threshold=0.95,
        packed_time_budget_ratio=budget,
    ).compress(model)
    exponent = result["stream_results"][-1]
    return {
        "full_pct": 100.0 * result["compression_ratio"],
        "exponent_pct": 100.0 * exponent["compression_ratio"],
        "encode_seconds": result["encode_seconds"],
        "packed_chunks": result["packed_ngram_chunks"],
        "selected_modes": result["selected_modes"],
    }


def diagnose(model):
    _, legacy = neuralzip_component_ablation(model, repeats=1)
    legacy_rows = {
        row["variant"]: {
            "exponent_pct": float(row["compression_pct"]),
            "packed_chunks": int(row["packed_ngram_chunks"]),
        }
        for row in legacy.to_dict("records")
    }
    return {
        "legacy_ablation": legacy_rows,
        "codec": {
            "zipnn": codec_result(model, "zipnn"),
            "packed_default": codec_result(model, "neuralzip", budget=0.10),
            "packed_unlimited": codec_result(model, "neuralzip", budget=1_000.0),
        },
    }


def main():
    torch.set_num_threads(32)
    models = {
        "torchvision_resnet18_default": resnet18(
            weights=ResNet18_Weights.DEFAULT
        ).eval().to(torch.bfloat16),
        "timm_resnet18_a1": timm.create_model(
            "hf_hub:timm/resnet18.a1_in1k",
            pretrained=True,
        ).eval().to(torch.bfloat16),
    }
    output = {name: diagnose(model) for name, model in models.items()}
    destination = Path(__file__).with_name("packed_regression_diagnosis_20260712.json")
    destination.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
