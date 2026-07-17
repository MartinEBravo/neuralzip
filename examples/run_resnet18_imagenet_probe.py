from __future__ import annotations

import json
from pathlib import Path

import torch
from torchvision.models import ResNet18_Weights, resnet18

from neuralzip import Compressor


def exponent_cost(result: dict) -> tuple[int, int]:
    raw = 0
    compressed = 0
    for row in result["tensor_results"]:
        payload = row.get("payload")
        if payload is None:
            continue
        stream_id = payload.element_size - 1
        group = payload.groups[stream_id]
        raw += int(group.symbol_count)
        compressed += int(group.compressed_bytes)
    return raw, compressed


def main() -> None:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval()
    tensors = {
        name: tensor.detach().cpu().contiguous().to(torch.bfloat16)
        for name, tensor in model.named_parameters()
        if tensor.is_floating_point()
    }

    rows = []
    for label, method in (
        ("ZipNN local", "zipnn-local"),
        ("NeuralZip", "neuralzip"),
    ):
        result = Compressor(
            method,
            verify=True,
            keep_payloads=True,
            threads=12,
        ).compress(tensors)
        exponent_raw, exponent_compressed = exponent_cost(result)
        rows.append(
            {
                "compressor": label,
                "exponent_raw_bytes": exponent_raw,
                "exponent_compressed_bytes": exponent_compressed,
                "exponent_ratio_pct": 100 * exponent_compressed / exponent_raw,
                "encode_seconds": float(result["encode_seconds"]),
                "precompute_seconds": float(result.get("precompute_seconds") or 0),
                "codebook_setup_seconds": float(
                    result.get("codebook_setup_seconds") or 0
                ),
                "codebooks": len(result.get("codebooks") or ()),
                "selected_modes": result.get("selected_modes") or {},
                "roundtrip_ok": bool(result["roundtrip_ok"]),
            }
        )

    output = Path(__file__).resolve().parent / "outputs" / "resnet18_imagenet_probe.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
