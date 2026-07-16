from __future__ import annotations

import csv
import json
import platform
from pathlib import Path

import torch
import torchvision
import zipnn
from torchvision.models import ResNet18_Weights, resnet18
from transformers import AutoModel

from neuralzip import Compressor


OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "outputs"
    / "zipnn_implementation_ablation"
)


def _bf16_parameters(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().contiguous().to(torch.bfloat16)
        for name, parameter in module.named_parameters()
        if parameter.is_floating_point()
    }


def _load_models() -> list[tuple[str, str, dict[str, torch.Tensor]]]:
    resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval()
    bert = AutoModel.from_pretrained("google-bert/bert-base-uncased").eval()
    return [
        (
            "ResNet-18",
            "torchvision ResNet18_Weights.IMAGENET1K_V1",
            _bf16_parameters(resnet),
        ),
        (
            "BERT Base",
            "google-bert/bert-base-uncased",
            _bf16_parameters(bert),
        ),
    ]


def _compress_once(
    tensors: dict[str, torch.Tensor],
    method: str,
) -> dict[str, object]:
    result = Compressor(
        method,
        verify=True,
        zipnn_method="HUFFMAN",
        compression_threshold=0.95,
    ).compress(tensors)
    if result["roundtrip_ok"] is not True:
        raise RuntimeError(f"Round-trip verification failed for {method}.")
    return {
        "seconds": float(result["encode_seconds"]),
        "raw_bytes": int(result["raw_bytes"]),
        "compressed_bytes": int(result["compressed_bytes"]),
        "roundtrip_ok": bool(result["roundtrip_ok"]),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    detailed_rows = []

    for model_name, source, tensors in _load_models():
        local = _compress_once(tensors, "zipnn-local")
        official = _compress_once(tensors, "zipnn-src")
        rows.append(
            {
                "model": model_name,
                "zipnn_local_seconds": local["seconds"],
                "zipnn_src_seconds": official["seconds"],
            }
        )
        detailed_rows.append(
            {
                "model": model_name,
                "source": source,
                "dtype": "bfloat16",
                "parameters": sum(tensor.numel() for tensor in tensors.values()),
                "raw_bytes": local["raw_bytes"],
                "zipnn_local": local,
                "zipnn_src": official,
            }
        )
        print(
            f"{model_name}: local={local['seconds']:.6f}s, "
            f"src={official['seconds']:.6f}s",
            flush=True,
        )

    csv_path = OUTPUT_DIR / "zipnn_implementation_ablation.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "setup": {
            "repetitions": 1,
            "warmup_runs": 0,
            "dtype": "bfloat16",
            "zipnn_method": "HUFFMAN",
            "compression_threshold": 0.95,
            "time_metric": "sum of per-tensor encode wall times",
            "verification": "bitwise round trip in the same run",
            "host": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "zipnn": getattr(zipnn, "__version__", "unknown"),
        },
        "results": detailed_rows,
    }
    json_path = OUTPUT_DIR / "zipnn_implementation_ablation.json"
    json_path.write_text(json.dumps(payload, indent=2))
    print(f"saved_csv={csv_path}", flush=True)
    print(f"saved_json={json_path}", flush=True)


if __name__ == "__main__":
    main()
