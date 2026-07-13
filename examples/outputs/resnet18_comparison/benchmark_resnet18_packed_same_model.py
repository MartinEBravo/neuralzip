import json
import statistics
from pathlib import Path

import timm
import torch

from neuralzip import Compressor


MODEL_ID = "timm/resnet18.a1_in1k"


def load_model():
    return timm.create_model(
        f"hf_hub:{MODEL_ID}",
        pretrained=True,
    ).eval().to(torch.bfloat16)


def run(model, method, *, packed_time_budget_ratio=0.10, verify=False):
    options = {
        "verify": verify,
        "compression_threshold": 0.95,
    }
    if method == "neuralzip":
        options["packed_time_budget_ratio"] = packed_time_budget_ratio
    result = Compressor(method, **options).compress(model)
    exponent = result["stream_results"][-1]
    return {
        "setup_seconds": (
            result["precompute_seconds"] + result["codebook_setup_seconds"]
        ),
        "compression_seconds": result["encode_seconds"],
        "one_shot_seconds": result["total_compression_seconds"],
        "compression_pct": 100.0 * result["compression_ratio"],
        "exponent_pct": 100.0 * exponent["compression_ratio"],
        "clusters": result["clusters"],
        "codebooks": len(result["codebooks"]),
        "selected_modes": result["selected_modes"],
        "selection_stats": result["selection_stats"],
        "packed_chunks": result["packed_ngram_chunks"],
        "ngram_hit_rate": result["ngram_hit_rate"],
        "roundtrip_ok": result["roundtrip_ok"],
    }


def median_result(rows):
    first = rows[0]
    output = {
        field: statistics.median(row[field] for row in rows)
        for field in ("setup_seconds", "compression_seconds", "one_shot_seconds")
    }
    for field in (
        "compression_pct",
        "exponent_pct",
        "clusters",
        "codebooks",
        "selected_modes",
        "selection_stats",
        "packed_chunks",
        "ngram_hit_rate",
    ):
        output[field] = first[field]
    return output


def main():
    torch.set_num_threads(32)
    model = load_model()
    verification = {
        "zipnn": run(model, "zipnn", verify=True)["roundtrip_ok"],
        "packed_default": run(
            model,
            "neuralzip",
            packed_time_budget_ratio=0.10,
            verify=True,
        )["roundtrip_ok"],
        "packed_unlimited": run(
            model,
            "neuralzip",
            packed_time_budget_ratio=1_000.0,
            verify=True,
        )["roundtrip_ok"],
    }

    rows = {"zipnn": [], "packed_default": [], "packed_unlimited": []}
    for repeat in range(1, 6):
        rows["zipnn"].append(run(model, "zipnn"))
        rows["packed_default"].append(
            run(model, "neuralzip", packed_time_budget_ratio=0.10)
        )
        rows["packed_unlimited"].append(
            run(model, "neuralzip", packed_time_budget_ratio=1_000.0)
        )
        print(f"repeat {repeat}/5", flush=True)

    summary = {name: median_result(values) for name, values in rows.items()}
    zipnn = summary["zipnn"]
    for name in ("packed_default", "packed_unlimited"):
        summary[name]["full_gain_vs_zipnn_pp"] = (
            zipnn["compression_pct"] - summary[name]["compression_pct"]
        )
        summary[name]["exponent_gain_vs_zipnn_pp"] = (
            zipnn["exponent_pct"] - summary[name]["exponent_pct"]
        )
        summary[name]["speedup_vs_zipnn"] = (
            zipnn["compression_seconds"] / summary[name]["compression_seconds"]
        )

    output = {
        "model_id": MODEL_ID,
        "dataset": "ImageNet-1K",
        "recipe": "timm ResNet Strikes Back A1",
        "dtype": "bfloat16",
        "verification": verification,
        "summary": summary,
    }
    destination = Path(__file__).with_name(
        "resnet18_packed_same_model_20260712.json"
    )
    destination.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
