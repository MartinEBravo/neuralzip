#!/usr/bin/env python3
"""Validate the final expanded distilled and quantized result artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from neuralzip.hashing import sha256_file as sha256


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = (
    REPOSITORY_ROOT / "results" / "final-lossy-lossless-expanded-20260719"
)

DISTILLED_MODELS = {
    "deepseek-r1-distill-llama-8b",
    "deepseek-r1-distill-qwen-7b",
    "gemma2-9b-distilled",
    "qwen25-7b-distilled",
}
QUANTIZED_REVISIONS = {
    "unsloth/gemma-4-31B-it-unsloth-bnb-4bit": (
        "8e256fc6d63003fc0ca8c91b976e6dcc38433385"
    ),
    "unsloth/phi-4-bnb-4bit": "098ea917f967c8811e6f579cc38872b2bf3d4bf4",
    "skkwowee/Qwen3.5-27B-bnb-4bit": (
        "26ba09cf1db425f2c170b3446f42de8891e28400"
    ),
    "Qwen/Qwen2.5-7B-Instruct-AWQ": (
        "b25037543e9394b818fdfca67ab2a00ecc7dd641"
    ),
    "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4": (
        "e9c932ac1893a49ae0fc497ad6e1e86e2e39af20"
    ),
    "jairo/Llama-3.1-70B-LatamGPT-SFT-1.0-GGUF": (
        "43aaa004bc882624cfed02c98a898516987da773"
    ),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json(path: Path):
    return json.loads(path.read_text())


def validate_quantized(root: Path) -> dict[str, object]:
    directory = root / "quantized-all-current"
    rows = list(csv.DictReader((directory / "quantized_nibble_compression.csv").open()))
    metadata = read_json(directory / "run_metadata.json")

    require(len(rows) == 12, "Quantized suite must contain 12 rows.")
    require(metadata["rows"] == 12, "Quantized metadata row count is stale.")
    require(metadata["host"]["hostname"] == "godel", "Quantized host is not Godel.")
    require(
        metadata["host"]["cpu_affinity"] == list(range(12, 24)),
        "Quantized affinity must be CPUs 12-23.",
    )
    require(metadata["host"]["threads"] == 12, "Quantized thread count changed.")
    require(
        metadata["backend"]["runner_sha256"]
        == sha256(REPOSITORY_ROOT / "examples" / "run_quantized_nibble_compression.py"),
        "Quantized runner hash does not match the current source.",
    )
    require(
        metadata["backend"]["config_sha256"]
        == sha256(REPOSITORY_ROOT / "examples" / "config.py"),
        "Quantized config hash does not match the current source.",
    )

    grouped = defaultdict(dict)
    for row in rows:
        require(row["status"] == "ok", f"Failed quantized row: {row['model']}")
        require(row["roundtrip_ok"] == "True", f"Bad quantized roundtrip: {row['model']}")
        require(row["threads"] == "12", "Quantized row changed thread count.")
        require(row["warmup_repetitions"] == "1", "Quantized warm-up count changed.")
        require(row["measured_repetitions"] == "1", "Quantized measured count changed.")
        require(
            QUANTIZED_REVISIONS.get(row["model_id"]) == row["resolved_revision"],
            f"Unexpected quantized revision for {row['model_id']}.",
        )
        require(row["compressor"] not in grouped[row["model"]], "Duplicate quantized row.")
        grouped[row["model"]][row["compressor"]] = row

    require(len(grouped) == 6, "Quantized suite must contain six models.")
    expected_compressors = {"zipnn-nibble-local", "neuralzip-nibble-js"}
    require(
        all(set(pair) == expected_compressors for pair in grouped.values()),
        "Every quantized model must contain both compressors.",
    )

    faster = 0
    time_factors = []
    ratio_deltas = []
    for pair in grouped.values():
        local = pair["zipnn-nibble-local"]
        shared = pair["neuralzip-nibble-js"]
        local_time = float(local["encode_seconds"])
        shared_time = float(shared["encode_seconds"])
        faster += int(shared_time < local_time)
        time_factors.append(max(local_time, shared_time) / min(local_time, shared_time))
        ratio_deltas.append(float(shared["ratio_pct"]) - float(local["ratio_pct"]))

    return {
        "models": len(grouped),
        "rows": len(rows),
        "shared_faster": faster,
        "shared_slower": len(grouped) - faster,
        "max_time_factor": max(time_factors),
        "ratio_delta_pp": [min(ratio_deltas), max(ratio_deltas)],
    }


def validate_distilled(root: Path) -> dict[str, object]:
    rows = []
    for directory in ("distilled-deepseek-current", "distilled-existing-current"):
        rows.extend(read_json(root / directory / "results.json"))

    require(len(rows) == 8, "Distilled suite must contain eight rows.")
    grouped = defaultdict(dict)
    code_digests = set()
    for row in rows:
        require(row["status"] == "ok", f"Failed distilled row: {row['model_key']}")
        require(row["host"] == "godel", "Distilled host is not Godel.")
        require(row["roundtrip_ok"] is True, f"Bad distilled roundtrip: {row['model_key']}")
        require(row["verified_warmup"] is True, "Distilled warm-up was not verified.")
        require(row["threads"] == 12, "Distilled thread count changed.")
        require(row["cpu_set"] == list(range(12)), "Distilled affinity must be CPUs 0-11.")
        require(row["warmup_runs"] == 1, "Distilled warm-up count changed.")
        require(row["measured_runs"] == 1, "Distilled measured count changed.")
        require(row["model_key"] in DISTILLED_MODELS, "Unexpected distilled model.")
        require(row["compressor"] not in grouped[row["model_key"]], "Duplicate distilled row.")
        grouped[row["model_key"]][row["compressor"]] = row
        code_digests.add(row["code_digest"])

    require(set(grouped) == DISTILLED_MODELS, "Distilled model coverage is incomplete.")
    expected_compressors = {"ZipNN src", "NeuralZip"}
    require(
        all(set(pair) == expected_compressors for pair in grouped.values()),
        "Every distilled model must contain both compressors.",
    )
    require(len(code_digests) == 1, "Distilled rows do not share one code digest.")

    faster = 0
    time_factors = []
    ratio_deltas = []
    for pair in grouped.values():
        source = pair["ZipNN src"]
        neural = pair["NeuralZip"]
        source_time = float(source["encode_seconds"])
        neural_time = float(neural["encode_seconds"])
        faster += int(neural_time < source_time)
        time_factors.append(max(source_time, neural_time) / min(source_time, neural_time))
        ratio_deltas.append(
            float(neural["exponent_ratio_pct"])
            - float(source["exponent_ratio_pct"])
        )

    return {
        "models": len(grouped),
        "rows": len(rows),
        "neuralzip_faster": faster,
        "neuralzip_slower": len(grouped) - faster,
        "time_factor": [min(time_factors), max(time_factors)],
        "ratio_delta_pp": [min(ratio_deltas), max(ratio_deltas)],
        "code_digest": next(iter(code_digests)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    summary = {
        "quantized": validate_quantized(args.results_dir),
        "distilled": validate_distilled(args.results_dir),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
