#!/usr/bin/env python3
"""Validate and summarize the final multi-target family-transfer suite."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from neuralzip.experiment_io import atomic_write_json

from examples.run_general_codec_rerun import (  # noqa: E402
    DEFAULT_CHUNK_SIZE,
    DEFAULT_COMPRESSION_THRESHOLD,
    code_provenance,
    host_provenance,
)
from examples.run_precompute_family_suite import (  # noqa: E402
    FAMILY_SPECS,
    MODEL_SPECS,
    PINNED_REVISIONS,
    dependency_sha256,
    runner_sha256,
)


EXPECTED = {
    "resnet-basic": ("resnet18",),
    "resnet-bottleneck": ("resnet101", "resnet50"),
    "efficientnet": ("efficientnet-b3", "efficientnet-b0"),
    "convnext": ("convnext-base", "convnext-small", "convnext-tiny"),
    "llama32": ("llama32-1b",),
    "siglip2": ("siglip2-so400m",),
    "whisper-v3": ("whisper-large-v3-turbo",),
    "gemma3": ("gemma3-4b", "gemma3-1b"),
    "olmo2": ("olmo2-7b", "olmo2-1b"),
    "qwen25": ("qwen25-7b", "qwen25-3b", "qwen25-15b"),
    "qwen35": ("qwen35-2b-base", "qwen35-08b-base"),
}


EXPECTED_SOURCES = {
    key: spec.source_key
    for key, spec in FAMILY_SPECS.items()
}


EXPECTED_VERIFICATIONS = {
    "neuralzip_transferred_direct",
    "neuralzip_target_specific_direct",
    "zipnn_source",
}


FIELDS = (
    "family_key",
    "family",
    "source_key",
    "source",
    "source_revision",
    "target_key",
    "target",
    "target_revision",
    "tensor_scope",
    "source_setup_seconds",
    "mapping_seconds",
    "mapped_tensors",
    "target_tensors",
    "encoding_seconds",
    "ratio_pct",
    "target_specific_encoding_seconds",
    "target_specific_ratio_pct",
    "zipnn_source_encoding_seconds",
    "zipnn_source_ratio_pct",
    "speedup_vs_zipnn_source",
    "speedup_vs_target_specific",
    "ratio_delta_vs_zipnn_source_pp",
    "ratio_delta_vs_target_specific_pp",
    "source_plan_sha256",
    "runner_sha256",
    "dependency_sha256",
    "code_digest",
    "host_fingerprint",
    "source_weights_fingerprint",
    "target_weights_fingerprint",
    "roundtrip_ok",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--json", type=Path)
    return parser.parse_args()


def atomic_json(path: Path, payload: object) -> None:
    atomic_write_json(path, payload)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def finite(value: object, label: str, *, positive: bool = False) -> float:
    number = float(value)
    require(math.isfinite(number), f"{label} is not finite.")
    if positive:
        require(number > 0.0, f"{label} must be positive.")
    else:
        require(number >= 0.0, f"{label} must be non-negative.")
    return number


def close(actual: object, expected: object, label: str) -> None:
    require(
        math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=1e-9),
        f"{label} mismatch: {actual!r} != {expected!r}.",
    )


def validate_revision_and_weights(
    key: str,
    revision: object,
    weights: object,
    label: str,
) -> str:
    require(isinstance(revision, str) and revision, f"{label} has no revision.")
    spec = MODEL_SPECS[key]
    if spec.backend == "torchvision":
        from torchvision import models

        constructor = getattr(models, str(spec.constructor))
        expected_revision = str(models.get_model_weights(constructor).DEFAULT)
    else:
        expected_revision = PINNED_REVISIONS[key]
    require(revision == expected_revision, f"{label} revision mismatch.")
    require(isinstance(weights, dict), f"{label} has no weight provenance.")
    require(weights.get("revision") == revision, f"{label} weight revision mismatch.")
    fingerprint = weights.get("weights_fingerprint")
    require(
        isinstance(fingerprint, str) and len(fingerprint) == 64,
        f"{label} has no valid weight fingerprint.",
    )
    require(int(weights.get("floating_numel", 0)) > 0, f"{label} has no floating weights.")
    artifacts = weights.get("weight_artifacts")
    require(isinstance(artifacts, list) and artifacts, f"{label} has no weight artifacts.")
    return fingerprint


def validate_family(key: str, payload: dict) -> list[dict[str, object]]:
    expected_family = FAMILY_SPECS[key]
    source_key = EXPECTED_SOURCES[key]
    require(payload.get("complete") is True, f"{key} is incomplete.")
    require(payload.get("schema_version") == 1, f"{key} schema mismatch.")
    require(payload.get("transfer_mode") == "direct", f"{key} is not direct.")
    require(payload.get("dtype") == "bfloat16", f"{key} is not BF16.")
    require(payload.get("threads") == 12, f"{key} did not use 12 threads.")
    require(payload.get("cpu_set") == list(range(12)), f"{key} CPU affinity mismatch.")
    require(payload.get("host") == "godel", f"{key} did not run on Godel.")
    require(payload.get("chunk_size") == DEFAULT_CHUNK_SIZE, f"{key} chunk size mismatch.")
    close(
        payload.get("compression_threshold"),
        DEFAULT_COMPRESSION_THRESHOLD,
        f"{key} compression threshold",
    )
    require(payload.get("warmup_repetitions") == 1, f"{key} warm-up mismatch.")
    require(payload.get("measured_repetitions") == 1, f"{key} repetition mismatch.")
    require(payload.get("target_histograms_computed") is False, f"{key} histogrammed target.")
    require(payload.get("target_js_matching") is False, f"{key} ran target JS.")
    require(payload.get("target_mode_probing") is False, f"{key} probed target modes.")
    require(payload.get("exact_target_cache") is False, f"{key} cached exact target data.")
    require(payload.get("source_plan_sha256"), f"{key} has no source plan digest.")

    family = payload.get("family") or {}
    require(family.get("key") == key, f"{key} family key mismatch.")
    require(family.get("label") == expected_family.label, f"{key} family label mismatch.")
    require(family.get("source_key") == source_key, f"{key} family source mismatch.")
    require(tuple(family.get("target_keys") or ()) == EXPECTED[key], f"{key} family targets mismatch.")
    require(payload.get("source_model", {}).get("key") == source_key, f"{key} source model mismatch.")
    require(
        payload.get("source_model", {}).get("label") == MODEL_SPECS[source_key].label,
        f"{key} source label mismatch.",
    )
    source_fingerprint = validate_revision_and_weights(
        source_key,
        payload.get("source_revision"),
        payload.get("source_weights"),
        f"{key} source",
    )
    expected_source_scope = (
        "Gemma 3 text backbone"
        if source_key.startswith("gemma3-")
        else "complete floating checkpoint"
    )
    require(payload.get("tensor_scope") == expected_source_scope, f"{key} source scope mismatch.")

    setup = payload.get("source_setup") or {}
    clustering_seconds = finite(setup.get("clustering_seconds"), f"{key} clustering time")
    codebook_seconds = finite(setup.get("codebook_seconds"), f"{key} codebook time")
    source_setup_seconds = finite(setup.get("total_seconds"), f"{key} setup time", positive=True)
    close(
        source_setup_seconds,
        clustering_seconds + codebook_seconds,
        f"{key} setup sum",
    )

    targets = payload.get("targets") or []
    observed = tuple(row.get("target_key") for row in targets)
    require(observed == EXPECTED[key], f"{key} targets {observed!r} != {EXPECTED[key]!r}.")
    output = []
    for target in targets:
        target_key = str(target["target_key"])
        mapping = target["transfer_setup"]
        transferred = target["neuralzip_transferred_direct"]
        native = target["neuralzip_target_specific_direct"]
        zipnn = target["zipnn_source"]
        comparison = target["comparison"]
        verification = target["verification"]
        require(target.get("target_model", {}).get("key") == target_key, f"{target_key} model mismatch.")
        require(
            target.get("target_model", {}).get("label") == MODEL_SPECS[target_key].label,
            f"{target_key} label mismatch.",
        )
        target_fingerprint = validate_revision_and_weights(
            target_key,
            target.get("target_revision"),
            target.get("target_weights"),
            target_key,
        )
        expected_target_scope = (
            "Gemma 3 text backbone"
            if target_key.startswith("gemma3-")
            else "complete floating checkpoint"
        )
        require(target.get("tensor_scope") == expected_target_scope, f"{target_key} scope mismatch.")
        require(mapping["target_histograms_computed"] is False, f"{target_key} histogrammed target.")
        require(mapping["target_js_matching"] is False, f"{target_key} ran JS.")
        require(mapping["target_mode_probing"] is False, f"{target_key} probed modes.")
        require(int(mapping["missing_streams"]) == 0, f"{target_key} has missing streams.")
        require(
            int(mapping["matched_tensors"]) == int(mapping["target_tensors"]),
            f"{target_key} does not map every tensor.",
        )
        require(
            int(mapping["matched_exponents"]) == int(mapping["target_exponents"]),
            f"{target_key} does not map every exponent.",
        )
        require(
            set(verification) == EXPECTED_VERIFICATIONS,
            f"{target_key} verification keys mismatch.",
        )
        require(all(value is True for value in verification.values()), f"{target_key} round-trip failed.")
        require(
            mapping.get("mapping") == "exact source tensor name, dtype, and stream",
            f"{target_key} mapping protocol mismatch.",
        )
        require(transferred["roundtrip_ok"] is True, f"{target_key} transfer verification missing.")
        require(native["roundtrip_ok"] is True, f"{target_key} native verification missing.")
        require(zipnn["roundtrip_ok"] is True, f"{target_key} ZipNN verification missing.")
        for method_name, method in (
            ("transfer", transferred),
            ("target-specific", native),
            ("ZipNN source", zipnn),
        ):
            require(
                method.get("roundtrip_verified_on") == "uncached warm-up",
                f"{target_key} {method_name} did not verify an uncached warm-up.",
            )

        mapped_tensors = int(mapping["matched_tensors"])
        target_tensors = int(mapping["target_tensors"])
        require(mapped_tensors > 0 and target_tensors > 0, f"{target_key} has no mapped tensors.")
        require(int(mapping["target_exponents"]) > 0, f"{target_key} has no exponents.")
        target_exponents = int(mapping["target_exponents"])
        require(
            int(target.get("target_parameters", 0)) == target_exponents,
            f"{target_key} parameter/exponent count mismatch.",
        )
        require(
            int(transferred["raw_exponents"]) == target_exponents,
            f"{target_key} transfer covers a different exponent count.",
        )
        require(
            int(native["raw_exponents"]) == target_exponents,
            f"{target_key} native control covers a different exponent count.",
        )
        require(
            int(zipnn["exponent_raw_bytes"]) == target_exponents,
            f"{target_key} ZipNN covers a different exponent count.",
        )
        mapping_seconds = finite(mapping["seconds"], f"{target_key} mapping time")
        encoding_seconds = finite(transferred["encode_seconds"], f"{target_key} encode time", positive=True)
        native_seconds = finite(native["encode_seconds"], f"{target_key} native encode time", positive=True)
        zipnn_seconds = finite(zipnn["encode_seconds"], f"{target_key} ZipNN encode time", positive=True)
        ratio_pct = finite(transferred["exponent_pct"], f"{target_key} ratio", positive=True)
        native_ratio_pct = finite(native["exponent_pct"], f"{target_key} native ratio", positive=True)
        zipnn_ratio_pct = finite(zipnn["exponent_ratio_pct"], f"{target_key} ZipNN ratio", positive=True)
        for ratio_name, ratio in (
            ("transfer", ratio_pct),
            ("target-specific", native_ratio_pct),
            ("ZipNN", zipnn_ratio_pct),
        ):
            require(ratio <= 100.0, f"{target_key} {ratio_name} ratio exceeds 100%.")
        close(
            ratio_pct,
            100.0 * float(transferred["compressed_exponent_bytes"]) / float(transferred["raw_exponents"]),
            f"{target_key} transfer ratio formula",
        )
        close(
            native_ratio_pct,
            100.0 * float(native["compressed_exponent_bytes"]) / float(native["raw_exponents"]),
            f"{target_key} native ratio formula",
        )
        close(
            zipnn_ratio_pct,
            100.0 * float(zipnn["exponent_compressed_bytes"]) / float(zipnn["exponent_raw_bytes"]),
            f"{target_key} ZipNN ratio formula",
        )
        speedup_zipnn = zipnn_seconds / encoding_seconds
        speedup_native = native_seconds / encoding_seconds
        delta_zipnn = ratio_pct - zipnn_ratio_pct
        delta_native = ratio_pct - native_ratio_pct
        close(
            comparison["direct_transfer_speedup_vs_zipnn_source"],
            speedup_zipnn,
            f"{target_key} ZipNN speedup",
        )
        close(
            comparison["direct_transfer_speedup_vs_target_specific"],
            speedup_native,
            f"{target_key} native speedup",
        )
        close(
            comparison["direct_transfer_ratio_delta_vs_zipnn_source_pp"],
            delta_zipnn,
            f"{target_key} ZipNN ratio delta",
        )
        close(
            comparison["direct_transfer_ratio_delta_vs_target_specific_pp"],
            delta_native,
            f"{target_key} native ratio delta",
        )

        output.append(
            {
                "family_key": key,
                "family": payload["family"]["label"],
                "source_key": payload["family"]["source_key"],
                "source": payload["source_model"]["label"],
                "source_revision": payload["source_revision"],
                "target_key": target_key,
                "target": target["target_model"]["label"],
                "target_revision": target["target_revision"],
                "tensor_scope": target["tensor_scope"],
                "source_setup_seconds": source_setup_seconds,
                "mapping_seconds": mapping_seconds,
                "mapped_tensors": mapped_tensors,
                "target_tensors": target_tensors,
                "encoding_seconds": encoding_seconds,
                "ratio_pct": ratio_pct,
                "target_specific_encoding_seconds": native_seconds,
                "target_specific_ratio_pct": native_ratio_pct,
                "zipnn_source_encoding_seconds": zipnn_seconds,
                "zipnn_source_ratio_pct": zipnn_ratio_pct,
                "speedup_vs_zipnn_source": speedup_zipnn,
                "speedup_vs_target_specific": speedup_native,
                "ratio_delta_vs_zipnn_source_pp": delta_zipnn,
                "ratio_delta_vs_target_specific_pp": delta_native,
                "source_plan_sha256": payload["source_plan_sha256"],
                "runner_sha256": payload["runner_sha256"],
                "dependency_sha256": json.dumps(payload["dependency_sha256"], sort_keys=True),
                "code_digest": payload["code_provenance"]["digest"],
                "host_fingerprint": payload["host_provenance"]["fingerprint"],
                "source_weights_fingerprint": source_fingerprint,
                "target_weights_fingerprint": target_fingerprint,
                "roundtrip_ok": True,
            }
        )
    return output


def main() -> int:
    args = parse_args()
    rows = []
    payloads = []
    for key in EXPECTED:
        path = args.input_dir / f"{key}.json"
        require(path.is_file(), f"Missing family result: {path}.")
        payload = json.loads(path.read_text())
        payloads.append(payload)
        rows.extend(validate_family(key, payload))

    require(len(rows) == 20, f"Expected 20 targets, found {len(rows)}.")
    require(
        len({row["source_key"] for row in rows} | {row["target_key"] for row in rows}) == 31,
        "Expected 31 unique checkpoints.",
    )
    require(len({row["runner_sha256"] for row in rows}) == 1, "Mixed runner hashes.")
    require(rows[0]["runner_sha256"] == runner_sha256(), "Results use a stale runner.")
    expected_dependencies = dependency_sha256()
    require(
        all(payload.get("dependency_sha256") == expected_dependencies for payload in payloads),
        "Results use stale or mixed dependencies.",
    )
    require(len({row["code_digest"] for row in rows}) == 1, "Mixed code digests.")
    require(rows[0]["code_digest"] == code_provenance()["digest"], "Results use stale codec code.")
    require(len({payload["host"] for payload in payloads}) == 1, "Mixed hosts.")
    current_host = host_provenance()
    require(current_host["hostname"] == "godel", "Summarizer is not running on Godel.")
    require(
        all(payload.get("host_provenance", {}).get("fingerprint") == current_host["fingerprint"] for payload in payloads),
        "Results use a different Godel host fingerprint.",
    )

    csv_path = args.csv or args.input_dir / "summary.csv"
    json_path = args.json or args.input_dir / "summary.json"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_temporary = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with csv_temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    csv_temporary.replace(csv_path)
    atomic_json(
        json_path,
        {
            "complete": True,
            "families": len(EXPECTED),
            "source_plans": len(EXPECTED),
            "targets": len(rows),
            "unique_checkpoints": len(
                {row["source_key"] for row in rows}
                | {row["target_key"] for row in rows}
            ),
            "host": payloads[0]["host"],
            "host_fingerprint": rows[0]["host_fingerprint"],
            "cpu_set": list(range(12)),
            "runner_sha256": rows[0]["runner_sha256"],
            "dependency_sha256": expected_dependencies,
            "code_digest": rows[0]["code_digest"],
            "rows": rows,
        },
    )
    print(json.dumps({"families": len(EXPECTED), "targets": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
