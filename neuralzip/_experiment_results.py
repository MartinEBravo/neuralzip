"""Export and validate the saved paper measurements."""

import csv
import json
from statistics import median

from neuralzip._experiment_inputs import RESULTS, catalogue, evidence


def export_cpu(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "model",
        "method",
        "exponent_ratio_pct",
        "full_ratio_pct",
        "setup_seconds",
        "encode_seconds",
        "cold_decode_seconds",
        "setup_codebooks",
        "archive_codebooks",
        "exponent_payload_bytes",
        "exponent_codebook_metadata_bytes",
        "exponent_chunk_metadata_bytes",
    )
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            values = {field: row.get(field) for field in fields}
            for field, value in values.items():
                if isinstance(value, dict):
                    values[field] = value["median"]
            writer.writerow(values)


def saved_dfloat_cpu(keys, prefix):
    """Include available official CPU controls; never synthesize missing cells."""
    members = json.loads((RESULTS / "paper/evidence-members.json").read_text())
    rows = []
    for key in keys:
        name = f"{prefix}/{key}-dfloat11.json"
        if name not in members:
            continue
        row = evidence(name)
        assert row["roundtrip_exact"] and row["input_unchanged"]
        assert (
            row["input"]["bf16_manifest_sha256"]
            == catalogue()["inputs"][key]["bf16_manifest_sha256"]
        )
        assert row["official_commit"] == "457733886ce6ebc6d8dda1621fad1ffa2661e028"
        assert (
            row["official_utils_sha256"]
            == "c50216fda9da2a516131ce84afec6003f566f6e0f88cb4fa8178bc18f83851fe"
        )
        assert (
            row["exponent_ratio_pct"]
            == 100 * row["exponent_compressed_bytes"] / row["exponent_raw_bytes"]
        )
        rows.append(row)
    return rows


def export_phases(rows, path):
    fields = ("model", "total_seconds", *rows[0]["phase_seconds"])
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            dict(
                model=row["model"],
                total_seconds=row["total_seconds"],
                **row["phase_seconds"],
            )
            for row in rows
        )


def check_catalogue_gpu(keys):
    members = json.loads((RESULTS / "paper/evidence-members.json").read_text())
    rows = []
    complete = []
    for key in keys:
        names = [
            f"all/gpu/{key}-{m}.json" for m in ("bf16", "own", "foreign", "dfloat11")
        ]
        available = [n for n in names if n in members]
        if not available:
            continue
        assert len(available) == 4, (
            "Saved GPU evidence must contain a whole same-device cohort"
        )
        cohort = [evidence(n) for n in names]
        bf16 = cohort[0]
        for row in cohort:
            for field in (
                "weight_bitwise",
                "output_bitwise",
                "input_unchanged",
                "setup_frozen",
                "full_decode_each_forward",
            ):
                assert row[field] is True, (key, field)
            assert row["profiling_in_timing"] is False
            for field in (
                "gpu_uuid",
                "protocol",
                "selected_weight_sha256",
                "manifest_sha256",
                "pinned_model",
                "forward_input_sha256",
                "torch_version",
                "cuda_version",
                "cupy_version",
            ):
                assert row[field] == bf16[field], (key, field)
            assert (
                len(row["inference_trials_ms"]) == 5
                and median(row["inference_trials_ms"]) == row["inference_ms"]
            )
            if row["method"] != "bf16":
                assert (
                    len(row["decode_trials_ms"]) == 5
                    and median(row["decode_trials_ms"]) == row["decode_ms"]
                )
            assert row["peak_gpu_bytes"] >= row["allocated_gpu_bytes"] > 0
            assert (
                row["manifest_sha256"]
                == catalogue()["inputs"][key]["bf16_manifest_sha256"]
            )
        rows.extend(cohort)
        complete.append(key)
    print(
        f"PASS: {len(complete)} saved whole GPU cohorts; {len(keys) - len(complete)} have no GPU evidence in this snapshot"
    )
    return rows


def export_gpu(rows, path):
    with path.open("x", newline="") as stream:
        fields = (
            "model",
            "method",
            "gpu_name",
            "physical_gpu",
            "inference_ms",
            "decode_ms",
            "allocated_gpu_bytes",
            "peak_gpu_bytes",
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            values = {field: row.get(field) for field in fields}
            values["physical_gpu"] = row["protocol"]["physical_gpu"]
            writer.writerow(values)
