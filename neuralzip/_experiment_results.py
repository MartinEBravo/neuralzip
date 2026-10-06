"""Check every packed-GPU acceptance gate; missing records never count as passing."""

import argparse
import copy
import csv
import hashlib
import json
import math
import zipfile
from pathlib import Path
from statistics import median

from neuralzip._experiment_inputs import RESULTS, catalogue, evidence

MODELS = ("qwen35", "gemma3", "dinov3-convnext", "dinov3-vit", "siglip2", "parakeet")
IDENTITY = (
    "manifest_sha256",
    "input_file_sha256",
    "selected_weight_sha256",
    "forward_input_sha256",
    "pinned_model",
    "protocol",
    "official_sha256",
    "gpu_uuid",
    "torch_version",
    "cuda_version",
    "cupy_version",
    "runner_sha256",
    "functional_consumer_adapter",
    "exporter_ast_sha256",
)
FLAGS = (
    "weight_bitwise",
    "output_bitwise",
    "input_unchanged",
    "setup_frozen",
    "full_decode_each_forward",
)


def check(rows):
    errors, comparisons = [], []
    records = {(r["campaign"], r["model"], r["variant"], r["method"]): r for r in rows}
    if len(records) != len(rows):
        errors.append("Duplicate records")
    expected_protocol = {
        "physical_gpu": 1,
        "threads": 8,
        "affinity": list(range(16)),
        "batch": 1,
        "attention": "eager",
        "cache": False,
        "offload": False,
        "warmups": 20,
        "trials": 5,
        "forwards_per_trial": 50,
    }
    if rows:
        expected_protocol["physical_gpu"] = rows[0]["protocol"]["physical_gpu"]
        if expected_protocol["physical_gpu"] not in (0, 1):
            errors.append("Invalid physical GPU")
    for campaign in ("1", "2"):
        for model in MODELS:
            keys = [
                ("control", "bf16"),
                ("control", "dfloat11"),
                ("baseline", "own"),
                ("baseline", "transfer"),
                ("candidate", "own"),
                ("candidate", "transfer"),
            ]
            cohort = [
                records.get((campaign, model, variant, method))
                for variant, method in keys
            ]
            if any(row is None for row in cohort):
                errors.append(f"{campaign}/{model}: incomplete six-control cohort")
                continue
            bf16, dfloat, *variants = cohort
            for row in cohort:
                label = f"{campaign}/{model}/{row['variant']}/{row['method']}"
                if (
                    any(row.get(flag) is not True for flag in FLAGS)
                    or row.get("profiling_in_timing") is not False
                ):
                    errors.append(f"{label}: correctness/timing flags")
                samples = row.get("inference_trials_ms", [])
                if (
                    len(samples) != 5
                    or not all(math.isfinite(s) and s > 0 for s in samples)
                    or row.get("inference_ms") != median(samples)
                ):
                    errors.append(f"{label}: invalid timing samples or median")
                if row["method"] != "bf16":
                    samples = row.get("decode_trials_ms", [])
                    if (
                        len(samples) != 5
                        or not all(math.isfinite(s) and s > 0 for s in samples)
                        or row.get("decode_ms") != median(samples)
                    ):
                        errors.append(
                            f"{label}: invalid decoder-only samples or median"
                        )
                if row.get("protocol") != expected_protocol:
                    errors.append(f"{label}: protocol changed")
                if (
                    row.get("functional_consumer_adapter")
                    != "multihead-attention-out-proj-v1"
                ):
                    errors.append(f"{label}: functional-consumer adapter changed")
                if any(
                    row.get(field) != bf16.get(field) or field not in row
                    for field in IDENTITY
                ):
                    errors.append(
                        f"{label}: source/input/environment identity mismatch"
                    )
            for baseline, candidate in zip(variants[:2], variants[2:], strict=True):
                label = f"{campaign}/{model}/{candidate['method']}"
                protected = (
                    "setup_sha256",
                    "payload_sha256",
                    "compressed_names",
                    "packed_chunks",
                    "groups",
                    "functional_hook_targets",
                )
                if any(
                    candidate.get(f) != baseline.get(f) or f not in candidate
                    for f in protected
                ):
                    errors.append(
                        f"{label}: setup, payload or compressed coverage changed"
                    )
                cpu = {
                    n: h
                    for n, h in baseline["source_hashes"].items()
                    if not n.startswith("gpu/")
                }
                if any(candidate["source_hashes"].get(n) != h for n, h in cpu.items()):
                    errors.append(f"{label}: CPU/native implementation changed")
                limits = {}
                for field in ("allocated_gpu_bytes", "peak_gpu_bytes"):
                    valid = (
                        0 < candidate[field] <= baseline[field] * 1.05
                        and candidate[field] < bf16[field]
                    )
                    limits[field] = valid
                    if not valid:
                        errors.append(f"{label}: {field} memory limit")
                ratio = candidate["inference_ms"] / dfloat["inference_ms"]
                if ratio > 1.10:
                    errors.append(
                        f"{label}: latency {ratio:.6f}x DFloat11 exceeds 1.10x"
                    )
                comparisons.append(
                    {
                        "campaign": campaign,
                        "model": model,
                        "method": candidate["method"],
                        "baseline_ms": baseline["inference_ms"],
                        "candidate_ms": candidate["inference_ms"],
                        "dfloat_ms": dfloat["inference_ms"],
                        "dfloat_ratio": ratio,
                        "memory_pass": all(limits.values()),
                        "baseline_live": baseline["allocated_gpu_bytes"],
                        "candidate_live": candidate["allocated_gpu_bytes"],
                        "baseline_peak": baseline["peak_gpu_bytes"],
                        "candidate_peak": candidate["peak_gpu_bytes"],
                        "bf16_live": bf16["allocated_gpu_bytes"],
                        "bf16_peak": bf16["peak_gpu_bytes"],
                        "correctness": all(
                            candidate.get(flag) is True for flag in FLAGS
                        ),
                    }
                )
    for model in MODELS:
        for variant, method in (
            ("control", "bf16"),
            ("control", "dfloat11"),
            ("baseline", "own"),
            ("baseline", "transfer"),
            ("candidate", "own"),
            ("candidate", "transfer"),
        ):
            a, b = (records.get((c, model, variant, method)) for c in ("1", "2"))
            if (
                a
                and b
                and (
                    a["source_hashes"] != b["source_hashes"]
                    or any(a[f] != b[f] for f in IDENTITY)
                )
            ):
                errors.append(
                    f"{model}/{variant}/{method}: campaigns did not test identical versions"
                )
    return {
        "passed": not errors and len(comparisons) == 24,
        "errors": errors,
        "comparisons": comparisons,
    }


def self_test():
    rows = []
    for campaign in ("1", "2"):
        for model in MODELS:
            for variant, method in (
                ("control", "bf16"),
                ("control", "dfloat11"),
                ("baseline", "own"),
                ("baseline", "transfer"),
                ("candidate", "own"),
                ("candidate", "transfer"),
            ):
                row = dict.fromkeys(IDENTITY, "same")
                row.update(
                    campaign=campaign,
                    model=model,
                    variant=variant,
                    method=method,
                    protocol={
                        "physical_gpu": 1,
                        "threads": 8,
                        "affinity": list(range(16)),
                        "batch": 1,
                        "attention": "eager",
                        "cache": False,
                        "offload": False,
                        "warmups": 20,
                        "trials": 5,
                        "forwards_per_trial": 50,
                    },
                    source_hashes={"compressor.py": "same"},
                    setup_sha256="same",
                    payload_sha256="same",
                    compressed_names=["weight"],
                    packed_chunks=1,
                    groups=1,
                    functional_hook_targets={},
                    functional_consumer_adapter="multihead-attention-out-proj-v1",
                    profiling_in_timing=False,
                    allocated_gpu_bytes=200 if method == "bf16" else 100,
                    peak_gpu_bytes=220 if method == "bf16" else 120,
                    inference_trials_ms=[1] * 5,
                    inference_ms=1,
                    **dict.fromkeys(FLAGS, True),
                )
                if method != "bf16":
                    row.update(decode_trials_ms=[0.2] * 5, decode_ms=0.2)
                rows.append(row)
    assert check(rows)["passed"]
    assert not check(rows[:-1])["passed"]
    assert not check(rows + rows[:1])["passed"]
    for field, value in (
        ("inference_trials_ms", [1.2] * 5),
        ("allocated_gpu_bytes", 106),
        ("peak_gpu_bytes", 127),
        ("payload_sha256", "changed"),
        ("output_bitwise", False),
        ("forward_input_sha256", "changed"),
        ("profiling_in_timing", True),
        ("functional_consumer_adapter", "unapproved"),
        ("functional_hook_targets", {"projection": "changed"}),
        ("decode_trials_ms", []),
        ("decode_ms", 0.3),
    ):
        altered = copy.deepcopy(rows)
        altered[-1][field] = value
        if field == "inference_trials_ms":
            altered[-1]["inference_ms"] = 1.2
        assert not check(altered)["passed"], field
    print(
        "PASS: all-model, two-campaign acceptance checks reject incomplete or noncompliant results"
    )
    for row in rows:
        row["record_sha256"] = "reference"
    ablations = []
    for row in rows:
        if row["variant"] == "control":
            continue
        row = copy.deepcopy(row)
        row["method"] = "sharing" if row["method"] == "own" else "packing"
        row["packed_chunks"] = 0 if row["method"] == "sharing" else 1
        row["ablation_cpu_record_sha256"] = "frozen-cpu-record"
        row["bf16_reference_record_sha256"] = "reference"
        row["bf16_reference_output_sha256"] = "frozen-output"
        ablations.append(row)
    assert check_ablations(ablations, rows)["passed"]
    assert not check_ablations(ablations[:-1], rows)["passed"]
    for field, value in (
        ("setup_sha256", "changed"),
        ("payload_sha256", "changed"),
        ("packed_chunks", 0),
        ("output_bitwise", False),
        ("bf16_reference_record_sha256", "changed"),
        ("inference_ms", 2),
        ("allocated_gpu_bytes", 106),
    ):
        altered = copy.deepcopy(ablations)
        altered[-1][field] = value
        assert not check_ablations(altered, rows)["passed"], field
    print(
        "PASS: frozen sharing/packing controls reject incomplete, changed or incorrect records"
    )


def check_ablations(rows, references):
    """Ablations have no DFloat11 speed target; preserve every measured outcome."""
    errors, comparisons = [], []
    records = {(r["campaign"], r["model"], r["variant"], r["method"]): r for r in rows}
    refs = {
        (r["campaign"], r["model"], r["variant"], r["method"]): r for r in references
    }
    expected = {
        (c, m, v, method)
        for c in ("1", "2")
        for m in MODELS
        for v in ("baseline", "candidate")
        for method in ("sharing", "packing")
    }
    if len(records) != len(rows) or set(records) != expected:
        errors.append("Incomplete, duplicate or unexpected ablation records")
    if len({r.get("runner_sha256") for r in rows}) != 1:
        errors.append("Ablation runner versions differ")
    protected = (
        "setup_sha256",
        "payload_sha256",
        "compressed_names",
        "packed_chunks",
        "groups",
        "functional_hook_targets",
        "ablation_cpu_record_sha256",
        "bf16_reference_output_sha256",
    )
    for campaign in ("1", "2"):
        for model in MODELS:
            bf = refs.get((campaign, model, "control", "bf16"))
            df = refs.get((campaign, model, "control", "dfloat11"))
            if not bf or not df:
                errors.append(f"{campaign}/{model}: missing validated references")
                continue
            for method in ("sharing", "packing"):
                baseline, candidate = (
                    records.get((campaign, model, v, method))
                    for v in ("baseline", "candidate")
                )
                if not baseline or not candidate:
                    continue
                label = f"{campaign}/{model}/{method}"
                for row in (baseline, candidate):
                    if (
                        any(row.get(f) is not True for f in FLAGS)
                        or row.get("profiling_in_timing") is not False
                    ):
                        errors.append(f"{label}: correctness/timing flags")
                    for samples_key, median_key in (
                        ("inference_trials_ms", "inference_ms"),
                        ("decode_trials_ms", "decode_ms"),
                    ):
                        samples = row.get(samples_key, [])
                        if (
                            len(samples) != 5
                            or not all(math.isfinite(s) and s > 0 for s in samples)
                            or row.get(median_key) != median(samples)
                        ):
                            errors.append(f"{label}: invalid {samples_key}")
                    if any(
                        row.get(f) != bf.get(f) or f not in row
                        for f in IDENTITY
                        if f != "runner_sha256"
                    ):
                        errors.append(f"{label}: pinned input/environment mismatch")
                    source = refs.get((campaign, model, row["variant"], "own"))
                    if (
                        not source
                        or row.get("source_hashes") != source["source_hashes"]
                    ):
                        errors.append(f"{label}: runtime/CPU source changed")
                    if row.get("bf16_reference_record_sha256") != bf.get(
                        "record_sha256"
                    ):
                        errors.append(f"{label}: BF16 reference changed")
                    if (
                        method == "sharing"
                        and row.get("packed_chunks") != 0
                        or method == "packing"
                        and row.get("packed_chunks", 0) <= 0
                    ):
                        errors.append(f"{label}: incorrect sharing/packing route")
                if any(
                    candidate.get(f) != baseline.get(f) or f not in candidate
                    for f in protected
                ):
                    errors.append(f"{label}: frozen setup/payload/coverage changed")
                limits = []
                for field in ("allocated_gpu_bytes", "peak_gpu_bytes"):
                    valid = (
                        0 < candidate[field] <= baseline[field] * 1.05
                        and candidate[field] < bf[field]
                    )
                    limits.append(valid)
                    if not valid:
                        errors.append(f"{label}: {field} memory limit")
                comparisons.append(
                    dict(
                        campaign=campaign,
                        model=model,
                        method=method,
                        baseline_ms=baseline["inference_ms"],
                        candidate_ms=candidate["inference_ms"],
                        dfloat_ms=df["inference_ms"],
                        dfloat_ratio=candidate["inference_ms"] / df["inference_ms"],
                        baseline_live=baseline["allocated_gpu_bytes"],
                        candidate_live=candidate["allocated_gpu_bytes"],
                        baseline_peak=baseline["peak_gpu_bytes"],
                        candidate_peak=candidate["peak_gpu_bytes"],
                        bf16_live=bf["allocated_gpu_bytes"],
                        bf16_peak=bf["peak_gpu_bytes"],
                        memory_pass=all(limits),
                        correctness=all(candidate.get(f) is True for f in FLAGS),
                    )
                )
    for model in MODELS:
        for variant in ("baseline", "candidate"):
            for method in ("sharing", "packing"):
                a, b = (records.get((c, model, variant, method)) for c in ("1", "2"))
                if (
                    a
                    and b
                    and (
                        a["source_hashes"] != b["source_hashes"]
                        or any(a.get(f) != b.get(f) for f in (*IDENTITY, *protected))
                    )
                ):
                    errors.append(
                        f"{model}/{variant}/{method}: campaign identities differ"
                    )
    return dict(
        passed=not errors and len(comparisons) == 24,
        errors=errors,
        comparisons=comparisons,
    )


def read_records(root):
    rows = []
    for campaign in ("1", "2"):
        for path in sorted((root / campaign).glob("*.json")):
            data = path.read_bytes()
            rows.append(
                dict(json.loads(data), record_sha256=hashlib.sha256(data).hexdigest())
            )
    return rows


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


def check_archived():
    root = Path(__file__).resolve().parents[1] / "results/packed-gpu/evidence"

    def records(path):
        with zipfile.ZipFile(path) as archive:
            assert archive.testzip() is None
            rows = []
            for name in archive.namelist():
                if not name.endswith(".json"):
                    continue
                data = archive.read(name)
                row = json.loads(data)
                if (
                    isinstance(row, dict)
                    and {
                        "campaign",
                        "model",
                        "variant",
                        "method",
                        "inference_trials_ms",
                    }
                    <= row.keys()
                ):
                    rows.append(
                        dict(row, record_sha256=hashlib.sha256(data).hexdigest())
                    )
            return rows

    references = records(root / "staging-functional/campaign-records.zip")
    ablations = records(root / "ablations-20261005/ablation-records.zip")
    self_test()
    main_check = check(references)
    ablation_check = check_ablations(ablations, references)
    assert main_check["passed"], main_check["errors"]
    assert ablation_check["passed"], ablation_check["errors"]
    proof = json.loads((root / "staging-functional/correctness.json").read_text())
    modules = json.loads(
        (root / "staging-functional/module-correctness.json").read_text()
    )
    assert proof["cpu"] == json.loads((root / "baseline-cpu.json").read_text())["cpu"]
    assert modules["_runtime_sha256"] == proof["runtime_sha256"]
    assert proof["gpu"]["segment_boundaries"]["code_bits"] == [12, 32, 33, 56]
    assert all(
        proof["gpu"]["segment_boundaries"][k]
        for k in ("bitwise", "odd_tails", "staged_and_direct")
    )
    for row in references:
        if row["variant"] == "candidate":
            assert all(
                row["source_hashes"][name] == digest
                for name, digest in proof["runtime_sha256"].items()
            )
    print(
        "PASS: 72 GPU controls, 48 sharing/packing controls, 24 acceptance and 24 ablation comparisons"
    )
    return dict(
        status="PASS",
        gpu_records=len(references),
        ablation_records=len(ablations),
        comparisons=len(main_check["comparisons"]),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="?")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--baseline-cpu", type=Path)
    parser.add_argument("--ablations", action="store_true")
    parser.add_argument("--reference-root", type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.root:
        parser.error("root is required")
    rows = read_records(args.root)
    if args.ablations:
        if not args.reference_root:
            parser.error("--ablations requires --reference-root")
        references = read_records(args.reference_root)
        result = check_ablations(rows, references)
        result["errors"].extend(check(references)["errors"])
    else:
        result = check(rows)
    proof_root = args.reference_root if args.ablations else args.root
    proof_paths = (
        proof_root / "correctness.json",
        proof_root / "module-correctness.json",
    )
    if (
        not args.baseline_cpu
        or not args.baseline_cpu.exists()
        or not all(p.exists() for p in proof_paths)
    ):
        result["errors"].append(
            "Missing independent CPU archive, CUDA edge-case or supported-module proof"
        )
    else:
        baseline = json.loads(args.baseline_cpu.read_text())
        proof, modules = (json.loads(p.read_text()) for p in proof_paths)
        if proof["cpu"] != baseline["cpu"] or set(proof["cpu"]) != {
            "float16",
            "bfloat16",
            "float32",
        }:
            result["errors"].append("CPU archives are not byte-identical")
        segments = proof["gpu"].get("segment_boundaries", {})
        if segments.get("code_bits") != [12, 32, 33, 56] or not all(
            segments.get(f) is True
            for f in ("bitwise", "odd_tails", "staged_and_direct")
        ):
            result["errors"].append("Missing long-code/tail/direct-fallback coverage")
        wrapper = proof["gpu"].get("public_hf_wrapper", {})
        if not all(
            wrapper.get(f) is True
            for f in (
                "packed",
                "ordinary_load",
                "compressed_reload",
                "weights_bitwise",
                "output_bitwise",
            )
        ):
            result["errors"].append("Compressed public save/reload failed")
        for name in (
            "resnet18",
            "convnext_tiny",
            "efficientnet_b0",
            "conv2d",
            "sequential",
            "nested",
            "gpt2",
            "multihead_attention",
        ):
            if not all(
                modules.get(name, {}).get(f) is True
                for f in ("discovery", "weights_bitwise", "outputs_bitwise")
            ):
                result["errors"].append(f"Missing supported-module proof: {name}")
        if (
            modules.get("multihead_attention", {}).get("functional_consumer_bitwise")
            is not True
        ):
            result["errors"].append(
                "Missing packed functional-consumer regression proof"
            )
        for row in rows:
            if row["variant"] == "candidate":
                if any(
                    row["source_hashes"].get(n) != h
                    for n, h in proof["runtime_sha256"].items()
                ):
                    result["errors"].append(
                        "CUDA proof did not test the measured candidate"
                    )
                if modules["_runtime_sha256"] != proof["runtime_sha256"]:
                    result["errors"].append(
                        "Module proof did not test the measured candidate"
                    )
    result["passed"] = result["passed"] and not result["errors"]
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
