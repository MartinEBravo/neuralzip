"""Check every packed-GPU acceptance gate; missing records never count as passing."""

import argparse
import copy
import json
import math
from pathlib import Path
from statistics import median

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="?")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--baseline-cpu", type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.root:
        parser.error("root is required")
    rows = [
        json.loads(p.read_text())
        for c in ("1", "2")
        for p in sorted((args.root / c).glob("*.json"))
    ]
    result = check(rows)
    proof_paths = (
        args.root / "correctness.json",
        args.root / "module-correctness.json",
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
