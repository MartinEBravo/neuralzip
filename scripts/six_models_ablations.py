"""Table 2 and B.2--B.5: pinned six-model compression, ablations and inference."""

import argparse
import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path

from neuralzip._experiment_cpu import benchmark, configure, summarize
from neuralzip._experiment_inputs import catalogue, evidence
from neuralzip._experiment_results import (
    export_cpu,
    export_phases,
    saved_dfloat_cpu,
)

METHODS = (
    "new",
    "transfer",
    "zipnn",
    "sharing",
    "packing",
    "level-global",
    "level-tensor",
    "js-0.005",
    "js-0.010",
    "js-0.025",
    "js-0.075",
    "js-0.100",
    "js-0.150",
    "js-0.200",
)


def run_gpu(args, key, method):
    # One fresh process per method preserves cold runtime initialization/allocator.
    command = [
        sys.executable,
        "-m",
        "neuralzip._experiment_gpu",
        "--model",
        key,
        "--method",
        method,
        "--gpu",
        str(args.gpu),
        "--cache",
        str(args.cache),
        "--inputs",
        str(args.inputs),
        "--setups",
        str(args.setups or args.output),
        "--output",
        str(args.output),
        "--campaign",
        args.campaign,
    ]
    subprocess.run(
        command, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu)), check=True
    )


def run_dfloat(args, key, reference=None):
    command = [
        sys.executable,
        "-m",
        "neuralzip._experiment_dfloat",
        "--model",
        key,
        "--output",
        str(args.output),
        "--cache",
        str(args.cache),
        "--inputs",
        str(args.inputs),
        "--threads",
        str(args.threads),
        "--cpus",
        *map(str, args.cpus),
    ]
    if reference:
        command += ["--reference", reference]
    subprocess.run(command, check=True)


def parser(description=__doc__, models=None):
    p = argparse.ArgumentParser(description=description)
    models = models or catalogue()["six"]
    p.add_argument("--models", nargs="+", choices=models, default=models)
    p.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    p.add_argument(
        "--methods",
        nargs="+",
        choices=(*METHODS, "bf16", "own", "dfloat11"),
        default=list(METHODS),
    )
    p.add_argument("--output", type=Path, default=Path("scripts/results/reruns/six"))
    p.add_argument("--cache", type=Path, default=Path(".cache/experiments/hf"))
    p.add_argument("--inputs", type=Path, default=Path(".cache/experiments/inputs"))
    p.add_argument("--setups", type=Path)
    p.add_argument("--repetitions", type=int, default=20)
    p.add_argument("--threads", type=int, default=12)
    p.add_argument("--cpus", type=int, nargs="+", default=list(range(12)))
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--campaign", choices=("1", "2", "pilot"), default="1")
    p.add_argument(
        "--verify-published",
        action="store_true",
        help="Require identical compressed sizes and archive SHA256",
    )
    p.add_argument(
        "--check",
        action="store_true",
        help="Recompute saved statistics only; no weights or benchmark",
    )
    p.add_argument(
        "--check-kernels",
        action="store_true",
        help="Run synthetic CUDA/module correctness checks, not a timing campaign",
    )
    p.add_argument(
        "--export-saved",
        action="store_true",
        help="Export verified published CPU tables; no weights or rerun",
    )
    return p


def check_cpu_row(row):
    for metric in row["samples"][0]:
        from_values = [s[metric] for s in row["samples"]]
        # Saved datasets use their actual repetition count, not the CLI override.
        assert summarize(from_values) == row[metric], (row["model"], metric)
    assert row["roundtrip_bitwise"] and row["input_unchanged"]
    sizes = row["sizes"]
    assert (
        row["exponent_ratio_pct"]
        == 100 * sizes["exponent_compressed_bytes"] / sizes["exponent_raw_bytes"]
    )


def main():
    args = parser().parse_args()
    if args.check_kernels:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu))
        for module in ("_experiment_kernel_checks", "_experiment_module_checks"):
            extra = ["--cuda"] if module == "_experiment_module_checks" else []
            subprocess.run(
                [sys.executable, "-m", "neuralzip." + module, *extra],
                env=env,
                check=True,
            )
        return
    if args.check or args.export_saved:
        rows = []
        for key in args.models:
            for method in METHODS:
                label = "bank-physical-v8-gate3" if method == "transfer" else method
                row = evidence(f"six/{key}-{label}.json")
                check_cpu_row(row)
                rows.append(row)
        print(
            f"PASS: {14 * len(args.models)} six-suite/ablation records, input identities, sizes and sample statistics"
        )
        phases = []
        for key in args.models:
            phase = evidence(f"six/{key}-phases.json")
            own = evidence(f"six/{key}-new.json")
            assert phase["total_seconds"] == own["setup_seconds"]["median"]
            assert (
                phase["shared_setup_execution_id"] == own["shared_setup_execution_id"]
            )
            assert phase["setup_sha256"] == own["own_setup_sha256"]
            assert (
                phase["profiler_ast_stripping_exact"]
                and phase["original_codebooks_and_assignments_identical"]
            )
            phases.append(phase)
        rows.extend(saved_dfloat_cpu(args.models, "six"))
        if args.export_saved:
            export_cpu(rows, args.output / "six-models-ablations.csv")
            export_phases(phases, args.output / "setup-phases.csv")
        return
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "timing.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.device == "gpu":
            assert len(args.models) == len(args.methods) == 1, (
                "Run each GPU method in a fresh process"
            )
            assert args.methods[0] in (
                "bf16",
                "own",
                "transfer",
                "sharing",
                "packing",
                "dfloat11",
            )
            run_gpu(args, args.models[0], args.methods[0])
            return
        assert args.repetitions > 0 and set(args.methods) <= set(METHODS) | {"dfloat11"}
        configure(args.threads, args.cpus)
        rows = []
        for key in args.models:
            for method in args.methods:
                if method == "dfloat11":
                    run_dfloat(
                        args,
                        key,
                        f"six/{key}-dfloat11.json" if args.verify_published else None,
                    )
                    rows.append(
                        json.loads((args.output / f"{key}-dfloat11.json").read_text())
                    )
                    continue
                label = "bank-physical-v8-gate3" if method == "transfer" else method
                rows.append(
                    benchmark(
                        key,
                        method,
                        args.output,
                        args.cache,
                        args.inputs,
                        args.repetitions,
                        args.threads,
                        reference=f"six/{key}-{label}.json"
                        if args.verify_published
                        else None,
                    )
                )
        export_cpu(rows, args.output / "six-models-ablations.csv")


if __name__ == "__main__":
    main()
