"""Table 3: the 18 retained models, pinned inputs and same-device GPU controls."""

import fcntl
import json
from pathlib import Path

from six_models_ablations import check_cpu_row, parser, run_dfloat, run_gpu

from neuralzip._experiment_cpu import benchmark, configure
from neuralzip._experiment_inputs import catalogue, evidence
from neuralzip._experiment_results import (
    check_catalogue_gpu,
    export_cpu,
    export_gpu,
    saved_dfloat_cpu,
)


def main():
    p = parser(__doc__, catalogue()["all"])
    p.set_defaults(
        methods=["new", "transfer", "zipnn"],
        repetitions=9,
        output=Path("scripts/results/reruns/all"),
    )
    args = p.parse_args()
    if args.check_kernels:
        p.error(
            "Use six_models_ablations.py --check-kernels for the shared CUDA checks"
        )
    if args.check or args.export_saved:
        rows = []
        for key in args.models:
            for method in ("new", "zipnn", "bank-physical-v8-gate3"):
                row = evidence(f"all/{key}-{method}.json")
                check_cpu_row(row)
                rows.append(row)
        print(f"PASS: {len(args.models)} Table 3 CPU cohorts")
        rows.extend(saved_dfloat_cpu(args.models, "all"))
        gpu = check_catalogue_gpu(args.models)
        if args.export_saved:
            export_cpu(rows, args.output / "all-models-cpu.csv")
            export_gpu(gpu, args.output / "all-models-gpu.csv")
        return
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "timing.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.device == "gpu":
            assert len(args.models) == len(args.methods) == 1, (
                "One model/method per GPU process"
            )
            assert args.methods[0] in ("bf16", "own", "transfer", "dfloat11")
            run_gpu(args, args.models[0], args.methods[0])
            return
        assert args.repetitions > 0 and set(args.methods) <= {
            "new",
            "transfer",
            "zipnn",
            "dfloat11",
        }
        configure(args.threads, args.cpus)
        rows = []
        for key in args.models:
            for method in args.methods:
                if method == "dfloat11":
                    run_dfloat(
                        args,
                        key,
                        f"all/{key}-dfloat11.json" if args.verify_published else None,
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
                        reference=f"all/{key}-{label}.json"
                        if args.verify_published
                        else None,
                    )
                )
        export_cpu(rows, args.output / "all-models-cpu.csv")


if __name__ == "__main__":
    main()
