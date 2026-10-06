"""Figure 6: 70 exact pinned BF16 checkpoints and frozen source-only transfer."""

import argparse
import csv
import fcntl
from pathlib import Path

from six_models_ablations import check_cpu_row

from neuralzip._experiment_cpu import benchmark, configure
from neuralzip._experiment_inputs import catalogue, evidence

FAMILIES = ("olmo2-1b", "tri-1.9b", "pythia-1b", "gaperon-1.5b")


def export_saved(keys, specs, out):
    out.mkdir(parents=True, exist_ok=True)
    for family in FAMILIES:
        selected = [k for k in keys if specs[k]["model"] == family]
        if not selected:
            continue
        with (out / f"checkpoint-bf16-{family}.csv").open("x", newline="") as stream:
            fields = (
                "step_k",
                "neuralzip_ratio",
                "neuralzip_seconds",
                "zipnn_ratio",
                "zipnn_seconds",
            )
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for key in selected:
                own = evidence(f"checkpoints/{key}-transfer.json")
                zipnn = evidence(f"checkpoints/{key}-zipnn.json")
                writer.writerow(
                    dict(
                        step_k=specs[key]["step"] / 1000,
                        neuralzip_ratio=own["exponent_ratio_pct"],
                        neuralzip_seconds=own["encode_seconds"]["median"],
                        zipnn_ratio=zipnn["exponent_ratio_pct"],
                        zipnn_seconds=zipnn["encode_seconds"]["median"],
                    )
                )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--families", nargs="+", choices=FAMILIES, default=FAMILIES)
    p.add_argument(
        "--steps", nargs="+", type=int, help="Select a few original training steps"
    )
    p.add_argument(
        "--methods",
        nargs="+",
        choices=("transfer", "zipnn"),
        default=["transfer", "zipnn"],
    )
    p.add_argument("--output", type=Path, default=Path("results/reruns/checkpoints"))
    p.add_argument("--inputs", type=Path, default=Path(".cache/experiments/inputs"))
    p.add_argument("--cache", type=Path, default=Path(".cache/experiments/hf"))
    p.add_argument("--repetitions", type=int, default=5)
    p.add_argument("--cpus", nargs="+", type=int, default=list(range(12)))
    p.add_argument("--verify-published", action="store_true")
    p.add_argument("--check", action="store_true")
    p.add_argument(
        "--export-saved",
        action="store_true",
        help="Export verified Figure 6 coordinates; no weights or rerun",
    )
    args = p.parse_args()
    specs = catalogue()["checkpoints"]
    keys = [
        k
        for k in sorted(specs, key=lambda k: (specs[k]["model"], specs[k]["step"]))
        if specs[k]["model"] in args.families
        and (args.steps is None or specs[k]["step"] in args.steps)
    ]
    assert keys, "No pinned checkpoint matches the requested steps"
    if args.check or args.export_saved:
        for key in keys:
            for method in ("transfer", "zipnn"):
                check_cpu_row(evidence(f"checkpoints/{key}-{method}.json"))
        print(
            f"PASS: {len(keys)} pinned checkpoints and their recorded NeuralZip/ZipNN samples"
        )
        if args.export_saved:
            export_saved(keys, specs, args.output)
        return
    configure(12, args.cpus)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "timing.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows = {family: [] for family in args.families}
        for key in keys:
            measurements = {}
            for method in args.methods:
                measurements[method] = benchmark(
                    key,
                    method,
                    args.output,
                    args.cache,
                    args.inputs,
                    args.repetitions,
                    gate_bits=specs[key]["gate_bits"],
                    reference=f"checkpoints/{key}-{method}.json"
                    if args.verify_published
                    else None,
                )
            if set(measurements) == {"transfer", "zipnn"}:
                rows[specs[key]["model"]].append(
                    dict(
                        step_k=specs[key]["step"] / 1000,
                        neuralzip_ratio=measurements["transfer"]["exponent_ratio_pct"],
                        neuralzip_seconds=measurements["transfer"]["encode_seconds"][
                            "median"
                        ],
                        zipnn_ratio=measurements["zipnn"]["exponent_ratio_pct"],
                        zipnn_seconds=measurements["zipnn"]["encode_seconds"]["median"],
                    )
                )
        for family, values in rows.items():
            if values:
                with (args.output / f"checkpoint-bf16-{family}.csv").open(
                    "x"
                ) as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(values[0]))
                    writer.writeheader()
                    writer.writerows(values)


if __name__ == "__main__":
    main()
