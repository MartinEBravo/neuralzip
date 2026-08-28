#!/usr/bin/env python3
"""Merge and render the four-model training trajectories.

The script does not run experiments.  It combines the language trajectories
produced by ``run_checkpoint_language.py`` and
``run_federated_language.py`` with the ResNet-18/34
trajectories. Before writing an artifact it checks that every requested
epoch/round contains exactly one valid row for each compressor.

The resulting figures use the thesis layout requested for both experiments:

* columns: GPT-2, BERT Base, ResNet-18, and ResNet-34;
* top row: exponent-field compression ratio;
* bottom row: direct encoding time.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

COMPRESSOR_ZIPNN = "ZipNN"
COMPRESSOR_NEURALZIP = "NeuralZip"
COMPRESSORS = (COMPRESSOR_ZIPNN, COMPRESSOR_NEURALZIP)
MODEL_ORDER = ("gpt2", "bert-base", "resnet18", "resnet34")
MODEL_LABELS = {
    "gpt2": "GPT-2",
    "bert-base": "BERT Base",
    "resnet18": "ResNet-18",
    "resnet34": "ResNet-34",
}
COLORS = {"ZipNN": "#4C78A8", "NeuralZip": "#E45756"}
MARKERS = {"ZipNN": "o", "NeuralZip": "s"}

CHECKPOINT_REQUIRED_COLUMNS = {
    "model",
    "epoch",
    "compressor",
    "exponent_ratio_pct",
    "encode_seconds",
    "roundtrip_exact",
    "input_unchanged",
    "status",
}
FEDERATED_REQUIRED_COLUMNS = {
    "architecture",
    "round",
    "compressor",
    "exponent_ratio_pct",
    "downlink_encode_seconds",
    "roundtrip_ok",
}
CANONICAL_FIELDS = (
    "experiment",
    "model",
    "model_label",
    "step",
    "compressor",
    "exponent_ratio_pct",
    "encode_seconds",
    "source_path",
)


@dataclass(frozen=True)
class InputSpec:
    path: Path
    models: tuple[str, ...]
    first_step: int
    last_step: int
    experiment: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--language-checkpoint-csv",
        type=Path,
        default=(ROOT / "results" / "checkpoints-wikitext2" / "checkpoint_results.csv"),
    )
    parser.add_argument(
        "--vision-checkpoint-csv",
        type=Path,
        default=(ROOT / "results" / "checkpoints-cifar100" / "checkpoint_results.csv"),
    )
    parser.add_argument(
        "--language-federated-root",
        type=Path,
        default=(ROOT / "results" / "federated-wikitext2-200x10"),
        help="Root containing gpt2/ and bert-base/ result directories.",
    )
    parser.add_argument(
        "--vision-federated-root",
        type=Path,
        default=(ROOT / "results" / "federated-cifar10-200x10"),
        help="Root containing resnet18/ and resnet34/ result directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "training-trajectories",
    )
    parser.add_argument("--language-checkpoint-epochs", type=int, default=20)
    parser.add_argument("--vision-checkpoint-epochs", type=int, default=50)
    parser.add_argument("--federated-rounds", type=int, default=200)
    parser.add_argument(
        "--checkpoint-only",
        action="store_true",
        help="Validate and render only the checkpoint figure.",
    )
    parser.add_argument(
        "--federated-only",
        action="store_true",
        help="Validate and render only the federated figure.",
    )
    args = parser.parse_args()
    if args.checkpoint_only and args.federated_only:
        parser.error("--checkpoint-only and --federated-only are mutually exclusive")
    for field in (
        "language_checkpoint_epochs",
        "vision_checkpoint_epochs",
        "federated_rounds",
    ):
        if int(getattr(args, field)) < 1:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    return args


def _read_rows(path: Path) -> tuple[list[dict[str, str]], set[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing result file: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"Result file is empty: {path}")
    return rows, fields


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _finite_nonnegative(value: object, *, field: str, path: Path) -> float:
    try:
        parsed = float(str(value))
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid {field}={value!r} in {path}") from error
    if parsed < 0 or not math.isfinite(parsed):
        raise ValueError(f"Invalid {field}={value!r} in {path}")
    return parsed


def _require_columns(path: Path, observed: set[str], required: set[str]) -> None:
    missing = sorted(required - observed)
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def _canonicalize_checkpoint(spec: InputSpec) -> list[dict[str, object]]:
    rows, fields = _read_rows(spec.path)
    _require_columns(spec.path, fields, CHECKPOINT_REQUIRED_COLUMNS)
    selected: list[dict[str, object]] = []
    for row in rows:
        model = str(row["model"])
        if model not in spec.models:
            continue
        try:
            step = int(row["epoch"])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid epoch={row['epoch']!r} in {spec.path}"
            ) from error
        if not spec.first_step <= step <= spec.last_step:
            continue
        if str(row["status"]).strip() != "ok":
            raise ValueError(
                f"Non-ok checkpoint row for {model} epoch {step} in {spec.path}"
            )
        if not _truthy(row["roundtrip_exact"]) or not _truthy(row["input_unchanged"]):
            raise ValueError(
                f"Invalid checkpoint verification for {model} epoch {step} "
                f"in {spec.path}"
            )
        selected.append(
            {
                "experiment": spec.experiment,
                "model": model,
                "model_label": MODEL_LABELS[model],
                "step": step,
                "compressor": str(row["compressor"]),
                "exponent_ratio_pct": _finite_nonnegative(
                    row["exponent_ratio_pct"],
                    field="exponent_ratio_pct",
                    path=spec.path,
                ),
                "encode_seconds": _finite_nonnegative(
                    row["encode_seconds"], field="encode_seconds", path=spec.path
                ),
                "source_path": str(spec.path.resolve()),
            }
        )
    _validate_complete(selected, spec)
    return selected


def _canonicalize_federated(spec: InputSpec) -> list[dict[str, object]]:
    rows, fields = _read_rows(spec.path)
    _require_columns(spec.path, fields, FEDERATED_REQUIRED_COLUMNS)
    selected: list[dict[str, object]] = []
    for row in rows:
        model = str(row["architecture"])
        if model not in spec.models:
            continue
        try:
            step = int(row["round"])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid round={row['round']!r} in {spec.path}"
            ) from error
        if not spec.first_step <= step <= spec.last_step:
            continue
        if not _truthy(row["roundtrip_ok"]):
            raise ValueError(
                f"Failed federated round trip for {model} round {step} in {spec.path}"
            )
        selected.append(
            {
                "experiment": spec.experiment,
                "model": model,
                "model_label": MODEL_LABELS[model],
                "step": step,
                "compressor": str(row["compressor"]),
                "exponent_ratio_pct": _finite_nonnegative(
                    row["exponent_ratio_pct"],
                    field="exponent_ratio_pct",
                    path=spec.path,
                ),
                "encode_seconds": _finite_nonnegative(
                    row["downlink_encode_seconds"],
                    field="downlink_encode_seconds",
                    path=spec.path,
                ),
                "source_path": str(spec.path.resolve()),
            }
        )
    _validate_complete(selected, spec)
    return selected


def _validate_complete(rows: Sequence[Mapping[str, object]], spec: InputSpec) -> None:
    expected = {
        (model, step, compressor)
        for model in spec.models
        for step in range(spec.first_step, spec.last_step + 1)
        for compressor in COMPRESSORS
    }
    observed_list = [
        (str(row["model"]), int(row["step"]), str(row["compressor"])) for row in rows
    ]
    observed = set(observed_list)
    duplicates = sorted(key for key in observed if observed_list.count(key) != 1)
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    if duplicates or missing or unexpected:
        details = []
        if missing:
            details.append(
                f"missing={missing[:8]}" + (" ..." if len(missing) > 8 else "")
            )
        if duplicates:
            details.append(
                f"duplicates={duplicates[:8]}" + (" ..." if len(duplicates) > 8 else "")
            )
        if unexpected:
            details.append(
                f"unexpected={unexpected[:8]}" + (" ..." if len(unexpected) > 8 else "")
            )
        raise ValueError(
            f"Incomplete {spec.experiment} input {spec.path}: {'; '.join(details)}"
        )


def _write_canonical_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=CANONICAL_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in CANONICAL_FIELDS})
    os.replace(temporary, path)


def _render(
    rows: Sequence[Mapping[str, object]],
    *,
    output_stem: Path,
    x_label: str,
    time_label: str,
    ratio_tick: float | None = None,
    time_zero: bool = False,
) -> tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ratio_limits: tuple[float, float] | None = None
    ratio_ticks: list[float] | None = None
    if ratio_tick is not None:
        if ratio_tick <= 0:
            raise ValueError("ratio_tick must be positive")
        ratio_values = [float(row["exponent_ratio_pct"]) for row in rows]
        lower = math.floor(min(ratio_values) / ratio_tick) * ratio_tick
        upper = math.ceil(max(ratio_values) / ratio_tick) * ratio_tick
        if upper <= lower:
            upper = lower + ratio_tick
        ratio_limits = (lower, upper)
        tick_count = round((upper - lower) / ratio_tick)
        ratio_ticks = [lower + index * ratio_tick for index in range(tick_count + 1)]

    fig, axes = plt.subplots(2, 4, figsize=(14.4, 6.2), sharex="col")
    for column, model in enumerate(MODEL_ORDER):
        ratio_axis = axes[0, column]
        time_axis = axes[1, column]
        ratio_axis.set_title(MODEL_LABELS[model], fontsize=11)
        for compressor in COMPRESSORS:
            group = sorted(
                (
                    row
                    for row in rows
                    if row["model"] == model and row["compressor"] == compressor
                ),
                key=lambda row: int(row["step"]),
            )
            steps = [int(row["step"]) for row in group]
            markevery = max(1, len(group) // 10)
            style = {
                "color": COLORS[compressor],
                "marker": MARKERS[compressor],
                "markersize": 2.8,
                "markevery": markevery,
                "linewidth": 1.5,
                "label": compressor,
            }
            ratio_axis.plot(
                steps,
                [float(row["exponent_ratio_pct"]) for row in group],
                **style,
            )
            time_axis.plot(
                steps,
                [float(row["encode_seconds"]) for row in group],
                **style,
            )
        if ratio_limits is not None and ratio_ticks is not None:
            ratio_axis.set_ylim(*ratio_limits)
            ratio_axis.set_yticks(ratio_ticks)
        if time_zero:
            time_axis.set_ylim(bottom=0.0)
        for axis in (ratio_axis, time_axis):
            axis.grid(True, axis="y", color="#dddddd", linewidth=0.6)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
        time_axis.set_xlabel(x_label)
    axes[0, 0].set_ylabel("Exponent ratio (%)")
    axes[1, 0].set_ylabel(time_label)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_stem.with_suffix(".pdf")
    png_path = output_stem.with_suffix(".png")
    pdf_tmp = pdf_path.with_name(pdf_path.stem + ".tmp.pdf")
    png_tmp = png_path.with_name(png_path.stem + ".tmp.png")
    fig.savefig(
        pdf_tmp,
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None, "Creator": "NeuralZip"},
    )
    fig.savefig(
        png_tmp,
        dpi=200,
        bbox_inches="tight",
        metadata={"Software": "NeuralZip"},
    )
    plt.close(fig)
    os.replace(pdf_tmp, pdf_path)
    os.replace(png_tmp, png_path)
    return pdf_path, png_path


def build_checkpoint_outputs(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    rows = []
    rows.extend(
        _canonicalize_checkpoint(
            InputSpec(
                path=args.language_checkpoint_csv,
                models=("gpt2", "bert-base"),
                first_step=0,
                last_step=int(args.language_checkpoint_epochs),
                experiment="checkpoint",
            )
        )
    )
    rows.extend(
        _canonicalize_checkpoint(
            InputSpec(
                path=args.vision_checkpoint_csv,
                models=("resnet18", "resnet34"),
                first_step=0,
                last_step=int(args.vision_checkpoint_epochs),
                experiment="checkpoint",
            )
        )
    )
    rows.sort(
        key=lambda row: (
            MODEL_ORDER.index(str(row["model"])),
            int(row["step"]),
            COMPRESSORS.index(str(row["compressor"])),
        )
    )
    csv_path = args.output_dir / "checkpoint_trajectories.csv"
    _write_canonical_csv(csv_path, rows)
    pdf_path, png_path = _render(
        rows,
        output_stem=args.output_dir / "figures" / "checkpoint_training",
        x_label="Epoch",
        time_label="Encoding time (s)",
    )
    return csv_path, pdf_path, png_path


def build_federated_outputs(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    rows = []
    for root, models in (
        (args.language_federated_root, ("gpt2", "bert-base")),
        (args.vision_federated_root, ("resnet18", "resnet34")),
    ):
        for model in models:
            rows.extend(
                _canonicalize_federated(
                    InputSpec(
                        path=root / model / "federated_round_results.csv",
                        models=(model,),
                        first_step=1,
                        last_step=int(args.federated_rounds),
                        experiment="federated",
                    )
                )
            )
    rows.sort(
        key=lambda row: (
            MODEL_ORDER.index(str(row["model"])),
            int(row["step"]),
            COMPRESSORS.index(str(row["compressor"])),
        )
    )
    csv_path = args.output_dir / "federated_trajectories.csv"
    _write_canonical_csv(csv_path, rows)
    pdf_path, png_path = _render(
        rows,
        output_stem=args.output_dir / "figures" / "federated_training",
        x_label="Round",
        time_label="Encoding time (s)",
        ratio_tick=0.1,
        time_zero=True,
    )
    return csv_path, pdf_path, png_path


def main() -> None:
    args = _parse_args()
    outputs: list[Path] = []
    if not args.federated_only:
        outputs.extend(build_checkpoint_outputs(args))
    if not args.checkpoint_only:
        outputs.extend(build_federated_outputs(args))
    for path in outputs:
        print(path.resolve())


if __name__ == "__main__":
    main()
