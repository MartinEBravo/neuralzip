"""Task-independent plumbing for the final checkpoint runners.

Model construction and training remain explicit in the vision and language
runners. Persistence, frozen-plan lifecycle, compression trajectories,
validation, and rendering live here so the two protocols cannot drift.
"""

from __future__ import annotations

import gc
import json
import random
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from neuralzip.experiment_io import (
    atomic_write_json,
    atomic_write_csv,
    read_csv,
)

from examples.checkpoint_loading import load_torch_checkpoint


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def limited(dataset: Dataset, limit: int | None) -> Dataset:
    if limit is None or int(limit) >= len(dataset):
        return dataset
    return Subset(dataset, range(max(0, int(limit))))


def atomic_torch_save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def upsert_model_epoch_row(
    path: Path,
    row: Mapping,
    *,
    fields: Iterable[str],
    model_order: Sequence[str],
    include_compressor: bool,
    lineterminator: str | None = None,
) -> None:
    """Upsert one result while preserving the runners' stable CSV ordering."""

    target_model = str(row["model"])
    target_epoch = int(row["epoch"])
    target_compressor = str(row["compressor"]) if include_compressor else None

    def is_target(existing: Mapping) -> bool:
        if existing.get("model") != target_model:
            return False
        if int(existing.get("epoch", -1)) != target_epoch:
            return False
        return not include_compressor or existing.get("compressor") == target_compressor

    rows = [existing for existing in read_csv(path) if not is_target(existing)]
    rows.append(dict(row))

    def sort_key(item: Mapping) -> tuple:
        base = (
            model_order.index(str(item["model"])),
            int(item["epoch"]),
        )
        if include_compressor:
            return (*base, str(item["compressor"]))
        return base

    rows.sort(key=sort_key)
    atomic_write_csv(
        path,
        fields,
        rows,
        lineterminator=lineterminator,
    )


def rng_state(loader_generator: torch.Generator) -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "loader_generator": loader_generator.get_state(),
    }


def restore_rng_state(state: Mapping, loader_generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])
    loader_generator.set_state(state["loader_generator"])


def checkpoint_epoch(path: Path) -> int:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    try:
        return int(checkpoint.get("epoch", int(path.stem.rsplit("_", 1)[-1])))
    finally:
        del checkpoint


def checkpoint_files(checkpoint_dir: Path, maximum_epoch: int) -> dict[int, Path]:
    files = {}
    for path in checkpoint_dir.glob("*.pt"):
        if path.name.endswith("training_state.pt") or path.name.endswith("latest_state.pt"):
            continue
        try:
            epoch = checkpoint_epoch(path)
        except (ValueError, KeyError, RuntimeError):
            continue
        if 0 <= epoch <= int(maximum_epoch):
            files[epoch] = path
    return files


def find_external_checkpoint_dir(
    root: Path | None,
    slug: str,
    epochs: int,
) -> Path | None:
    if root is None:
        return None
    for candidate in (root / slug / "checkpoints", root / slug):
        if not candidate.is_dir():
            continue
        files = checkpoint_files(candidate, epochs)
        if all(epoch in files for epoch in range(int(epochs) + 1)):
            return candidate
    return None


def find_complete_local_checkpoint_dir(
    output_dir: Path,
    slug: str,
    epochs: int,
) -> Path | None:
    candidate = output_dir / slug / "checkpoints"
    if not candidate.is_dir():
        return None
    files = checkpoint_files(candidate, epochs)
    if all(epoch in files for epoch in range(int(epochs) + 1)):
        return candidate
    return None


def load_or_fit_frozen_codec(
    *,
    slug: str,
    checkpoint_dir: Path,
    model_dir: Path,
    epochs: int,
    threads: int,
    prepare_epoch_zero: Callable[[Mapping], tuple[Sequence, Mapping, Mapping]],
):
    """Load or fit the one epoch-0 codec plan used by a trajectory.

    ``prepare_epoch_zero`` returns ``(named_tensors, identity, metadata)``.
    Identity fields are checked when resuming an existing plan, while metadata
    is recorded only to describe a newly fitted plan.
    """

    from examples.checkpoint_compression import (
        load_codec_plan,
        save_codec_plan,
        setup_neuralzip,
    )

    files = checkpoint_files(checkpoint_dir, epochs)
    if 0 not in files:
        raise FileNotFoundError(f"Missing epoch-0 checkpoint in {checkpoint_dir}.")

    plan_path = model_dir / "neuralzip_epoch0_plan.pkl"
    setup_path = model_dir / "neuralzip_epoch0_setup.json"
    checkpoint = load_torch_checkpoint(files[0])
    named_tensors, identity, metadata = prepare_epoch_zero(checkpoint)
    named_tensors = list(named_tensors)

    if plan_path.is_file() and setup_path.is_file():
        setup = json.loads(setup_path.read_text())
        for key, expected in identity.items():
            if setup.get(key) != expected:
                raise RuntimeError(
                    f"Frozen plan in {plan_path} has a different {key}."
                )
        if int(setup.get("threads", -1)) != int(threads):
            raise RuntimeError(
                f"Frozen plan in {plan_path} used threads={setup.get('threads')}, "
                f"not {threads}; use a fresh output directory."
            )
        plan = load_codec_plan(plan_path)
        if plan.digest != setup["codebook_digest"]:
            raise RuntimeError(f"Frozen NeuralZip plan is corrupt: {plan_path}.")
        print(f"[{slug}] loaded frozen epoch-0 NeuralZip plan", flush=True)
    else:
        plan, setup = setup_neuralzip(named_tensors, threads=threads)
        setup = {
            **setup,
            "model": slug,
            "dtype": "float32",
            "epoch0_checkpoint": str(files[0]),
            **dict(identity),
            "threads": threads,
            "enable_prepacked_cache": False,
            **dict(metadata),
        }
        save_codec_plan(plan_path, plan, named_tensors)
        atomic_write_json(setup_path, setup)

    print(
        f"[{slug}] epoch-0 NeuralZip setup={setup['total_seconds']:.3f}s",
        flush=True,
    )
    del checkpoint, named_tensors
    gc.collect()
    return plan, setup, files


def run_checkpoint_trajectory(
    *,
    spec,
    checkpoint_dir: Path,
    checkpoint_source: str,
    args,
    result_path: Path,
    frozen_codec_for_trajectory: Callable,
    benchmark_checkpoint: Callable,
    after_epoch: Callable[[int], None] | None = None,
) -> None:
    """Run the task-independent compression loop for one trajectory."""

    model_dir = args.output_dir / spec.slug
    model_dir.mkdir(parents=True, exist_ok=True)
    plan, setup, files = frozen_codec_for_trajectory(
        spec,
        checkpoint_dir,
        model_dir,
        args,
    )
    missing = [epoch for epoch in range(int(args.epochs) + 1) if epoch not in files]
    if missing:
        raise FileNotFoundError(
            f"Missing checkpoints for {spec.slug}: {missing[:8]}"
            + (f" (+{len(missing) - 8} more)" if len(missing) > 8 else "")
        )
    for epoch in range(int(args.epochs) + 1):
        benchmark_checkpoint(
            spec=spec,
            epoch=epoch,
            path=files[epoch],
            checkpoint_source=checkpoint_source,
            plan=plan,
            setup=setup,
            args=args,
            result_path=result_path,
        )
        if after_epoch is not None:
            after_epoch(epoch)
    del plan
    gc.collect()


def render_checkpoint_figure(
    result_path: Path,
    output_dir: Path,
    *,
    models: Sequence[str],
    model_labels: Mapping[str, str],
    output_stem: str,
    figure_size: tuple[float, float],
    is_current_result: Callable[[Mapping], bool],
    legend_from_any_axis: bool = False,
) -> tuple[Path, Path] | None:
    """Render the shared ratio/time checkpoint figure."""

    if not result_path.is_file():
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    frame = pd.read_csv(result_path)
    frame = frame[
        frame["status"].eq("ok")
        & frame["roundtrip_exact"].eq(True)  # noqa: E712
        & frame["input_unchanged"].eq(True)  # noqa: E712
        & frame.apply(is_current_result, axis=1)
    ].copy()
    if frame.empty:
        return None

    colors = {"ZipNN src": "#4C78A8", "NeuralZip": "#E45756"}
    markers = {"ZipNN src": "o", "NeuralZip": "s"}
    fig, axes = plt.subplots(
        2,
        len(models),
        figsize=figure_size,
        sharex="col",
        squeeze=False,
    )
    for column, slug in enumerate(models):
        subset = frame[frame["model"].eq(slug)].sort_values("epoch")
        ratio_axis = axes[0, column]
        time_axis = axes[1, column]
        ratio_axis.set_title(model_labels[slug], fontsize=11)
        if subset.empty:
            for axis in (ratio_axis, time_axis):
                axis.text(
                    0.5,
                    0.5,
                    "Pending",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )
        for method in ("ZipNN src", "NeuralZip"):
            group = subset[subset["compressor"].eq(method)]
            if group.empty:
                continue
            style = {
                "color": colors[method],
                "marker": markers[method],
                "markersize": 2.8,
                "markevery": max(1, len(group) // 10),
                "linewidth": 1.5,
                "label": method,
            }
            ratio_axis.plot(group["epoch"], group["exponent_ratio_pct"], **style)
            time_axis.plot(group["epoch"], group["encode_seconds"], **style)
        for axis in (ratio_axis, time_axis):
            axis.grid(True, axis="y", color="#dddddd", linewidth=0.6)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
        time_axis.set_xlabel("Epoch")

    axes[0, 0].set_ylabel("Exponent ratio (%)")
    axes[1, 0].set_ylabel("Encoding time (s)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if legend_from_any_axis and not handles:
        for axis in axes[0, 1:]:
            handles, labels = axis.get_legend_handles_labels()
            if handles:
                break
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = figure_dir / f"{output_stem}.pdf"
    png_path = figure_dir / f"{output_stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def validate_checkpoint_results(
    result_path: Path,
    *,
    models: Iterable[str],
    epochs: int,
    is_current_result: Callable[[Mapping], bool],
    error_label: str,
) -> None:
    """Reject incomplete, duplicated, or stale checkpoint measurements."""

    selected_models = set(models)
    rows = read_csv(result_path)
    expected = {
        (slug, epoch, compressor)
        for slug in selected_models
        for epoch in range(int(epochs) + 1)
        for compressor in ("ZipNN src", "NeuralZip")
    }
    selected_rows = [row for row in rows if row.get("model") in selected_models]
    keys = [
        (str(row["model"]), int(row["epoch"]), str(row["compressor"]))
        for row in selected_rows
    ]
    observed = set(keys)
    invalid = [
        row
        for row in selected_rows
        if row.get("status") != "ok"
        or row.get("roundtrip_exact", "").lower() != "true"
        or row.get("input_unchanged", "").lower() != "true"
        or not is_current_result(row)
    ]
    if expected - observed or len(keys) != len(observed) or invalid:
        raise RuntimeError(
            f"{error_label} are incomplete or invalid: "
            f"missing={len(expected - observed)}, "
            f"duplicates={len(keys) - len(observed)}, invalid={len(invalid)}."
        )
