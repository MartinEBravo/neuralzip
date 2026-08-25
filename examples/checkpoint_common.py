"""Shared persistence and trajectory helpers for final checkpoint runners.

This module intentionally contains only task-agnostic plumbing.  Model
construction, training, compression, and rendering stay in their respective
runners so their experiment protocols remain explicit.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from neuralzip.experiment_io import (
    atomic_write_csv,
    read_csv,
)


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
