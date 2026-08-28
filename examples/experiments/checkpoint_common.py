"""Task-independent plumbing for checkpoint experiments.

Model construction and training remain explicit in the vision and language
runners. Persistence, reusable-setup lifecycle, compression trajectories, and
validation live here so the two protocols cannot drift.
"""

from __future__ import annotations

import gc
import json
import random
from argparse import BooleanOptionalAction, Namespace
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from neuralzip.checkpoint_loading import load_torch_checkpoint
from neuralzip.experiment_io import (
    atomic_write_csv,
    atomic_write_json,
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


def add_checkpoint_run_arguments(
    parser,
    *,
    epochs: int,
    batch_size: int | None,
    batch_size_help: str | None = None,
    data_dir: Path,
    output_dir: Path,
) -> None:
    """Add execution options shared by vision and language trajectories."""

    parser.add_argument("--epochs", type=int, default=epochs)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=batch_size,
        help=batch_size_help,
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--data-dir", type=Path, default=data_dir)
    parser.add_argument("--output-dir", type=Path, default=output_dir)
    parser.add_argument(
        "--existing-checkpoint-root",
        type=Path,
        help="Optional root containing complete <model>/checkpoints trajectories.",
    )
    parser.add_argument(
        "--train-missing",
        action=BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--compression-only", action="store_true")
    parser.add_argument("--no-amp", action="store_true")


def initialize_checkpoint_run(args: Namespace) -> Path:
    """Validate common modes and initialize paths and CPU timing controls."""

    modes = (args.train_only, args.compression_only)
    if sum(map(bool, modes)) > 1:
        raise ValueError("--train-only and --compression-only are mutually exclusive.")
    args.output_dir = args.output_dir.resolve()
    args.data_dir = args.data_dir.resolve()
    if args.existing_checkpoint_root is not None:
        args.existing_checkpoint_root = args.existing_checkpoint_root.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, int(args.threads)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    return args.output_dir / "checkpoint_results.csv"


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
        if path.name.endswith("training_state.pt") or path.name.endswith(
            "latest_state.pt"
        ):
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


def missing_checkpoint_slugs(
    *,
    slugs: Iterable[str],
    external_root: Path | None,
    output_dir: Path,
    epochs: int,
) -> list[str]:
    """Return models without a complete external or local trajectory."""

    return [
        slug
        for slug in slugs
        if find_external_checkpoint_dir(external_root, slug, epochs) is None
        and find_complete_local_checkpoint_dir(output_dir, slug, epochs) is None
    ]


def run_checkpoint_models(
    *,
    args: Namespace,
    specs: Mapping,
    result_path: Path,
    local_source: str,
    train_trajectory: Callable,
    process_trajectory: Callable,
    validate_results: Callable,
) -> None:
    """Reuse, train, and compress each requested checkpoint trajectory."""

    for slug in args.models:
        spec = specs[slug]
        checkpoint_dir = find_external_checkpoint_dir(
            args.existing_checkpoint_root,
            slug,
            args.epochs,
        )
        if checkpoint_dir is not None:
            source = "external-existing"
            print(f"[{slug}] reuse checkpoints from {checkpoint_dir}", flush=True)
        else:
            checkpoint_dir = find_complete_local_checkpoint_dir(
                args.output_dir,
                slug,
                args.epochs,
            )
            if checkpoint_dir is not None:
                source = local_source
                print(
                    f"[{slug}] reuse locally trained checkpoints from {checkpoint_dir}",
                    flush=True,
                )
            else:
                if args.compression_only or not args.train_missing:
                    raise FileNotFoundError(
                        f"No complete epoch 0..{args.epochs} trajectory for {slug}."
                    )
                checkpoint_dir = train_trajectory(spec)
                source = local_source
        if args.train_only:
            print(f"[{slug}] train-only complete: {checkpoint_dir}", flush=True)
            continue
        process_trajectory(spec, checkpoint_dir, source, args, result_path)

    if not args.train_only:
        validate_results(result_path, models=args.models, epochs=args.epochs)
    print(f"results={result_path}", flush=True)


def load_or_fit_frozen_codec(
    *,
    slug: str,
    checkpoint_dir: Path,
    model_dir: Path,
    epochs: int,
    threads: int,
    prepare_epoch_zero: Callable[[Mapping], tuple[Sequence, Mapping, Mapping]],
):
    """Load or fit the one epoch-0 codec setup used by a trajectory.

    ``prepare_epoch_zero`` returns ``(named_tensors, identity, metadata)``.
    Identity fields are checked when resuming an existing setup, while metadata
    is recorded only to describe a newly fitted setup.
    """

    from examples.experiments.checkpoint_compression import setup_neuralzip
    from neuralzip import Compressor

    files = checkpoint_files(checkpoint_dir, epochs)
    if 0 not in files:
        raise FileNotFoundError(f"Missing epoch-0 checkpoint in {checkpoint_dir}.")

    setup_state_path = model_dir / "neuralzip_epoch0_setup.pkl"
    setup_path = model_dir / "neuralzip_epoch0_setup.json"
    checkpoint = load_torch_checkpoint(files[0])
    named_tensors, identity, metadata = prepare_epoch_zero(checkpoint)
    named_tensors = list(named_tensors)

    if setup_state_path.is_file() and setup_path.is_file():
        setup = json.loads(setup_path.read_text())
        for key, expected in identity.items():
            if setup.get(key) != expected:
                raise RuntimeError(
                    f"Reusable setup in {setup_state_path} has a different {key}."
                )
        if int(setup.get("threads", -1)) != int(threads):
            raise RuntimeError(
                f"Reusable setup in {setup_state_path} used threads={setup.get('threads')}, "
                f"not {threads}; use a fresh output directory."
            )
        compressor = Compressor(
            "neuralzip",
            threads=threads,
            compression_threshold=0.95,
            setup=setup_state_path,
        )
        if compressor.setup_digest != setup["codebook_digest"]:
            raise RuntimeError(
                f"Reusable NeuralZip setup is corrupt: {setup_state_path}."
            )
        print(f"[{slug}] loaded reusable epoch-0 NeuralZip setup", flush=True)
    else:
        compressor, setup = setup_neuralzip(named_tensors, threads=threads)
        setup = {
            **setup,
            "model": slug,
            "dtype": "float32",
            "epoch0_checkpoint": str(files[0]),
            **dict(identity),
            "threads": threads,
            **dict(metadata),
        }
        compressor.save_setup(setup_state_path)
        atomic_write_json(setup_path, setup)

    print(
        f"[{slug}] epoch-0 NeuralZip setup={setup['setup_seconds']:.3f}s",
        flush=True,
    )
    del checkpoint, named_tensors
    gc.collect()
    return compressor, setup, files


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
    compressor, setup, files = frozen_codec_for_trajectory(
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
            compressor=compressor,
            setup=setup,
            args=args,
            result_path=result_path,
        )
        if after_epoch is not None:
            after_epoch(epoch)
    del compressor
    gc.collect()


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
        for compressor in ("ZipNN", "NeuralZip")
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
