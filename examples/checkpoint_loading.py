"""Strict, reusable loaders for experiment checkpoint artifacts.

The experiment runners intentionally accept only authoritative checkpoint
layouts.  In particular, a sharded safetensors checkpoint must be described by
an index; otherwise alternate, quantized, or incomplete shards could silently
be mixed into one measurement.
"""

from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping

import torch


@dataclass(frozen=True)
class SnapshotSelection:
    """Authoritative files selected from one Hugging Face snapshot."""

    files: tuple[Path, ...]
    index_path: Path | None = None
    expected_keys: frozenset[str] | None = None


def _safe_snapshot_path(root: Path, value: object) -> Path:
    relative = Path(str(value))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe checkpoint shard path: {value!r}")
    return root / relative


def select_safetensors_weights(
    path: Path,
    *,
    require_config: bool = False,
) -> SnapshotSelection:
    """Select exactly one canonical safetensors checkpoint under ``path``.

    An index is authoritative when present.  Every referenced shard must be a
    non-empty safetensors file, and unrelated files are excluded.  Without an
    index, only the canonical ``model.safetensors`` filename is accepted.
    """

    path = Path(path)
    if require_config and not (path / "config.json").is_file():
        raise FileNotFoundError(f"Snapshot has no config.json: {path}")

    indexes = sorted(path.glob("*.safetensors.index.json"))
    if indexes:
        canonical = [
            index for index in indexes if index.name == "model.safetensors.index.json"
        ]
        if len(canonical) == 1:
            index_path = canonical[0]
        elif len(indexes) == 1:
            index_path = indexes[0]
        else:
            raise ValueError(
                "Snapshot contains ambiguous safetensors indexes: "
                + ", ".join(index.name for index in indexes)
            )

        index = json.loads(index_path.read_text())
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Checkpoint index has no weight_map: {index_path}")
        shard_names = sorted({str(value) for value in weight_map.values()})
        if any(not name.endswith(".safetensors") for name in shard_names):
            raise ValueError(
                f"Safetensors index references a non-safetensors shard: {index_path}"
            )
        files = tuple(_safe_snapshot_path(path, name) for name in shard_names)
        missing = [
            file for file in files if not file.is_file() or file.stat().st_size <= 0
        ]
        if missing:
            raise FileNotFoundError(
                "Checkpoint index references missing or empty shards: "
                + ", ".join(str(file) for file in missing)
            )
        return SnapshotSelection(
            files=files,
            index_path=index_path,
            expected_keys=frozenset(str(key) for key in weight_map),
        )

    canonical = path / "model.safetensors"
    if canonical.is_file() and canonical.stat().st_size > 0:
        return SnapshotSelection(files=(canonical,))

    orphan_shards = sorted(path.glob("*.safetensors"))
    if orphan_shards:
        raise ValueError(
            "Snapshot has safetensors files but no authoritative index or "
            f"model.safetensors: {path}"
        )
    raise FileNotFoundError(f"Snapshot has no authoritative weight file: {path}")


def download_safetensors_snapshot(
    repo_id: str,
    *,
    revision: str | None = None,
    local_files_only: bool = True,
    cache_dir: Path | None = None,
    local_dir: Path | None = None,
    include_config: bool = False,
) -> tuple[Path, SnapshotSelection]:
    """Resolve a snapshot and return its strictly selected weight files."""

    from huggingface_hub import snapshot_download

    patterns = ["*.safetensors", "*.safetensors.index.json"]
    if include_config:
        patterns.extend(("*.json", "*.model", "*.txt"))
    kwargs = {
        "repo_id": repo_id,
        "local_files_only": bool(local_files_only),
        "allow_patterns": tuple(patterns),
    }
    if revision is not None:
        kwargs["revision"] = revision
    if cache_dir is not None:
        kwargs["cache_dir"] = Path(cache_dir)
    if local_dir is not None:
        kwargs["local_dir"] = Path(local_dir)
        kwargs["ignore_patterns"] = (
            "*.bin",
            "*.gguf",
            "*.h5",
            "*.msgpack",
            "*.onnx",
        )
    snapshot = Path(snapshot_download(**kwargs))
    return snapshot, select_safetensors_weights(
        snapshot,
        require_config=include_config,
    )


def iter_safetensors_shards(
    selection: SnapshotSelection,
    *,
    device: str = "cpu",
) -> Iterator[tuple[Path, Mapping[str, torch.Tensor]]]:
    """Yield selected shards while validating index keys and duplicates.

    Consumers should exhaust the iterator so its final index validation runs.
    Only one shard is retained by this generator at a time.
    """

    from safetensors.torch import load_file

    loaded_keys: set[str] = set()
    for file_path in selection.files:
        state = load_file(str(file_path), device=device)
        duplicate = loaded_keys.intersection(state)
        if duplicate:
            preview = ", ".join(sorted(duplicate)[:5])
            raise ValueError(
                f"Duplicate checkpoint tensor key(s) in {file_path}: {preview}."
            )
        loaded_keys.update(str(name) for name in state)
        try:
            yield file_path, state
        finally:
            del state
            gc.collect()

    if selection.expected_keys is not None:
        missing = sorted(selection.expected_keys - loaded_keys)
        unexpected = sorted(loaded_keys - selection.expected_keys)
        if missing:
            preview = ", ".join(missing[:5])
            raise ValueError(
                f"Checkpoint is missing {len(missing)} indexed tensor(s), "
                f"including: {preview}."
            )
        if unexpected:
            preview = ", ".join(unexpected[:5])
            raise ValueError(
                f"Checkpoint contains {len(unexpected)} tensor(s) not declared "
                f"by its index, including: {preview}."
            )


def load_torch_checkpoint(path: Path) -> Mapping:
    """Load the explicit mapping stored by the final checkpoint runners."""

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Checkpoint is not a mapping: {path}")
    return checkpoint
