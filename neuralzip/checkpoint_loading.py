"""Strict, reusable loaders for checkpoint artifacts.

The experiment runners intentionally accept only authoritative checkpoint
layouts.  In particular, a sharded safetensors checkpoint must be described by
an index; otherwise alternate, quantized, or incomplete shards could silently
be mixed into one measurement.
"""

from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Iterable, Iterator, Mapping

import torch


@dataclass(frozen=True)
class SnapshotSelection:
    """Authoritative files selected from one Hugging Face snapshot."""

    files: tuple[Path, ...]
    index_path: Path | None = None
    expected_keys: frozenset[str] | None = None


@dataclass(frozen=True)
class WeightMapSelection:
    """Validated repository-relative artifacts declared by a weight map."""

    files: tuple[str, ...]
    expected_keys: frozenset[str]


def _safe_weight_path(value: object) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"Unsafe checkpoint shard path: {value!r}")
    if "\\" in value:
        raise ValueError(f"Unsafe checkpoint shard path: {value!r}")
    posix_path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    if (
        posix_path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or ".." in posix_path.parts
        or posix_path.as_posix() in {"", "."}
        or value.endswith("/")
    ):
        raise ValueError(f"Unsafe checkpoint shard path: {value!r}")
    return posix_path.as_posix()


def resolve_weight_map(
    weight_map: object,
    *,
    available_files: Mapping[str, int],
    allowed_suffixes: Iterable[str],
) -> WeightMapSelection:
    """Purely validate and resolve an indexed checkpoint's ``weight_map``.

    ``available_files`` is a repository-relative path-to-size catalog supplied
    by either a local snapshot or a remote repository listing.  Keeping I/O
    outside this function gives every frontend the same path, format, presence,
    non-empty-file, and expected-key rules.
    """

    if not isinstance(weight_map, Mapping) or not weight_map:
        raise ValueError("Checkpoint index has no weight_map.")

    suffixes = tuple(sorted({str(value).lower() for value in allowed_suffixes}))
    if not suffixes or any(not suffix.startswith(".") for suffix in suffixes):
        raise ValueError("allowed_suffixes must contain file extensions.")

    expected_keys = set()
    referenced_files = set()
    for key, value in weight_map.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"Invalid checkpoint tensor key: {key!r}")
        relative_path = _safe_weight_path(value)
        if not relative_path.lower().endswith(suffixes):
            allowed = ", ".join(suffixes)
            raise ValueError(
                f"Checkpoint shard {relative_path!r} does not use an allowed "
                f"suffix ({allowed})."
            )
        if relative_path not in available_files:
            raise FileNotFoundError(
                f"Checkpoint index references missing shard: {relative_path}"
            )
        try:
            size = int(available_files[relative_path])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Checkpoint shard {relative_path!r} has an invalid size."
            ) from error
        if size <= 0:
            raise ValueError(f"Checkpoint index references empty shard: {relative_path}")
        expected_keys.add(key)
        referenced_files.add(relative_path)

    return WeightMapSelection(
        files=tuple(sorted(referenced_files)),
        expected_keys=frozenset(expected_keys),
    )


def _local_file_sizes(root: Path) -> dict[str, int]:
    return {
        path.relative_to(root).as_posix(): int(path.stat().st_size)
        for path in root.rglob("*")
        if path.is_file()
    }


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
        try:
            resolved = resolve_weight_map(
                index.get("weight_map"),
                available_files=_local_file_sizes(index_path.parent),
                allowed_suffixes=(".safetensors",),
            )
        except (FileNotFoundError, ValueError) as error:
            error_type = type(error)
            raise error_type(f"{error} ({index_path})") from error
        files = tuple(index_path.parent / name for name in resolved.files)
        return SnapshotSelection(
            files=files,
            index_path=index_path,
            expected_keys=resolved.expected_keys,
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
