"""Strict, reusable loaders for checkpoint artifacts.

The experiment runners intentionally accept only authoritative checkpoint
layouts.  In particular, a sharded safetensors checkpoint must be described by
an index; otherwise alternate, quantized, or incomplete shards could silently
be mixed into one measurement.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

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
            raise ValueError(
                f"Checkpoint index references empty shard: {relative_path}"
            )
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


def select_indexed_weights(
    path: Path,
    *,
    allow_bin: bool = False,
) -> SnapshotSelection | None:
    """Select and validate an authoritative sharded checkpoint index.

    Safetensors is preferred when both formats are present.  ``None`` means
    that no supported index exists; malformed or incomplete indexes raise.
    """

    path = Path(path)
    formats = [
        (
            "*.safetensors.index.json",
            "model.safetensors.index.json",
            (".safetensors",),
        ),
    ]
    if allow_bin:
        formats.append(("*.bin.index.json", "pytorch_model.bin.index.json", (".bin",)))

    for pattern, canonical_name, suffixes in formats:
        indexes = sorted(path.glob(pattern))
        if not indexes:
            continue
        canonical = [index for index in indexes if index.name == canonical_name]
        if len(canonical) == 1:
            index_path = canonical[0]
        elif len(indexes) == 1:
            index_path = indexes[0]
        else:
            raise ValueError(
                "Checkpoint contains ambiguous shard indexes: "
                + ", ".join(index.name for index in indexes)
            )

        index = json.loads(index_path.read_text())
        try:
            resolved = resolve_weight_map(
                index.get("weight_map"),
                available_files=_local_file_sizes(index_path.parent),
                allowed_suffixes=suffixes,
            )
        except (FileNotFoundError, ValueError) as error:
            raise type(error)(f"{error} ({index_path})") from error
        return SnapshotSelection(
            files=tuple(index_path.parent / name for name in resolved.files),
            index_path=index_path,
            expected_keys=resolved.expected_keys,
        )
    return None


def load_torch_checkpoint(path: Path) -> Mapping:
    """Load the explicit mapping stored by the checkpoint runners."""

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Checkpoint is not a mapping: {path}")
    return checkpoint
