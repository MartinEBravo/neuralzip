"""Shared strict and crash-safe persistence helpers for GPU artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def canonical_json(
    value: object,
    *,
    error_type: type[Exception],
    error_message: str,
) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise error_type(error_message) from error


def strict_json_loads(
    source: str,
    *,
    error_type: type[Exception],
    document: str,
    invalid_message: str,
) -> object:
    def reject_constant(value: str) -> None:
        raise error_type(f"Invalid JSON constant {value!r} in {document}.")

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        output = {}
        for key, value in pairs:
            if key in output:
                raise error_type(f"Duplicate JSON field {key!r} in {document}.")
            output[key] = value
        return output

    try:
        return json.loads(
            source,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except error_type:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise error_type(invalid_message) from error


def require_fields(
    value: object,
    expected: set[str],
    *,
    context: str,
    error_type: type[Exception],
    compact: bool = False,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise error_type(f"{context} must be a JSON object.")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        if compact:
            details = []
            if missing:
                details.append(f"missing {missing}")
            if unexpected:
                details.append(f"unexpected {unexpected}")
            suffix = ", ".join(details)
        else:
            suffix = f"missing={missing}, unexpected={unexpected}"
        raise error_type(f"Invalid {context} fields: {suffix}.")
    return value


def require_string(
    value: object,
    *,
    context: str,
    error_type: type[Exception],
) -> str:
    if not isinstance(value, str) or not value:
        raise error_type(f"{context} must be a non-empty string.")
    return value


def require_int(
    value: object,
    *,
    context: str,
    error_type: type[Exception],
    minimum: int = 0,
    maximum: int | None = None,
    combined_error: bool = False,
) -> int:
    invalid_type = isinstance(value, bool) or not isinstance(value, int)
    outside_range = not invalid_type and (
        value < minimum or (maximum is not None and value > maximum)
    )
    if invalid_type or outside_range:
        if maximum is None:
            expected = f">= {minimum}"
        else:
            expected = f"in [{minimum}, {maximum}]"
        if combined_error:
            raise error_type(f"{context} must be an integer {expected}.")
        if invalid_type:
            raise error_type(f"{context} must be an integer.")
        raise error_type(f"{context} must be {expected}.")
    return value


def require_dotted_name(
    value: object,
    *,
    context: str,
    kind: str,
    error_type: type[Exception],
) -> str:
    name = require_string(value, context=context, error_type=error_type)
    segments = name.split(".")
    if any(
        not segment
        or not (segment.isidentifier() or (segment.isascii() and segment.isdecimal()))
        for segment in segments
    ):
        raise error_type(f"{context} is not a valid dotted {kind} path.")
    return name


def path_argument(
    value: str | os.PathLike[str],
    *,
    type_message: str,
    bytes_message: str,
    invalid_message: str,
) -> Path:
    try:
        source = os.fspath(value)
    except TypeError as error:
        raise TypeError(type_message) from error
    if isinstance(source, bytes):
        raise TypeError(bytes_message)
    if not source or "\x00" in source:
        raise ValueError(invalid_message)
    return Path(source)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:  # pragma: no cover - platform/filesystem dependent.
        return
    try:
        os.fsync(descriptor)
    except OSError:  # pragma: no cover - platform/filesystem dependent.
        pass
    finally:
        os.close(descriptor)


def atomic_text(path: Path, source: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii", newline="\n") as handle:
            handle.write(source)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def safetensors_functions(error_message: str):
    try:
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ImportError as error:  # pragma: no cover - dependency configuration.
        raise RuntimeError(error_message) from error
    return safe_open, save_file


def atomic_safetensors(
    tensors: Mapping[str, torch.Tensor],
    path: Path,
    *,
    metadata: Mapping[str, str],
    dependency_error: str,
) -> None:
    _safe_open, save_file = safetensors_functions(dependency_error)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        save_file(dict(tensors), str(temporary), metadata=dict(metadata))
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
