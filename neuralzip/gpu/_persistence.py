"""Canonical and crash-safe primitives for NeuralZip GPU artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

from safetensors.torch import save_file


class GPUPretrainedError(ValueError):
    """Raised when a persistent GPU artifact is malformed or unsupported."""


def canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise GPUPretrainedError("The checkpoint manifest is not JSON-safe.") from error


def canonical_json_loads(source: str, *, document: str) -> object:
    def reject_constant(value: str) -> None:
        raise GPUPretrainedError(f"Invalid JSON constant {value!r} in {document}.")

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        output = {}
        for key, value in pairs:
            if key in output:
                raise GPUPretrainedError(f"Duplicate JSON field {key!r} in {document}.")
            output[key] = value
        return output

    try:
        return json.loads(
            source,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except GPUPretrainedError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise GPUPretrainedError(f"Unable to parse {document}.") from error


def require_fields(
    value: object,
    expected: set[str],
    *,
    context: str,
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != expected:
        raise GPUPretrainedError(f"Invalid {context} fields.")
    return value


def require_string(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise GPUPretrainedError(f"{context} must be a non-empty string.")
    return value


def require_int(
    value: object,
    *,
    context: str,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        raise GPUPretrainedError(f"{context} is outside its integer range.")
    return value


def require_name(value: object, *, context: str) -> str:
    name = require_string(value, context=context)
    if any(
        not segment
        or not (segment.isidentifier() or (segment.isascii() and segment.isdecimal()))
        for segment in name.split(".")
    ):
        raise GPUPretrainedError(f"{context} is not a valid dotted path.")
    return name


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
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
        dir=path.parent, prefix=f".{path.name}."
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii", newline="\n") as handle:
            handle.write(source)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_safetensors(
    tensors: Mapping[str, object],
    path: Path,
    *,
    metadata: Mapping[str, str],
) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}."
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        save_file(dict(tensors), str(temporary), metadata=dict(metadata))
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
