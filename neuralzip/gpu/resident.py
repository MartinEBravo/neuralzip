"""Safe, sharded persistence for the state not encoded by NeuralZip.

Resident tensors are stored in deterministic Safetensors shards.  A canonical
JSON index records tensor descriptors, exact storage aliases, and a SHA-256
digest for every shard.  The index always exists, including for empty state,
and readers can consume one validated shard at a time.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from types import MappingProxyType

import torch

FORMAT_NAME = "neuralzip-resident-state"
FORMAT_VERSION = 1
INDEX_FILENAME = "neuralzip-resident.safetensors.index.json"
SHARD_PATTERN = "neuralzip-resident-{index:05d}-of-{count:05d}.safetensors"

_SHARD_RE = re.compile(r"^neuralzip-resident-(\d{5})-of-(\d{5})\.safetensors$")
_SIZE_RE = re.compile(
    r"^\s*(\d+(?:\.\d+)?)\s*(B|KB|MB|GB|TB|KIB|MIB|GIB|TIB)\s*$",
    re.IGNORECASE,
)
_SIZE_UNITS = {
    "B": 1,
    "KB": 1000,
    "MB": 1000**2,
    "GB": 1000**3,
    "TB": 1000**4,
    "KIB": 1 << 10,
    "MIB": 1 << 20,
    "GIB": 1 << 30,
    "TIB": 1 << 40,
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _dtype_table() -> dict[str, tuple[torch.dtype, str]]:
    definitions = (
        ("bool", "bool", "BOOL"),
        ("int8", "int8", "I8"),
        ("uint8", "uint8", "U8"),
        ("int16", "int16", "I16"),
        ("uint16", "uint16", "U16"),
        ("int32", "int32", "I32"),
        ("uint32", "uint32", "U32"),
        ("int64", "int64", "I64"),
        ("uint64", "uint64", "U64"),
        ("float16", "float16", "F16"),
        ("bfloat16", "bfloat16", "BF16"),
        ("float32", "float32", "F32"),
        ("float64", "float64", "F64"),
        ("complex64", "complex64", "C64"),
        ("float8_e4m3fn", "float8_e4m3fn", "F8_E4M3"),
        ("float8_e4m3fnuz", "float8_e4m3fnuz", "F8_E4M3FNUZ"),
        ("float8_e5m2", "float8_e5m2", "F8_E5M2"),
        ("float8_e5m2fnuz", "float8_e5m2fnuz", "F8_E5M2FNUZ"),
    )
    output = {}
    for name, attribute, safetensors_name in definitions:
        dtype = getattr(torch, attribute, None)
        if dtype is not None:
            output[name] = (dtype, safetensors_name)
    return output


_DTYPES = _dtype_table()
_DTYPE_NAMES = {dtype: name for name, (dtype, _safe_name) in _DTYPES.items()}


class ResidentStateError(ValueError):
    """Raised when resident state is unsafe, corrupt, or structurally invalid."""


@dataclass(frozen=True)
class ResidentTensorDescriptor:
    """Logical description of one physically stored tensor."""

    dtype: str
    shape: tuple[int, ...]
    nbytes: int
    shard: str


@dataclass(frozen=True)
class ResidentStateManifest:
    """Validated file and tensor inventory for one resident-state directory."""

    directory: Path
    index_path: Path
    shard_paths: tuple[Path, ...]
    descriptors: Mapping[str, ResidentTensorDescriptor]
    aliases: Mapping[str, str]
    shard_hashes: Mapping[str, str]
    weight_map: Mapping[str, str]
    total_size: int
    index_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "directory", Path(self.directory))
        object.__setattr__(self, "index_path", Path(self.index_path))
        object.__setattr__(
            self, "shard_paths", tuple(Path(path) for path in self.shard_paths)
        )
        for name in ("descriptors", "aliases", "shard_hashes", "weight_map"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))

    @property
    def tensor_names(self) -> tuple[str, ...]:
        """Return all logical tensor names, including aliases."""

        return tuple(sorted((*self.descriptors, *self.aliases)))


def _require_safetensors():
    try:
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ImportError as error:  # pragma: no cover - required core dependency.
        raise RuntimeError(
            "Resident state requires the 'safetensors' package."
        ) from error
    return safe_open, save_file


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:  # pragma: no cover - internal invariant.
        raise ResidentStateError("The resident index is not JSON-safe.") from error


def _reject_constant(value: str) -> None:
    raise ResidentStateError(f"Invalid JSON constant {value!r} in resident index.")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output = {}
    for key, value in pairs:
        if key in output:
            raise ResidentStateError(f"Duplicate JSON field {key!r} in resident index.")
        output[key] = value
    return output


def _parse_index(path: Path) -> dict[str, object]:
    try:
        source = path.read_text(encoding="ascii")
        value = json.loads(
            source,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except FileNotFoundError:
        raise
    except ResidentStateError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ResidentStateError("Unable to read the resident-state index.") from error
    if not isinstance(value, dict):
        raise ResidentStateError("The resident-state index must be a JSON object.")
    if source != _canonical_json(value) + "\n":
        raise ResidentStateError("The resident-state index is not canonical JSON.")
    return value


def _require_fields(
    value: object,
    expected: set[str],
    *,
    context: str,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ResidentStateError(f"{context} must be a JSON object.")
    actual = set(value)
    if actual != expected:
        raise ResidentStateError(
            f"Invalid {context} fields: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}."
        )
    return value


def _require_int(value: object, *, context: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ResidentStateError(f"{context} must be an integer >= {minimum}.")
    return value


def _require_string(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ResidentStateError(f"{context} must be a non-empty string.")
    return value


def _require_tensor_name(value: object, *, context: str) -> str:
    name = _require_string(value, context=context)
    segments = name.split(".")
    if any(
        not segment
        or not (segment.isidentifier() or (segment.isascii() and segment.isdecimal()))
        for segment in segments
    ):
        raise ResidentStateError(f"{context} is not a valid dotted tensor path.")
    return name


def _directory(value: str | os.PathLike[str], *, create: bool) -> Path:
    try:
        source = os.fspath(value)
    except TypeError as error:
        raise TypeError("directory must be a string or path-like value.") from error
    if isinstance(source, bytes):
        raise TypeError("directory must be a text path, not bytes.")
    if not source or "\x00" in source:
        raise ValueError("directory must be a non-empty valid path.")
    path = Path(source)
    if path.exists() and not path.is_dir():
        raise ValueError("Resident state must be stored in a directory.")
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def _parse_max_shard_size(value: int | str) -> int:
    if isinstance(value, bool):
        raise TypeError("max_shard_size must be a positive integer or size string.")
    if isinstance(value, int):
        if value <= 0:
            raise ValueError("max_shard_size must be positive.")
        return value
    if not isinstance(value, str):
        raise TypeError("max_shard_size must be a positive integer or size string.")
    match = _SIZE_RE.fullmatch(value)
    if match is None:
        raise ValueError(
            "max_shard_size must use B, KB, MB, GB, TB, KiB, MiB, GiB, or TiB."
        )
    amount, unit = match.groups()
    try:
        size = int(Decimal(amount) * _SIZE_UNITS[unit.upper()])
    except (InvalidOperation, OverflowError) as error:
        raise ValueError("max_shard_size is outside the supported range.") from error
    if size <= 0:
        raise ValueError("max_shard_size must resolve to at least one byte.")
    return size


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:  # pragma: no cover - filesystem dependent.
        return
    try:
        os.fsync(descriptor)
    except OSError:  # pragma: no cover - filesystem dependent.
        pass
    finally:
        os.close(descriptor)


def _atomic_text(path: Path, source: str) -> None:
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
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_safetensors(
    tensors: Mapping[str, torch.Tensor],
    path: Path,
) -> None:
    _safe_open, save_file = _require_safetensors()
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        save_file(
            dict(tensors),
            str(temporary),
            metadata={"format": FORMAT_NAME, "version": str(FORMAT_VERSION)},
        )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _storage_key(tensor: torch.Tensor) -> tuple[object, ...]:
    storage = tensor.untyped_storage()
    return (
        tensor.device.type,
        tensor.device.index,
        storage._cdata,
        storage.data_ptr(),
        storage.nbytes(),
        tensor.storage_offset(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        bool(tensor.is_conj()),
        bool(tensor.is_neg()),
    )


def _snapshot_state(
    tensors: Mapping[str, torch.Tensor],
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, str],
    dict[str, tuple[str, tuple[int, ...], int]],
]:
    if not isinstance(tensors, Mapping):
        raise TypeError("tensors must be a mapping of names to torch.Tensor values.")
    names = [_require_tensor_name(name, context="tensor name") for name in tensors]
    physical = {}
    aliases = {}
    descriptors = {}
    canonical_by_storage = {}
    for name in sorted(names):
        tensor = tensors[name]
        if not torch.is_tensor(tensor):
            raise TypeError(f"Resident state {name!r} must be a torch.Tensor.")
        if tensor.device.type == "meta":
            raise ResidentStateError(f"Resident state {name!r} cannot be meta.")
        if tensor.layout != torch.strided or tensor.is_quantized:
            raise ResidentStateError(
                f"Resident state {name!r} must be a dense strided tensor."
            )
        dtype_name = _DTYPE_NAMES.get(tensor.dtype)
        if dtype_name is None:
            raise ResidentStateError(
                f"Resident state {name!r} has unsupported dtype {tensor.dtype}."
            )

        storage_key = _storage_key(tensor)
        canonical = canonical_by_storage.get(storage_key)
        if canonical is not None:
            aliases[name] = canonical
            continue
        canonical_by_storage[storage_key] = name
        snapshot = tensor.detach().to(device="cpu").contiguous().clone()
        physical[name] = snapshot
        descriptors[name] = (
            dtype_name,
            tuple(snapshot.shape),
            snapshot.numel() * snapshot.element_size(),
        )
    return physical, aliases, descriptors


def _partition(
    tensors: Mapping[str, torch.Tensor],
    *,
    max_bytes: int,
) -> list[dict[str, torch.Tensor]]:
    shards = []
    current = {}
    current_bytes = 0
    for name, tensor in tensors.items():
        nbytes = tensor.numel() * tensor.element_size()
        if current and current_bytes + nbytes > max_bytes:
            shards.append(current)
            current = {}
            current_bytes = 0
        current[name] = tensor
        current_bytes += nbytes
    if current:
        shards.append(current)
    if len(shards) > 99_999:
        raise ResidentStateError("Resident state requires more than 99,999 shards.")
    return shards


def save_resident_state(
    tensors: Mapping[str, torch.Tensor],
    directory: str | os.PathLike[str],
    max_shard_size: int | str = "5GB",
) -> ResidentStateManifest:
    """Save a deterministic resident-state snapshot and return its manifest."""

    max_bytes = _parse_max_shard_size(max_shard_size)
    target = _directory(directory, create=True)
    physical, aliases, source_descriptors = _snapshot_state(tensors)
    partitions = _partition(physical, max_bytes=max_bytes)
    shard_count = len(partitions)
    weight_map = {}
    shard_records = {}
    expected_shards = set()

    for index, shard in enumerate(partitions, start=1):
        filename = SHARD_PATTERN.format(index=index, count=shard_count)
        expected_shards.add(filename)
        path = target / filename
        _atomic_safetensors(shard, path)
        shard_records[filename] = {
            "nbytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for name in shard:
            weight_map[name] = filename

    tensor_records = {
        name: {
            "dtype": dtype,
            "nbytes": nbytes,
            "shape": list(shape),
        }
        for name, (dtype, shape, nbytes) in source_descriptors.items()
    }
    total_size = sum(record["nbytes"] for record in tensor_records.values())
    index = {
        "aliases": aliases,
        "format": FORMAT_NAME,
        "metadata": {"total_size": total_size},
        "shards": shard_records,
        "tensors": tensor_records,
        "version": FORMAT_VERSION,
        "weight_map": weight_map,
    }
    _atomic_text(target / INDEX_FILENAME, _canonical_json(index) + "\n")

    for path in target.glob("neuralzip-resident-*-of-*.safetensors"):
        if path.name not in expected_shards and _SHARD_RE.fullmatch(path.name):
            path.unlink()
    descriptors = {
        name: ResidentTensorDescriptor(
            dtype,
            shape,
            nbytes,
            weight_map[name],
        )
        for name, (dtype, shape, nbytes) in source_descriptors.items()
    }
    index_path = target / INDEX_FILENAME
    return ResidentStateManifest(
        directory=target,
        index_path=index_path,
        shard_paths=tuple(target / filename for filename in sorted(shard_records)),
        descriptors=descriptors,
        aliases=aliases,
        shard_hashes={
            filename: record["sha256"] for filename, record in shard_records.items()
        },
        weight_map=weight_map,
        total_size=total_size,
        index_sha256=_sha256_file(index_path),
    )


def _validate_descriptor(
    name: str,
    value: object,
) -> tuple[ResidentTensorDescriptor, str]:
    record = _require_fields(
        value,
        {"dtype", "nbytes", "shape"},
        context=f"tensor descriptor {name!r}",
    )
    dtype_name = _require_string(record["dtype"], context=f"tensor {name!r} dtype")
    dtype_record = _DTYPES.get(dtype_name)
    if dtype_record is None:
        raise ResidentStateError(f"Tensor {name!r} has an unsupported dtype.")
    shape_source = record["shape"]
    if not isinstance(shape_source, list):
        raise ResidentStateError(f"Tensor {name!r} shape must be a JSON array.")
    shape = tuple(
        _require_int(value, context=f"tensor {name!r} shape entry")
        for value in shape_source
    )
    nbytes = _require_int(record["nbytes"], context=f"tensor {name!r} nbytes")
    expected_numel = 1
    for dimension in shape:
        expected_numel *= dimension
    if nbytes != expected_numel * torch.empty((), dtype=dtype_record[0]).element_size():
        raise ResidentStateError(f"Tensor {name!r} byte size is inconsistent.")
    return ResidentTensorDescriptor(dtype_name, shape, nbytes, ""), dtype_record[1]


def _validate_shard(
    path: Path,
    *,
    expected_names: set[str],
    descriptors: Mapping[str, ResidentTensorDescriptor],
) -> None:
    safe_open, _save_file = _require_safetensors()
    try:
        with safe_open(str(path), framework="pt", device="cpu") as archive:
            if archive.metadata() != {
                "format": FORMAT_NAME,
                "version": str(FORMAT_VERSION),
            }:
                raise ResidentStateError(f"Shard {path.name!r} metadata is invalid.")
            names = set(archive.keys())
            if names != expected_names:
                raise ResidentStateError(
                    f"Shard {path.name!r} tensor names do not match the index."
                )
            for name in names:
                slice_ = archive.get_slice(name)
                descriptor = descriptors[name]
                safe_dtype = _DTYPES[descriptor.dtype][1]
                if slice_.get_dtype() != safe_dtype:
                    raise ResidentStateError(f"Tensor {name!r} has the wrong dtype.")
                if tuple(slice_.get_shape()) != descriptor.shape:
                    raise ResidentStateError(f"Tensor {name!r} has the wrong shape.")
    except ResidentStateError:
        raise
    except Exception as error:
        raise ResidentStateError(
            f"Unable to read resident shard {path.name!r}."
        ) from error


def load_resident_state(
    directory: str | os.PathLike[str],
) -> ResidentStateManifest:
    """Validate a resident-state index and shard inventory without loading tensors."""

    target = _directory(directory, create=False)
    index_path = target / INDEX_FILENAME
    index = _parse_index(index_path)
    root = _require_fields(
        index,
        {"aliases", "format", "metadata", "shards", "tensors", "version", "weight_map"},
        context="resident index",
    )
    if root["format"] != FORMAT_NAME or root["version"] != FORMAT_VERSION:
        raise ResidentStateError("Unsupported resident-state format or version.")
    metadata = _require_fields(root["metadata"], {"total_size"}, context="metadata")
    total_size = _require_int(metadata["total_size"], context="metadata.total_size")

    tensor_source = root["tensors"]
    if not isinstance(tensor_source, dict):
        raise ResidentStateError("tensors must be a JSON object.")
    provisional = {}
    for raw_name, value in tensor_source.items():
        name = _require_tensor_name(raw_name, context="tensor descriptor name")
        descriptor, _safe_dtype = _validate_descriptor(name, value)
        provisional[name] = descriptor
    if total_size != sum(descriptor.nbytes for descriptor in provisional.values()):
        raise ResidentStateError("metadata.total_size is inconsistent.")

    weight_map_source = root["weight_map"]
    if not isinstance(weight_map_source, dict) or set(weight_map_source) != set(
        provisional
    ):
        raise ResidentStateError("weight_map must cover every physical tensor exactly.")
    weight_map = {}
    for raw_name, raw_filename in weight_map_source.items():
        name = _require_tensor_name(raw_name, context="weight_map tensor name")
        filename = _require_string(raw_filename, context=f"weight_map[{name!r}]")
        if Path(filename).name != filename or _SHARD_RE.fullmatch(filename) is None:
            raise ResidentStateError("weight_map contains a non-canonical shard path.")
        weight_map[name] = filename

    shard_source = root["shards"]
    if not isinstance(shard_source, dict):
        raise ResidentStateError("shards must be a JSON object.")
    if set(shard_source) != set(weight_map.values()):
        raise ResidentStateError(
            "shards must match the files referenced by weight_map."
        )
    shard_count = len(shard_source)
    expected_filenames = {
        SHARD_PATTERN.format(index=index, count=shard_count)
        for index in range(1, shard_count + 1)
    }
    if set(shard_source) != expected_filenames:
        raise ResidentStateError(
            "Resident shard names are not consecutive and canonical."
        )

    shard_hashes = {}
    shard_paths = []
    for filename in sorted(expected_filenames):
        record = _require_fields(
            shard_source[filename],
            {"nbytes", "sha256"},
            context=f"shard {filename!r}",
        )
        expected_nbytes = _require_int(
            record["nbytes"], context=f"shard {filename!r} nbytes", minimum=1
        )
        digest = _require_string(record["sha256"], context=f"shard {filename!r} sha256")
        if _SHA256_RE.fullmatch(digest) is None:
            raise ResidentStateError(f"Shard {filename!r} SHA-256 is invalid.")
        path = target / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != expected_nbytes:
            raise ResidentStateError(f"Shard {filename!r} has the wrong file size.")
        if _sha256_file(path) != digest:
            raise ResidentStateError(f"Shard {filename!r} failed its SHA-256 check.")
        shard_hashes[filename] = digest
        shard_paths.append(path)

    aliases_source = root["aliases"]
    if not isinstance(aliases_source, dict):
        raise ResidentStateError("aliases must be a JSON object.")
    aliases = {}
    for raw_alias, raw_canonical in aliases_source.items():
        alias = _require_tensor_name(raw_alias, context="alias name")
        canonical = _require_tensor_name(raw_canonical, context=f"alias {alias!r}")
        if alias in provisional or canonical not in provisional or alias == canonical:
            raise ResidentStateError(
                "Alias entries must target physical tensors directly."
            )
        aliases[alias] = canonical

    descriptors = {
        name: ResidentTensorDescriptor(
            descriptor.dtype,
            descriptor.shape,
            descriptor.nbytes,
            weight_map[name],
        )
        for name, descriptor in provisional.items()
    }
    names_by_shard = {
        filename: {name for name, mapped in weight_map.items() if mapped == filename}
        for filename in expected_filenames
    }
    for path in shard_paths:
        _validate_shard(
            path,
            expected_names=names_by_shard[path.name],
            descriptors=descriptors,
        )

    return ResidentStateManifest(
        directory=target,
        index_path=index_path,
        shard_paths=tuple(shard_paths),
        descriptors=descriptors,
        aliases=aliases,
        shard_hashes=shard_hashes,
        weight_map=weight_map,
        total_size=total_size,
        index_sha256=_sha256_file(index_path),
    )


def iter_resident_shards(
    manifest_or_directory: ResidentStateManifest | str | os.PathLike[str],
) -> Iterator[tuple[Path, dict[str, torch.Tensor]]]:
    """Yield one revalidated resident Safetensors shard at a time."""

    manifest = (
        manifest_or_directory
        if isinstance(manifest_or_directory, ResidentStateManifest)
        else load_resident_state(manifest_or_directory)
    )
    safe_open, _save_file = _require_safetensors()
    for path in manifest.shard_paths:
        expected_digest = manifest.shard_hashes[path.name]
        if not path.is_file() or _sha256_file(path) != expected_digest:
            raise ResidentStateError(f"Shard {path.name!r} changed after validation.")
        expected_names = {
            name
            for name, filename in manifest.weight_map.items()
            if filename == path.name
        }
        _validate_shard(
            path,
            expected_names=expected_names,
            descriptors=manifest.descriptors,
        )
        try:
            with safe_open(str(path), framework="pt", device="cpu") as archive:
                tensors = {
                    name: archive.get_tensor(name)
                    for name in archive.keys()  # noqa: SIM118 - safe_open is not iterable.
                }
        except Exception as error:
            raise ResidentStateError(f"Unable to load shard {path.name!r}.") from error
        yield path, tensors


__all__ = [
    "FORMAT_NAME",
    "FORMAT_VERSION",
    "INDEX_FILENAME",
    "ResidentStateError",
    "ResidentStateManifest",
    "ResidentTensorDescriptor",
    "iter_resident_shards",
    "load_resident_state",
    "save_resident_state",
]
