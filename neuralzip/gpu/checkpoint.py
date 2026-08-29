"""Persistent, versioned checkpoints for GPU-ready NeuralZip exports.

The archive is one atomic Safetensors file.  Byte streams and GPU lookup
tables are stored as typed tensors; bindings and structural metadata live in
one canonical JSON manifest embedded in the Safetensors header.  Loading never
uses pickle and validates the complete schema before constructing runtime
objects.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Mapping
from functools import partial
from pathlib import Path

import torch

from ._persistence import (
    atomic_safetensors,
    canonical_json,
    path_argument,
    require_dotted_name,
    require_fields,
    require_int,
    require_string,
    safetensors_functions,
    strict_json_loads,
)
from .export import GPUCodebook, GPUEncodedBF16, GPUEncodedExponents
from .inference import BF16WeightGroup, LinearWeightBinding, NeuralZipModelExport

FORMAT_NAME = "neuralzip-gpu-export"
FORMAT_VERSION = 1
DEFAULT_FILENAME = "neuralzip-gpu-export.safetensors"

_MANIFEST_METADATA_KEY = "neuralzip.manifest"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_DTYPES = {
    "uint8": torch.uint8,
    "uint16": torch.uint16,
    "int32": torch.int32,
}
_DTYPE_NAMES = {value: key for key, value in _DTYPES.items()}
_MAX_INT32 = (1 << 31) - 1


class GPUCheckpointError(ValueError):
    """Raised when a GPU checkpoint is malformed, corrupt, or unsupported."""


_canonical_json = partial(
    canonical_json,
    error_type=GPUCheckpointError,
    error_message="The GPU checkpoint manifest is not JSON-safe.",
)
_require_fields = partial(
    require_fields,
    error_type=GPUCheckpointError,
    compact=True,
)
_require_string = partial(require_string, error_type=GPUCheckpointError)
_require_int = partial(require_int, error_type=GPUCheckpointError)
_require_name = partial(
    require_dotted_name,
    kind="module",
    error_type=GPUCheckpointError,
)
_require_safetensors = partial(
    safetensors_functions,
    "GPU checkpoint persistence requires the 'safetensors' package.",
)
_path_argument = partial(
    path_argument,
    type_message="directory_or_file must be a string or path-like value.",
    bytes_message="directory_or_file must be a text path, not bytes.",
    invalid_message="directory_or_file must be a non-empty valid path.",
)


def _parse_canonical_json(source: str) -> dict[str, object]:
    if not isinstance(source, str):
        raise GPUCheckpointError("The embedded GPU checkpoint manifest is missing.")
    value = strict_json_loads(
        source,
        error_type=GPUCheckpointError,
        document="GPU checkpoint",
        invalid_message="The embedded GPU checkpoint manifest is invalid JSON.",
    )
    if not isinstance(value, dict):
        raise GPUCheckpointError("The GPU checkpoint manifest must be a JSON object.")
    if _canonical_json(value) != source:
        raise GPUCheckpointError("The GPU checkpoint manifest is not canonical JSON.")
    return value


def _require_list(value: object, *, context: str) -> list[object]:
    if not isinstance(value, list):
        raise GPUCheckpointError(f"{context} must be a JSON array.")
    return value


def _require_bool(value: object, *, context: str) -> bool:
    if not isinstance(value, bool):
        raise GPUCheckpointError(f"{context} must be a boolean.")
    return value


def _validate_binding_paths(binding: LinearWeightBinding) -> None:
    block_name = _require_name(binding.block_name, context="binding.block_name")
    module_name = _require_name(binding.module_name, context="binding.module_name")
    tensor_name = _require_name(binding.tensor_name, context="binding.tensor_name")
    if module_name != block_name and not module_name.startswith(f"{block_name}."):
        raise GPUCheckpointError(
            "A binding module must be inside its transformer block."
        )
    if tensor_name != f"{module_name}.weight":
        raise GPUCheckpointError(
            "A binding tensor path must be '<module_name>.weight'."
        )


def _validate_export_paths(exported: NeuralZipModelExport) -> None:
    for block_name in exported.block_order:
        _require_name(block_name, context="block_order entry")
    referenced_blocks = {group.block_name for group in exported.groups}
    if referenced_blocks != set(exported.block_order):
        raise GPUCheckpointError("Every block_order entry must own a compressed group.")
    for group in exported.groups:
        _require_name(group.block_name, context="group.block_name")
        if group.encoded.exponents.symbol_count > _MAX_INT32:
            raise GPUCheckpointError(
                "A group exceeds the GPU ABI's int32 symbol limit."
            )
        if any(
            position > _MAX_INT32
            for position in group.encoded.exponents.output_positions
        ):
            raise GPUCheckpointError(
                "An output position exceeds the GPU ABI's int32 limit."
            )
        for binding in group.bindings:
            _validate_binding_paths(binding)
    for name in exported.resident_weight_names:
        resident_name = _require_name(name, context="resident_weight_names entry")
        if not resident_name.endswith(".weight"):
            raise GPUCheckpointError("Resident tensor paths must end in '.weight'.")


def _checkpoint_path(
    directory_or_file: str | os.PathLike[str],
    *,
    create_directory: bool,
) -> Path:
    path = _path_argument(directory_or_file)
    if path.exists() and path.is_dir():
        directory = path
        if create_directory:
            directory.mkdir(parents=True, exist_ok=True)
        return directory / DEFAULT_FILENAME
    if path.suffix == ".safetensors":
        if create_directory:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path
    if path.suffix:
        raise ValueError("A GPU checkpoint file must use the '.safetensors' suffix.")
    if path.exists() and not path.is_dir():
        raise ValueError("A suffixless GPU checkpoint path must be a directory.")
    if create_directory:
        path.mkdir(parents=True, exist_ok=True)
    return path / DEFAULT_FILENAME


def _bytes_tensor(value: bytes) -> torch.Tensor:
    if not value:
        return torch.empty(0, dtype=torch.uint8)
    return torch.frombuffer(bytearray(value), dtype=torch.uint8).clone()


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    if tensor.device.type != "cpu":
        tensor = tensor.cpu()
    return tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def _tensor_descriptor(tensor: torch.Tensor) -> dict[str, object]:
    dtype = _DTYPE_NAMES.get(tensor.dtype)
    if dtype is None:  # pragma: no cover - construction only uses known dtypes.
        raise GPUCheckpointError(f"Unsupported checkpoint tensor dtype {tensor.dtype}.")
    payload = _tensor_bytes(tensor)
    return {
        "dtype": dtype,
        "nbytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "shape": list(tensor.shape),
    }


def _binding_manifest(binding: LinearWeightBinding) -> dict[str, object]:
    return {
        "block_name": binding.block_name,
        "group_offset": binding.group_offset,
        "module_kind": binding.module_kind,
        "module_name": binding.module_name,
        "numel": binding.numel,
        "requires_grad": binding.requires_grad,
        "shape": list(binding.shape),
        "tensor_name": binding.tensor_name,
        "workspace_offset": binding.workspace_offset,
    }


def _build_archive(
    exported: NeuralZipModelExport,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    _validate_export_paths(exported)
    tensors: dict[str, torch.Tensor] = {}

    codebooks_by_id: dict[int, GPUCodebook] = {}
    for group in exported.groups:
        previous = codebooks_by_id.setdefault(group.codebook_id, group.codebook)
        if previous != group.codebook:  # Also enforced by NeuralZipModelExport.
            raise GPUCheckpointError("A codebook ID maps to inconsistent codebooks.")

    lut_names: dict[GPUCodebook, str] = {}
    codebook_manifest = []
    for codebook_id, codebook in sorted(codebooks_by_id.items()):
        lut_name = lut_names.get(codebook)
        if lut_name is None:
            lut_name = f"codebooks.{len(lut_names):06d}.lut"
            lut_names[codebook] = lut_name
            tensors[lut_name] = torch.tensor(codebook.decode_lut, dtype=torch.uint16)
        codebook_manifest.append(
            {
                "code_lengths": list(codebook.code_lengths),
                "code_values": list(codebook.code_values),
                "id": codebook_id,
                "lut_tensor": lut_name,
                "max_bits": codebook.max_bits,
            }
        )

    groups_manifest = []
    for index, group in enumerate(exported.groups):
        prefix = f"groups.{index:06d}"
        buffer_names = {
            "bitstream": f"{prefix}.bitstream",
            "output_positions": f"{prefix}.output_positions",
            "packed_gaps": f"{prefix}.packed_gaps",
            "sign_mantissa": f"{prefix}.sign_mantissa",
        }
        exponents = group.encoded.exponents
        tensors[buffer_names["bitstream"]] = _bytes_tensor(exponents.bitstream)
        tensors[buffer_names["sign_mantissa"]] = _bytes_tensor(
            group.encoded.sign_mantissa
        )
        tensors[buffer_names["packed_gaps"]] = _bytes_tensor(exponents.packed_gaps)
        tensors[buffer_names["output_positions"]] = torch.tensor(
            exponents.output_positions,
            dtype=torch.int32,
        )
        groups_manifest.append(
            {
                "bindings": [_binding_manifest(binding) for binding in group.bindings],
                "block_name": group.block_name,
                "buffers": buffer_names,
                "codebook_id": group.codebook_id,
                "encoded": {
                    "bit_length": exponents.bit_length,
                    "bytes_per_thread": exponents.bytes_per_thread,
                    "symbol_count": exponents.symbol_count,
                    "threads_per_block": exponents.threads_per_block,
                },
                "workspace_offset": group.workspace_offset,
            }
        )

    manifest: dict[str, object] = {
        "codebooks": codebook_manifest,
        "export": {
            "block_order": list(exported.block_order),
            "groups": groups_manifest,
            "resident_raw_bytes": exported.resident_raw_bytes,
            "resident_weight_names": list(exported.resident_weight_names),
        },
        "format": FORMAT_NAME,
        "tensors": {
            name: _tensor_descriptor(tensor) for name, tensor in tensors.items()
        },
        "version": FORMAT_VERSION,
    }
    return tensors, manifest


def save_gpu_export(
    exported: NeuralZipModelExport,
    directory_or_file: str | os.PathLike[str],
) -> Path:
    """Atomically save a validated GPU export and return its Safetensors path.

    A directory target receives :data:`DEFAULT_FILENAME`; an explicit file
    target must end in ``.safetensors``.
    """

    if not isinstance(exported, NeuralZipModelExport):
        raise TypeError("exported must be a NeuralZipModelExport.")
    _require_safetensors()
    tensors, manifest = _build_archive(exported)
    checkpoint = _checkpoint_path(directory_or_file, create_directory=True)
    if checkpoint.exists() and not checkpoint.is_file():
        raise ValueError("The GPU checkpoint target is not a regular file.")

    metadata = {_MANIFEST_METADATA_KEY: _canonical_json(manifest)}
    atomic_safetensors(
        tensors,
        checkpoint,
        metadata=metadata,
        dependency_error="GPU checkpoint persistence requires the 'safetensors' package.",
    )
    return checkpoint


def _validate_manifest_root(
    value: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    root = _require_fields(
        value,
        {"codebooks", "export", "format", "tensors", "version"},
        context="manifest",
    )
    if root["format"] != FORMAT_NAME:
        raise GPUCheckpointError("This is not a NeuralZip GPU checkpoint.")
    version = _require_int(root["version"], context="manifest.version", minimum=1)
    if version != FORMAT_VERSION:
        raise GPUCheckpointError(
            f"Unsupported NeuralZip GPU checkpoint version {version}; "
            f"expected {FORMAT_VERSION}."
        )
    export = _require_fields(
        root["export"],
        {"block_order", "groups", "resident_raw_bytes", "resident_weight_names"},
        context="manifest.export",
    )
    tensors = root["tensors"]
    if not isinstance(tensors, dict):
        raise GPUCheckpointError("manifest.tensors must be a JSON object.")
    return export, tensors


def _validate_tensor_descriptors(
    source: Mapping[str, object],
    tensors: Mapping[str, torch.Tensor],
) -> None:
    if set(source) != set(tensors):
        missing = sorted(set(source) - set(tensors))
        unexpected = sorted(set(tensors) - set(source))
        raise GPUCheckpointError(
            f"Safetensors members do not match the manifest "
            f"(missing={missing}, unexpected={unexpected})."
        )
    for name, tensor in tensors.items():
        descriptor = _require_fields(
            source[name],
            {"dtype", "nbytes", "sha256", "shape"},
            context=f"tensor descriptor {name!r}",
        )
        dtype_name = _require_string(
            descriptor["dtype"], context=f"tensor {name!r} dtype"
        )
        expected_dtype = _DTYPES.get(dtype_name)
        if expected_dtype is None or tensor.dtype != expected_dtype:
            raise GPUCheckpointError(f"Tensor {name!r} has an invalid dtype.")
        shape_values = _require_list(
            descriptor["shape"], context=f"tensor {name!r} shape"
        )
        shape = tuple(
            _require_int(value, context=f"tensor {name!r} shape entry")
            for value in shape_values
        )
        if len(shape) != 1 or tuple(tensor.shape) != shape:
            raise GPUCheckpointError(f"Tensor {name!r} has an invalid shape.")
        payload = _tensor_bytes(tensor)
        nbytes = _require_int(descriptor["nbytes"], context=f"tensor {name!r} nbytes")
        if nbytes != len(payload):
            raise GPUCheckpointError(f"Tensor {name!r} has an invalid byte size.")
        digest = _require_string(
            descriptor["sha256"], context=f"tensor {name!r} sha256"
        )
        if not _SHA256_RE.fullmatch(digest):
            raise GPUCheckpointError(f"Tensor {name!r} has an invalid SHA-256 value.")
        if hashlib.sha256(payload).hexdigest() != digest:
            raise GPUCheckpointError(f"Tensor {name!r} failed its SHA-256 check.")


def _int_array(
    value: object,
    *,
    context: str,
    length: int | None = None,
    maximum: int | None = None,
) -> tuple[int, ...]:
    source = _require_list(value, context=context)
    if length is not None and len(source) != length:
        raise GPUCheckpointError(f"{context} must contain exactly {length} integers.")
    return tuple(
        _require_int(item, context=f"{context} entry", maximum=maximum)
        for item in source
    )


def _load_codebooks(
    source: object,
    tensors: Mapping[str, torch.Tensor],
) -> tuple[dict[int, GPUCodebook], set[str]]:
    records = _require_list(source, context="manifest.codebooks")
    codebooks: dict[int, GPUCodebook] = {}
    lut_names: list[str] = []
    previous_id = -1
    for index, raw_record in enumerate(records):
        record = _require_fields(
            raw_record,
            {"code_lengths", "code_values", "id", "lut_tensor", "max_bits"},
            context=f"codebook {index}",
        )
        codebook_id = _require_int(record["id"], context=f"codebook {index} id")
        if codebook_id <= previous_id:
            raise GPUCheckpointError(
                "Codebooks must be ordered by unique increasing IDs."
            )
        previous_id = codebook_id
        lengths = _int_array(
            record["code_lengths"],
            context=f"codebook {codebook_id} code_lengths",
            length=256,
            maximum=12,
        )
        values = _int_array(
            record["code_values"],
            context=f"codebook {codebook_id} code_values",
            length=256,
            maximum=(1 << 12) - 1,
        )
        try:
            codebook = GPUCodebook.from_codes(lengths, values)
        except (TypeError, ValueError) as error:
            raise GPUCheckpointError(f"Codebook {codebook_id} is invalid.") from error
        max_bits = _require_int(
            record["max_bits"], context=f"codebook {codebook_id} max_bits", maximum=12
        )
        if max_bits != codebook.max_bits:
            raise GPUCheckpointError(
                f"Codebook {codebook_id} max_bits is inconsistent."
            )
        lut_name = _require_string(
            record["lut_tensor"], context=f"codebook {codebook_id} lut_tensor"
        )
        if lut_name not in lut_names:
            expected = f"codebooks.{len(lut_names):06d}.lut"
            if lut_name != expected:
                raise GPUCheckpointError("Codebook LUT tensor names are not canonical.")
            lut_names.append(lut_name)
        lut = tensors.get(lut_name)
        if lut is None or lut.dtype != torch.uint16 or lut.ndim != 1:
            raise GPUCheckpointError(f"Codebook {codebook_id} LUT tensor is missing.")
        if tuple(int(value) for value in lut.tolist()) != codebook.decode_lut:
            raise GPUCheckpointError(f"Codebook {codebook_id} LUT is inconsistent.")
        codebooks[codebook_id] = codebook
    return codebooks, set(lut_names)


def _load_binding(
    value: object, *, group_index: int, binding_index: int
) -> LinearWeightBinding:
    context = f"group {group_index} binding {binding_index}"
    record = _require_fields(
        value,
        {
            "block_name",
            "group_offset",
            "module_kind",
            "module_name",
            "numel",
            "requires_grad",
            "shape",
            "tensor_name",
            "workspace_offset",
        },
        context=context,
    )
    shape = _int_array(record["shape"], context=f"{context} shape", length=2)
    try:
        binding = LinearWeightBinding(
            block_name=_require_name(
                record["block_name"], context=f"{context} block_name"
            ),
            module_name=_require_name(
                record["module_name"], context=f"{context} module_name"
            ),
            tensor_name=_require_name(
                record["tensor_name"], context=f"{context} tensor_name"
            ),
            shape=shape,
            numel=_require_int(record["numel"], context=f"{context} numel"),
            workspace_offset=_require_int(
                record["workspace_offset"], context=f"{context} workspace_offset"
            ),
            group_offset=_require_int(
                record["group_offset"], context=f"{context} group_offset"
            ),
            requires_grad=_require_bool(
                record["requires_grad"], context=f"{context} requires_grad"
            ),
            module_kind=_require_string(
                record["module_kind"], context=f"{context} module_kind"
            ),
        )
        _validate_binding_paths(binding)
        return binding
    except GPUCheckpointError:
        raise
    except (TypeError, ValueError) as error:
        raise GPUCheckpointError(f"{context} is invalid.") from error


def _bytes_from_tensor(tensor: torch.Tensor) -> bytes:
    if tensor.dtype != torch.uint8 or tensor.ndim != 1:
        raise GPUCheckpointError("A byte buffer has an invalid dtype or shape.")
    return _tensor_bytes(tensor)


def _load_groups(
    source: object,
    codebooks: Mapping[int, GPUCodebook],
    tensors: Mapping[str, torch.Tensor],
) -> tuple[tuple[BF16WeightGroup, ...], set[str]]:
    records = _require_list(source, context="manifest.export.groups")
    groups = []
    used_tensors = set()
    for index, raw_record in enumerate(records):
        context = f"group {index}"
        record = _require_fields(
            raw_record,
            {
                "bindings",
                "block_name",
                "buffers",
                "codebook_id",
                "encoded",
                "workspace_offset",
            },
            context=context,
        )
        buffers = _require_fields(
            record["buffers"],
            {"bitstream", "output_positions", "packed_gaps", "sign_mantissa"},
            context=f"{context} buffers",
        )
        prefix = f"groups.{index:06d}"
        expected_buffers = {
            "bitstream": f"{prefix}.bitstream",
            "output_positions": f"{prefix}.output_positions",
            "packed_gaps": f"{prefix}.packed_gaps",
            "sign_mantissa": f"{prefix}.sign_mantissa",
        }
        if buffers != expected_buffers:
            raise GPUCheckpointError(f"{context} buffer names are not canonical.")
        if used_tensors.intersection(expected_buffers.values()):
            raise GPUCheckpointError("Group buffers cannot be shared.")
        used_tensors.update(expected_buffers.values())
        try:
            selected = {key: tensors[name] for key, name in expected_buffers.items()}
        except KeyError as error:
            raise GPUCheckpointError(
                f"{context} references a missing buffer."
            ) from error

        encoded_record = _require_fields(
            record["encoded"],
            {"bit_length", "bytes_per_thread", "symbol_count", "threads_per_block"},
            context=f"{context} encoded metadata",
        )
        positions_tensor = selected["output_positions"]
        if positions_tensor.dtype != torch.int32 or positions_tensor.ndim != 1:
            raise GPUCheckpointError(f"{context} output_positions has an invalid type.")
        positions = tuple(int(value) for value in positions_tensor.tolist())
        try:
            exponents = GPUEncodedExponents(
                bitstream=_bytes_from_tensor(selected["bitstream"]),
                bit_length=_require_int(
                    encoded_record["bit_length"], context=f"{context} bit_length"
                ),
                symbol_count=_require_int(
                    encoded_record["symbol_count"],
                    context=f"{context} symbol_count",
                    maximum=_MAX_INT32,
                ),
                packed_gaps=_bytes_from_tensor(selected["packed_gaps"]),
                output_positions=positions,
                bytes_per_thread=_require_int(
                    encoded_record["bytes_per_thread"],
                    context=f"{context} bytes_per_thread",
                    minimum=1,
                ),
                threads_per_block=_require_int(
                    encoded_record["threads_per_block"],
                    context=f"{context} threads_per_block",
                    minimum=1,
                ),
            )
            encoded = GPUEncodedBF16(
                sign_mantissa=_bytes_from_tensor(selected["sign_mantissa"]),
                exponents=exponents,
            )
        except GPUCheckpointError:
            raise
        except (TypeError, ValueError) as error:
            raise GPUCheckpointError(
                f"{context} encoded buffers are inconsistent."
            ) from error

        codebook_id = _require_int(
            record["codebook_id"], context=f"{context} codebook_id"
        )
        codebook = codebooks.get(codebook_id)
        if codebook is None:
            raise GPUCheckpointError(f"{context} references an unknown codebook.")
        binding_records = _require_list(
            record["bindings"], context=f"{context} bindings"
        )
        bindings = tuple(
            _load_binding(value, group_index=index, binding_index=binding_index)
            for binding_index, value in enumerate(binding_records)
        )
        try:
            groups.append(
                BF16WeightGroup(
                    block_name=_require_name(
                        record["block_name"], context=f"{context} block_name"
                    ),
                    codebook_id=codebook_id,
                    codebook=codebook,
                    encoded=encoded,
                    bindings=bindings,
                    workspace_offset=_require_int(
                        record["workspace_offset"],
                        context=f"{context} workspace_offset",
                    ),
                )
            )
        except GPUCheckpointError:
            raise
        except (TypeError, ValueError) as error:
            raise GPUCheckpointError(f"{context} structure is inconsistent.") from error
    return tuple(groups), used_tensors


def load_gpu_export(
    directory_or_file: str | os.PathLike[str],
) -> NeuralZipModelExport:
    """Load and fully validate one GPU-ready NeuralZip export."""

    safe_open, _save_file = _require_safetensors()
    checkpoint = _checkpoint_path(directory_or_file, create_directory=False)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    if not checkpoint.is_file():
        raise ValueError("The GPU checkpoint path is not a regular file.")
    try:
        with safe_open(str(checkpoint), framework="pt", device="cpu") as archive:
            metadata = archive.metadata()
            if metadata is None or set(metadata) != {_MANIFEST_METADATA_KEY}:
                raise GPUCheckpointError(
                    "GPU checkpoint metadata has an invalid schema."
                )
            manifest = _parse_canonical_json(metadata[_MANIFEST_METADATA_KEY])
            tensors = {
                name: archive.get_tensor(name)
                for name in archive.keys()  # noqa: SIM118 - safe_open is not iterable.
            }
    except GPUCheckpointError:
        raise
    except Exception as error:
        raise GPUCheckpointError(
            "Unable to read the Safetensors GPU checkpoint."
        ) from error

    export_record, descriptors = _validate_manifest_root(manifest)
    _validate_tensor_descriptors(descriptors, tensors)
    codebooks, lut_tensors = _load_codebooks(manifest["codebooks"], tensors)
    groups, group_tensors = _load_groups(export_record["groups"], codebooks, tensors)
    if set(codebooks) != {group.codebook_id for group in groups}:
        raise GPUCheckpointError("Every stored codebook must be used by a group.")
    if lut_tensors | group_tensors != set(tensors):
        raise GPUCheckpointError("The GPU checkpoint contains unreferenced tensors.")

    block_order = tuple(
        _require_name(value, context="block_order entry")
        for value in _require_list(
            export_record["block_order"], context="manifest.export.block_order"
        )
    )
    resident_names = tuple(
        _require_name(value, context="resident_weight_names entry")
        for value in _require_list(
            export_record["resident_weight_names"],
            context="manifest.export.resident_weight_names",
        )
    )
    try:
        exported = NeuralZipModelExport(
            groups=groups,
            block_order=block_order,
            resident_weight_names=resident_names,
            resident_raw_bytes=_require_int(
                export_record["resident_raw_bytes"], context="resident_raw_bytes"
            ),
        )
        _validate_export_paths(exported)
        return exported
    except GPUCheckpointError:
        raise
    except (TypeError, ValueError) as error:
        raise GPUCheckpointError("The GPU export structure is inconsistent.") from error


__all__ = [
    "DEFAULT_FILENAME",
    "FORMAT_NAME",
    "FORMAT_VERSION",
    "GPUCheckpointError",
    "load_gpu_export",
    "save_gpu_export",
]
