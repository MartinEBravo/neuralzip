"""Persistent, directly loadable NeuralZip inference checkpoints.

One canonical JSON manifest binds the model metadata, compressed BF16 streams,
and sharded resident Safetensors without deserializing Python objects.
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from collections.abc import Callable, Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn

from .. import _persistence
from .export import GPUCodebook, GPUEncodedBF16, GPUEncodedExponents
from .inference import (
    BF16WeightGroup,
    LinearWeightBinding,
    NeuralZipModelExport,
    NeuralZipRuntime,
    _offloaded_blocks,
    export_model_for_gpu,
)

FORMAT_NAME = "neuralzip-gpu-pretrained"
FORMAT_VERSION = 2
MANIFEST_FILENAME = "neuralzip_config.json"
GPU_EXPORT_FILENAME = "neuralzip-gpu-export.safetensors"
RESIDENT_SHARD_PATTERN = "neuralzip-resident{suffix}.safetensors"
_GPU_EXPORT_FORMAT = "neuralzip-gpu-export"
_RESIDENT_FORMAT = "neuralzip-resident-state"
_PLACEMENT_BUFFER_NAME = "_neuralzip_placement_payload"
_RESIDENT_SHARD_RE = re.compile(
    r"^neuralzip-resident(?:-\d{5}-of-\d{5})?\.safetensors$"
)
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
GPUPretrainedError = _persistence.GPUPretrainedError
_atomic_safetensors = _persistence.atomic_safetensors
_atomic_text = _persistence.atomic_text
_canonical_json = _persistence.canonical_json
_require_fields = _persistence.require_fields
_require_int = _persistence.require_int
_require_name = _persistence.require_name
_require_string = _persistence.require_string
_sha256_file = _persistence.sha256_file


def _directory_argument(value: str | os.PathLike[str]) -> Path:
    try:
        source = os.fspath(value)
    except TypeError as error:
        raise TypeError("directory must be a string or path-like value.") from error
    if isinstance(source, bytes):
        raise TypeError("directory must be a text path, not bytes.")
    if not source or "\x00" in source:
        raise ValueError("directory must be a non-empty valid path.")
    return Path(source)


def _parse_manifest(path: Path) -> dict[str, object]:
    try:
        source = path.read_text(encoding="ascii")
        value = _persistence.canonical_json_loads(
            source, document="checkpoint manifest"
        )
    except FileNotFoundError:
        raise
    except (OSError, UnicodeError) as error:
        raise GPUPretrainedError("Unable to read neuralzip_config.json.") from error
    if not isinstance(value, dict):
        raise GPUPretrainedError("The checkpoint manifest must be a JSON object.")
    if _canonical_json(value) + "\n" != source:
        raise GPUPretrainedError("The checkpoint manifest is not canonical JSON.")
    return value


def _require_list(value: object, *, context: str) -> list[object]:
    if not isinstance(value, list):
        raise GPUPretrainedError(f"{context} must be a JSON array.")
    return value


def _checkpoint_directory(value: str | os.PathLike[str], *, create: bool) -> Path:
    path = _directory_argument(value)
    if path.exists() and not path.is_dir():
        raise ValueError("A NeuralZip pretrained checkpoint must be a directory.")
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_pretrained_source(
    value: str | os.PathLike[str],
    *,
    revision: str,
    token: str | bool | None,
    cache_dir: str | os.PathLike[str] | None,
    local_files_only: bool,
) -> Path:
    try:
        source = os.fspath(value)
    except TypeError as error:
        raise TypeError(
            "pretrained_model_name_or_path must be a string or path-like value."
        ) from error
    if isinstance(source, bytes):
        raise TypeError("pretrained_model_name_or_path must be text, not bytes.")
    local = Path(source)
    if local.exists() or isinstance(value, os.PathLike) or local.is_absolute():
        directory = _checkpoint_directory(local, create=False)
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        return directory
    if source.startswith((".", "~")):
        raise FileNotFoundError(local)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:  # pragma: no cover - transformers dependency.
        raise RuntimeError(
            "Loading a Hub checkpoint requires huggingface_hub."
        ) from error
    try:
        snapshot = snapshot_download(
            repo_id=source,
            revision=revision,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
    except Exception as error:
        raise GPUPretrainedError(
            f"Unable to resolve NeuralZip checkpoint {source!r} from the Hub."
        ) from error
    return Path(snapshot)


def _validate_auto_class(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        import transformers
    except ImportError as error:  # pragma: no cover - core dependency declaration.
        raise RuntimeError("auto_class requires transformers.") from error
    candidate = getattr(transformers, value, None)
    if not value.startswith("AutoModel") or not hasattr(candidate, "_model_mapping"):
        raise ValueError("auto_class must name a Transformers AutoModel class.")
    return value


def _infer_auto_class(model: nn.Module) -> str | None:
    config = getattr(model, "config", None)
    if config is None:
        return None
    try:
        import transformers
    except ImportError:  # pragma: no cover - core dependency declaration.
        return None
    auto_classes = tuple(
        (name, candidate)
        for name in dir(transformers)
        if name.startswith("AutoModel")
        and hasattr((candidate := getattr(transformers, name, None)), "_model_mapping")
    )
    auto_map = getattr(config, "auto_map", None)
    if isinstance(auto_map, Mapping):
        for class_name, _auto_class in auto_classes:
            if class_name in auto_map:
                return class_name
    for class_name, auto_class in auto_classes:
        mapping = auto_class._model_mapping
        try:
            candidate = mapping[type(config)]
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
        candidates = candidate if isinstance(candidate, tuple) else (candidate,)
        if any(type(model) is value for value in candidates):
            return class_name
    return None


def _model_library(model: nn.Module) -> str:
    module = type(model).__module__
    if module == "transformers" or module.startswith("transformers."):
        return "transformers"
    if module == "diffusers" or module.startswith("diffusers."):
        return "diffusers"
    return "custom"


def _save_model_metadata(
    model: nn.Module,
    directory: Path,
    *,
    library: str,
) -> dict[str, object]:
    config = getattr(model, "config", None)
    config_record = None
    config_saved = False
    if config is not None and callable(getattr(config, "save_pretrained", None)):
        config.save_pretrained(directory)
        config_saved = True
    elif library == "diffusers" and callable(getattr(model, "save_config", None)):
        model.save_config(directory)
        config_saved = True
    if config_saved:
        config_path = directory / "config.json"
        if not config_path.is_file():
            raise GPUPretrainedError("The model did not create config.json.")
        config_record = _file_record(config_path)

    generation_record = None
    generation = getattr(model, "generation_config", None)
    if (
        library == "transformers"
        and generation is not None
        and callable(getattr(generation, "save_pretrained", None))
    ):
        generation.save_pretrained(directory)
        generation_path = directory / "generation_config.json"
        if generation_path.is_file():
            generation_record = _file_record(generation_path)
    return {"config": config_record, "generation_config": generation_record}


def _file_record(path: Path) -> dict[str, str | int]:
    return {
        "name": path.name,
        "nbytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _bytes_tensor(value: bytes) -> torch.Tensor:
    if not value:
        return torch.empty(0, dtype=torch.uint8)
    return torch.frombuffer(bytearray(value), dtype=torch.uint8).clone()


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def _save_gpu_export(
    exported: NeuralZipModelExport,
    directory: Path,
) -> tuple[Path, dict[str, object]]:
    codebooks = {}
    for group in exported.groups:
        codebooks.setdefault(group.codebook_id, group.codebook)

    tensors = {}
    groups = []
    for index, group in enumerate(exported.groups):
        prefix = f"groups.{index:06d}"
        exponents = group.encoded.exponents
        tensors[f"{prefix}.bitstream"] = _bytes_tensor(exponents.bitstream)
        tensors[f"{prefix}.packed_gaps"] = _bytes_tensor(exponents.packed_gaps)
        tensors[f"{prefix}.sign_mantissa"] = _bytes_tensor(group.encoded.sign_mantissa)
        tensors[f"{prefix}.output_positions"] = torch.tensor(
            exponents.output_positions,
            dtype=torch.int32,
        )
        groups.append(
            {
                "bindings": [
                    {
                        "kind": binding.module_kind,
                        "module": binding.module_name,
                        "requires_grad": binding.requires_grad,
                        "shape": list(binding.shape),
                    }
                    for binding in group.bindings
                ],
                "bit_length": exponents.bit_length,
                "block": group.block_name,
                "codebook": group.codebook_id,
            }
        )

    structure = {
        "codebooks": [
            {
                "id": codebook_id,
                "lengths": list(codebook.code_lengths),
                "values": list(codebook.code_values),
            }
            for codebook_id, codebook in sorted(codebooks.items())
        ],
        "groups": groups,
        "resident_raw_bytes": exported.resident_raw_bytes,
        "resident_weight_names": list(exported.resident_weight_names),
    }
    path = directory / GPU_EXPORT_FILENAME
    _atomic_safetensors(
        tensors,
        path,
        metadata={"format": _GPU_EXPORT_FORMAT, "version": str(FORMAT_VERSION)},
    )
    return path, structure


def _storage_key(tensor: torch.Tensor) -> tuple[object, ...]:
    storage = tensor.untyped_storage()
    return (
        tensor.device,
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


def _snapshot_resident(
    tensors: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    physical = {}
    aliases = {}
    canonical_by_storage = {}
    for name in sorted(tensors):
        _require_name(name, context="resident tensor name")
        tensor = tensors[name]
        if (
            not torch.is_tensor(tensor)
            or tensor.device.type == "meta"
            or tensor.layout != torch.strided
            or tensor.is_quantized
        ):
            raise GPUPretrainedError(
                f"Resident state {name!r} must be a dense, materialized tensor."
            )
        storage_key = _storage_key(tensor)
        canonical = canonical_by_storage.get(storage_key)
        if canonical is not None:
            aliases[name] = canonical
            continue
        canonical_by_storage[storage_key] = name
        physical[name] = tensor.detach().to(device="cpu").contiguous().clone()
    return physical, aliases


def _parse_max_shard_size(value: int | str) -> int:
    if isinstance(value, bool):
        raise TypeError("max_shard_size must be a positive integer or size string.")
    if isinstance(value, int):
        size = value
    elif isinstance(value, str):
        match = _SIZE_RE.fullmatch(value)
        if match is None:
            raise ValueError(
                "max_shard_size must use B, KB, MB, GB, TB, KiB, MiB, GiB, or TiB."
            )
        amount, unit = match.groups()
        try:
            size = int(Decimal(amount) * _SIZE_UNITS[unit.upper()])
        except (InvalidOperation, OverflowError) as error:
            raise ValueError(
                "max_shard_size is outside the supported range."
            ) from error
    else:
        raise TypeError("max_shard_size must be a positive integer or size string.")
    if size <= 0:
        raise ValueError("max_shard_size must be positive.")
    return size


def _resident_state(model: nn.Module, exported: NeuralZipModelExport):
    compressed = {binding.tensor_name for binding in exported.bindings}
    state = model.state_dict()
    if not compressed.issubset(state):
        raise GPUPretrainedError("The model state is missing compressed tensors.")
    return {name: tensor for name, tensor in state.items() if name not in compressed}


def _save_resident_state(
    tensors: Mapping[str, torch.Tensor],
    directory: Path,
    *,
    max_shard_size: int | str,
) -> tuple[list[dict[str, str | int]], dict[str, object]]:
    try:
        from huggingface_hub import split_torch_state_dict_into_shards
    except ImportError as error:  # pragma: no cover - transformers dependency.
        raise RuntimeError("Resident sharding requires huggingface_hub.") from error

    physical, aliases = _snapshot_resident(tensors)
    split = split_torch_state_dict_into_shards(
        physical,
        filename_pattern=RESIDENT_SHARD_PATTERN,
        max_shard_size=_parse_max_shard_size(max_shard_size),
    )
    filenames = sorted(split.filename_to_tensors)
    for filename in filenames:
        shard = {name: physical[name] for name in split.filename_to_tensors[filename]}
        _atomic_safetensors(
            shard,
            directory / filename,
            metadata={"format": _RESIDENT_FORMAT, "version": str(FORMAT_VERSION)},
        )
    expected = set(filenames)
    for path in directory.glob("neuralzip-resident*.safetensors"):
        if _RESIDENT_SHARD_RE.fullmatch(path.name) and path.name not in expected:
            path.unlink()
    return (
        [_file_record(directory / filename) for filename in filenames],
        {
            "aliases": aliases,
            "weight_map": dict(sorted(split.tensor_to_filename.items())),
        },
    )


def save_gpu_pretrained(
    model: nn.Module,
    directory: str | os.PathLike[str],
    *,
    export: NeuralZipModelExport | None = None,
    auto_class: str | None = None,
    tokenizer: object | None = None,
    max_shard_size: int | str = "5GB",
) -> Path:
    """Save a complete, directly loadable NeuralZip inference checkpoint.

    Compressed weights come from ``model.neuralzip_runtime`` and are never
    materialized merely to save the checkpoint.
    """

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module.")
    runtime = getattr(model, "neuralzip_runtime", None)
    if export is None:
        if not isinstance(runtime, NeuralZipRuntime):
            raise TypeError("model must have an attached NeuralZip runtime.")
        exported = runtime.export
        resident = runtime.resident_state_dict()
    elif runtime is not None:
        raise ValueError("For an attached model, omit export= and save its runtime.")
    elif isinstance(export, NeuralZipModelExport):
        exported = export
        resident = _resident_state(model, exported)
    else:
        raise TypeError("export must be a NeuralZipModelExport.")
    if not exported.groups:
        raise ValueError("The GPU export contains no compressed weights.")

    library = _model_library(model)
    selected_auto_class = _infer_auto_class(model) if auto_class is None else auto_class
    selected_auto_class = _validate_auto_class(selected_auto_class)
    if library != "transformers" and selected_auto_class is not None:
        raise ValueError("auto_class is only valid for Transformers models.")

    target = _checkpoint_directory(directory, create=True)
    resident_files, resident_structure = _save_resident_state(
        resident,
        target,
        max_shard_size=max_shard_size,
    )
    export_path, export_structure = _save_gpu_export(exported, target)
    model_files = _save_model_metadata(model, target, library=library)

    if tokenizer is not None:
        save_tokenizer = getattr(tokenizer, "save_pretrained", None)
        if not callable(save_tokenizer):
            raise TypeError("tokenizer must provide save_pretrained().")
        save_tokenizer(target)

    manifest = {
        "auto_class": selected_auto_class,
        "export": export_structure,
        "files": {
            "config": model_files["config"],
            "generation_config": model_files["generation_config"],
            "gpu_export": _file_record(export_path),
            "resident_shards": resident_files,
        },
        "format": FORMAT_NAME,
        "library": library,
        "model_class": type(model).__name__,
        "resident": resident_structure,
        "version": FORMAT_VERSION,
    }
    _atomic_text(target / MANIFEST_FILENAME, _canonical_json(manifest) + "\n")
    return target


def _file_from_record(
    directory: Path,
    value: object,
    *,
    expected_name: str | None,
    context: str,
    required: bool,
) -> Path | None:
    if value is None:
        if required:
            raise GPUPretrainedError(f"{context} is required.")
        return None
    record = _require_fields(value, {"name", "nbytes", "sha256"}, context=context)
    name = _require_string(record["name"], context=f"{context}.name")
    if Path(name).name != name or name in {".", ".."}:
        raise GPUPretrainedError(f"{context}.name must be a plain filename.")
    if expected_name is not None and name != expected_name:
        raise GPUPretrainedError(f"{context}.name must be {expected_name!r}.")
    digest = _require_string(record["sha256"], context=f"{context}.sha256")
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise GPUPretrainedError(f"{context}.sha256 is invalid.")
    path = directory / name
    if not path.is_file():
        raise FileNotFoundError(path)
    nbytes = record["nbytes"]
    if (
        isinstance(nbytes, bool)
        or not isinstance(nbytes, int)
        or nbytes < 0
        or path.stat().st_size != nbytes
    ):
        raise GPUPretrainedError(f"{name} has the wrong file size.")
    if _sha256_file(path) != digest:
        raise GPUPretrainedError(f"{name} failed its SHA-256 check.")
    return path


def _load_checkpoint_metadata(
    directory: Path,
) -> tuple[
    dict[str, object],
    Path,
    Path | None,
    tuple[Path, ...],
    dict[str, str],
    dict[str, str],
]:
    manifest = _parse_manifest(directory / MANIFEST_FILENAME)
    root = _require_fields(
        manifest,
        {
            "auto_class",
            "export",
            "files",
            "format",
            "library",
            "model_class",
            "resident",
            "version",
        },
        context="manifest",
    )
    if (
        root["format"] != FORMAT_NAME
        or isinstance(root["version"], bool)
        or root["version"] != FORMAT_VERSION
    ):
        raise GPUPretrainedError("Unsupported NeuralZip pretrained checkpoint format.")
    _require_string(root["model_class"], context="manifest.model_class")
    if root["library"] not in {"transformers", "diffusers", "custom"}:
        raise GPUPretrainedError("Unsupported checkpoint model library.")
    try:
        _validate_auto_class(root["auto_class"])
    except ValueError as error:
        raise GPUPretrainedError("Unsupported checkpoint auto_class.") from error
    if root["library"] != "transformers" and root["auto_class"] is not None:
        raise GPUPretrainedError(
            "Only Transformers checkpoints may declare an auto_class."
        )
    files = _require_fields(
        root["files"],
        {
            "config",
            "generation_config",
            "gpu_export",
            "resident_shards",
        },
        context="manifest.files",
    )
    export_path = _file_from_record(
        directory,
        files["gpu_export"],
        expected_name=GPU_EXPORT_FILENAME,
        context="manifest.files.gpu_export",
        required=True,
    )
    config_path = _file_from_record(
        directory,
        files["config"],
        expected_name="config.json",
        context="manifest.files.config",
        required=False,
    )
    _file_from_record(
        directory,
        files["generation_config"],
        expected_name="generation_config.json",
        context="manifest.files.generation_config",
        required=False,
    )
    shard_records = files["resident_shards"]
    if not isinstance(shard_records, list):
        raise GPUPretrainedError("manifest.files.resident_shards must be an array.")
    shard_paths = tuple(
        _file_from_record(
            directory,
            record,
            expected_name=None,
            context=f"manifest.files.resident_shards[{index}]",
            required=True,
        )
        for index, record in enumerate(shard_records)
    )

    resident = _require_fields(
        root["resident"], {"aliases", "weight_map"}, context="manifest.resident"
    )
    if not isinstance(resident["weight_map"], dict) or not isinstance(
        resident["aliases"], dict
    ):
        raise GPUPretrainedError("Resident mappings must be JSON objects.")
    weight_map = {
        _require_name(name, context="resident tensor name"): _require_string(
            filename, context=f"resident shard for {name!r}"
        )
        for name, filename in resident["weight_map"].items()
    }
    aliases = {
        _require_name(alias, context="resident alias name"): _require_name(
            canonical, context=f"resident alias target for {alias!r}"
        )
        for alias, canonical in resident["aliases"].items()
    }
    physical_names = set(weight_map)
    if any(
        alias in physical_names or canonical not in physical_names or alias == canonical
        for alias, canonical in aliases.items()
    ):
        raise GPUPretrainedError("Resident aliases must target physical tensors.")
    paths = tuple(path for path in shard_paths if path is not None)
    if len(paths) != len(set(paths)) or set(weight_map.values()) != {
        path.name for path in paths
    }:
        raise GPUPretrainedError("Resident shards and weight_map disagree.")
    assert export_path is not None
    return root, export_path, config_path, paths, aliases, weight_map


def _load_gpu_export(path: Path, value: object) -> NeuralZipModelExport:
    structure = _require_fields(
        value,
        {"codebooks", "groups", "resident_raw_bytes", "resident_weight_names"},
        context="manifest.export",
    )
    try:
        with safe_open(str(path), framework="pt", device="cpu") as archive:
            if archive.metadata() != {
                "format": _GPU_EXPORT_FORMAT,
                "version": str(FORMAT_VERSION),
            }:
                raise GPUPretrainedError("GPU export metadata is invalid.")
            tensors = {
                name: archive.get_tensor(name)
                for name in archive.keys()  # noqa: SIM118 - safe_open is not iterable.
            }
    except GPUPretrainedError:
        raise
    except Exception as error:
        raise GPUPretrainedError("Unable to read the GPU export.") from error

    codebooks = {}
    previous_id = -1
    for index, raw_record in enumerate(
        _require_list(structure["codebooks"], context="export.codebooks")
    ):
        record = _require_fields(
            raw_record, {"id", "lengths", "values"}, context=f"codebook {index}"
        )
        codebook_id = _require_int(record["id"], context=f"codebook {index} id")
        if codebook_id <= previous_id:
            raise GPUPretrainedError("Codebook IDs must be increasing and unique.")
        previous_id = codebook_id
        try:
            codebooks[codebook_id] = GPUCodebook.from_codes(
                record["lengths"],
                record["values"],
            )
        except (TypeError, ValueError) as error:
            raise GPUPretrainedError(f"Codebook {codebook_id} is invalid.") from error

    groups = []
    block_order = []
    workspace_offsets = defaultdict(int)
    expected_tensors = set()
    for index, raw_record in enumerate(
        _require_list(structure["groups"], context="export.groups")
    ):
        context = f"group {index}"
        record = _require_fields(
            raw_record,
            {"bindings", "bit_length", "block", "codebook"},
            context=context,
        )
        block_name = _require_name(record["block"], context=f"{context} block")
        if block_name not in block_order:
            block_order.append(block_name)
        codebook_id = _require_int(record["codebook"], context=f"{context} codebook")
        if codebook_id not in codebooks:
            raise GPUPretrainedError(f"{context} references an unknown codebook.")

        group_offset = 0
        workspace_offset = workspace_offsets[block_name]
        bindings = []
        for binding_index, raw_binding in enumerate(
            _require_list(record["bindings"], context=f"{context} bindings")
        ):
            binding_context = f"{context} binding {binding_index}"
            binding = _require_fields(
                raw_binding,
                {"kind", "module", "requires_grad", "shape"},
                context=binding_context,
            )
            module_name = _require_name(
                binding["module"], context=f"{binding_context} module"
            )
            if module_name != block_name and not module_name.startswith(
                f"{block_name}."
            ):
                raise GPUPretrainedError(
                    f"{binding_context} is outside its transformer block."
                )
            try:
                shape = tuple(binding["shape"])
                numel = shape[0] * shape[1]
                bindings.append(
                    LinearWeightBinding(
                        block_name=block_name,
                        module_name=module_name,
                        tensor_name=f"{module_name}.weight",
                        shape=shape,
                        numel=numel,
                        workspace_offset=workspace_offset + group_offset,
                        group_offset=group_offset,
                        requires_grad=binding["requires_grad"],
                        module_kind=binding["kind"],
                    )
                )
            except (IndexError, TypeError, ValueError) as error:
                raise GPUPretrainedError(f"{binding_context} is invalid.") from error
            group_offset += numel

        prefix = f"groups.{index:06d}"
        names = {
            kind: f"{prefix}.{kind}"
            for kind in (
                "bitstream",
                "output_positions",
                "packed_gaps",
                "sign_mantissa",
            )
        }
        expected_tensors.update(names.values())
        try:
            selected = {kind: tensors[name] for kind, name in names.items()}
        except KeyError as error:
            raise GPUPretrainedError(f"{context} has a missing buffer.") from error
        if any(
            selected[name].dtype != torch.uint8 or selected[name].ndim != 1
            for name in ("bitstream", "packed_gaps", "sign_mantissa")
        ):
            raise GPUPretrainedError(f"{context} has an invalid byte buffer.")
        positions = selected["output_positions"]
        if positions.dtype != torch.int32 or positions.ndim != 1:
            raise GPUPretrainedError(f"{context} has invalid output positions.")
        try:
            encoded = GPUEncodedBF16(
                sign_mantissa=_tensor_bytes(selected["sign_mantissa"]),
                exponents=GPUEncodedExponents(
                    bitstream=_tensor_bytes(selected["bitstream"]),
                    bit_length=_require_int(
                        record["bit_length"], context=f"{context} bit_length"
                    ),
                    symbol_count=group_offset,
                    packed_gaps=_tensor_bytes(selected["packed_gaps"]),
                    output_positions=tuple(int(item) for item in positions.tolist()),
                ),
            )
            groups.append(
                BF16WeightGroup(
                    block_name=block_name,
                    codebook_id=codebook_id,
                    codebook=codebooks[codebook_id],
                    encoded=encoded,
                    bindings=tuple(bindings),
                    workspace_offset=workspace_offset,
                )
            )
        except (TypeError, ValueError) as error:
            raise GPUPretrainedError(f"{context} is inconsistent.") from error
        workspace_offsets[block_name] += group_offset

    if expected_tensors != set(tensors):
        raise GPUPretrainedError("GPU export buffers do not match its manifest.")
    if set(codebooks) != {group.codebook_id for group in groups}:
        raise GPUPretrainedError("Every stored codebook must be used.")
    resident_names = tuple(
        _require_name(name, context="resident weight name")
        for name in _require_list(
            structure["resident_weight_names"], context="resident_weight_names"
        )
    )
    if any(not name.endswith(".weight") for name in resident_names):
        raise GPUPretrainedError("Resident weight names must end in '.weight'.")
    try:
        return NeuralZipModelExport(
            groups=tuple(groups),
            block_order=tuple(block_order),
            resident_weight_names=resident_names,
            resident_raw_bytes=_require_int(
                structure["resident_raw_bytes"], context="resident_raw_bytes"
            ),
        )
    except (TypeError, ValueError) as error:
        raise GPUPretrainedError("The GPU export structure is invalid.") from error


def _empty_weights():
    try:
        from accelerate import init_empty_weights
    except ImportError:
        return nullcontext()
    return init_empty_weights(include_buffers=False)


def _auto_model(
    directory: Path,
    auto_class_name: str | None,
    *,
    trust_remote_code: bool,
) -> nn.Module:
    if auto_class_name is None:
        raise GPUPretrainedError(
            "This custom checkpoint requires model= or model_factory=."
        )
    try:
        import transformers
        from transformers import AutoConfig
    except ImportError as error:  # pragma: no cover - core dependency declaration.
        raise RuntimeError("Automatic model loading requires transformers.") from error
    auto_class = getattr(transformers, auto_class_name, None)
    if auto_class is None:
        raise RuntimeError(
            f"The installed transformers version does not provide {auto_class_name}."
        )
    config = AutoConfig.from_pretrained(
        directory,
        local_files_only=True,
        trust_remote_code=trust_remote_code,
    )
    try:
        with _empty_weights():
            return auto_class.from_config(config, trust_remote_code=trust_remote_code)
    except Exception as error:
        raise GPUPretrainedError(
            f"Unable to instantiate {auto_class_name} from the saved configuration."
        ) from error


def _diffusers_model(directory: Path, model_class_name: str) -> nn.Module:
    try:
        import diffusers
    except ImportError as error:
        raise RuntimeError(
            "Loading a diffusion component requires the 'diffusion' extra."
        ) from error
    model_class = getattr(diffusers, model_class_name, None)
    if (
        not isinstance(model_class, type)
        or not issubclass(model_class, nn.Module)
        or not model_class.__module__.startswith("diffusers.")
        or not callable(getattr(model_class, "load_config", None))
        or not callable(getattr(model_class, "from_config", None))
    ):
        raise GPUPretrainedError(
            f"Unsupported Diffusers component class {model_class_name!r}."
        )
    try:
        config = model_class.load_config(directory)
        with _empty_weights():
            return model_class.from_config(config)
    except Exception as error:
        raise GPUPretrainedError(
            f"Unable to instantiate Diffusers component {model_class_name!r}."
        ) from error


def _validated_compressed_weights(
    model: nn.Module,
    exported: NeuralZipModelExport,
) -> tuple[tuple[nn.Module, nn.Parameter], ...]:
    validated = []
    for binding in exported.bindings:
        try:
            module = model.get_submodule(binding.module_name)
        except AttributeError as error:
            raise GPUPretrainedError(
                f"The model has no module {binding.module_name!r}."
            ) from error
        expected = nn.Embedding if binding.module_kind == "embedding" else nn.Linear
        if not isinstance(module, expected):
            raise GPUPretrainedError(
                f"{binding.module_name!r} must be an {expected.__name__}."
            )
        parameter = module._parameters.get("weight")
        if (
            not isinstance(parameter, nn.Parameter)
            or tuple(parameter.shape) != binding.shape
        ):
            raise GPUPretrainedError(
                f"{binding.tensor_name!r} does not match the compressed checkpoint."
            )
        validated.append((module, parameter))
    return tuple(validated)


def _tensor_slot(model: nn.Module, tensor_name: str) -> tuple[nn.Module, str]:
    parent, _, leaf = tensor_name.rpartition(".")
    try:
        return model.get_submodule(parent), leaf
    except AttributeError as error:
        raise GPUPretrainedError(
            f"The model has no tensor path {tensor_name!r}."
        ) from error


def _link_resident_alias(
    model: nn.Module,
    *,
    canonical_name: str,
    alias_name: str,
) -> None:
    """Restore one exact storage alias without another device allocation."""

    canonical_module, canonical_leaf = _tensor_slot(model, canonical_name)
    alias_module, alias_leaf = _tensor_slot(model, alias_name)
    canonical = canonical_module._parameters.get(canonical_leaf)
    if canonical is None:
        canonical = canonical_module._buffers.get(canonical_leaf)
    if not torch.is_tensor(canonical) or canonical.device.type == "meta":
        raise GPUPretrainedError(
            f"Resident alias source {canonical_name!r} was not loaded."
        )

    if alias_leaf in alias_module._parameters:
        previous = alias_module._parameters[alias_leaf]
        if not isinstance(previous, nn.Parameter):
            raise GPUPretrainedError(
                f"Resident alias target {alias_name!r} is not a parameter."
            )
        if isinstance(canonical, nn.Parameter):
            if canonical.requires_grad != previous.requires_grad:
                raise GPUPretrainedError(
                    f"Resident alias {alias_name!r} changed requires_grad."
                )
            alias_module._parameters[alias_leaf] = canonical
        else:
            alias_module._parameters[alias_leaf] = nn.Parameter(
                canonical,
                requires_grad=previous.requires_grad,
            )
    elif alias_leaf in alias_module._buffers:
        alias_module._buffers[alias_leaf] = (
            canonical.detach() if isinstance(canonical, nn.Parameter) else canonical
        )
    else:
        raise GPUPretrainedError(
            f"Resident alias target {alias_name!r} does not exist."
        )


def _load_model_state(
    model: nn.Module,
    resident_shards: tuple[Path, ...],
    resident_aliases: Mapping[str, str],
    resident_weight_map: Mapping[str, str],
    exported: NeuralZipModelExport,
    placement: _Placement,
) -> nn.Module:
    model_state = model.state_dict()
    compressed_names = {binding.tensor_name for binding in exported.bindings}
    expected_names = set(model_state) - compressed_names
    resident_names = set(resident_weight_map) | set(resident_aliases)
    if expected_names != resident_names:
        raise GPUPretrainedError(
            "The model architecture does not match the resident checkpoint state; "
            f"missing={sorted(expected_names - resident_names)}, "
            f"unexpected={sorted(resident_names - expected_names)}."
        )
    for module, _parameter in _validated_compressed_weights(model, exported):
        module._parameters.pop("weight")
        module.__dict__.pop("weight", None)

    try:
        from accelerate.utils import set_module_tensor_to_device
    except ImportError as error:
        raise RuntimeError(
            "Direct sharded checkpoint loading requires the 'accelerate' package."
        ) from error

    aliases_by_canonical = {}
    for alias, canonical in resident_aliases.items():
        aliases_by_canonical.setdefault(canonical, []).append(alias)
    loaded_names = set()
    try:
        for path in resident_shards:
            expected = {
                name
                for name, filename in resident_weight_map.items()
                if filename == path.name
            }
            with safe_open(str(path), framework="pt", device="cpu") as archive:
                if (
                    archive.metadata()
                    != {
                        "format": _RESIDENT_FORMAT,
                        "version": str(FORMAT_VERSION),
                    }
                    or set(archive.keys()) != expected
                ):
                    raise GPUPretrainedError(
                        f"Resident shard {path.name!r} does not match its manifest."
                    )
                shard = {name: archive.get_tensor(name) for name in expected}
            for canonical, tensor in shard.items():
                logical_names = (canonical, *aliases_by_canonical.get(canonical, ()))
                if any(
                    tuple(model_state[name].shape) != tuple(tensor.shape)
                    for name in logical_names
                ):
                    raise GPUPretrainedError(
                        f"Resident tensor {canonical!r} does not match the model."
                    )
                destinations = {
                    _storage_device_for_tensor(placement, name)
                    for name in logical_names
                }
                if len(destinations) != 1:
                    raise GPUPretrainedError(
                        f"Tied tensor group {logical_names!r} spans multiple devices."
                    )
                destination = destinations.pop()
                set_module_tensor_to_device(
                    model,
                    canonical,
                    destination,
                    value=tensor,
                    dtype=tensor.dtype,
                    clear_cache=False,
                )
                loaded_names.add(canonical)
                for alias in logical_names[1:]:
                    _link_resident_alias(
                        model,
                        canonical_name=canonical,
                        alias_name=alias,
                    )
                    loaded_names.add(alias)
    except Exception as error:
        if isinstance(error, GPUPretrainedError):
            raise
        raise GPUPretrainedError("Unable to load the resident model state.") from error
    if loaded_names != resident_names:
        raise GPUPretrainedError("The resident model state is incomplete.")
    remaining_meta = [
        name
        for name, value in tuple(model.named_parameters())
        + tuple(model.named_buffers())
        if value.device.type == "meta"
    ]
    if remaining_meta:
        raise GPUPretrainedError(
            "The model factory left non-persistent state on the meta device: "
            f"{remaining_meta}."
        )
    tie_weights = getattr(model, "tie_weights", None)
    if callable(tie_weights):
        tie_weights()
    return model.eval()


def _load_generation_config(
    model: nn.Module,
    directory: Path,
    *,
    enabled: bool,
    trust_remote_code: bool,
) -> None:
    if not enabled:
        return
    try:
        from transformers import GenerationConfig
    except ImportError as error:  # pragma: no cover - core dependency declaration.
        raise RuntimeError(
            "Loading generation configuration requires transformers."
        ) from error
    try:
        model.generation_config = GenerationConfig.from_pretrained(
            directory,
            local_files_only=True,
            trust_remote_code=trust_remote_code,
        )
    except Exception as error:
        raise GPUPretrainedError("Unable to load generation_config.json.") from error


def _require_accelerate():
    try:
        from accelerate import dispatch_model, infer_auto_device_map
        from accelerate.utils import get_balanced_memory, get_max_memory
    except ImportError as error:
        raise RuntimeError(
            "Automatic or mapped GPU placement requires the 'accelerate' package."
        ) from error
    return dispatch_model, infer_auto_device_map, get_balanced_memory, get_max_memory


def _device_from_map(value: object) -> torch.device:
    if isinstance(value, int) and not isinstance(value, bool):
        return torch.device("cuda", value)
    if value == "disk":
        raise GPUPretrainedError(
            "Compressed NeuralZip blocks cannot be disk-offloaded."
        )
    return torch.device(value)


def _mapped_device(name: str, device_map: Mapping[str, object]) -> torch.device:
    candidates = [
        (prefix, value)
        for prefix, value in device_map.items()
        if prefix == "" or name == prefix or name.startswith(f"{prefix}.")
    ]
    if not candidates:
        raise GPUPretrainedError(f"device_map does not cover module {name!r}.")
    _prefix, value = max(candidates, key=lambda item: len(item[0]))
    return _device_from_map(value)


@dataclass(frozen=True)
class _Placement:
    direct_device: torch.device | None
    device_map: dict[str, object] | None
    main_device: torch.device
    block_devices: dict[str, torch.device]
    cpu_stored_blocks: tuple[str, ...]


def _storage_device_for_tensor(
    placement: _Placement,
    tensor_name: str,
) -> torch.device:
    if placement.device_map is None:
        assert placement.direct_device is not None
        return placement.direct_device
    return _mapped_device(tensor_name, placement.device_map)


def _block_payload_bytes(exported: NeuralZipModelExport) -> dict[str, int]:
    payloads = {name: 0 for name in exported.block_order}
    for group in exported.groups:
        payloads[group.block_name] += group.payload_bytes
    return payloads


@contextmanager
def _deployment_sizing_skeleton(
    model: nn.Module,
    exported: NeuralZipModelExport,
    *,
    cpu_offload: bool,
    cpu_offload_blocks: int | None,
):
    """Temporarily expose deployment-sized payloads to Accelerate's mapper."""

    validated = _validated_compressed_weights(model, exported)
    offloaded = _offloaded_blocks(
        exported,
        cpu_offload=cpu_offload,
        cpu_offload_blocks=cpu_offload_blocks,
    )
    payloads = _block_payload_bytes(exported)
    placeholder_bytes = {
        name: (1 if name in offloaded else max(payloads[name], 1))
        for name in exported.block_order
    }
    blocks = {name: model.get_submodule(name) for name in exported.block_order}
    for name, block in blocks.items():
        if hasattr(block, _PLACEMENT_BUFFER_NAME):
            raise GPUPretrainedError(
                f"Block {name!r} already defines {_PLACEMENT_BUFFER_NAME!r}."
            )

    removed = []
    placeholders = []
    try:
        for module, parameter in validated:
            module._parameters.pop("weight")
            module.__dict__.pop("weight", None)
            removed.append((module, parameter))
        for name, block in blocks.items():
            block.register_buffer(
                _PLACEMENT_BUFFER_NAME,
                torch.empty(
                    placeholder_bytes[name],
                    dtype=torch.uint8,
                    device="meta",
                ),
                persistent=False,
            )
            placeholders.append(block)
        yield placeholder_bytes
    finally:
        for block in reversed(placeholders):
            block._buffers.pop(_PLACEMENT_BUFFER_NAME, None)
            block._non_persistent_buffers_set.discard(_PLACEMENT_BUFFER_NAME)
        for module, parameter in removed:
            if "weight" in module._parameters:
                raise RuntimeError(
                    "Compressed placement sizing unexpectedly replaced a weight."
                )
            module._parameters["weight"] = parameter


def _gpu_runtime_reservation_bytes(exported: NeuralZipModelExport) -> int:
    largest_payload = max(_block_payload_bytes(exported).values(), default=0)
    return (
        exported.workspace_bytes
        + exported.metadata_bytes
        + exported.codebook_bytes
        + exported.launch_descriptor_bytes
        + largest_payload
    )


def _cpu_runtime_reservation_bytes(exported: NeuralZipModelExport) -> int:
    # The immutable export remains alive after attachment. Reserve one more
    # payload copy for the worst case where every compressed block is offloaded.
    return exported.compressed_bytes + exported.payload_bytes


def _resolve_placement(
    model: nn.Module,
    exported: NeuralZipModelExport,
    *,
    device: object | None,
    device_map: str | Mapping[str, object] | None,
    max_memory: Mapping[object, object] | None,
    execution_device: object | None,
    cpu_offload: bool,
    cpu_offload_blocks: int | None,
) -> _Placement:
    _offloaded_blocks(
        exported,
        cpu_offload=cpu_offload,
        cpu_offload_blocks=cpu_offload_blocks,
    )
    if device is not None and device_map is not None:
        raise ValueError("Pass either device or device_map, not both.")
    if max_memory is not None and device_map != "auto":
        raise ValueError("max_memory requires device_map='auto'.")
    if execution_device is not None and device_map is None:
        raise ValueError("execution_device only applies to device_map placement.")

    if device_map is None:
        target = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        block_devices = {name: target for name in exported.block_order}
        placement = _Placement(
            direct_device=target,
            device_map=None,
            main_device=target,
            block_devices=block_devices,
            cpu_stored_blocks=(),
        )
    else:
        _dispatch_model, infer_auto_device_map, get_balanced_memory, get_max_memory = (
            _require_accelerate()
        )
        if isinstance(device_map, str):
            if device_map != "auto":
                raise ValueError("The only supported string device_map is 'auto'.")
            no_split = sorted(
                {
                    type(model.get_submodule(name)).__name__
                    for name in exported.block_order
                }
            )
            with _deployment_sizing_skeleton(
                model,
                exported,
                cpu_offload=cpu_offload,
                cpu_offload_blocks=cpu_offload_blocks,
            ):
                available_memory = get_max_memory(
                    dict(max_memory) if max_memory is not None else None
                )
                gpu_reservation = _gpu_runtime_reservation_bytes(exported)
                cpu_reservation = _cpu_runtime_reservation_bytes(exported)
                for key, available in tuple(available_memory.items()):
                    mapped = _device_from_map(key)
                    reservation = (
                        gpu_reservation
                        if mapped.type == "cuda"
                        else cpu_reservation
                        if mapped.type == "cpu"
                        else 0
                    )
                    remaining = int(available) - reservation
                    if remaining <= 0:
                        raise GPUPretrainedError(
                            f"{mapped} lacks room for NeuralZip runtime state."
                        )
                    available_memory[key] = remaining
                balanced = get_balanced_memory(
                    model,
                    max_memory=available_memory,
                    no_split_module_classes=no_split,
                    dtype=torch.bfloat16,
                )
                resolved_map = infer_auto_device_map(
                    model,
                    max_memory=balanced,
                    no_split_module_classes=no_split,
                    dtype=torch.bfloat16,
                    clean_result=True,
                )
        elif isinstance(device_map, Mapping):
            resolved_map = dict(device_map)
            if not resolved_map:
                raise ValueError("device_map must not be empty.")
        else:
            raise TypeError("device_map must be 'auto' or a mapping.")
        storage_devices = {
            name: _mapped_device(name, resolved_map) for name in exported.block_order
        }
        mapped_cuda = tuple(
            dict.fromkeys(
                value
                for value in (
                    _device_from_map(raw_value) for raw_value in resolved_map.values()
                )
                if value.type == "cuda"
            )
        )
        if execution_device is not None:
            main_device = torch.device(execution_device)
        elif mapped_cuda:
            main_device = mapped_cuda[0]
        elif torch.cuda.is_available():
            main_device = torch.device("cuda", torch.cuda.current_device())
        else:
            main_device = torch.device("cpu")
        block_devices = {
            name: (main_device if value.type == "cpu" else value)
            for name, value in storage_devices.items()
        }
        placement = _Placement(
            direct_device=None,
            device_map=resolved_map,
            main_device=main_device,
            block_devices=block_devices,
            cpu_stored_blocks=tuple(
                name for name, value in storage_devices.items() if value.type == "cpu"
            ),
        )

    non_cuda = {
        name: str(value)
        for name, value in placement.block_devices.items()
        if value.type != "cuda"
    }
    if non_cuda:
        raise GPUPretrainedError(
            "The native decoder requires every compressed block on CUDA; "
            f"non-CUDA blocks={non_cuda}. Increase GPU max_memory or provide "
            "an explicit CUDA device_map."
        )
    return placement


def _apply_placement(
    model: nn.Module,
    placement: _Placement,
) -> nn.Module:
    if placement.device_map is None:
        return model.to(placement.direct_device)
    all_cpu = all(
        _device_from_map(value).type == "cpu" for value in placement.device_map.values()
    )
    if all_cpu:
        if placement.main_device.type == "cpu":
            return model.to("cpu")
        try:
            from accelerate import cpu_offload
        except ImportError as error:  # pragma: no cover - GPU extra declaration.
            raise RuntimeError("Full CPU offload requires accelerate.") from error
        return cpu_offload(
            model,
            execution_device=placement.main_device,
            offload_buffers=False,
            state_dict=model.state_dict(),
        )
    dispatch_model, _infer_auto_device_map, _get_balanced_memory, _get_max_memory = (
        _require_accelerate()
    )
    return dispatch_model(
        model,
        device_map=placement.device_map,
        main_device=placement.main_device,
        offload_buffers=False,
        force_hooks=(
            bool(placement.cpu_stored_blocks)
            or len(set(placement.block_devices.values())) > 1
        ),
    )


def attach_gpu_model(
    model: nn.Module,
    setup,
    *,
    threads: int = 0,
    device: object | None = None,
    device_map: str | Mapping[str, object] | None = None,
    max_memory: Mapping[object, object] | None = None,
    execution_device: object | None = None,
    include_embeddings: bool = False,
    cpu_offload: bool = False,
    cpu_offload_blocks: int | None = None,
    pin_memory: bool = True,
) -> nn.Module:
    """Export, place, and attach a model using an explicit NeuralZip setup."""

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module.")
    if setup is None:
        raise TypeError("setup is required.")
    if hasattr(model, "neuralzip_runtime"):
        raise RuntimeError("This model already has a NeuralZip runtime.")
    exported = export_model_for_gpu(
        model,
        setup,
        threads=threads,
        include_embeddings=include_embeddings,
    )
    if not exported.groups:
        raise ValueError("The setup produced no GPU-compressible weights.")
    placement = _resolve_placement(
        model,
        exported,
        device=device,
        device_map=device_map,
        max_memory=max_memory,
        execution_device=execution_device,
        cpu_offload=cpu_offload,
        cpu_offload_blocks=cpu_offload_blocks,
    )
    compressed_parameters = _validated_compressed_weights(model, exported)
    runtime = NeuralZipRuntime._prepare_runtime(
        model,
        exported,
        block_devices=placement.block_devices,
        precompressed=True,
        cpu_offload=cpu_offload,
        cpu_offload_blocks=cpu_offload_blocks,
        cpu_offload_block_names=placement.cpu_stored_blocks,
        pin_memory=pin_memory,
    )
    removed = False
    try:
        for module, _parameter in compressed_parameters:
            module._parameters.pop("weight")
            module.__dict__.pop("weight", None)
        removed = True
        model = _apply_placement(model, placement)
        runtime.model = model
        runtime._install()
    except Exception:
        if removed:
            for module, parameter in compressed_parameters:
                module.__dict__.pop("weight", None)
                if "weight" not in module._parameters:
                    module.register_parameter("weight", parameter)
        runtime._release_device_state()
        raise
    return model


def load_gpu_pretrained(
    pretrained_model_name_or_path: str | os.PathLike[str],
    *,
    model: nn.Module | None = None,
    model_factory: Callable[[Path], nn.Module] | None = None,
    device: object | None = None,
    device_map: str | Mapping[str, object] | None = None,
    max_memory: Mapping[object, object] | None = None,
    execution_device: object | None = None,
    cpu_offload: bool = False,
    cpu_offload_blocks: int | None = None,
    pin_memory: bool = True,
    trust_remote_code: bool = False,
    revision: str = "main",
    token: str | bool | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    local_files_only: bool = False,
) -> nn.Module:
    """Load and attach a complete NeuralZip inference checkpoint.

    Transformers architectures are reconstructed from their saved local
    configuration. Other PyTorch or diffusion components can pass ``model``
    or a trusted ``model_factory``. The safe format itself never requires code
    execution; ``trust_remote_code=True`` is an explicit trusted-code opt-in.
    """

    if not isinstance(revision, str) or not revision:
        raise ValueError("revision must be a non-empty string.")
    if not isinstance(local_files_only, bool):
        raise TypeError("local_files_only must be a boolean.")
    target = _resolve_pretrained_source(
        pretrained_model_name_or_path,
        revision=revision,
        token=token,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    (
        manifest,
        export_path,
        config_path,
        resident_shards,
        resident_aliases,
        resident_weight_map,
    ) = _load_checkpoint_metadata(target)
    exported = _load_gpu_export(export_path, manifest["export"])
    if model is not None and model_factory is not None:
        raise ValueError("Pass either model or model_factory, not both.")
    if model is None:
        if model_factory is not None:
            model = model_factory(target)
            if not isinstance(model, nn.Module):
                raise TypeError("model_factory must return torch.nn.Module.")
        else:
            if config_path is None:
                raise GPUPretrainedError(
                    "This checkpoint has no loadable model config; pass model=."
                )
            if manifest["library"] == "transformers":
                model = _auto_model(
                    target,
                    manifest["auto_class"],
                    trust_remote_code=trust_remote_code,
                )
            elif manifest["library"] == "diffusers":
                model = _diffusers_model(target, manifest["model_class"])
            else:
                raise GPUPretrainedError(
                    "This custom checkpoint requires model= or model_factory=."
                )
            if type(model).__name__ != manifest["model_class"]:
                raise GPUPretrainedError(
                    "The saved model loader no longer resolves to the checkpoint's "
                    f"model class {manifest['model_class']!r}."
                )
    elif not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module.")

    placement = _resolve_placement(
        model,
        exported,
        device=device,
        device_map=device_map,
        max_memory=max_memory,
        execution_device=execution_device,
        cpu_offload=cpu_offload,
        cpu_offload_blocks=cpu_offload_blocks,
    )
    model = _load_model_state(
        model,
        resident_shards,
        resident_aliases,
        resident_weight_map,
        exported,
        placement,
    )
    generation_record = manifest["files"]["generation_config"]
    _load_generation_config(
        model,
        target,
        enabled=generation_record is not None,
        trust_remote_code=trust_remote_code,
    )
    model = _apply_placement(model, placement)
    NeuralZipRuntime.attach(
        model,
        exported,
        block_devices=placement.block_devices,
        precompressed=True,
        cpu_offload=cpu_offload,
        cpu_offload_blocks=cpu_offload_blocks,
        cpu_offload_block_names=placement.cpu_stored_blocks,
        pin_memory=pin_memory,
    )
    return model


__all__ = [
    "FORMAT_NAME",
    "FORMAT_VERSION",
    "MANIFEST_FILENAME",
    "GPUPretrainedError",
    "attach_gpu_model",
    "load_gpu_pretrained",
    "save_gpu_pretrained",
]
