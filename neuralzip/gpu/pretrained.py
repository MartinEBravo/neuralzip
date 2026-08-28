"""Persistent, directly loadable NeuralZip inference checkpoints.

The format deliberately separates ordinary model state from NeuralZip's GPU
representation.  Resident parameters and buffers live in Safetensors, while
compressed BF16 weights live in :mod:`neuralzip.gpu.checkpoint`.  A strict
JSON manifest binds the two files to the model configuration without ever
deserializing executable Python objects.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ..compressor import Compressor
from .adapters import GPUModelAdapter
from .checkpoint import DEFAULT_FILENAME as GPU_EXPORT_FILENAME
from .checkpoint import load_gpu_export, save_gpu_export
from .inference import (
    NeuralZipModelExport,
    NeuralZipRuntime,
    _offloaded_blocks,
    export_model_for_gpu,
)
from .resident import (
    INDEX_FILENAME as RESIDENT_INDEX_FILENAME,
)
from .resident import (
    ResidentStateManifest,
    iter_resident_shards,
    load_resident_state,
    save_resident_state,
)

FORMAT_NAME = "neuralzip-gpu-pretrained"
FORMAT_VERSION = 1
MANIFEST_FILENAME = "neuralzip_config.json"
# Kept as a public alias for callers that imported the early prototype name.
RESIDENT_FILENAME = RESIDENT_INDEX_FILENAME
_AUTO_CLASS_NAMES = (
    "AutoModelForSpeechSeq2Seq",
    "AutoModelForVision2Seq",
    "AutoModelForMultimodalLM",
    "AutoModelForImageTextToText",
    "AutoModelForVisualQuestionAnswering",
    "AutoModelForAudioClassification",
    "AutoModelForAudioFrameClassification",
    "AutoModelForAudioTokenization",
    "AutoModelForAudioXVector",
    "AutoModelForCTC",
    "AutoModelForRNNT",
    "AutoModelForTDT",
    "AutoModelForTextToSpectrogram",
    "AutoModelForTextToWaveform",
    "AutoModelForImageClassification",
    "AutoModelForVideoClassification",
    "AutoModelForZeroShotImageClassification",
    "AutoModelForImageToImage",
    "AutoModelForImageMatting",
    "AutoModelForMaskedImageModeling",
    "AutoModelForDepthEstimation",
    "AutoModelForNormalEstimation",
    "AutoModelForPointmapEstimation",
    "AutoModelForPoseEstimation",
    "AutoModelForKeypointDetection",
    "AutoModelForKeypointMatching",
    "AutoModelForMaskGeneration",
    "AutoModelForObjectDetection",
    "AutoModelForZeroShotObjectDetection",
    "AutoModelForInstanceSegmentation",
    "AutoModelForImageSegmentation",
    "AutoModelForSemanticSegmentation",
    "AutoModelForUniversalSegmentation",
    "AutoModelForQuestionAnswering",
    "AutoModelForDocumentQuestionAnswering",
    "AutoModelForTableQuestionAnswering",
    "AutoModelForTableRecognition",
    "AutoModelForTokenClassification",
    "AutoModelForSequenceClassification",
    "AutoModelForMultipleChoice",
    "AutoModelForNextSentencePrediction",
    "AutoModelForPreTraining",
    "AutoModelForMaskedLM",
    "AutoModelForSeq2SeqLM",
    "AutoModelForCausalLM",
    "AutoModelForTextEncoding",
    "AutoModelForTextRecognition",
    "AutoModelForTimeSeriesPrediction",
    "AutoModel",
)
_PLACEMENT_BUFFER_NAME = "_neuralzip_placement_payload"


class GPUPretrainedError(ValueError):
    """Raised when a complete NeuralZip inference checkpoint is invalid."""


def _canonical_json(value: object) -> str:
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


def _parse_manifest(path: Path) -> dict[str, object]:
    try:
        source = path.read_text(encoding="ascii")
        value = json.loads(source)
    except FileNotFoundError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise GPUPretrainedError("Unable to read neuralzip_config.json.") from error
    if not isinstance(value, dict):
        raise GPUPretrainedError("The checkpoint manifest must be a JSON object.")
    if _canonical_json(value) + "\n" != source:
        raise GPUPretrainedError("The checkpoint manifest is not canonical JSON.")
    return value


def _require_fields(
    value: object,
    expected: set[str],
    *,
    context: str,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise GPUPretrainedError(f"{context} must be a JSON object.")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise GPUPretrainedError(
            f"Invalid {context} fields: missing={missing}, unexpected={unexpected}."
        )
    return value


def _require_string(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise GPUPretrainedError(f"{context} must be a non-empty string.")
    return value


def _require_string_list(value: object, *, context: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise GPUPretrainedError(f"{context} must be an array of non-empty strings.")
    result = tuple(value)
    if len(result) != len(set(result)):
        raise GPUPretrainedError(f"{context} must not contain duplicates.")
    return result


def _checkpoint_directory(value: str | os.PathLike[str], *, create: bool) -> Path:
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


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
    finally:
        temporary.unlink(missing_ok=True)


def _resident_state(
    model: nn.Module,
    exported: NeuralZipModelExport,
    *,
    state: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    compressed = {binding.tensor_name for binding in exported.bindings}
    state = model.state_dict() if state is None else state
    missing = compressed - set(state)
    runtime = getattr(model, "neuralzip_runtime", None)
    if missing and runtime is None:
        raise GPUPretrainedError(
            f"The model state is missing compressed tensors: {sorted(missing)}."
        )
    if runtime is not None and missing != compressed:
        raise GPUPretrainedError(
            "An attached model's resident state unexpectedly contains only part of "
            "the compressed tensors."
        )
    resident = {}
    for name, value in state.items():
        if name in compressed:
            continue
        if not torch.is_tensor(value) or value.layout != torch.strided:
            raise GPUPretrainedError(
                f"Resident state {name!r} is not a dense tensor and cannot be "
                "stored safely."
            )
        resident[name] = value.detach()
    return resident


def _validate_auto_class(value: str | None) -> str | None:
    if value is not None and value not in _AUTO_CLASS_NAMES:
        raise ValueError(
            f"auto_class must be one of {list(_AUTO_CLASS_NAMES)}, or None."
        )
    return value


def _infer_auto_class(model: nn.Module) -> str | None:
    config = getattr(model, "config", None)
    if config is None:
        return None
    try:
        import transformers
    except ImportError:  # pragma: no cover - core dependency declaration.
        return None
    auto_map = getattr(config, "auto_map", None)
    if isinstance(auto_map, Mapping):
        for class_name in _AUTO_CLASS_NAMES:
            if class_name in auto_map:
                return class_name
    for class_name in _AUTO_CLASS_NAMES:
        auto_class = getattr(transformers, class_name, None)
        mapping = getattr(auto_class, "_model_mapping", None)
        if mapping is None:
            continue
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


def save_gpu_pretrained(
    model: nn.Module,
    compressor_or_export: Compressor
    | NeuralZipModelExport
    | str
    | os.PathLike[str]
    | None = None,
    directory: str | os.PathLike[str] | None = None,
    *,
    adapter: GPUModelAdapter | None = None,
    include_embeddings: bool = False,
    auto_class: str | None = None,
    tokenizer: object | None = None,
    max_shard_size: int | str = "5GB",
) -> Path:
    """Save a complete, directly loadable NeuralZip inference checkpoint.

    ``compressor_or_export`` may be a fitted Compressor or an encoded GPU
    export. Compressed weights are excluded from the sharded resident state and
    therefore never need to be materialized before loading the checkpoint.
    """

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module.")
    if directory is None and isinstance(compressor_or_export, (str, os.PathLike)):
        directory = compressor_or_export
        compressor_or_export = None
    if directory is None:
        raise TypeError("directory is required.")
    runtime = getattr(model, "neuralzip_runtime", None)
    resident_source = None
    if compressor_or_export is None:
        if not isinstance(runtime, NeuralZipRuntime):
            raise ValueError(
                "compressor_or_export is required unless model already has a "
                "NeuralZip runtime."
            )
        exported = runtime.export
        resident_source = runtime.resident_state_dict()
    elif runtime is not None:
        raise ValueError(
            "For an attached model, omit compressor_or_export and save its current export."
        )
    elif isinstance(compressor_or_export, Compressor):
        exported = export_model_for_gpu(
            model,
            compressor_or_export,
            adapter=adapter,
            include_embeddings=include_embeddings,
        )
    elif isinstance(compressor_or_export, NeuralZipModelExport):
        if adapter is not None or include_embeddings:
            raise ValueError(
                "adapter/include_embeddings only apply when exporting a Compressor."
            )
        exported = compressor_or_export
    else:
        raise TypeError(
            "compressor_or_export must be Compressor or NeuralZipModelExport."
        )
    if not exported.groups:
        raise ValueError("The GPU export contains no compressed weights.")

    library = _model_library(model)
    selected_auto_class = _infer_auto_class(model) if auto_class is None else auto_class
    selected_auto_class = _validate_auto_class(selected_auto_class)
    if library != "transformers" and selected_auto_class is not None:
        raise ValueError("auto_class is only valid for Transformers models.")

    target = _checkpoint_directory(directory, create=True)
    resident = _resident_state(model, exported, state=resident_source)
    resident_manifest = save_resident_state(
        resident,
        target,
        max_shard_size=max_shard_size,
    )
    export_path = save_gpu_export(exported, target)
    model_files = _save_model_metadata(model, target, library=library)

    if tokenizer is not None:
        save_tokenizer = getattr(tokenizer, "save_pretrained", None)
        if not callable(save_tokenizer):
            raise TypeError("tokenizer must provide save_pretrained().")
        save_tokenizer(target)

    manifest = {
        "compressed_weight_names": sorted(
            binding.tensor_name for binding in exported.bindings
        ),
        "files": {
            "config": model_files["config"],
            "generation_config": model_files["generation_config"],
            "gpu_export": _file_record(export_path),
            "resident_index": _file_record(resident_manifest.index_path),
            "resident_shards": [
                _file_record(path) for path in resident_manifest.shard_paths
            ],
        },
        "format": FORMAT_NAME,
        "auto_class": selected_auto_class,
        "library": library,
        "model_class": type(model).__name__,
        "resident_tensor_names": list(resident_manifest.tensor_names),
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
) -> tuple[dict[str, object], ResidentStateManifest, Path, Path | None]:
    manifest = _parse_manifest(directory / MANIFEST_FILENAME)
    root = _require_fields(
        manifest,
        {
            "auto_class",
            "compressed_weight_names",
            "files",
            "format",
            "library",
            "model_class",
            "resident_tensor_names",
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
    _require_string_list(
        root["compressed_weight_names"], context="compressed_weight_names"
    )
    files = _require_fields(
        root["files"],
        {
            "config",
            "generation_config",
            "gpu_export",
            "resident_index",
            "resident_shards",
        },
        context="manifest.files",
    )
    _file_from_record(
        directory,
        files["resident_index"],
        expected_name=RESIDENT_INDEX_FILENAME,
        context="manifest.files.resident_index",
        required=True,
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
    resident_manifest = load_resident_state(directory)
    if (
        tuple(path for path in shard_paths if path is not None)
        != resident_manifest.shard_paths
    ):
        raise GPUPretrainedError(
            "The full checkpoint and resident shard index disagree."
        )
    resident_names = _require_string_list(
        root["resident_tensor_names"], context="resident_tensor_names"
    )
    if resident_names != resident_manifest.tensor_names:
        raise GPUPretrainedError("The full checkpoint and resident state disagree.")
    assert export_path is not None
    return root, resident_manifest, export_path, config_path


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
        try:
            from accelerate import init_empty_weights
        except ImportError:
            return auto_class.from_config(config, trust_remote_code=trust_remote_code)
        with init_empty_weights(include_buffers=False):
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
        try:
            from accelerate import init_empty_weights
        except ImportError:
            return model_class.from_config(config)
        with init_empty_weights(include_buffers=False):
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


def _validate_and_remove_compressed_weights(
    model: nn.Module,
    exported: NeuralZipModelExport,
) -> None:
    for module, _parameter in _validated_compressed_weights(model, exported):
        module._parameters.pop("weight")
        module.__dict__.pop("weight", None)


def _tensor_slot(model: nn.Module, tensor_name: str) -> tuple[nn.Module, str]:
    segments = tensor_name.split(".")
    module = model
    for segment in segments[:-1]:
        child = getattr(module, segment, None)
        if not isinstance(child, nn.Module):
            raise GPUPretrainedError(f"The model has no tensor path {tensor_name!r}.")
        module = child
    return module, segments[-1]


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
    resident: ResidentStateManifest,
    exported: NeuralZipModelExport,
    placement: _Placement,
) -> nn.Module:
    compressed_names = {binding.tensor_name for binding in exported.bindings}
    expected_names = set(model.state_dict()) - compressed_names
    resident_names = set(resident.tensor_names)
    if expected_names != resident_names:
        raise GPUPretrainedError(
            "The model architecture does not match the resident checkpoint state; "
            f"missing={sorted(expected_names - resident_names)}, "
            f"unexpected={sorted(resident_names - expected_names)}."
        )
    model_state = model.state_dict()
    for name in resident.tensor_names:
        canonical = resident.aliases.get(name, name)
        descriptor = resident.descriptors[canonical]
        if tuple(model_state[name].shape) != descriptor.shape:
            raise GPUPretrainedError(
                f"Resident tensor {name!r} does not match the model architecture."
            )
    _validate_and_remove_compressed_weights(model, exported)

    try:
        from accelerate.utils import set_module_tensor_to_device
    except ImportError as error:
        raise RuntimeError(
            "Direct sharded checkpoint loading requires the 'accelerate' package."
        ) from error

    aliases_by_canonical = {}
    for alias, canonical in resident.aliases.items():
        aliases_by_canonical.setdefault(canonical, []).append(alias)
    loaded_names = set()
    try:
        for _path, shard in iter_resident_shards(resident):
            for canonical, tensor in shard.items():
                logical_names = (canonical, *aliases_by_canonical.get(canonical, ()))
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
    offload_folder: str | os.PathLike[str] | None,
    allow_cpu_blocks: bool,
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
    if offload_folder is not None and device_map is None:
        raise ValueError("offload_folder requires device_map.")
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
        elif not allow_cpu_blocks and torch.cuda.is_available():
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

    if not allow_cpu_blocks:
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
    *,
    offload_folder: str | os.PathLike[str] | None,
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
        offload_dir=offload_folder,
        offload_buffers=False,
        force_hooks=(
            bool(placement.cpu_stored_blocks)
            or len(set(placement.block_devices.values())) > 1
        ),
    )


def load_gpu_pretrained(
    pretrained_model_name_or_path: str | os.PathLike[str],
    *,
    model: nn.Module | None = None,
    model_factory: Callable[[Path], nn.Module] | None = None,
    device: object | None = None,
    device_map: str | Mapping[str, object] | None = None,
    max_memory: Mapping[object, object] | None = None,
    offload_folder: str | os.PathLike[str] | None = None,
    execution_device: object | None = None,
    decoder: Callable[..., Any] | object | None = None,
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
    manifest, resident_manifest, export_path, config_path = _load_checkpoint_metadata(
        target
    )
    exported = load_gpu_export(export_path)
    compressed_names = _require_string_list(
        manifest["compressed_weight_names"], context="compressed_weight_names"
    )
    if compressed_names != tuple(
        sorted(binding.tensor_name for binding in exported.bindings)
    ):
        raise GPUPretrainedError("The full checkpoint and GPU export disagree.")
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
        offload_folder=offload_folder,
        allow_cpu_blocks=decoder is not None,
        execution_device=execution_device,
        cpu_offload=cpu_offload,
        cpu_offload_blocks=cpu_offload_blocks,
    )
    model = _load_model_state(model, resident_manifest, exported, placement)
    generation_record = manifest["files"]["generation_config"]
    _load_generation_config(
        model,
        target,
        enabled=generation_record is not None,
        trust_remote_code=trust_remote_code,
    )
    model = _apply_placement(
        model,
        placement,
        offload_folder=offload_folder,
    )
    NeuralZipRuntime.attach(
        model,
        exported,
        block_devices=placement.block_devices,
        decoder=decoder,
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
    "RESIDENT_FILENAME",
    "GPUPretrainedError",
    "load_gpu_pretrained",
    "save_gpu_pretrained",
]
