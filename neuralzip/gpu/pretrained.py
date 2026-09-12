import json
import inspect
import time
from dataclasses import asdict, replace
from pathlib import Path
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, GenerationConfig, PreTrainedModel
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME
from neuralzip._storage_checkpoint import (
    NeuralZipStats,
    _bytes_tensor,
    decode_model_storage,
    encode_model_storage,
    setup_seconds,
    storage_stats,
)
from neuralzip.gpu.inference import (
    NeuralZipRuntime,
    export_model_for_gpu,
)

FORMAT_NAME = "neuralzip"
FORMAT_VERSION = 2
PREFIX = "__neuralzip__."
STORAGE_DATA = f"{PREFIX}storage.data"
STORAGE_METADATA = f"{PREFIX}storage.metadata"
STATS_ATTRIBUTE = "neuralzip_stats"
_MISSING = object()
_ORIGINAL_SAVE = PreTrainedModel.save_pretrained
_ORIGINAL_LOAD = PreTrainedModel.__dict__["from_pretrained"].__func__
_HF_OPTIONS = {}

def _tensor_bytes(tensor):
    return tensor.detach().cpu().contiguous().numpy().tobytes()

def _checkpoint_metadata(config):
    metadata = getattr(config, "neuralzip", None)
    if not isinstance(metadata, dict) or metadata.get("format") != FORMAT_NAME:
        return None
    version = metadata.get("version")
    if type(version) is not int or version != FORMAT_VERSION:
        raise ValueError("Unsupported NeuralZip checkpoint.")
    if metadata.get("encoding") != "storage":
        raise ValueError("Invalid NeuralZip checkpoint format metadata.")
    return metadata

def _export_stats(model, exported, setup, encoding_seconds):
    compressed_names = {binding.tensor_name for binding in exported.bindings}
    resident_bytes = 0
    resident_exponent_bytes = 0
    seen = set()
    for name, tensor in model.state_dict().items():
        key = (
            tensor.device,
            tensor.data_ptr(),
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.dtype,
        )
        if name in compressed_names or key in seen:
            continue
        seen.add(key)
        resident_bytes += tensor.numel() * tensor.element_size()
        if tensor.is_floating_point():
            resident_exponent_bytes += tensor.numel()
    compressed_numel = sum(group.encoded.numel for group in exported.groups)
    exponent_raw_bytes = resident_exponent_bytes + compressed_numel
    exponent_payload_bytes = sum(
        len(group.encoded.exponents.bitstream)
        + len(group.encoded.exponents.packed_gaps)
        + 4 * len(group.encoded.exponents.output_positions)
        for group in exported.groups
    )
    codebook_bytes = 1024 * len(
        {group.codebook_id for group in exported.groups}
    )
    exponent_compressed_bytes = (
        resident_exponent_bytes + exponent_payload_bytes + codebook_bytes
    )
    sign_mantissa_bytes = sum(
        len(group.encoded.sign_mantissa) for group in exported.groups
    )
    return NeuralZipStats(
        setup_seconds=setup_seconds(setup),
        encoding_seconds=float(encoding_seconds),
        raw_bytes=resident_bytes + 2 * compressed_numel,
        compressed_bytes=(
            resident_bytes
            + sign_mantissa_bytes
            + exponent_payload_bytes
            + codebook_bytes
        ),
        exponent_raw_bytes=exponent_raw_bytes,
        exponent_compressed_bytes=exponent_compressed_bytes,
        representation="gpu",
    )

def _encode_export(model, setup):
    started = time.perf_counter()
    exported = export_model_for_gpu(model, setup)
    stats = _export_stats(model, exported, setup, time.perf_counter() - started)
    setattr(model, STATS_ATTRIBUTE, stats)
    return exported

def _attach_export(model, exported, device):
    if exported.groups:
        NeuralZipRuntime.attach(model, exported, device=device)
    elif device is not None:
        model.to(device).eval()

def _checkpoint_directory(value, **hub_kwargs):
    path = Path(value).expanduser()
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=str(value),
            **hub_kwargs,
        )
    )

def _checkpoint_state(directory):
    special = {}
    resident = {}
    index = directory / SAFE_WEIGHTS_INDEX_NAME
    if index.is_file():
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        files = tuple(directory / name for name in dict.fromkeys(weight_map.values()))
    else:
        files = (directory / SAFE_WEIGHTS_NAME,)
    for path in files:
        shard = load_file(path, device="cpu")
        duplicates = (set(special) | set(resident)) & set(shard)
        if duplicates:
            raise ValueError("A checkpoint tensor occurs in more than one shard.")
        for name, tensor in shard.items():
            target = special if name.startswith(PREFIX) else resident
            target[name] = tensor
    return special, resident

def _save_pretrained(
    model,
    directory,
    *,
    setup=None,
    save_args=(),
    save_kwargs=None,
):
    runtime = getattr(model, "neuralzip_runtime", None)
    blob, storage, resident, aliases, sizes, seconds = encode_model_storage(
        model,
        setup,
        runtime if isinstance(runtime, NeuralZipRuntime) else None,
    )
    stats = storage_stats(setup, sizes, seconds)
    setattr(model, STATS_ATTRIBUTE, stats)
    bound = inspect.signature(_ORIGINAL_SAVE).bind(
        model,
        directory,
        *save_args,
        **dict(save_kwargs or {}),
    )
    bound.apply_defaults()
    options = dict(bound.arguments.pop("kwargs"))
    options.update(bound.arguments)
    options.pop("self")
    options.pop("save_directory")
    if options.pop("state_dict") is not None:
        raise ValueError("NeuralZip cannot compress an alternate state_dict.")
    if options.pop("variant") is not None:
        raise ValueError("NeuralZip checkpoints do not support variant filenames.")
    if options.pop("distributed_checkpoint"):
        raise ValueError("NeuralZip does not support distributed checkpoint saving.")
    state = dict(resident)
    state[STORAGE_DATA] = blob
    state[STORAGE_METADATA] = _bytes_tensor(
        json.dumps(storage, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )
    metadata = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "encoding": "storage",
        "aliases": aliases,
        "stats": asdict(stats),
    }
    config = model.config
    previous = getattr(config, "neuralzip", _MISSING)
    config.neuralzip = metadata
    options["save_original_format"] = False
    try:
        result = _ORIGINAL_SAVE(model, directory, state_dict=state, **options)
    finally:
        if previous is _MISSING:
            delattr(config, "neuralzip")
        else:
            config.neuralzip = previous
    return result

def _load_storage(
    directory,
    device,
    model_class,
    model_args,
    config,
    metadata,
    setup,
):
    special, resident = _checkpoint_state(directory)
    required = {STORAGE_DATA, STORAGE_METADATA}
    if set(special) != required:
        raise ValueError("The NeuralZip storage checkpoint is incomplete.")
    storage = json.loads(_tensor_bytes(special[STORAGE_METADATA]).decode("utf-8"))
    started = time.perf_counter()
    state = decode_model_storage(
        special[STORAGE_DATA],
        storage,
        resident,
        metadata.get("aliases", {}),
    )
    decoding_seconds = time.perf_counter() - started
    model, loading_info = _ORIGINAL_LOAD(
        model_class,
        None,
        *model_args,
        config=config,
        state_dict=state,
        output_loading_info=True,
    )
    if any(loading_info.values()):
        raise ValueError("The decoded state does not match the model architecture.")
    model.config.name_or_path = str(directory)
    if (directory / "generation_config.json").is_file():
        model.generation_config = GenerationConfig.from_pretrained(
            directory, local_files_only=True
        )
    started = time.perf_counter()
    exported = export_model_for_gpu(model, setup)
    runtime_encoding_seconds = time.perf_counter() - started
    _attach_export(model, exported, device)
    stats = replace(
        NeuralZipStats(**metadata["stats"]),
        decoding_seconds=decoding_seconds,
        runtime_encoding_seconds=runtime_encoding_seconds,
    )
    setattr(model, STATS_ATTRIBUTE, stats)
    return model

def _patched_save(model, directory, *args, **kwargs):
    return _save_pretrained(
        model,
        directory,
        setup=_HF_OPTIONS["setup"],
        save_args=args,
        save_kwargs=kwargs,
    )

def _load_device(kwargs):
    value = kwargs.pop("device_map", None)
    if value in (None, "auto"):
        return _HF_OPTIONS["device"]
    if isinstance(value, int):
        return f"cuda:{value}"
    try:
        device = torch.device(value)
    except (RuntimeError, TypeError) as error:
        raise TypeError("NeuralZip inference needs one CUDA device.") from error
    if device.type != "cuda":
        raise TypeError("NeuralZip inference needs one CUDA device.")
    return device

def _patched_load(cls, source, *model_args, config=None, **kwargs):
    hub_keys = ("revision", "token", "cache_dir", "local_files_only", "force_download")
    hub = {name: kwargs[name] for name in hub_keys if name in kwargs}
    if _checkpoint_metadata(config) is None:
        config_class = getattr(cls, "config_class", None) or AutoConfig
        try:
            source_config = config_class.from_pretrained(
                source,
                subfolder=kwargs.get("subfolder", ""),
                **hub,
            )
        except (OSError, ValueError):
            source_config = None
        if _checkpoint_metadata(source_config) is not None:
            config = source_config
    if _checkpoint_metadata(config) is None:
        if kwargs.get("quantization_config") is not None or getattr(
            config,
            "quantization_config",
            None,
        ) is not None:
            raise ValueError("NeuralZip cannot attach to a quantized model.")
        device = _load_device(kwargs)
        kwargs.pop("torch_dtype", None)
        kwargs["dtype"] = torch.bfloat16
        result = _ORIGINAL_LOAD(cls, source, *model_args, config=config, **kwargs)
        model = result[0] if isinstance(result, tuple) else result
        exported = _encode_export(model, _HF_OPTIONS["setup"])
        _attach_export(model, exported, device)
        return result
    device = _load_device(kwargs)
    output_loading_info = kwargs.pop("output_loading_info", False)
    if kwargs.pop("use_safetensors", True) is False:
        raise ValueError("NeuralZip checkpoints require Safetensors.")
    dtype = kwargs.pop("dtype", kwargs.pop("torch_dtype", None))
    if dtype not in (None, "auto", "bfloat16", torch.bfloat16):
        raise ValueError("NeuralZip GPU checkpoints require BF16 weights.")
    if kwargs.pop("ignore_mismatched_sizes", False):
        raise ValueError("NeuralZip cannot ignore mismatched tensor sizes.")
    kwargs.pop("weights_only", None)
    kwargs.pop("trust_remote_code", None)
    kwargs.pop("_from_auto", None)
    adapter_kwargs = kwargs.pop("adapter_kwargs", None)
    if adapter_kwargs and "_adapter_model_path" in adapter_kwargs:
        raise ValueError("NeuralZip checkpoints do not support PEFT adapters.")
    unsupported = sorted(set(kwargs) - set(hub_keys) - {"subfolder"})
    if unsupported:
        names = ", ".join(unsupported)
        raise TypeError(f"Unsupported NeuralZip loading options: {names}.")
    subfolder = kwargs.get("subfolder", "")
    pattern = f"{subfolder}/*" if subfolder else "*"
    directory = _checkpoint_directory(
        source,
        allow_patterns=(f"{pattern}.json", f"{pattern}.safetensors"),
        **hub,
    ) / subfolder
    model = _load_storage(
        directory,
        device,
        cls,
        model_args,
        config,
        _checkpoint_metadata(config),
        _HF_OPTIONS["setup"],
    )
    if output_loading_info:
        return model, {
            "missing_keys": set(),
            "unexpected_keys": set(),
            "mismatched_keys": set(),
            "error_msgs": [],
        }
    return model

def enable(setup=None, *, device=None):
    _HF_OPTIONS.update(setup=setup, device=device)
    if PreTrainedModel.save_pretrained is not _patched_save:
        PreTrainedModel.save_pretrained = _patched_save
        PreTrainedModel.from_pretrained = classmethod(_patched_load)
