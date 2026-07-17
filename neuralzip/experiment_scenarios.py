from __future__ import annotations

import gc
import math
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

from .compressor import Compressor
from .neuralzip_codec import NeuralZipCodec
from .zipnn_codec import ZipNNCodec


DEFAULT_METHODS = (
    ("NeuralZip", "neuralzip"),
)


FULL_CHECKPOINT_METHODS = DEFAULT_METHODS


@dataclass
class MappingModel:
    model_name: str
    model: dict[str, torch.Tensor]


def _ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _bit_entropy_from_counts(ones, total):
    if total <= 0:
        return 0.0
    p = ones / total
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p)))


def _byte_bitplane_counts(data):
    if not data:
        return [0] * 8, 0
    tensor = torch.frombuffer(bytearray(data), dtype=torch.uint8)
    counts = [int(((tensor >> bit) & 1).sum().item()) for bit in range(8)]
    return counts, int(tensor.numel())


def _update_bitplane_counts(counts, data):
    bit_counts, byte_count = _byte_bitplane_counts(data)
    for bit, value in enumerate(bit_counts):
        counts[bit] += value
    return byte_count


def _tensor_mapping_from_module(module, *, dtype=None, predicate=None):
    tensors = {}
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            if not parameter.is_floating_point():
                continue
            if predicate is not None and not predicate(name):
                continue
            tensor = parameter.detach().cpu().contiguous()
            if dtype is not None:
                tensor = tensor.to(dtype=dtype)
            tensors[name] = tensor
    return tensors


def _normalise_dtype(dtype):
    if dtype is None:
        return None
    if isinstance(dtype, torch.dtype):
        return dtype
    dtype_key = str(dtype).lower().replace("torch.", "").replace("-", "_")
    dtype_map = {
        "auto": "auto",
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    if dtype_key not in dtype_map:
        choices = ", ".join(sorted(dtype_map))
        raise ValueError(f"Unsupported dtype {dtype!r}. Expected one of: {choices}.")
    return dtype_map[dtype_key]


def _model_spec_value(spec, key, default=None):
    if isinstance(spec, dict):
        return spec.get(key, default)
    if key == "name":
        return spec[0]
    if key == "model_id":
        return spec[1]
    return default


def _load_transformers_module(
    model_id,
    *,
    dtype=torch.bfloat16,
    local_files_only=True,
    trust_remote_code=True,
    **kwargs,
):
    from transformers import AutoModel, AutoModelForCausalLM

    dtype = _normalise_dtype(dtype)
    load_kwargs = {
        "local_files_only": bool(local_files_only),
        "trust_remote_code": bool(trust_remote_code),
        "low_cpu_mem_usage": True,
    }
    load_kwargs.update(kwargs)
    if dtype is not None and "dtype" not in load_kwargs and "torch_dtype" not in load_kwargs:
        load_kwargs["dtype"] = dtype

    errors = []
    for loader in (AutoModelForCausalLM, AutoModel):
        try:
            return loader.from_pretrained(model_id, **load_kwargs)
        except Exception as error:
            errors.append(f"{loader.__name__}: {type(error).__name__}: {error}")
    raise ValueError(
        f"Could not load {model_id} with supported model loaders:\n"
        + "\n".join(errors)
    )


def _load_safetensors_mapping(
    model_id,
    *,
    dtype=torch.bfloat16,
    local_files_only=True,
):
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    snapshot_path = Path(
        snapshot_download(
            model_id,
            local_files_only=bool(local_files_only),
            allow_patterns=("*.safetensors", "*.safetensors.index.json"),
        )
    )
    files = sorted(snapshot_path.glob("*.safetensors"))
    if not files:
        raise ValueError(f"No safetensors files found for {model_id}.")

    target_dtype = _normalise_dtype(dtype)
    if target_dtype == "auto":
        target_dtype = None

    tensors = {}
    for file_index, file_path in enumerate(files):
        state = load_file(str(file_path), device="cpu")
        for name, tensor in state.items():
            if not torch.is_tensor(tensor) or not tensor.is_floating_point():
                continue
            value = tensor.detach().cpu().contiguous()
            if target_dtype is not None:
                value = value.to(dtype=target_dtype)
            key = name if name not in tensors else f"{file_path.stem}.{name}"
            tensors[key] = value
        del state
        gc.collect()

    if not tensors:
        raise ValueError(f"No floating-point tensors found in safetensors for {model_id}.")
    return tensors


def _result_exponent_metrics(result):
    raw_bytes = result.get("raw_bytes")
    compressed_bytes = result.get("compressed_bytes")
    scope = str(result.get("scope", "")).lower()
    if "exponent" in scope:
        return {
            "exponent_raw_bytes": raw_bytes,
            "exponent_compressed_bytes": compressed_bytes,
            "exponent_compressed_pct": (
                100 * compressed_bytes / raw_bytes if raw_bytes else None
            ),
        }

    stream_results = result.get("stream_results") or ()
    if not stream_results:
        return {
            "exponent_raw_bytes": None,
            "exponent_compressed_bytes": None,
            "exponent_compressed_pct": None,
        }

    # ZipNN byte grouping places the exponent/significant high byte in the last
    # stream for bf16 checkpoints. The large-model experiments use bf16.
    exponent_stream = max(stream_results, key=lambda row: int(row["stream_id"]))
    exponent_raw = int(exponent_stream["raw_bytes"])
    exponent_compressed = int(exponent_stream["compressed_bytes"])
    return {
        "exponent_raw_bytes": exponent_raw,
        "exponent_compressed_bytes": exponent_compressed,
        "exponent_compressed_pct": (
            100 * exponent_compressed / exponent_raw if exponent_raw else None
        ),
    }


def _compress_mapping(model_name, tensor_mapping, *, methods=DEFAULT_METHODS):
    rows = []
    wrapped = MappingModel(model_name=model_name, model=tensor_mapping)
    for compressor_name, method in methods:
        start = time.perf_counter()
        try:
            result = Compressor(method, verify=False).compress(wrapped)
            elapsed = time.perf_counter() - start
            exponent_metrics = _result_exponent_metrics(result)
            rows.append(
                {
                    "model": model_name,
                    "compressor": compressor_name,
                    "method": result.get("method"),
                    "scope": result.get("scope", "full checkpoint"),
                    "status": "ok",
                    "raw_bytes": result.get("raw_bytes"),
                    "compressed_bytes": result.get("compressed_bytes"),
                    "compression_ratio": result.get("compression_ratio"),
                    "compressed_pct": 100 * result.get("compression_ratio"),
                    "full_compressed_pct": 100 * result.get("compression_ratio"),
                    **exponent_metrics,
                    "precompute_seconds": result.get("precompute_seconds"),
                    "codebook_setup_seconds": result.get(
                        "codebook_setup_seconds"
                    ),
                    "encode_seconds": result.get("encode_seconds"),
                    "compression_seconds": result.get("encode_seconds", elapsed),
                    "total_compression_seconds": result.get("encode_seconds", elapsed),
                    "cold_encode_seconds": result.get("encode_seconds", elapsed),
                    "clusters": result.get("clusters"),
                    "codebooks": (
                        len(result.get("codebooks"))
                        if result.get("codebooks") is not None
                        else None
                    ),
                    "huffman_chunks": result.get("huffman_chunks"),
                    "raw_chunks": result.get("raw_chunks"),
                    "error": None,
                }
            )
        except Exception as error:
            rows.append(
                {
                    "model": model_name,
                    "compressor": compressor_name,
                    "method": method,
                    "scope": None,
                    "status": "failed",
                    "raw_bytes": None,
                    "compressed_bytes": None,
                    "compression_ratio": None,
                    "compressed_pct": None,
                    "full_compressed_pct": None,
                    "exponent_raw_bytes": None,
                    "exponent_compressed_bytes": None,
                    "exponent_compressed_pct": None,
                    "precompute_seconds": None,
                    "codebook_setup_seconds": None,
                    "encode_seconds": None,
                    "compression_seconds": None,
                    "total_compression_seconds": None,
                    "cold_encode_seconds": None,
                    "clusters": None,
                    "codebooks": None,
                    "huffman_chunks": None,
                    "raw_chunks": None,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    return rows


def run_hf_model_suite_experiment(
    output_dir,
    *,
    model_suite,
    experiment,
    methods=DEFAULT_METHODS,
    default_dtype=torch.bfloat16,
    local_files_only=True,
    trust_remote_code=True,
):
    output_dir = _ensure_dir(output_dir)
    result_path = output_dir / f"{experiment.replace('-', '_')}_results.csv"
    rows = []
    for spec in model_suite:
        model_name = _model_spec_value(spec, "name")
        model_id = _model_spec_value(spec, "model_id")
        dtype = _model_spec_value(spec, "dtype", None)
        if dtype is None:
            dtype = _model_spec_value(spec, "default_dtype", default_dtype)
        representation = _model_spec_value(spec, "representation", None)
        category = _model_spec_value(spec, "category", None)
        load_kwargs = dict(_model_spec_value(spec, "load_kwargs", {}) or {})
        model = None
        try:
            try:
                tensors = _load_safetensors_mapping(
                    model_id,
                    dtype=dtype,
                    local_files_only=local_files_only,
                )
                load_backend = "safetensors"
            except Exception as safetensors_error:
                model = _load_transformers_module(
                    model_id,
                    dtype=dtype,
                    local_files_only=local_files_only,
                    trust_remote_code=trust_remote_code,
                    **load_kwargs,
                )
                tensor_dtype = (
                    None
                    if _normalise_dtype(dtype) == "auto"
                    else _normalise_dtype(dtype)
                )
                tensors = _tensor_mapping_from_module(model, dtype=tensor_dtype)
                load_backend = "transformers"
            if not tensors:
                rows.append(
                    {
                        "experiment": experiment,
                        "model": model_name,
                        "model_name": model_name,
                        "model_id": model_id,
                        "representation": representation,
                        "category": category,
                        "status": "no-floating-tensors",
                        "error": "Loaded model exposes no floating-point tensors.",
                        "load_backend": None,
                    }
                )
                continue
            for row in _compress_mapping(model_name, tensors, methods=methods):
                row["experiment"] = experiment
                row["model_name"] = model_name
                row["model_id"] = model_id
                row["representation"] = representation
                row["category"] = category
                row["dtype"] = str(dtype).replace("torch.", "")
                row["tensor_count"] = len(tensors)
                row["load_backend"] = load_backend
                rows.append(row)
        except Exception as error:
            rows.append(
                {
                    "experiment": experiment,
                    "model": model_name,
                    "model_name": model_name,
                    "model_id": model_id,
                    "representation": representation,
                    "category": category,
                    "status": "load-failed",
                    "error": f"{type(error).__name__}: {error}",
                    "load_backend": None,
                }
            )
        finally:
            if model is not None:
                del model
            gc.collect()
            if rows:
                pd.DataFrame(rows).to_csv(result_path, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(result_path, index=False)
    return df


def _resolve_torchvision_weights(models_module, constructor_name, weights):
    if weights is None:
        return None, "none"
    if str(weights).upper() == "DEFAULT":
        try:
            weights_enum = models_module.get_model_weights(constructor_name)
            return weights_enum.DEFAULT, "DEFAULT"
        except Exception:
            return "DEFAULT", "DEFAULT"
    return weights, str(weights)


def _load_torchvision_mapping(spec, *, dtype):
    from torchvision import models

    constructor_name = _model_spec_value(spec, "constructor")
    if not constructor_name:
        raise ValueError("torchvision specs require a constructor name.")
    builder = getattr(models, constructor_name)
    weights, weights_label = _resolve_torchvision_weights(
        models,
        constructor_name,
        _model_spec_value(spec, "weights", "DEFAULT"),
    )

    try:
        module = builder(weights=weights)
    except TypeError:
        module = builder(pretrained=weights is not None)
    tensor_dtype = _normalise_dtype(dtype)
    if tensor_dtype == "auto":
        tensor_dtype = None
    tensors = _tensor_mapping_from_module(module, dtype=tensor_dtype)
    return tensors, {
        "load_backend": "torchvision",
        "source": f"torchvision.models.{constructor_name}",
        "weights": weights_label,
    }


def _load_diffusers_unet_mapping(
    spec,
    *,
    dtype,
    local_files_only,
):
    loader_name = _model_spec_value(spec, "loader", "UNet2DConditionModel")
    if loader_name == "UNet2DConditionModel":
        from diffusers import UNet2DConditionModel as loader_cls
    elif loader_name == "UNet2DModel":
        from diffusers import UNet2DModel as loader_cls
    else:
        raise ValueError(
            "Unsupported diffusers UNet loader "
            f"{loader_name!r}. Expected UNet2DConditionModel or UNet2DModel."
        )

    model_id = _model_spec_value(spec, "model_id")
    if not model_id:
        raise ValueError("diffusers specs require a model_id.")
    load_kwargs = dict(_model_spec_value(spec, "load_kwargs", {}) or {})
    subfolder = _model_spec_value(spec, "subfolder", None)
    if subfolder:
        load_kwargs["subfolder"] = subfolder
    load_kwargs["local_files_only"] = bool(local_files_only)

    tensor_dtype = _normalise_dtype(dtype)
    if tensor_dtype != "auto" and tensor_dtype is not None:
        load_kwargs.setdefault("torch_dtype", tensor_dtype)
    load_kwargs.setdefault("low_cpu_mem_usage", True)

    module = loader_cls.from_pretrained(model_id, **load_kwargs)
    if tensor_dtype == "auto":
        tensor_dtype = None
    tensors = _tensor_mapping_from_module(module, dtype=tensor_dtype)
    source = model_id if not subfolder else f"{model_id}/{subfolder}"
    return tensors, {
        "load_backend": f"diffusers:{loader_name}",
        "source": source,
        "weights": "pretrained",
    }


def _load_vision_diffusion_mapping(
    spec,
    *,
    dtype,
    local_files_only,
):
    backend = str(_model_spec_value(spec, "backend", "")).lower()
    if backend == "torchvision":
        return _load_torchvision_mapping(spec, dtype=dtype)
    if backend == "diffusers-unet":
        return _load_diffusers_unet_mapping(
            spec,
            dtype=dtype,
            local_files_only=local_files_only,
        )
    raise ValueError(f"Unsupported vision/diffusion backend {backend!r}.")


def _load_failure_row(spec, *, experiment, error, dtype, fallback_for=None):
    model_name = _model_spec_value(spec, "name")
    return {
        "experiment": experiment,
        "model": model_name,
        "model_name": model_name,
        "model_id": _model_spec_value(spec, "model_id", None),
        "family": _model_spec_value(spec, "family", None),
        "backend": _model_spec_value(spec, "backend", None),
        "source": _model_spec_value(spec, "model_id", None)
        or _model_spec_value(spec, "constructor", None),
        "weights": _model_spec_value(spec, "weights", None),
        "fallback_for": fallback_for,
        "dtype": str(dtype).replace("torch.", ""),
        "status": "load-failed",
        "error": f"{type(error).__name__}: {error}",
        "load_backend": None,
    }


def _model_methods_completed(existing_rows, model_name, methods):
    completed = {
        str(row.get("compressor"))
        for row in existing_rows
        if row.get("model_name") == model_name
        and row.get("status") == "ok"
        and row.get("compressor") is not None
    }
    return all(name in completed for name, _ in methods)


def run_vision_diffusion_experiment(
    output_dir,
    *,
    model_suite,
    methods=DEFAULT_METHODS,
    dtype=torch.bfloat16,
    local_files_only=False,
    resume=True,
):
    output_dir = _ensure_dir(output_dir)
    result_path = output_dir / "vision_diffusion_results.csv"
    experiment = "vision-diffusion"
    rows = []
    if resume and result_path.exists():
        rows = pd.read_csv(result_path).to_dict(orient="records")

    for spec in model_suite:
        model_name = _model_spec_value(spec, "name")
        if resume and _model_methods_completed(rows, model_name, methods):
            continue

        candidate_specs = [spec]
        fallback = _model_spec_value(spec, "fallback", None)
        if fallback:
            candidate_specs.append(fallback)

        loaded = False
        for attempt_index, candidate in enumerate(candidate_specs):
            fallback_for = _model_spec_value(candidate, "fallback_for", None)
            candidate_name = _model_spec_value(candidate, "name")
            if resume and _model_methods_completed(rows, candidate_name, methods):
                loaded = True
                break

            load_start = time.perf_counter()
            try:
                candidate_dtype = _model_spec_value(
                    candidate,
                    "dtype",
                    _model_spec_value(candidate, "default_dtype", dtype),
                )
                tensors, load_metadata = _load_vision_diffusion_mapping(
                    candidate,
                    dtype=candidate_dtype,
                    local_files_only=local_files_only,
                )
            except Exception as error:
                if attempt_index == 0:
                    rows.append(
                        _load_failure_row(
                            candidate,
                            experiment=experiment,
                            error=error,
                            dtype=_model_spec_value(
                                candidate,
                                "dtype",
                                _model_spec_value(candidate, "default_dtype", dtype),
                            ),
                            fallback_for=fallback_for,
                        )
                    )
                    pd.DataFrame(rows).to_csv(result_path, index=False)
                if attempt_index + 1 < len(candidate_specs):
                    continue
                rows.append(
                    _load_failure_row(
                        candidate,
                        experiment=experiment,
                        error=error,
                        dtype=_model_spec_value(
                            candidate,
                            "dtype",
                            _model_spec_value(candidate, "default_dtype", dtype),
                        ),
                        fallback_for=fallback_for,
                    )
                )
                pd.DataFrame(rows).to_csv(result_path, index=False)
                break

            load_seconds = time.perf_counter() - load_start
            if not tensors:
                rows.append(
                    {
                        "experiment": experiment,
                        "model": candidate_name,
                        "model_name": candidate_name,
                        "model_id": _model_spec_value(candidate, "model_id", None),
                        "family": _model_spec_value(candidate, "family", None),
                        "backend": _model_spec_value(candidate, "backend", None),
                        "source": load_metadata.get("source"),
                        "weights": load_metadata.get("weights"),
                        "fallback_for": fallback_for,
                        "dtype": str(candidate_dtype).replace("torch.", ""),
                        "status": "no-floating-tensors",
                        "error": "Loaded model exposes no floating-point tensors.",
                        "load_backend": load_metadata.get("load_backend"),
                    }
                )
                pd.DataFrame(rows).to_csv(result_path, index=False)
                break

            parameter_count = sum(int(tensor.numel()) for tensor in tensors.values())
            floating_raw_bytes = sum(
                int(tensor.numel() * tensor.element_size())
                for tensor in tensors.values()
            )
            for row in _compress_mapping(candidate_name, tensors, methods=methods):
                row["experiment"] = experiment
                row["model_name"] = candidate_name
                row["model_id"] = _model_spec_value(candidate, "model_id", None)
                row["family"] = _model_spec_value(candidate, "family", None)
                row["backend"] = _model_spec_value(candidate, "backend", None)
                row["source"] = load_metadata.get("source")
                row["weights"] = load_metadata.get("weights")
                row["fallback_for"] = fallback_for
                row["dtype"] = str(candidate_dtype).replace("torch.", "")
                row["tensor_count"] = len(tensors)
                row["parameter_count"] = parameter_count
                row["floating_raw_bytes"] = floating_raw_bytes
                row["load_seconds"] = load_seconds
                row["load_backend"] = load_metadata.get("load_backend")
                rows.append(row)

            pd.DataFrame(rows).to_csv(result_path, index=False)
            del tensors
            gc.collect()
            loaded = True
            break

        if not loaded:
            gc.collect()

    df = pd.DataFrame(rows)
    df.to_csv(result_path, index=False)
    return df


def run_moe_experiment(
    output_dir,
    *,
    model_suite=None,
    methods=DEFAULT_METHODS,
    dtype=torch.bfloat16,
    local_files_only=True,
    trust_remote_code=True,
):
    if model_suite is not None:
        return run_hf_model_suite_experiment(
            output_dir,
            model_suite=model_suite,
            experiment="moe",
            methods=methods,
            default_dtype=dtype,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )

    from transformers import MixtralConfig, MixtralForCausalLM

    output_dir = _ensure_dir(output_dir)
    model_configs = (
        {
            "name": "Tiny-Mixtral-4E",
            "num_local_experts": 4,
            "num_experts_per_tok": 2,
        },
        {
            "name": "Tiny-Mixtral-8E",
            "num_local_experts": 8,
            "num_experts_per_tok": 2,
        },
    )
    rows = []
    for index, spec in enumerate(model_configs):
        torch.manual_seed(1000 + index)
        config = MixtralConfig(
            vocab_size=512,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            num_local_experts=spec["num_local_experts"],
            num_experts_per_tok=spec["num_experts_per_tok"],
            max_position_embeddings=128,
        )
        model = MixtralForCausalLM(config).to(dtype=dtype)
        full = _tensor_mapping_from_module(model, dtype=dtype)
        experts = _tensor_mapping_from_module(
            model,
            dtype=dtype,
            predicate=lambda name: ".experts." in name,
        )
        for scope, tensors in (("full-checkpoint", full), ("experts-only", experts)):
            for row in _compress_mapping(
                f'{spec["name"]}:{scope}',
                tensors,
                methods=methods,
            ):
                row["experiment"] = "moe"
                row["checkpoint_scope"] = scope
                row["num_experts"] = spec["num_local_experts"]
                row["active_experts"] = spec["num_experts_per_tok"]
                row["tensor_count"] = len(tensors)
                rows.append(row)
        del model, full, experts
        gc.collect()

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "moe_results.csv", index=False)
    return df


def run_lossy_lossless_experiment(
    output_dir,
    *,
    model_suite=None,
    methods=DEFAULT_METHODS,
    local_files_only=True,
    trust_remote_code=True,
):
    from transformers import AutoModel

    output_dir = _ensure_dir(output_dir)
    if model_suite is not None and any(isinstance(spec, dict) for spec in model_suite):
        return run_hf_model_suite_experiment(
            output_dir,
            model_suite=model_suite,
            experiment="lossy-lossless",
            methods=methods,
            default_dtype=torch.bfloat16,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )

    if model_suite is None:
        model_suite = (
            ("GPT-2", "gpt2"),
            ("BERT Base", "google-bert/bert-base-uncased"),
        )

    representations = (
        ("fp32-original", torch.float32, FULL_CHECKPOINT_METHODS),
        ("bf16-rounded", torch.bfloat16, methods),
    )

    rows = []
    for model_name, model_id in model_suite:
        for representation, dtype, selected_methods in representations:
            try:
                model = AutoModel.from_pretrained(
                    model_id,
                    dtype=dtype,
                    local_files_only=local_files_only,
                )
            except Exception as error:
                rows.append(
                    {
                        "experiment": "lossy-lossless",
                        "model": model_name,
                        "model_id": model_id,
                        "representation": representation,
                        "status": "load-failed",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                continue
            tensors = _tensor_mapping_from_module(model, dtype=dtype)
            for row in _compress_mapping(
                f"{model_name}:{representation}",
                tensors,
                methods=selected_methods,
            ):
                row["experiment"] = "lossy-lossless"
                row["model_name"] = model_name
                row["model_id"] = model_id
                row["representation"] = representation
                row["dtype"] = str(dtype).replace("torch.", "")
                row["tensor_count"] = len(tensors)
                rows.append(row)
            del model, tensors
            gc.collect()

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "lossy_lossless_results.csv", index=False)
    return df


def _hf_model_files(model_id, *, extensions):
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    files = []
    for item in fs.ls(model_id, detail=True):
        if item.get("type") != "file":
            continue
        name = item["name"]
        if any(str(name).lower().endswith(extension) for extension in extensions):
            files.append(
                {
                    "path": name,
                    "size": int(item.get("size") or 0),
                }
            )
    return fs, files


def _sample_hf_binary_file(
    fs,
    path,
    *,
    file_size,
    max_sample_bytes,
    chunk_size=4 << 20,
):
    suffix = str(path).lower()
    with fs.open(path, "rb") as handle:
        offset = 0
        if suffix.endswith(".safetensors"):
            header_size_bytes = handle.read(8)
            if len(header_size_bytes) == 8:
                header_size = int.from_bytes(header_size_bytes, "little")
                offset = 8 + header_size
        elif suffix.endswith(".gguf"):
            offset = min(4096, int(file_size))

        try:
            handle.seek(offset)
        except Exception:
            if offset:
                handle.read(offset)

        remaining = max(0, int(file_size) - int(offset))
        target = min(int(max_sample_bytes), remaining)
        counts = [0] * 8
        sampled = 0
        while sampled < target:
            data = handle.read(min(int(chunk_size), target - sampled))
            if not data:
                break
            sampled += _update_bitplane_counts(counts, data)
        return counts, sampled


def run_quantized_bitplane_experiment(
    output_dir,
    *,
    model_suite,
    max_sample_bytes_per_file=64 << 20,
    low_entropy_threshold=0.5,
):
    output_dir = _ensure_dir(output_dir)
    rows = []
    extensions = (".safetensors", ".bin", ".pt", ".gguf")

    for spec in model_suite:
        if _model_spec_value(spec, "category") != "quantized":
            continue
        model_name = _model_spec_value(spec, "name")
        model_id = _model_spec_value(spec, "model_id")
        try:
            fs, files = _hf_model_files(model_id, extensions=extensions)
            if not files:
                raise ValueError("No supported model files found.")

            total_counts = [0] * 8
            total_bytes = 0
            sampled_files = 0
            for item in files:
                counts, sampled = _sample_hf_binary_file(
                    fs,
                    item["path"],
                    file_size=item["size"],
                    max_sample_bytes=max_sample_bytes_per_file,
                )
                if sampled <= 0:
                    continue
                sampled_files += 1
                total_bytes += sampled
                for bit, value in enumerate(counts):
                    total_counts[bit] += value

            entropies = [
                _bit_entropy_from_counts(total_counts[bit], total_bytes)
                for bit in range(8)
            ]
            min_entropy = min(entropies) if entropies else None
            low_bits = [
                bit
                for bit, entropy in enumerate(entropies)
                if entropy < float(low_entropy_threshold)
            ]
            row = {
                "experiment": "lossy-lossless-bitplanes",
                "model": model_name,
                "model_name": model_name,
                "model_id": model_id,
                "status": "ok",
                "files": len(files),
                "sampled_files": sampled_files,
                "sampled_bytes": total_bytes,
                "min_bit_entropy": min_entropy,
                "mean_bit_entropy": (
                    sum(entropies) / len(entropies) if entropies else None
                ),
                "low_entropy_threshold": float(low_entropy_threshold),
                "low_entropy_bits": ",".join(str(bit) for bit in low_bits),
                "low_entropy_bit_count": len(low_bits),
                "error": None,
            }
            for bit, entropy in enumerate(entropies):
                row[f"bit_{bit}_entropy"] = entropy
            rows.append(row)
        except Exception as error:
            rows.append(
                {
                    "experiment": "lossy-lossless-bitplanes",
                    "model": model_name,
                    "model_name": model_name,
                    "model_id": model_id,
                    "status": "failed",
                    "files": None,
                    "sampled_files": None,
                    "sampled_bytes": None,
                    "min_bit_entropy": None,
                    "mean_bit_entropy": None,
                    "low_entropy_threshold": float(low_entropy_threshold),
                    "low_entropy_bits": None,
                    "low_entropy_bit_count": None,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
        pd.DataFrame(rows).to_csv(
            output_dir / "quantized_bitplane_entropy.csv",
            index=False,
        )

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "quantized_bitplane_entropy.csv", index=False)
    return df


_CIFAR_ARCHITECTURES = ("resnet18", "resnet34", "vgg16_bn", "vit_b_16")


def _canonical_cifar_architecture(architecture):
    key = str(architecture).lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "resnet_18": "resnet18",
        "resnet_34": "resnet34",
        "vgg16": "vgg16_bn",
        "vgg_16": "vgg16_bn",
        "vgg_16_bn": "vgg16_bn",
        "vit": "vit_b_16",
        "vit_base": "vit_b_16",
        "vit_b16": "vit_b_16",
    }
    key = aliases.get(key, key)
    if key not in _CIFAR_ARCHITECTURES:
        choices = ", ".join(_CIFAR_ARCHITECTURES)
        raise ValueError(f"Unsupported CIFAR architecture {architecture!r}: {choices}.")
    return key


def build_cifar_model(architecture="resnet18", *, num_classes=100):
    """Build a 32x32 CIFAR classifier with a consistent output head."""

    import torch.nn as nn
    from torchvision import models

    architecture = _canonical_cifar_architecture(architecture)
    if architecture in {"resnet18", "resnet34"}:
        constructor = getattr(models, architecture)
        model = constructor(weights=None, num_classes=int(num_classes))
        model.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        model.maxpool = nn.Identity()
        return model

    if architecture == "vgg16_bn":
        model = models.vgg16_bn(weights=None)
        model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model.classifier = nn.Linear(512, int(num_classes))
        return model

    return models.vit_b_16(
        weights=None,
        image_size=32,
        num_classes=int(num_classes),
    )


def build_resnet18_cifar(*, num_classes=10):
    return build_cifar_model("resnet18", num_classes=num_classes)


def _cifar_transforms(dataset, *, train):
    from torchvision import transforms

    dataset_key = str(dataset).lower().replace("-", "_")
    if "100" in dataset_key:
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    else:
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)

    steps = []
    if train:
        steps.extend(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        )
    steps.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
    return transforms.Compose(steps)


def _load_cifar_datasets(dataset, data_dir, *, download=True, include_test=True):
    from torchvision import datasets

    dataset_key = str(dataset).lower().replace("-", "_")
    is_cifar100 = "100" in dataset_key
    num_classes = 100 if is_cifar100 else 10
    train_transform = _cifar_transforms(dataset_key, train=True)
    test_transform = _cifar_transforms(dataset_key, train=False)

    if dataset_key in {"cifar10", "cifar100"}:
        dataset_cls = datasets.CIFAR100 if is_cifar100 else datasets.CIFAR10
        train_data = dataset_cls(
            root=data_dir,
            train=True,
            download=download,
            transform=train_transform,
        )
        test_data = (
            dataset_cls(
                root=data_dir,
                train=False,
                download=download,
                transform=test_transform,
            )
            if include_test
            else None
        )
        return train_data, test_data, num_classes

    if dataset_key in {
        "hf_cifar10",
        "cifar10_hf",
        "parquet_cifar10",
        "hf_cifar100",
        "cifar100_hf",
        "parquet_cifar100",
    }:
        folder = "cifar100" if is_cifar100 else "plain_text"
        train_path = data_dir / folder / "train-00000-of-00001.parquet"
        test_path = data_dir / folder / "test-00000-of-00001.parquet"
        required = [train_path] + ([test_path] if include_test else [])
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            repo = "cifar100" if is_cifar100 else "cifar10"
            raise FileNotFoundError(
                "Missing Hugging Face CIFAR parquet files: "
                + ", ".join(missing)
                + f". Download them from uoft-cs/{repo}."
            )
        train_data = _HFCIFARParquet(train_path, transform=train_transform)
        test_data = (
            _HFCIFARParquet(test_path, transform=test_transform)
            if include_test
            else None
        )
        return train_data, test_data, num_classes

    if dataset_key in {"fake", "fake_cifar", "synthetic"}:
        train_data = datasets.FakeData(
            size=50_000,
            image_size=(3, 32, 32),
            num_classes=num_classes,
            transform=train_transform,
            random_offset=0,
        )
        test_data = (
            datasets.FakeData(
                size=10_000,
                image_size=(3, 32, 32),
                num_classes=num_classes,
                transform=test_transform,
                random_offset=50_000,
            )
            if include_test
            else None
        )
        return train_data, test_data, num_classes

    raise ValueError(
        "dataset must be CIFAR-10/CIFAR-100, their hf-* parquet variant, or fake."
    )


def _limited_dataset(dataset, limit):
    from torch.utils.data import Subset

    if limit is None or int(limit) >= len(dataset):
        return dataset
    return Subset(dataset, list(range(max(0, int(limit)))))


def _checkpoint_path(checkpoint_dir, architecture, epoch):
    return checkpoint_dir / f"{architecture}_cifar100_epoch_{int(epoch):03d}.pt"


def _save_atomic_torch(payload, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train_cifar_checkpoints(
    output_dir,
    *,
    architecture="resnet18",
    epochs=200,
    train_subset=None,
    test_subset=None,
    batch_size=128,
    learning_rate=None,
    seed=2026,
    download=False,
    dataset="hf-cifar100",
    data_dir=None,
    evaluate_each_epoch=True,
    device=None,
    num_workers=4,
    amp=True,
    resume=True,
):
    """Train one CIFAR model and save epochs 0 through ``epochs``."""

    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader

    architecture = _canonical_cifar_architecture(architecture)
    output_dir = _ensure_dir(output_dir)
    checkpoint_dir = _ensure_dir(output_dir / "checkpoints")
    if data_dir is None:
        data_dir = output_dir / "data"
    data_dir = _ensure_dir(data_dir)

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    train_data, test_data, num_classes = _load_cifar_datasets(
        dataset,
        data_dir,
        download=download,
        include_test=True,
    )
    train_data = _limited_dataset(train_data, train_subset)
    test_data = _limited_dataset(test_data, test_subset)
    loader_generator = torch.Generator().manual_seed(int(seed))
    loader_kwargs = {
        "num_workers": int(num_workers),
        "pin_memory": device.type == "cuda",
        "persistent_workers": int(num_workers) > 0,
    }
    train_loader = DataLoader(
        train_data,
        batch_size=int(batch_size),
        shuffle=True,
        generator=loader_generator,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=int(batch_size),
        shuffle=False,
        **loader_kwargs,
    )

    model = build_cifar_model(architecture, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    if architecture == "vit_b_16":
        effective_lr = 3e-4 if learning_rate is None else float(learning_rate)
        optimizer = optim.AdamW(model.parameters(), lr=effective_lr, weight_decay=0.05)
    else:
        effective_lr = 0.1 if learning_rate is None else float(learning_rate)
        optimizer = optim.SGD(
            model.parameters(),
            lr=effective_lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(epochs)),
    )
    amp_enabled = bool(amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    state_path = output_dir / "training_state.pt"
    metrics_path = output_dir / "training_metrics.csv"
    metrics = (
        pd.read_csv(metrics_path).to_dict("records")
        if resume and metrics_path.exists()
        else []
    )
    start_epoch = 0
    if resume and state_path.exists():
        training_state = torch.load(state_path, map_location=device)
        model.load_state_dict(training_state["model_state"], strict=True)
        optimizer.load_state_dict(training_state["optimizer_state"])
        scheduler.load_state_dict(training_state["scheduler_state"])
        scaler.load_state_dict(training_state.get("scaler_state", {}))
        start_epoch = int(training_state["epoch"])

    def evaluate():
        model.eval()
        correct = 0
        total = 0
        loss_total = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=amp_enabled,
                ):
                    logits = model(images)
                    loss = criterion(logits, labels)
                loss_total += float(loss.item()) * labels.numel()
                correct += int((logits.argmax(dim=1) == labels).sum().item())
                total += int(labels.numel())
        return loss_total / max(total, 1), correct / max(total, 1)

    def cpu_model_state():
        return {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }

    if start_epoch == 0:
        initial_path = _checkpoint_path(checkpoint_dir, architecture, 0)
        if not initial_path.exists():
            _save_atomic_torch(
                {"epoch": 0, "architecture": architecture, "model_state": cpu_model_state()},
                initial_path,
            )
        if not any(int(row["epoch"]) == 0 for row in metrics):
            test_loss, test_accuracy = evaluate()
            metrics.append(
                {
                    "architecture": architecture,
                    "dataset": str(dataset),
                    "epoch": 0,
                    "device": str(device),
                    "learning_rate": effective_lr,
                    "train_loss": None,
                    "train_accuracy": None,
                    "test_loss": test_loss,
                    "test_accuracy": test_accuracy,
                    "train_seconds": 0.0,
                    "eval_seconds": 0.0,
                    "checkpoint_path": str(initial_path),
                }
            )
            pd.DataFrame(metrics).to_csv(metrics_path, index=False)

    for epoch in range(start_epoch + 1, int(epochs) + 1):
        model.train()
        train_started = time.perf_counter()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += float(loss.item()) * labels.numel()
            train_correct += int((logits.argmax(dim=1) == labels).sum().item())
            train_total += int(labels.numel())
        train_seconds = time.perf_counter() - train_started
        scheduler.step()

        eval_started = time.perf_counter()
        if evaluate_each_epoch:
            test_loss, test_accuracy = evaluate()
        else:
            test_loss, test_accuracy = None, None
        eval_seconds = time.perf_counter() - eval_started

        model_state = cpu_model_state()
        checkpoint_path = _checkpoint_path(checkpoint_dir, architecture, epoch)
        _save_atomic_torch(
            {
                "epoch": int(epoch),
                "architecture": architecture,
                "model_state": model_state,
            },
            checkpoint_path,
        )
        _save_atomic_torch(
            {
                "epoch": int(epoch),
                "architecture": architecture,
                "model_state": model_state,
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict(),
            },
            state_path,
        )
        metrics = [row for row in metrics if int(row["epoch"]) != int(epoch)]
        metrics.append(
            {
                "architecture": architecture,
                "dataset": str(dataset),
                "epoch": int(epoch),
                "device": str(device),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "train_loss": train_loss / max(train_total, 1),
                "train_accuracy": train_correct / max(train_total, 1),
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "train_seconds": train_seconds,
                "eval_seconds": eval_seconds,
                "checkpoint_path": str(checkpoint_path),
            }
        )
        metrics.sort(key=lambda row: int(row["epoch"]))
        pd.DataFrame(metrics).to_csv(metrics_path, index=False)
        print(
            f"{architecture} epoch={epoch}/{epochs} "
            f"train_loss={metrics[-1]['train_loss']:.4f} "
            f"test_accuracy={test_accuracy if test_accuracy is not None else float('nan'):.4f} "
            f"seconds={train_seconds + eval_seconds:.1f}",
            flush=True,
        )
        del model_state

    return pd.DataFrame(metrics).sort_values("epoch").reset_index(drop=True)


def train_resnet18_cifar_checkpoints(
    output_dir,
    *,
    epochs=3,
    train_subset=5000,
    test_subset=1000,
    batch_size=128,
    learning_rate=0.05,
    seed=2026,
    download=True,
    dataset="cifar10",
    data_dir=None,
    evaluate_each_epoch=True,
    device=None,
):
    import torch.nn as nn
    import torch.nn as nn
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms

    output_dir = _ensure_dir(output_dir)
    checkpoint_dir = _ensure_dir(output_dir / "checkpoints")
    if data_dir is None:
        data_dir = output_dir / "data"
    data_dir = _ensure_dir(data_dir)

    torch.manual_seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    dataset_key = str(dataset).lower().replace("-", "_")
    if dataset_key == "cifar10":
        train_data = datasets.CIFAR10(
            root=data_dir,
            train=True,
            download=download,
            transform=transform,
        )
        test_data = datasets.CIFAR10(
            root=data_dir,
            train=False,
            download=download,
            transform=transform,
        )
    elif dataset_key in {"hf_cifar10", "cifar10_hf", "parquet_cifar10"}:
        train_parquet = data_dir / "plain_text" / "train-00000-of-00001.parquet"
        test_parquet = data_dir / "plain_text" / "test-00000-of-00001.parquet"
        if not train_parquet.exists() or not test_parquet.exists():
            raise FileNotFoundError(
                "Hugging Face CIFAR-10 parquet files were not found under "
                f"{data_dir / 'plain_text'}. Download them from "
                "https://huggingface.co/datasets/uoft-cs/cifar10/tree/main/plain_text"
            )
        train_data = _HFCIFAR10Parquet(train_parquet, transform=transform)
        test_data = _HFCIFAR10Parquet(test_parquet, transform=transform)
    elif dataset_key in {"fake", "fake_cifar", "synthetic"}:
        train_data = datasets.FakeData(
            size=int(train_subset),
            image_size=(3, 32, 32),
            num_classes=10,
            transform=transform,
            random_offset=0,
        )
        test_data = datasets.FakeData(
            size=int(test_subset),
            image_size=(3, 32, 32),
            num_classes=10,
            transform=transform,
            random_offset=int(train_subset),
        )
    else:
        raise ValueError("dataset must be 'cifar10', 'hf-cifar10', or 'fake'.")
    train_indices = list(range(min(int(train_subset), len(train_data))))
    test_indices = list(range(min(int(test_subset), len(test_data))))
    train_loader = DataLoader(
        Subset(train_data, train_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        Subset(test_data, test_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )

    model = build_resnet18_cifar().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=5e-4,
    )

    metrics = []

    def evaluate():
        model.eval()
        correct = 0
        total = 0
        loss_total = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(images)
                loss = criterion(logits, labels)
                loss_total += float(loss.item()) * labels.numel()
                correct += int((logits.argmax(dim=1) == labels).sum().item())
                total += int(labels.numel())
        return loss_total / max(total, 1), correct / max(total, 1)

    def save_checkpoint(epoch):
        path = checkpoint_dir / f"resnet18_cifar10_epoch_{epoch:02d}.pt"
        torch.save(
            {
                "epoch": int(epoch),
                "model_state": {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                },
            },
            path,
        )
        return path

    save_checkpoint(0)
    for epoch in range(1, int(epochs) + 1):
        model.train()
        start = time.perf_counter()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += float(loss.item()) * labels.numel()
            train_correct += int((logits.argmax(dim=1) == labels).sum().item())
            train_total += int(labels.numel())

        if evaluate_each_epoch:
            test_loss, test_accuracy = evaluate()
        else:
            test_loss, test_accuracy = None, None
        checkpoint_path = save_checkpoint(epoch)
        metrics.append(
            {
                "epoch": epoch,
                "device": str(device),
                "train_loss": train_loss / max(train_total, 1),
                "train_accuracy": train_correct / max(train_total, 1),
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "seconds": time.perf_counter() - start,
                "checkpoint_path": str(checkpoint_path),
            }
        )
        pd.DataFrame(metrics).to_csv(output_dir / "training_metrics.csv", index=False)
        if test_accuracy is None:
            print(
                f"epoch={epoch} train_loss={metrics[-1]['train_loss']:.4f}",
                flush=True,
            )
        else:
            print(
                f"epoch={epoch} train_loss={metrics[-1]['train_loss']:.4f} "
                f"test_accuracy={test_accuracy:.4f}",
                flush=True,
            )

    df = pd.DataFrame(metrics)
    df.to_csv(output_dir / "training_metrics.csv", index=False)
    return df


def compress_checkpoint_directory(
    checkpoint_dir,
    output_dir,
    *,
    methods=DEFAULT_METHODS,
    dtype=torch.bfloat16,
):
    checkpoint_dir = Path(checkpoint_dir)
    output_dir = _ensure_dir(output_dir)
    rows = []
    for path in sorted(checkpoint_dir.glob("*.pt")):
        checkpoint = torch.load(path, map_location="cpu")
        state = checkpoint.get("model_state", checkpoint)
        tensors = {
            name: tensor.detach().cpu().contiguous().to(dtype=dtype)
            for name, tensor in state.items()
            if torch.is_tensor(tensor) and tensor.is_floating_point()
        }
        for row in _compress_mapping(path.stem, tensors, methods=methods):
            row["experiment"] = "checkpoints"
            row["checkpoint"] = path.name
            row["epoch"] = int(checkpoint.get("epoch", -1))
            row["dtype"] = str(dtype).replace("torch.", "")
            row["tensor_count"] = len(tensors)
            rows.append(row)
        del checkpoint, state, tensors
        gc.collect()

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "checkpoint_compression_results.csv", index=False)
    return df


def _average_floating_states(states):
    averaged = {}
    first = states[0]
    for key, value in first.items():
        if torch.is_tensor(value) and value.is_floating_point():
            stacked = torch.stack([state[key].to(torch.float32) for state in states])
            averaged[key] = stacked.mean(dim=0).to(dtype=value.dtype)
        else:
            averaged[key] = value.clone() if torch.is_tensor(value) else value
    return averaged


def run_federated_resnet_synthetic(
    output_dir,
    *,
    clients=4,
    rounds=2,
    local_batches=8,
    batch_size=64,
    learning_rate=0.02,
    seed=2026,
    device=None,
):
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    output_dir = _ensure_dir(output_dir)
    checkpoint_dir = _ensure_dir(output_dir / "client_checkpoints")

    torch.manual_seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    criterion = nn.CrossEntropyLoss()
    server = build_resnet18_cifar()
    server_state = {
        key: value.detach().cpu().clone()
        for key, value in server.state_dict().items()
    }
    rows = []

    for round_index in range(1, int(rounds) + 1):
        client_states = []
        for client_index in range(int(clients)):
            model = build_resnet18_cifar().to(device)
            model.load_state_dict(server_state, strict=True)
            model.train()
            optimizer = optim.SGD(
                model.parameters(),
                lr=learning_rate,
                momentum=0.9,
                weight_decay=5e-4,
            )
            data = datasets.FakeData(
                size=int(local_batches) * int(batch_size),
                image_size=(3, 32, 32),
                num_classes=10,
                transform=transform,
                random_offset=round_index * 10000 + client_index * 1000,
            )
            loader = DataLoader(
                data,
                batch_size=batch_size,
                shuffle=True,
                num_workers=2,
                pin_memory=device.type == "cuda",
            )

            start = time.perf_counter()
            total_loss = 0.0
            total_correct = 0
            total_examples = 0
            for images, labels in loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                total_loss += float(loss.item()) * labels.numel()
                total_correct += int((logits.argmax(dim=1) == labels).sum().item())
                total_examples += int(labels.numel())

            client_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            client_states.append(client_state)
            checkpoint_path = (
                checkpoint_dir
                / f"round_{round_index:02d}_client_{client_index:02d}.pt"
            )
            torch.save(
                {
                    "round": round_index,
                    "client": client_index,
                    "model_state": client_state,
                },
                checkpoint_path,
            )
            rows.append(
                {
                    "round": round_index,
                    "client": client_index,
                    "device": str(device),
                    "examples": total_examples,
                    "train_loss": total_loss / max(total_examples, 1),
                    "train_accuracy": total_correct / max(total_examples, 1),
                    "seconds": time.perf_counter() - start,
                    "checkpoint_path": str(checkpoint_path),
                }
            )
            del model, data, loader, client_state
            if device.type == "cuda":
                torch.cuda.empty_cache()

        server_state = _average_floating_states(client_states)
        del client_states
        gc.collect()

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "federated_training_metrics.csv", index=False)
    return df


def compress_federated_directory(
    checkpoint_dir,
    output_dir,
    *,
    methods=FULL_CHECKPOINT_METHODS,
    dtype=torch.bfloat16,
):
    checkpoint_dir = Path(checkpoint_dir)
    output_dir = _ensure_dir(output_dir)
    rows = []
    for path in sorted(checkpoint_dir.glob("round_*_client_*.pt")):
        checkpoint = torch.load(path, map_location="cpu")
        state = checkpoint["model_state"]
        tensors = {
            name: tensor.detach().cpu().contiguous().to(dtype=dtype)
            for name, tensor in state.items()
            if torch.is_tensor(tensor) and tensor.is_floating_point()
        }
        for row in _compress_mapping(path.stem, tensors, methods=methods):
            row["experiment"] = "federated-learning"
            row["checkpoint"] = path.name
            row["round"] = int(checkpoint["round"])
            row["client"] = int(checkpoint["client"])
            row["dtype"] = str(dtype).replace("torch.", "")
            row["tensor_count"] = len(tensors)
            rows.append(row)
        del checkpoint, state, tensors
        gc.collect()

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "federated_compression_results.csv", index=False)
    return df


def compress_federated_directory_amortized(
    checkpoint_dir,
    output_dir,
    *,
    dtype=torch.bfloat16,
    chunk_size=256 << 10,
    compression_threshold=1.0,
    clustering="js-distance",
    cluster_js_threshold=0.05,
    max_codebook_sample_bytes=8 << 20,
    calibration_rounds=1,
):
    checkpoint_dir = Path(checkpoint_dir)
    output_dir = _ensure_dir(output_dir)
    checkpoint_paths = sorted(checkpoint_dir.glob("round_*_client_*.pt"))
    if not checkpoint_paths:
        raise ValueError(f"No federated checkpoints found in {checkpoint_dir}.")

    def load_tensors(path):
        checkpoint = torch.load(path, map_location="cpu")
        state = checkpoint["model_state"]
        tensors = {
            name: tensor.detach().cpu().contiguous().to(dtype=dtype)
            for name, tensor in state.items()
            if torch.is_tensor(tensor) and tensor.is_floating_point()
        }
        return checkpoint, tensors

    if calibration_rounds is None:
        calibration_paths = checkpoint_paths[:1]
    else:
        calibration_paths = [
            path
            for path in checkpoint_paths
            if int(path.name.split("_")[1]) <= int(calibration_rounds)
        ]
    if not calibration_paths:
        calibration_paths = checkpoint_paths[:1]

    codec = NeuralZipCodec(
        chunk_size=chunk_size,
        compression_threshold=compression_threshold,
        clustering=clustering,
        cluster_js_threshold=cluster_js_threshold,
        max_codebook_sample_bytes=max_codebook_sample_bytes,
    )
    calibration_named = []
    for path in calibration_paths:
        calibration_checkpoint, calibration_tensors = load_tensors(path)
        calibration_named.extend(list(calibration_tensors.items()))
        del calibration_checkpoint, calibration_tensors
    setup_timings = _prepare_neuralzip_codec(codec, calibration_named)
    setup_seconds = setup_timings["total_setup_seconds"]
    codebooks = codec.export_codebooks()
    codebook_metadata_bytes = codec.codebook_metadata_bytes()
    total_updates = len(checkpoint_paths)

    rows = []
    for path in checkpoint_paths:
        checkpoint, tensors = load_tensors(path)
        raw_bytes = 0
        payload_bytes = 0
        chunk_metadata_bytes = 0
        huffman_chunks = 0
        raw_chunks = 0
        shared_refs = 0
        start = time.perf_counter()
        for name, tensor in tensors.items():
            compressed = codec.compress_tensor(name, tensor)
            raw_bytes += compressed.raw_bytes
            payload_bytes += compressed.payload_bytes
            chunk_metadata_bytes += compressed.compressed_bytes - compressed.payload_bytes
            huffman_chunks += compressed.huffman_chunks
            raw_chunks += compressed.raw_chunks
            shared_refs += compressed.shared_codebook_refs
        encode_seconds = time.perf_counter() - start
        compressed_bytes = payload_bytes + chunk_metadata_bytes
        amortized_compressed_bytes = (
            compressed_bytes + codebook_metadata_bytes / total_updates
        )
        rows.append(
            {
                "experiment": "federated-learning",
                "checkpoint": path.name,
                "round": int(checkpoint["round"]),
                "client": int(checkpoint["client"]),
                "compressor": "NeuralZip amortized",
                "method": "neuralzip-amortized",
                "scope": "full checkpoint",
                "status": "ok",
                "raw_bytes": raw_bytes,
                "compressed_bytes": compressed_bytes,
                "amortized_compressed_bytes": amortized_compressed_bytes,
                "compression_ratio": compressed_bytes / raw_bytes if raw_bytes else 0.0,
                "amortized_compression_ratio": (
                    amortized_compressed_bytes / raw_bytes if raw_bytes else 0.0
                ),
                "compressed_pct": (
                    100 * compressed_bytes / raw_bytes if raw_bytes else 0.0
                ),
                "amortized_compressed_pct": (
                    100 * amortized_compressed_bytes / raw_bytes
                    if raw_bytes
                    else 0.0
                ),
                "setup_seconds_once": setup_seconds,
                "precompute_seconds_once": setup_timings["precompute_seconds"],
                "codebook_setup_seconds_once": setup_timings[
                    "codebook_setup_seconds"
                ],
                "encode_seconds": encode_seconds,
                "amortized_cold_seconds": encode_seconds
                + setup_seconds / total_updates,
                "clusters": codec.cluster_count,
                "codebooks": len(codebooks),
                "codebook_metadata_bytes_once": codebook_metadata_bytes,
                "shared_huff0_chunks": shared_refs,
                "huffman_chunks": huffman_chunks,
                "raw_chunks": raw_chunks,
                "dtype": str(dtype).replace("torch.", ""),
                "tensor_count": len(tensors),
                "calibration_updates": len(calibration_paths),
                "calibration_rounds": calibration_rounds,
                "calibration_scope": ", ".join(path.name for path in calibration_paths),
            }
        )
        del checkpoint, tensors
        gc.collect()

    del calibration_named
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "federated_amortized_compression_results.csv", index=False)
    return df


def _floating_tensors_from_state(state, *, dtype):
    tensors = {}
    for name, tensor in state.items():
        if not torch.is_tensor(tensor) or not tensor.is_floating_point():
            continue
        value = tensor.detach().cpu().contiguous()
        if dtype is not None:
            value = value.to(dtype=dtype)
        tensors[name] = value
    return tensors


def _initial_resnet18_cifar_tensors(*, dtype, seed=2026):
    torch.manual_seed(seed)
    model = build_resnet18_cifar()
    tensors = _floating_tensors_from_state(model.state_dict(), dtype=dtype)
    del model
    return tensors


def _parse_federated_checkpoint_path(path):
    parts = Path(path).stem.split("_")
    return int(parts[1]), int(parts[3])


def _load_federated_client_tensors(path, *, dtype):
    checkpoint = torch.load(path, map_location="cpu")
    tensors = _floating_tensors_from_state(checkpoint["model_state"], dtype=dtype)
    round_index = int(checkpoint.get("round", _parse_federated_checkpoint_path(path)[0]))
    client_index = int(
        checkpoint.get("client", _parse_federated_checkpoint_path(path)[1])
    )
    return round_index, client_index, tensors


class _HFCIFARParquet(torch.utils.data.Dataset):
    def __init__(self, parquet_path, *, transform=None):
        import io

        from PIL import Image
        import pyarrow.parquet as pq

        schema_names = set(pq.read_schema(parquet_path).names)
        if "label" in schema_names:
            label_column = "label"
        elif "fine_label" in schema_names:
            label_column = "fine_label"
        else:
            raise ValueError(
                f"No CIFAR label column found in {parquet_path}; "
                f"available columns: {sorted(schema_names)}"
            )
        table = pq.read_table(parquet_path, columns=["img", label_column])
        rows = table.to_pylist()
        self._images = [row["img"]["bytes"] for row in rows]
        self._labels = [int(row[label_column]) for row in rows]
        self._transform = transform
        self._image_cls = Image
        self._io = io

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, index):
        image = self._image_cls.open(
            self._io.BytesIO(self._images[int(index)])
        ).convert("RGB")
        if self._transform is not None:
            image = self._transform(image)
        return image, self._labels[int(index)]


_HFCIFAR10Parquet = _HFCIFARParquet


def _codec_compress_tensors(codec, tensors):
    payloads = []
    raw_bytes = 0
    compressed_bytes = 0
    huffman_chunks = 0
    raw_chunks = 0
    shared_refs = 0

    start = time.perf_counter()
    for name, tensor in tensors.items():
        payload = codec.compress_tensor(name, tensor)
        payloads.append(payload)
        raw_bytes += payload.raw_bytes
        compressed_bytes += payload.compressed_bytes
        huffman_chunks += payload.huffman_chunks
        raw_chunks += payload.raw_chunks
        shared_refs += getattr(payload, "shared_codebook_refs", 0)
    seconds = time.perf_counter() - start

    return {
        "payloads": payloads,
        "raw_bytes": raw_bytes,
        "compressed_bytes": compressed_bytes,
        "seconds": seconds,
        "huffman_chunks": huffman_chunks,
        "raw_chunks": raw_chunks,
        "shared_refs": shared_refs,
    }


def _prepare_neuralzip_codec(codec, named_tensors):
    named_tensors = list(named_tensors)
    started = time.perf_counter()
    codec.prepare_clusters(named_tensors)
    precompute_seconds = time.perf_counter() - started

    started = time.perf_counter()
    codec.build_codebooks_and_select_modes(named_tensors)
    codebook_setup_seconds = time.perf_counter() - started
    return {
        "precompute_seconds": precompute_seconds,
        "codebook_setup_seconds": codebook_setup_seconds,
        "total_setup_seconds": precompute_seconds + codebook_setup_seconds,
    }


def _codec_decompress_tensors(codec, payloads):
    start = time.perf_counter()
    tensors = {payload.name: codec.decompress_tensor(payload) for payload in payloads}
    seconds = time.perf_counter() - start
    return tensors, seconds


def _payload_exponent_sizes(payload):
    if not payload.groups:
        return 0, 0
    exponent_group = payload.groups[int(payload.element_size) - 1]
    return int(exponent_group.symbol_count), int(exponent_group.compressed_bytes)


def _assert_tensor_roundtrip(expected, decoded, *, method, checkpoint):
    if expected.keys() != decoded.keys():
        raise AssertionError(
            f"{method} changed the tensor names for checkpoint {checkpoint}."
        )
    for name, tensor in expected.items():
        if not torch.equal(tensor, decoded[name]):
            raise AssertionError(
                f"{method} failed bitwise roundtrip for {checkpoint}:{name}."
            )


def compress_training_checkpoints(
    checkpoint_dir,
    output_dir,
    *,
    architecture,
    dtype=None,
    chunk_size=256 << 10,
    compression_threshold=0.95,
    clustering="js-distance",
    cluster_js_threshold=0.05,
    max_codebook_sample_bytes=8 << 20,
    ngram_values=(2, 3, 4),
    top_k=254,
    packed_source_budget_ratio=0.30,
    verify=True,
    resume=True,
):
    """Compress each checkpoint with a NeuralZip setup fitted to that checkpoint."""

    architecture = _canonical_cifar_architecture(architecture)
    checkpoint_dir = Path(checkpoint_dir)
    output_dir = _ensure_dir(output_dir)
    result_path = output_dir / "checkpoint_compression_results.csv"
    summary_path = output_dir / "checkpoint_compression_summary.csv"
    checkpoint_paths = sorted(
        checkpoint_dir.glob(f"{architecture}_cifar100_epoch_*.pt")
    )
    if not checkpoint_paths:
        raise ValueError(
            f"No {architecture} CIFAR-100 checkpoints found in {checkpoint_dir}."
        )

    zipnn_codec = ZipNNCodec(
        chunk_size=chunk_size,
        compression_threshold=compression_threshold,
    )

    rows = (
        pd.read_csv(result_path).to_dict("records")
        if resume and result_path.exists()
        else []
    )
    if rows and any(row.get("setup_scope") != "per-checkpoint" for row in rows):
        rows = []
    completed = {
        (str(row["method"]), int(row["epoch"]))
        for row in rows
        if str(row.get("status", "ok")) == "ok"
    }

    for checkpoint_path in checkpoint_paths:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        epoch = int(checkpoint["epoch"])
        tensors = _floating_tensors_from_state(checkpoint["model_state"], dtype=dtype)
        protocols = [
            (
                "ZipNN",
                zipnn_codec,
                {
                    "setup_seconds": 0.0,
                    "cluster_setup_seconds": 0.0,
                    "codebook_setup_seconds": 0.0,
                    "setup_bytes": 0,
                    "exponent_codebook_bytes": 0,
                    "clusters": 0,
                    "codebooks": 0,
                },
            )
        ]
        if ("NeuralZip", epoch) not in completed:
            neuralzip_codec = NeuralZipCodec(
                chunk_size=chunk_size,
                compression_threshold=compression_threshold,
                clustering=clustering,
                cluster_js_threshold=cluster_js_threshold,
                max_codebook_sample_bytes=max_codebook_sample_bytes,
                ngram_values=ngram_values,
                top_k=top_k,
                packed_source_budget_ratio=packed_source_budget_ratio,
            )
            neuralzip_setup = _prepare_neuralzip_codec(
                neuralzip_codec,
                tensors.items(),
            )
            codebooks = neuralzip_codec.export_codebooks()
            element_sizes = {"float16": 2, "bfloat16": 2, "float32": 4}
            exponent_codebook_bytes = sum(
                int(codebook.metadata_bytes)
                for codebook in codebooks
                if int(codebook.stream_id)
                == element_sizes[str(codebook.dtype)] - 1
            )
            protocols.append(
                (
                    "NeuralZip",
                    neuralzip_codec,
                    {
                        "setup_seconds": neuralzip_setup["total_setup_seconds"],
                        "cluster_setup_seconds": neuralzip_setup[
                            "precompute_seconds"
                        ],
                        "codebook_setup_seconds": neuralzip_setup[
                            "codebook_setup_seconds"
                        ],
                        "setup_bytes": int(
                            neuralzip_codec.codebook_metadata_bytes()
                        ),
                        "exponent_codebook_bytes": int(
                            exponent_codebook_bytes
                        ),
                        "clusters": int(neuralzip_codec.cluster_count),
                        "codebooks": int(len(codebooks)),
                    },
                )
            )
        for method, codec, setup in protocols:
            if (method, epoch) in completed:
                continue
            encoded = _codec_compress_tensors(codec, tensors)
            exponent_raw_bytes = 0
            exponent_compressed_bytes = 0
            for payload in encoded["payloads"]:
                raw_size, compressed_size = _payload_exponent_sizes(payload)
                exponent_raw_bytes += raw_size
                exponent_compressed_bytes += compressed_size
            exponent_compressed_bytes += int(setup["exponent_codebook_bytes"])
            stored_bytes = int(encoded["compressed_bytes"]) + int(
                setup["setup_bytes"]
            )

            decoded, decode_seconds = _codec_decompress_tensors(
                codec,
                encoded["payloads"],
            )
            if verify:
                _assert_tensor_roundtrip(
                    tensors,
                    decoded,
                    method=method,
                    checkpoint=checkpoint_path.name,
                )
            row = {
                "experiment": "training-checkpoints",
                "architecture": architecture,
                "checkpoint": checkpoint_path.name,
                "epoch": epoch,
                "method": method,
                "setup_scope": "per-checkpoint",
                "dtype": (
                    str(dtype).replace("torch.", "")
                    if dtype is not None
                    else "checkpoint"
                ),
                "tensor_count": len(tensors),
                "raw_bytes": int(encoded["raw_bytes"]),
                "payload_compressed_bytes": int(encoded["compressed_bytes"]),
                "stored_bytes": stored_bytes,
                "compression_ratio_pct": (
                    100 * stored_bytes / encoded["raw_bytes"]
                    if encoded["raw_bytes"]
                    else 0.0
                ),
                "exponent_raw_bytes": int(exponent_raw_bytes),
                "exponent_compressed_bytes": int(exponent_compressed_bytes),
                "exponent_compression_ratio_pct": (
                    100 * exponent_compressed_bytes / exponent_raw_bytes
                    if exponent_raw_bytes
                    else 0.0
                ),
                "encode_seconds": float(encoded["seconds"]),
                "total_compression_seconds": float(encoded["seconds"])
                + float(setup["setup_seconds"]),
                "decode_seconds": float(decode_seconds),
                "huffman_chunks": int(encoded["huffman_chunks"]),
                "raw_chunks": int(encoded["raw_chunks"]),
                "shared_codebook_refs": int(encoded["shared_refs"]),
                "roundtrip_exact": True,
                "status": "ok",
                **setup,
            }
            rows = [
                existing
                for existing in rows
                if not (
                    str(existing["method"]) == method
                    and int(existing["epoch"]) == epoch
                )
            ]
            rows.append(row)
            rows.sort(key=lambda item: (int(item["epoch"]), str(item["method"])))
            pd.DataFrame(rows).to_csv(result_path, index=False)
            print(
                f"{architecture} checkpoint={epoch} method={method} "
                f"ratio={row['compression_ratio_pct']:.3f}% "
                f"exponents={row['exponent_compression_ratio_pct']:.3f}% "
                f"setup={row['setup_seconds']:.3f}s "
                f"encode={row['encode_seconds']:.3f}s",
                flush=True,
            )
            del decoded, encoded
        del checkpoint, tensors
        gc.collect()

    result_df = pd.DataFrame(rows)
    summary = (
        result_df.groupby("method", as_index=False)
        .agg(
            architecture=("architecture", "first"),
            checkpoints=("epoch", "count"),
            setup_seconds=("setup_seconds", "sum"),
            setup_bytes=("setup_bytes", "sum"),
            clusters_mean=("clusters", "mean"),
            codebooks_mean=("codebooks", "mean"),
            raw_bytes=("raw_bytes", "sum"),
            payload_compressed_bytes=("payload_compressed_bytes", "sum"),
            stored_bytes=("stored_bytes", "sum"),
            exponent_raw_bytes=("exponent_raw_bytes", "sum"),
            exponent_compressed_bytes=("exponent_compressed_bytes", "sum"),
            encode_seconds=("encode_seconds", "sum"),
            total_compression_seconds=("total_compression_seconds", "sum"),
            decode_seconds=("decode_seconds", "sum"),
        )
    )
    summary["compression_ratio_pct"] = (
        100 * summary["stored_bytes"] / summary["raw_bytes"]
    )
    summary["exponent_compression_ratio_pct"] = (
        100
        * summary["exponent_compressed_bytes"]
        / summary["exponent_raw_bytes"]
    )
    summary.to_csv(summary_path, index=False)
    return result_df, summary


def run_federated_codec_protocol(
    checkpoint_dir,
    output_dir,
    *,
    training_metrics_path=None,
    dtype=torch.bfloat16,
    seed=2026,
    chunk_size=256 << 10,
    compression_threshold=1.0,
    clustering="js-distance",
    cluster_js_threshold=0.05,
    max_codebook_sample_bytes=8 << 20,
):
    """Benchmark FedZipNN vs FedNeuralZip as communication protocols.

    The saved checkpoints represent the client updates produced after local
    training. For each round we measure:
    server downlink compression, client downlink decompression, client uplink
    compression, server uplink decompression, and FedAvg aggregation. Training
    time is read from ``federated_training_metrics.csv`` when available.
    """

    checkpoint_dir = Path(checkpoint_dir)
    output_dir = _ensure_dir(output_dir)
    checkpoint_paths = sorted(checkpoint_dir.glob("round_*_client_*.pt"))
    if not checkpoint_paths:
        raise ValueError(f"No federated checkpoints found in {checkpoint_dir}.")

    round_paths = {}
    for path in checkpoint_paths:
        round_index, client_index = _parse_federated_checkpoint_path(path)
        round_paths.setdefault(round_index, {})[client_index] = path

    if training_metrics_path is None:
        training_metrics_path = checkpoint_dir.parent / "federated_training_metrics.csv"
    training_metrics_path = Path(training_metrics_path)
    training_metrics = None
    if training_metrics_path.exists():
        training_metrics = pd.read_csv(training_metrics_path)

    try:
        initial_server = _initial_resnet18_cifar_tensors(dtype=dtype, seed=seed)
    except ModuleNotFoundError:
        first_round = min(round_paths)
        first_round_states = [
            _load_federated_client_tensors(path, dtype=dtype)[2]
            for path in round_paths[first_round].values()
        ]
        initial_server = _average_floating_states(first_round_states)
        del first_round_states

    protocols = []
    protocols.append(
        {
            "name": "FedZipNN",
            "codec": ZipNNCodec(
                chunk_size=chunk_size,
                compression_threshold=compression_threshold,
            ),
            "setup_seconds": 0.0,
            "setup_bytes": 0,
            "clusters": 0,
            "codebooks": 0,
        }
    )

    neuralzip_codec = NeuralZipCodec(
        chunk_size=chunk_size,
        compression_threshold=compression_threshold,
        clustering=clustering,
        cluster_js_threshold=cluster_js_threshold,
        max_codebook_sample_bytes=max_codebook_sample_bytes,
    )
    neuralzip_setup = _prepare_neuralzip_codec(
        neuralzip_codec, initial_server.items()
    )
    neuralzip_setup_seconds = neuralzip_setup["total_setup_seconds"]
    neuralzip_codebooks = neuralzip_codec.export_codebooks()
    protocols.append(
        {
            "name": "FedNeuralZip",
            "codec": neuralzip_codec,
            "setup_seconds": neuralzip_setup_seconds,
            "precompute_seconds": neuralzip_setup["precompute_seconds"],
            "codebook_setup_seconds": neuralzip_setup["codebook_setup_seconds"],
            "setup_bytes": neuralzip_codec.codebook_metadata_bytes(),
            "clusters": neuralzip_codec.cluster_count,
            "codebooks": len(neuralzip_codebooks),
        }
    )

    rows = []
    for protocol in protocols:
        server_tensors = {
            name: tensor.detach().cpu().contiguous().clone()
            for name, tensor in initial_server.items()
        }
        codec = protocol["codec"]

        for round_index in sorted(round_paths):
            client_paths = round_paths[round_index]
            client_count = len(client_paths)

            if training_metrics is not None:
                round_training = training_metrics[
                    training_metrics["round"] == round_index
                ]
                train_client_sum_seconds = float(round_training["seconds"].sum())
                train_wall_proxy_seconds = float(round_training["seconds"].max())
            else:
                train_client_sum_seconds = 0.0
                train_wall_proxy_seconds = 0.0

            downlink = _codec_compress_tensors(codec, server_tensors)

            downlink_decode_seconds = []
            for _ in range(client_count):
                _, seconds = _codec_decompress_tensors(codec, downlink["payloads"])
                downlink_decode_seconds.append(seconds)
            downlink_decode_sum = float(sum(downlink_decode_seconds))
            downlink_decode_wall = float(max(downlink_decode_seconds, default=0.0))

            uplink_encode_seconds = []
            uplink_decode_seconds = []
            uplink_bytes = 0
            uplink_raw_bytes = 0
            client_states = []
            huffman_chunks = downlink["huffman_chunks"]
            raw_chunks = downlink["raw_chunks"]
            shared_refs = downlink["shared_refs"]

            for client_index in sorted(client_paths):
                _, _, client_tensors = _load_federated_client_tensors(
                    client_paths[client_index],
                    dtype=dtype,
                )
                uplink = _codec_compress_tensors(codec, client_tensors)
                decoded, decode_seconds = _codec_decompress_tensors(
                    codec,
                    uplink["payloads"],
                )
                client_states.append(decoded)
                uplink_encode_seconds.append(float(uplink["seconds"]))
                uplink_decode_seconds.append(float(decode_seconds))
                uplink_bytes += int(uplink["compressed_bytes"])
                uplink_raw_bytes += int(uplink["raw_bytes"])
                huffman_chunks += int(uplink["huffman_chunks"])
                raw_chunks += int(uplink["raw_chunks"])
                shared_refs += int(uplink["shared_refs"])

            start = time.perf_counter()
            server_tensors = _average_floating_states(client_states)
            aggregation_seconds = time.perf_counter() - start

            uplink_encode_sum = float(sum(uplink_encode_seconds))
            uplink_decode_sum = float(sum(uplink_decode_seconds))
            uplink_encode_wall = float(max(uplink_encode_seconds, default=0.0))
            codec_serial_seconds = (
                downlink["seconds"]
                + downlink_decode_sum
                + uplink_encode_sum
                + uplink_decode_sum
            )
            codec_wall_proxy_seconds = (
                downlink["seconds"]
                + downlink_decode_wall
                + uplink_encode_wall
                + uplink_decode_sum
            )
            total_serial_seconds = (
                codec_serial_seconds
                + train_client_sum_seconds
                + aggregation_seconds
            )
            total_wall_proxy_seconds = (
                codec_wall_proxy_seconds
                + train_wall_proxy_seconds
                + aggregation_seconds
            )
            message_bytes = downlink["compressed_bytes"] * client_count + uplink_bytes
            raw_message_bytes = downlink["raw_bytes"] * client_count + uplink_raw_bytes

            rows.append(
                {
                    "protocol": protocol["name"],
                    "round": round_index,
                    "clients": client_count,
                    "dtype": str(dtype).replace("torch.", ""),
                    "setup_seconds_once": protocol["setup_seconds"],
                    "precompute_seconds_once": protocol.get(
                        "precompute_seconds", 0.0
                    ),
                    "codebook_setup_seconds_once": protocol.get(
                        "codebook_setup_seconds", 0.0
                    ),
                    "setup_bytes_once": protocol["setup_bytes"],
                    "clusters": protocol["clusters"],
                    "codebooks": protocol["codebooks"],
                    "train_client_sum_seconds": train_client_sum_seconds,
                    "train_wall_proxy_seconds": train_wall_proxy_seconds,
                    "downlink_encode_seconds": float(downlink["seconds"]),
                    "downlink_decode_sum_seconds": downlink_decode_sum,
                    "downlink_decode_wall_proxy_seconds": downlink_decode_wall,
                    "uplink_encode_sum_seconds": uplink_encode_sum,
                    "uplink_encode_wall_proxy_seconds": uplink_encode_wall,
                    "uplink_decode_sum_seconds": uplink_decode_sum,
                    "aggregation_seconds": aggregation_seconds,
                    "codec_serial_seconds": codec_serial_seconds,
                    "codec_wall_proxy_seconds": codec_wall_proxy_seconds,
                    "total_serial_seconds": total_serial_seconds,
                    "total_wall_proxy_seconds": total_wall_proxy_seconds,
                    "downlink_bytes": int(downlink["compressed_bytes"]),
                    "uplink_bytes": int(uplink_bytes),
                    "message_bytes": int(message_bytes),
                    "raw_message_bytes": int(raw_message_bytes),
                    "message_compressed_pct": (
                        100 * message_bytes / raw_message_bytes
                        if raw_message_bytes
                        else 0.0
                    ),
                    "huffman_chunks": int(huffman_chunks),
                    "raw_chunks": int(raw_chunks),
                    "shared_codebook_refs": int(shared_refs),
                }
            )

        del server_tensors
        gc.collect()

    rounds = len(round_paths)
    round_df = pd.DataFrame(rows)
    summary = (
        round_df.groupby("protocol", as_index=False)
        .agg(
            rounds=("round", "count"),
            clients=("clients", "mean"),
            setup_seconds_once=("setup_seconds_once", "first"),
            setup_bytes_once=("setup_bytes_once", "first"),
            clusters=("clusters", "first"),
            codebooks=("codebooks", "first"),
            codec_serial_seconds=("codec_serial_seconds", "sum"),
            codec_wall_proxy_seconds=("codec_wall_proxy_seconds", "sum"),
            train_client_sum_seconds=("train_client_sum_seconds", "sum"),
            train_wall_proxy_seconds=("train_wall_proxy_seconds", "sum"),
            aggregation_seconds=("aggregation_seconds", "sum"),
            total_serial_seconds=("total_serial_seconds", "sum"),
            total_wall_proxy_seconds=("total_wall_proxy_seconds", "sum"),
            message_bytes=("message_bytes", "sum"),
            raw_message_bytes=("raw_message_bytes", "sum"),
            huffman_chunks=("huffman_chunks", "sum"),
            raw_chunks=("raw_chunks", "sum"),
            shared_codebook_refs=("shared_codebook_refs", "sum"),
        )
    )
    summary["amortized_total_serial_seconds"] = (
        summary["total_serial_seconds"] + summary["setup_seconds_once"]
    )
    summary["amortized_total_wall_proxy_seconds"] = (
        summary["total_wall_proxy_seconds"] + summary["setup_seconds_once"]
    )
    summary["avg_codec_wall_proxy_seconds"] = (
        summary["codec_wall_proxy_seconds"] / rounds
    )
    summary["avg_total_wall_proxy_seconds"] = (
        summary["total_wall_proxy_seconds"] / rounds
    )
    summary["message_compressed_pct"] = (
        100 * summary["message_bytes"] / summary["raw_message_bytes"]
    )

    round_df.to_csv(output_dir / "federated_protocol_round_results.csv", index=False)
    summary.to_csv(output_dir / "federated_protocol_summary.csv", index=False)
    return round_df, summary


def run_federated_codec_protocol_online(
    output_dir,
    *,
    architecture="resnet18",
    clients=10,
    rounds=200,
    local_batches=1,
    batch_size=64,
    learning_rate=0.02,
    dtype=None,
    seed=2026,
    dataset="hf-cifar100",
    data_dir=None,
    download=False,
    device=None,
    chunk_size=256 << 10,
    compression_threshold=0.95,
    clustering="js-distance",
    cluster_js_threshold=0.05,
    max_codebook_sample_bytes=8 << 20,
    num_workers=0,
    progress_every=10,
    train_subset=None,
    amp=True,
    verify=True,
):
    """Run a FedZipNN vs FedNeuralZip protocol benchmark online.

    This version does not save per-client checkpoints. It executes the FedAvg
    loop and measures the communication codec at each round:

    server compress -> clients decompress -> clients train -> clients compress
    -> server decompress -> server average.
    """

    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Subset
    architecture = _canonical_cifar_architecture(architecture)

    output_dir = _ensure_dir(output_dir)
    compression_dir = _ensure_dir(output_dir / "compression")
    if data_dir is None:
        data_dir = output_dir / "data"
    data_dir = _ensure_dir(data_dir)

    torch.manual_seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    criterion = nn.CrossEntropyLoss()
    train_data, _, num_classes = _load_cifar_datasets(
        dataset,
        data_dir,
        download=download,
        include_test=False,
    )
    train_data = _limited_dataset(train_data, train_subset)

    split_generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(train_data), generator=split_generator).tolist()
    client_indices = [indices[index :: int(clients)] for index in range(int(clients))]
    client_loaders = [
        DataLoader(
            Subset(train_data, client_indices[client_index]),
            batch_size=batch_size,
            shuffle=True,
            num_workers=int(num_workers),
            pin_memory=device.type == "cuda",
        )
        for client_index in range(int(clients))
    ]
    client_iterators = [iter(loader) for loader in client_loaders]

    def next_client_batches(client_index):
        batches = []
        while len(batches) < int(local_batches):
            try:
                batch = next(client_iterators[client_index])
            except StopIteration:
                client_iterators[client_index] = iter(client_loaders[client_index])
                batch = next(client_iterators[client_index])
            batches.append(batch)
        return batches

    server = build_cifar_model(architecture, num_classes=num_classes)
    server_state = {
        key: value.detach().cpu().clone()
        for key, value in server.state_dict().items()
    }
    del server

    initial_tensors = _floating_tensors_from_state(server_state, dtype=dtype)

    protocols = [
        {
            "name": "ZipNN",
            "codec": ZipNNCodec(
                chunk_size=chunk_size,
                compression_threshold=compression_threshold,
            ),
            "setup_seconds": 0.0,
            "setup_bytes": 0,
            "clusters": 0,
            "codebooks": 0,
        }
    ]

    neuralzip_codec = NeuralZipCodec(
        chunk_size=chunk_size,
        compression_threshold=compression_threshold,
        clustering=clustering,
        cluster_js_threshold=cluster_js_threshold,
        max_codebook_sample_bytes=max_codebook_sample_bytes,
    )
    neuralzip_setup = _prepare_neuralzip_codec(
        neuralzip_codec, initial_tensors.items()
    )
    neuralzip_setup_seconds = neuralzip_setup["total_setup_seconds"]
    neuralzip_codebooks = neuralzip_codec.export_codebooks()
    protocols.append(
        {
            "name": "NeuralZip",
            "codec": neuralzip_codec,
            "setup_seconds": neuralzip_setup_seconds,
            "precompute_seconds": neuralzip_setup["precompute_seconds"],
            "codebook_setup_seconds": neuralzip_setup["codebook_setup_seconds"],
            "setup_bytes": neuralzip_codec.codebook_metadata_bytes(),
            "clusters": neuralzip_codec.cluster_count,
            "codebooks": len(neuralzip_codebooks),
        }
    )

    rows = []
    for round_index in range(1, int(rounds) + 1):
        server_tensors = _floating_tensors_from_state(server_state, dtype=dtype)
        downlink_metrics = {}
        for protocol in protocols:
            downlink = _codec_compress_tensors(protocol["codec"], server_tensors)
            decoded_downlink, downlink_decode_seconds = _codec_decompress_tensors(
                protocol["codec"],
                downlink["payloads"],
            )
            if verify:
                _assert_tensor_roundtrip(
                    server_tensors,
                    decoded_downlink,
                    method=protocol["name"],
                    checkpoint=f"round-{round_index}-downlink",
                )
            downlink_metrics[protocol["name"]] = {
                "encode_seconds": float(downlink["seconds"]),
                "decode_one_seconds": float(downlink_decode_seconds),
                "raw_bytes": int(downlink["raw_bytes"]),
                "compressed_bytes": int(downlink["compressed_bytes"]),
                "huffman_chunks": int(downlink["huffman_chunks"]),
                "raw_chunks": int(downlink["raw_chunks"]),
                "shared_refs": int(downlink["shared_refs"]),
            }
            del decoded_downlink, downlink

        client_states = []
        train_seconds = []
        for client_index in range(int(clients)):
            model = build_cifar_model(
                architecture,
                num_classes=num_classes,
            ).to(device)
            model.load_state_dict(server_state, strict=True)
            model.train()
            if architecture == "vit_b_16":
                optimizer = optim.AdamW(
                    model.parameters(),
                    lr=float(learning_rate),
                    weight_decay=0.05,
                )
            else:
                optimizer = optim.SGD(
                    model.parameters(),
                    lr=float(learning_rate),
                    momentum=0.9,
                    weight_decay=5e-4,
                )
            amp_enabled = bool(amp and device.type == "cuda")
            scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
            start = time.perf_counter()
            for images, labels in next_client_batches(client_index):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=amp_enabled,
                ):
                    logits = model(images)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            train_seconds.append(time.perf_counter() - start)
            client_states.append(
                {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
            )
            del model, optimizer, scaler

        for protocol in protocols:
            codec = protocol["codec"]
            uplink_encode_seconds = []
            uplink_decode_seconds = []
            uplink_bytes = 0
            uplink_raw_bytes = 0
            huffman_chunks = downlink_metrics[protocol["name"]]["huffman_chunks"]
            raw_chunks = downlink_metrics[protocol["name"]]["raw_chunks"]
            shared_refs = downlink_metrics[protocol["name"]]["shared_refs"]

            for client_state in client_states:
                client_tensors = _floating_tensors_from_state(client_state, dtype=dtype)
                uplink = _codec_compress_tensors(codec, client_tensors)
                decoded_uplink, decode_seconds = _codec_decompress_tensors(
                    codec,
                    uplink["payloads"],
                )
                if verify:
                    _assert_tensor_roundtrip(
                        client_tensors,
                        decoded_uplink,
                        method=protocol["name"],
                        checkpoint=f"round-{round_index}-uplink",
                    )
                uplink_encode_seconds.append(float(uplink["seconds"]))
                uplink_decode_seconds.append(float(decode_seconds))
                uplink_bytes += int(uplink["compressed_bytes"])
                uplink_raw_bytes += int(uplink["raw_bytes"])
                huffman_chunks += int(uplink["huffman_chunks"])
                raw_chunks += int(uplink["raw_chunks"])
                shared_refs += int(uplink["shared_refs"])
                del client_tensors, decoded_uplink, uplink

            downlink = downlink_metrics[protocol["name"]]
            downlink_decode_sum = downlink["decode_one_seconds"] * int(clients)
            downlink_decode_wall = downlink["decode_one_seconds"]
            uplink_encode_sum = float(sum(uplink_encode_seconds))
            uplink_decode_sum = float(sum(uplink_decode_seconds))
            uplink_encode_wall = float(max(uplink_encode_seconds, default=0.0))
            codec_serial_seconds = (
                downlink["encode_seconds"]
                + downlink_decode_sum
                + uplink_encode_sum
                + uplink_decode_sum
            )
            codec_wall_proxy_seconds = (
                downlink["encode_seconds"]
                + downlink_decode_wall
                + uplink_encode_wall
                + uplink_decode_sum
            )
            message_bytes = downlink["compressed_bytes"] * int(clients) + uplink_bytes
            raw_message_bytes = downlink["raw_bytes"] * int(clients) + uplink_raw_bytes

            rows.append(
                {
                    "architecture": architecture,
                    "dataset": str(dataset),
                    "protocol": protocol["name"],
                    "round": round_index,
                    "clients": int(clients),
                    "local_batches": int(local_batches),
                    "batch_size": int(batch_size),
                    "device": str(device),
                    "dtype": (
                        str(dtype).replace("torch.", "")
                        if dtype is not None
                        else "model-state"
                    ),
                    "setup_seconds_once": protocol["setup_seconds"],
                    "precompute_seconds_once": protocol.get(
                        "precompute_seconds", 0.0
                    ),
                    "codebook_setup_seconds_once": protocol.get(
                        "codebook_setup_seconds", 0.0
                    ),
                    "setup_bytes_once": protocol["setup_bytes"],
                    "clusters": protocol["clusters"],
                    "codebooks": protocol["codebooks"],
                    "train_client_sum_seconds": float(sum(train_seconds)),
                    "train_wall_proxy_seconds": float(max(train_seconds, default=0.0)),
                    "downlink_encode_seconds": downlink["encode_seconds"],
                    "downlink_decode_sum_seconds": downlink_decode_sum,
                    "downlink_decode_wall_proxy_seconds": downlink_decode_wall,
                    "uplink_encode_sum_seconds": uplink_encode_sum,
                    "uplink_encode_wall_proxy_seconds": uplink_encode_wall,
                    "uplink_decode_sum_seconds": uplink_decode_sum,
                    "codec_serial_seconds": codec_serial_seconds,
                    "codec_wall_proxy_seconds": codec_wall_proxy_seconds,
                    "total_serial_seconds": codec_serial_seconds
                    + float(sum(train_seconds)),
                    "total_wall_proxy_seconds": codec_wall_proxy_seconds
                    + float(max(train_seconds, default=0.0)),
                    "downlink_bytes": downlink["compressed_bytes"],
                    "uplink_bytes": int(uplink_bytes),
                    "message_bytes": int(message_bytes),
                    "raw_message_bytes": int(raw_message_bytes),
                    "message_compressed_pct": (
                        100 * message_bytes / raw_message_bytes
                        if raw_message_bytes
                        else 0.0
                    ),
                    "huffman_chunks": int(huffman_chunks),
                    "raw_chunks": int(raw_chunks),
                    "shared_codebook_refs": int(shared_refs),
                }
            )

        start = time.perf_counter()
        server_state = _average_floating_states(client_states)
        aggregation_seconds = time.perf_counter() - start
        for row in rows[-len(protocols) :]:
            row["aggregation_seconds"] = aggregation_seconds
            row["total_serial_seconds"] += aggregation_seconds
            row["total_wall_proxy_seconds"] += aggregation_seconds

        if device.type == "cuda":
            torch.cuda.empty_cache()
        del client_states, server_tensors

        if progress_every and round_index % int(progress_every) == 0:
            partial = pd.DataFrame(rows)
            partial.to_csv(
                compression_dir / "federated_protocol_online_round_results.csv",
                index=False,
            )
            print(
                f"finished round {round_index}/{rounds}",
                flush=True,
            )

    round_df = pd.DataFrame(rows)
    summary = (
        round_df.groupby("protocol", as_index=False)
        .agg(
            architecture=("architecture", "first"),
            dataset=("dataset", "first"),
            rounds=("round", "count"),
            clients=("clients", "mean"),
            local_batches=("local_batches", "first"),
            batch_size=("batch_size", "first"),
            setup_seconds_once=("setup_seconds_once", "first"),
            setup_bytes_once=("setup_bytes_once", "first"),
            clusters=("clusters", "first"),
            codebooks=("codebooks", "first"),
            codec_serial_seconds=("codec_serial_seconds", "sum"),
            codec_wall_proxy_seconds=("codec_wall_proxy_seconds", "sum"),
            train_client_sum_seconds=("train_client_sum_seconds", "sum"),
            train_wall_proxy_seconds=("train_wall_proxy_seconds", "sum"),
            aggregation_seconds=("aggregation_seconds", "sum"),
            total_serial_seconds=("total_serial_seconds", "sum"),
            total_wall_proxy_seconds=("total_wall_proxy_seconds", "sum"),
            message_bytes=("message_bytes", "sum"),
            raw_message_bytes=("raw_message_bytes", "sum"),
            huffman_chunks=("huffman_chunks", "sum"),
            raw_chunks=("raw_chunks", "sum"),
            shared_codebook_refs=("shared_codebook_refs", "sum"),
        )
    )
    summary["amortized_total_serial_seconds"] = (
        summary["total_serial_seconds"] + summary["setup_seconds_once"]
    )
    summary["amortized_total_wall_proxy_seconds"] = (
        summary["total_wall_proxy_seconds"] + summary["setup_seconds_once"]
    )
    summary["avg_codec_wall_proxy_seconds"] = (
        summary["codec_wall_proxy_seconds"] / summary["rounds"]
    )
    summary["avg_total_wall_proxy_seconds"] = (
        summary["total_wall_proxy_seconds"] / summary["rounds"]
    )
    summary["message_compressed_pct"] = (
        100 * summary["message_bytes"] / summary["raw_message_bytes"]
    )

    round_df.to_csv(
        compression_dir / "federated_protocol_online_round_results.csv",
        index=False,
    )
    summary.to_csv(
        compression_dir / "federated_protocol_online_summary.csv",
        index=False,
    )
    return round_df, summary


def run_checkpoint_experiment_suite(
    output_dir,
    *,
    architectures=_CIFAR_ARCHITECTURES,
    data_dir,
    epochs=200,
    batch_size=128,
    seed=2026,
    device=None,
    num_workers=4,
    train_subset=None,
    test_subset=None,
    verify=True,
):
    """Train and compress the full CIFAR-100 checkpoint suite."""

    output_dir = _ensure_dir(output_dir)
    training_frames = []
    result_frames = []
    summary_frames = []
    for architecture in architectures:
        architecture = _canonical_cifar_architecture(architecture)
        architecture_dir = _ensure_dir(output_dir / architecture)
        training = train_cifar_checkpoints(
            architecture_dir,
            architecture=architecture,
            epochs=epochs,
            train_subset=train_subset,
            test_subset=test_subset,
            batch_size=batch_size,
            seed=seed,
            download=False,
            dataset="hf-cifar100",
            data_dir=data_dir,
            evaluate_each_epoch=True,
            device=device,
            num_workers=num_workers,
            resume=True,
        )
        results, summary = compress_training_checkpoints(
            architecture_dir / "checkpoints",
            architecture_dir / "compression",
            architecture=architecture,
            dtype=None,
            verify=verify,
            resume=True,
        )
        training_frames.append(training)
        result_frames.append(results)
        summary_frames.append(summary)
        pd.concat(training_frames, ignore_index=True).to_csv(
            output_dir / "training_metrics.csv",
            index=False,
        )
        pd.concat(result_frames, ignore_index=True).to_csv(
            output_dir / "checkpoint_compression_results.csv",
            index=False,
        )
        pd.concat(summary_frames, ignore_index=True).to_csv(
            output_dir / "checkpoint_compression_summary.csv",
            index=False,
        )

    return (
        pd.concat(training_frames, ignore_index=True),
        pd.concat(result_frames, ignore_index=True),
        pd.concat(summary_frames, ignore_index=True),
    )


def run_distributed_experiment_suite(
    output_dir,
    *,
    architectures=_CIFAR_ARCHITECTURES,
    data_dir,
    clients=10,
    rounds=200,
    local_batches=1,
    batch_size=64,
    seed=2026,
    device=None,
    num_workers=2,
    train_subset=None,
    verify=True,
):
    """Run the online CIFAR-100 communication suite architecture by architecture."""

    output_dir = _ensure_dir(output_dir)
    round_frames = []
    summary_frames = []
    for architecture in architectures:
        architecture = _canonical_cifar_architecture(architecture)
        learning_rate = 3e-4 if architecture == "vit_b_16" else 0.02
        architecture_dir = _ensure_dir(output_dir / architecture)
        round_df, summary = run_federated_codec_protocol_online(
            architecture_dir,
            architecture=architecture,
            clients=clients,
            rounds=rounds,
            local_batches=local_batches,
            batch_size=batch_size,
            learning_rate=learning_rate,
            dtype=None,
            seed=seed,
            dataset="hf-cifar100",
            data_dir=data_dir,
            download=False,
            device=device,
            compression_threshold=0.95,
            num_workers=num_workers,
            progress_every=1,
            train_subset=train_subset,
            verify=verify,
        )
        round_frames.append(round_df)
        summary_frames.append(summary)
        pd.concat(round_frames, ignore_index=True).to_csv(
            output_dir / "federated_protocol_round_results.csv",
            index=False,
        )
        pd.concat(summary_frames, ignore_index=True).to_csv(
            output_dir / "federated_protocol_summary.csv",
            index=False,
        )

    return (
        pd.concat(round_frames, ignore_index=True),
        pd.concat(summary_frames, ignore_index=True),
    )
