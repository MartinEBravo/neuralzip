"""Model-loading experiment implementations kept for notebook compatibility."""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

from .codec_protocol import (
    CODEC_PROTOCOL_VERSION,
    is_current_codec_protocol,
)
from .compressor import Compressor


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
    for file_path in files:
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


def _encoder_path_for_method(method):
    key = str(method).strip().lower().replace("_", "-")
    if key in {"zipnn-src", "zipnn-source", "official-zipnn"}:
        return "zipnn-src"
    if key in {"neuralzip", "neural-zip"}:
        return "compress_tensors"
    return "compress_tensor"


def _compress_mapping(model_name, tensor_mapping, *, methods=DEFAULT_METHODS):
    rows = []
    wrapped = MappingModel(model_name=model_name, model=tensor_mapping)
    for compressor_name, method in methods:
        start = time.perf_counter()
        try:
            result = Compressor(method, verify=False).compress(wrapped)
            elapsed = time.perf_counter() - start
            exponent_metrics = _result_exponent_metrics(result)
            encode_seconds = result.get("encode_seconds")
            compression_seconds = result.get("compression_seconds")
            if compression_seconds is None:
                compression_seconds = (
                    encode_seconds if encode_seconds is not None else elapsed
                )
            total_compression_seconds = result.get("total_compression_seconds")
            if total_compression_seconds is None:
                total_compression_seconds = compression_seconds
            cold_encode_seconds = result.get("cold_encode_seconds")
            if cold_encode_seconds is None:
                cold_encode_seconds = total_compression_seconds
            rows.append(
                {
                    "model": model_name,
                    "compressor": compressor_name,
                    "method": result.get("method"),
                    "codec_protocol_version": CODEC_PROTOCOL_VERSION,
                    "encoder_path": _encoder_path_for_method(
                        result.get("method", method)
                    ),
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
                    "encode_seconds": encode_seconds,
                    "compression_seconds": compression_seconds,
                    "total_compression_seconds": total_compression_seconds,
                    "cold_encode_seconds": cold_encode_seconds,
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
                    "codec_protocol_version": CODEC_PROTOCOL_VERSION,
                    "encoder_path": _encoder_path_for_method(method),
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
            except Exception:
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
        and is_current_codec_protocol(row.get("codec_protocol_version"))
        and row.get("encoder_path")
        == _encoder_path_for_method(row.get("method"))
    }
    return all(name in completed for name, _ in methods)


def _upsert_model_compressor_row(rows, row):
    key = (row.get("model_name"), row.get("compressor"))
    rows[:] = [
        existing
        for existing in rows
        if (existing.get("model_name"), existing.get("compressor")) != key
    ]
    rows.append(row)


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
                _upsert_model_compressor_row(rows, row)

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
