from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

from .codec_protocol import (
    CODEC_PROTOCOL_VERSION,
    assert_tensor_roundtrip,
    decode_tensor_batch,
    encode_tensor_batch,
    is_current_codec_protocol,
    payload_exponent_sizes,
    prepare_neuralzip_codec,
)
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
    expected_encoder_paths = {
        "ZipNN": "compress_tensor",
        "NeuralZip": "compress_tensors",
    }
    rows = [
        row
        for row in rows
        if is_current_codec_protocol(row.get("codec_protocol_version"))
        and row.get("encoder_path")
        == expected_encoder_paths.get(str(row.get("method")))
    ]
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
            neuralzip_setup = prepare_neuralzip_codec(
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
                        "setup_seconds": neuralzip_setup.total_setup_seconds,
                        "cluster_setup_seconds": neuralzip_setup.precompute_seconds,
                        "codebook_setup_seconds": (
                            neuralzip_setup.codebook_setup_seconds
                        ),
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
            encoded = encode_tensor_batch(codec, tensors)
            exponent_raw_bytes = 0
            exponent_compressed_bytes = 0
            for payload in encoded.payloads:
                raw_size, compressed_size = payload_exponent_sizes(payload)
                exponent_raw_bytes += raw_size
                exponent_compressed_bytes += compressed_size
            exponent_compressed_bytes += int(setup["exponent_codebook_bytes"])
            stored_bytes = int(encoded.compressed_bytes) + int(
                setup["setup_bytes"]
            )

            decoded, decode_seconds = decode_tensor_batch(
                codec,
                encoded.payloads,
            )
            if verify:
                assert_tensor_roundtrip(
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
                "codec_protocol_version": CODEC_PROTOCOL_VERSION,
                "encoder_path": encoded.encoder_path,
                "setup_scope": "per-checkpoint",
                "dtype": (
                    str(dtype).replace("torch.", "")
                    if dtype is not None
                    else "checkpoint"
                ),
                "tensor_count": len(tensors),
                "raw_bytes": int(encoded.raw_bytes),
                "payload_compressed_bytes": int(encoded.compressed_bytes),
                "stored_bytes": stored_bytes,
                "compression_ratio_pct": (
                    100 * stored_bytes / encoded.raw_bytes
                    if encoded.raw_bytes
                    else 0.0
                ),
                "exponent_raw_bytes": int(exponent_raw_bytes),
                "exponent_compressed_bytes": int(exponent_compressed_bytes),
                "exponent_compression_ratio_pct": (
                    100 * exponent_compressed_bytes / exponent_raw_bytes
                    if exponent_raw_bytes
                    else 0.0
                ),
                "encode_seconds": float(encoded.encode_seconds),
                "total_compression_seconds": float(encoded.encode_seconds)
                + float(setup["setup_seconds"]),
                "decode_seconds": float(decode_seconds),
                "huffman_chunks": int(encoded.huffman_chunks),
                "raw_chunks": int(encoded.raw_chunks),
                "shared_codebook_refs": int(encoded.shared_codebook_refs),
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
    if any(
        group[["codec_protocol_version", "encoder_path"]].nunique().max() != 1
        for _, group in result_df.groupby("method")
    ):
        raise RuntimeError("Checkpoint summary mixes incompatible codec protocols.")
    summary = (
        result_df.groupby("method", as_index=False)
        .agg(
            architecture=("architecture", "first"),
            codec_protocol_version=("codec_protocol_version", "first"),
            encoder_path=("encoder_path", "first"),
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
    neuralzip_setup = prepare_neuralzip_codec(
        neuralzip_codec, initial_tensors.items()
    )
    neuralzip_setup_seconds = neuralzip_setup.total_setup_seconds
    neuralzip_codebooks = neuralzip_codec.export_codebooks()
    protocols.append(
        {
            "name": "NeuralZip",
            "codec": neuralzip_codec,
            "setup_seconds": neuralzip_setup_seconds,
            "precompute_seconds": neuralzip_setup.precompute_seconds,
            "codebook_setup_seconds": neuralzip_setup.codebook_setup_seconds,
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
            downlink = encode_tensor_batch(protocol["codec"], server_tensors)
            decoded_downlink, downlink_decode_seconds = decode_tensor_batch(
                protocol["codec"],
                downlink.payloads,
            )
            if verify:
                assert_tensor_roundtrip(
                    server_tensors,
                    decoded_downlink,
                    method=protocol["name"],
                    checkpoint=f"round-{round_index}-downlink",
                )
            downlink_metrics[protocol["name"]] = {
                "encoder_path": downlink.encoder_path,
                "encode_seconds": float(downlink.encode_seconds),
                "decode_one_seconds": float(downlink_decode_seconds),
                "raw_bytes": int(downlink.raw_bytes),
                "compressed_bytes": int(downlink.compressed_bytes),
                "huffman_chunks": int(downlink.huffman_chunks),
                "raw_chunks": int(downlink.raw_chunks),
                "shared_refs": int(downlink.shared_codebook_refs),
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
                uplink = encode_tensor_batch(codec, client_tensors)
                if uplink.encoder_path != downlink_metrics[protocol["name"]][
                    "encoder_path"
                ]:
                    raise RuntimeError("Codec changed encoder path within one round.")
                decoded_uplink, decode_seconds = decode_tensor_batch(
                    codec,
                    uplink.payloads,
                )
                if verify:
                    assert_tensor_roundtrip(
                        client_tensors,
                        decoded_uplink,
                        method=protocol["name"],
                        checkpoint=f"round-{round_index}-uplink",
                    )
                uplink_encode_seconds.append(float(uplink.encode_seconds))
                uplink_decode_seconds.append(float(decode_seconds))
                uplink_bytes += int(uplink.compressed_bytes)
                uplink_raw_bytes += int(uplink.raw_bytes)
                huffman_chunks += int(uplink.huffman_chunks)
                raw_chunks += int(uplink.raw_chunks)
                shared_refs += int(uplink.shared_codebook_refs)
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
                    "codec_protocol_version": CODEC_PROTOCOL_VERSION,
                    "encoder_path": downlink["encoder_path"],
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
    if any(
        group[["codec_protocol_version", "encoder_path"]].nunique().max() != 1
        for _, group in round_df.groupby("protocol")
    ):
        raise RuntimeError("Federated summary mixes incompatible codec protocols.")
    summary = (
        round_df.groupby("protocol", as_index=False)
        .agg(
            architecture=("architecture", "first"),
            dataset=("dataset", "first"),
            codec_protocol_version=("codec_protocol_version", "first"),
            encoder_path=("encoder_path", "first"),
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
