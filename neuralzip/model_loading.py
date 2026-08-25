"""Small model loaders shared by experiment runners."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from .dtypes import resolve_torch_dtype


def _target_dtype(dtype):
    resolved = resolve_torch_dtype(dtype)
    return None if resolved == "auto" else resolved


def _floating_parameters(module, *, dtype):
    tensors = {}
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            if not parameter.is_floating_point():
                continue
            tensor = parameter.detach().cpu().contiguous()
            if dtype is not None:
                tensor = tensor.to(dtype=dtype)
            tensors[name] = tensor
    return tensors


def _load_torchvision(spec: Mapping, *, dtype):
    from torchvision import models

    constructor = spec.get("constructor")
    if not constructor:
        raise ValueError("torchvision specs require a constructor name.")

    weights = spec.get("weights", "DEFAULT")
    weights_label = "none" if weights is None else str(weights)
    if isinstance(weights, str) and weights.upper() == "DEFAULT":
        weights = models.get_model_weights(constructor).DEFAULT
        weights_label = "DEFAULT"

    builder = getattr(models, constructor)
    try:
        module = builder(weights=weights)
    except TypeError:
        module = builder(pretrained=weights is not None)
    return _floating_parameters(module, dtype=_target_dtype(dtype)), {
        "load_backend": "torchvision",
        "source": f"torchvision.models.{constructor}",
        "weights": weights_label,
    }


def _load_diffusers_unet(spec: Mapping, *, dtype, local_files_only):
    loader_name = spec.get("loader", "UNet2DConditionModel")
    if loader_name == "UNet2DConditionModel":
        from diffusers import UNet2DConditionModel as loader
    elif loader_name == "UNet2DModel":
        from diffusers import UNet2DModel as loader
    else:
        raise ValueError(f"Unsupported diffusers UNet loader {loader_name!r}.")

    model_id = spec.get("model_id")
    if not model_id:
        raise ValueError("diffusers specs require a model_id.")

    kwargs = dict(spec.get("load_kwargs") or {})
    if spec.get("subfolder"):
        kwargs["subfolder"] = spec["subfolder"]
    kwargs["local_files_only"] = bool(local_files_only)
    dtype = _target_dtype(dtype)
    if dtype is not None:
        kwargs.setdefault("torch_dtype", dtype)
    kwargs.setdefault("low_cpu_mem_usage", True)

    module = loader.from_pretrained(model_id, **kwargs)
    source = model_id
    if spec.get("subfolder"):
        source = f"{model_id}/{spec['subfolder']}"
    return _floating_parameters(module, dtype=dtype), {
        "load_backend": f"diffusers:{loader_name}",
        "source": source,
        "weights": "pretrained",
    }


def load_vision_model(spec: Mapping, *, dtype, local_files_only=True):
    """Load the floating tensors selected by a vision experiment spec."""

    backend = str(spec.get("backend", "")).lower()
    if backend == "torchvision":
        return _load_torchvision(spec, dtype=dtype)
    if backend == "diffusers-unet":
        return _load_diffusers_unet(
            spec,
            dtype=dtype,
            local_files_only=local_files_only,
        )
    raise ValueError(f"Unsupported vision backend {backend!r}.")
