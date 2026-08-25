"""Canonical conversion helpers for torch dtype values and names."""

import torch


FLOAT_DTYPES_BY_NAME = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

_DTYPE_ALIASES = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "f16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
    "f32": torch.float32,
}


def dtype_name(dtype):
    """Return the stable, unqualified name used in NeuralZip payloads."""

    return str(dtype).replace("torch.", "")


def resolve_torch_dtype(dtype):
    """Resolve accepted model dtype aliases while preserving ``auto``."""

    if dtype is None or dtype == "auto":
        return "auto"
    if isinstance(dtype, torch.dtype):
        return dtype

    dtype_key = dtype_name(dtype).lower().replace("-", "_")
    if dtype_key not in _DTYPE_ALIASES:
        choices = ", ".join(("auto", *sorted(_DTYPE_ALIASES)))
        raise ValueError(f"Unsupported dtype {dtype!r}. Expected one of: {choices}.")
    return _DTYPE_ALIASES[dtype_key]
