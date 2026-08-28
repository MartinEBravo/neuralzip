"""Canonical conversion helpers for torch dtype values and names."""

import torch

FLOAT_DTYPES_BY_NAME = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def dtype_name(dtype):
    """Return the stable, unqualified name used in NeuralZip payloads."""

    return str(dtype).replace("torch.", "")
