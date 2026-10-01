import math
import time
from collections.abc import Mapping
from itertools import chain

import numpy as np
import torch

from neuralzip.dtypes import DTYPES


def _named_tensors(value, prefix=""):
    if torch.is_tensor(value):
        yield prefix, value
    elif isinstance(value, Mapping):
        for key, child in value.items():
            name = str(key) if not prefix else f"{prefix}.{key}"
            yield from _named_tensors(child, name)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            name = str(index) if not prefix else f"{prefix}.{index}"
            yield from _named_tensors(child, name)


def _model_tensors(model):
    if isinstance(model, Mapping):
        items = _named_tensors(model)
    elif isinstance(model, torch.nn.Module):
        persistent = set(model.state_dict())
        items = chain(
            model.named_parameters(),
            (
                (name, tensor)
                for name, tensor in model.named_buffers()
                if name in persistent
            ),
        )
    else:
        raise TypeError("Expected a tensor mapping or torch.nn.Module.")
    tensors = [
        (str(name), tensor)
        for name, tensor in items
        if torch.is_tensor(tensor) and tensor.is_floating_point()
    ]
    if not tensors:
        raise ValueError("No floating tensors were found.")
    if len({name for name, _ in tensors}) != len(tensors):
        raise ValueError("Tensor names must be unique.")
    return tensors


def _floating_tensors(named_tensors):
    for name, tensor in sorted(named_tensors, key=lambda item: item[0]):
        if torch.is_tensor(tensor) and tensor.is_floating_point() and tensor.numel():
            yield name, tensor


def _tensor_raw_bytes(tensor):
    tensor = tensor.detach().cpu().contiguous()
    return tensor.reshape(-1).view(torch.uint8).numpy().tobytes()


def _tensor_from_raw_bytes(raw_bytes, dtype, shape):
    # Native decoders return an owning bytearray that PyTorch can wrap
    # directly. Preserve the defensive copy for immutable/archive inputs.
    owner = raw_bytes if isinstance(raw_bytes, bytearray) else bytearray(raw_bytes)
    shape = tuple(int(dimension) for dimension in shape)
    expected_bytes = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
    if len(owner) != expected_bytes:
        raise ValueError(
            f"Decoded tensor byte length mismatch: {len(owner)} != {expected_bytes}."
        )
    if not owner:
        return torch.empty(shape, dtype=dtype)
    return torch.frombuffer(owner, dtype=torch.uint8).view(dtype).reshape(shape)


def exponent_distribution(tensor):
    """Return the normalized exponent histogram of a floating tensor."""

    config = DTYPES[tensor.dtype]
    exponent = config.extract_exponents(_tensor_raw_bytes(tensor))
    counts = np.bincount(exponent, minlength=256).astype(np.float64)
    total = counts.sum()
    return counts / total if total else counts


def jensen_shannon_distance(left, right):
    """Return the base-2 Jensen--Shannon distance between distributions."""

    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    middle = 0.5 * (left + right)
    left_mask = left > 0
    right_mask = right > 0
    divergence = 0.5 * np.sum(
        left[left_mask] * np.log2(left[left_mask] / middle[left_mask])
    )
    divergence += 0.5 * np.sum(
        right[right_mask] * np.log2(right[right_mask] / middle[right_mask])
    )
    return float(np.sqrt(max(float(divergence), 0.0)))




def _stream_chunk_size(chunk_size, element_size):
    return max(1, chunk_size // int(element_size))


def time_call(func, /, *args, **kwargs):
    """Call a function and return the result and elapsed time in seconds."""

    start = time.perf_counter()
    return func(*args, **kwargs), time.perf_counter() - start
