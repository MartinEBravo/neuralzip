import math

import numpy as np
import torch

from .dtypes import FLOAT_DTYPES_BY_NAME

DTYPE_BY_NAME = FLOAT_DTYPES_BY_NAME

BYTE_GROUP_DTYPE_CONFIG = {
    torch.float32: {
        "element_size": 4,
        "num_buffers": 4,
        "byte_reorder": 220,
        "bit_reorder": 1,
    },
    torch.bfloat16: {
        "element_size": 2,
        "num_buffers": 2,
        "byte_reorder": 10,
        "bit_reorder": 1,
    },
    torch.float16: {
        "element_size": 2,
        "num_buffers": 2,
        "byte_reorder": 10,
        "bit_reorder": 0,
    },
}


def _tensor_raw_bytes(tensor):
    tensor = tensor.detach().cpu().contiguous()
    return tensor.reshape(-1).view(torch.uint8).numpy().tobytes()


def _tensor_from_raw_bytes(raw_bytes, dtype, shape):
    # Native decoders return an owning bytearray that PyTorch can wrap
    # directly.  Preserve the defensive copy for immutable/archive inputs.
    owner = raw_bytes if isinstance(raw_bytes, bytearray) else bytearray(raw_bytes)
    shape = tuple(int(dimension) for dimension in shape)
    expected_bytes = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
    if len(owner) != expected_bytes:
        raise ValueError(
            f"Decoded tensor byte length mismatch: {len(owner)} != {expected_bytes}."
        )
    if not owner:
        return torch.empty(shape, dtype=dtype)
    raw = torch.frombuffer(owner, dtype=torch.uint8)
    return raw.view(dtype).reshape(shape)


def _byte_group_dtype_config(dtype):
    config = BYTE_GROUP_DTYPE_CONFIG.get(dtype)
    if config is None:
        raise TypeError(
            "NeuralZip supports torch.float32, torch.bfloat16, and torch.float16 "
            f"tensors; got {dtype}."
        )
    return config


def _reorder_float32(raw_bytes):
    values = np.frombuffer(raw_bytes, dtype=np.dtype("<u4")).copy()
    sign = (values >> np.uint32(8)) & np.uint32(0x00800000)
    exponent = (values << np.uint32(1)) & np.uint32(0xFF000000)
    mantissa = values & np.uint32(0x007FFFFF)
    return (exponent | sign | mantissa).astype(np.dtype("<u4"), copy=False).tobytes()


def _restore_float32(raw_bytes):
    values = np.frombuffer(raw_bytes, dtype=np.dtype("<u4")).copy()
    sign = (values << np.uint32(8)) & np.uint32(0x80000000)
    exponent = (values >> np.uint32(1)) & np.uint32(0x7F800000)
    mantissa = values & np.uint32(0x007FFFFF)
    return (sign | exponent | mantissa).astype(np.dtype("<u4"), copy=False).tobytes()


def _reorder_bfloat16(raw_bytes):
    values = np.frombuffer(raw_bytes, dtype=np.dtype("<u2")).copy()
    sign = (values >> np.uint16(8)) & np.uint16(0x0080)
    exponent = (values << np.uint16(1)) & np.uint16(0xFF00)
    mantissa = values & np.uint16(0x007F)
    return (exponent | sign | mantissa).astype(np.dtype("<u2"), copy=False).tobytes()


def _restore_bfloat16(raw_bytes):
    values = np.frombuffer(raw_bytes, dtype=np.dtype("<u2")).copy()
    sign = (values << np.uint16(8)) & np.uint16(0x8000)
    exponent = (values >> np.uint16(1)) & np.uint16(0x7F80)
    mantissa = values & np.uint16(0x007F)
    return (sign | exponent | mantissa).astype(np.dtype("<u2"), copy=False).tobytes()


def _byte_group_reordered_bytes(tensor, config):
    raw_bytes = _tensor_raw_bytes(tensor)
    if not config["bit_reorder"]:
        return raw_bytes
    if tensor.dtype == torch.float32:
        return _reorder_float32(raw_bytes)
    if tensor.dtype == torch.bfloat16:
        return _reorder_bfloat16(raw_bytes)
    return raw_bytes


def _byte_group_restored_bytes(raw_bytes, dtype, config):
    if not config["bit_reorder"]:
        return raw_bytes
    if dtype == torch.float32:
        return _restore_float32(raw_bytes)
    if dtype == torch.bfloat16:
        return _restore_bfloat16(raw_bytes)
    return raw_bytes


class ByteGroupCodec:
    """Internal dtype-aware byte grouping with Huffman and raw fallback."""

    def __init__(
        self,
        *,
        min_saving_bytes=0,
        chunk_size=256 << 10,
        compression_threshold=0.95,
        threads=0,
    ):
        self.min_saving_bytes = int(min_saving_bytes)
        self.chunk_size = int(chunk_size)
        self.compression_threshold = float(compression_threshold)
        self.threads = int(threads)
        if self.min_saving_bytes < 0:
            raise ValueError("min_saving_bytes must be non-negative.")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        if not 0 < self.compression_threshold <= 1:
            raise ValueError("compression_threshold must be in (0, 1].")
        if self.threads < 0:
            raise ValueError("threads must be non-negative.")

    def _stream_chunk_size(self, element_size):
        return max(1, self.chunk_size // int(element_size))
