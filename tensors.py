from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch


# ---- Data classes ----

@dataclass(frozen=True)
class FloatLayout:
    total_bits: int
    exponent_bits: int
    mantissa_bits: int
    bits_dtype: torch.dtype


@dataclass(frozen=True)
class BlockInfo:
    name: str
    exponent_counts: np.ndarray
    num_weights: int
    tensor_names: tuple[str, ...]


@dataclass(frozen=True)
class FieldEntropySummary:
    field: str
    total_symbols: int
    h0_bits: float
    bits_per_symbol: float


@dataclass(frozen=True)
class ScanSummary:
    block_infos: list[BlockInfo]
    original_bits: int
    payload_bits: int
    dtype_summary: str
    num_blocks: int
    loader: str
    field_entropies: tuple[FieldEntropySummary, ...]

    def field_entropy(self, field: str) -> FieldEntropySummary:
        for summary in self.field_entropies:
            if summary.field == field:
                return summary
        raise KeyError(f"Unknown field entropy summary: {field}")


# ---- Block grouping ----

def block_name_from_tensor_name(tensor_name: str, grouping: str = "module") -> str:
    if grouping == "tensor":
        return tensor_name
    if tensor_name.endswith(".weight"):
        return tensor_name[: -len(".weight")]
    if tensor_name.endswith(".bias"):
        return tensor_name[: -len(".bias")]
    return tensor_name


def is_known_non_parameter_buffer(name: str, tensor) -> bool:
    return (name.endswith("attn.bias") and tensor.ndim >= 3) or name.endswith("masked_bias")


def natural_sort_key(text: str) -> tuple[object, ...]:
    parts = re.split(r"(\d+)", text)
    key: list[object] = []
    for part in parts:
        if not part:
            continue
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part)
    return tuple(key)


# ---- Tensor utilities ----

FLOAT_LAYOUTS = {
    torch.float16: FloatLayout(total_bits=16, exponent_bits=5, mantissa_bits=10, bits_dtype=torch.uint16),
    torch.bfloat16: FloatLayout(total_bits=16, exponent_bits=8, mantissa_bits=7, bits_dtype=torch.uint16),
    torch.float32: FloatLayout(total_bits=32, exponent_bits=8, mantissa_bits=23, bits_dtype=torch.uint32),
}
EXPONENT_CHUNK_SIZE = 1 << 22


def ensure_cpu_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.device.type != "cpu":
        tensor = tensor.cpu()
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    return tensor


def concat_arrays(parts: list[np.ndarray], dtype=np.uint8) -> np.ndarray:
    if not parts:
        return np.array([], dtype=dtype)
    if len(parts) == 1:
        return parts[0]
    return np.concatenate(parts)


@dataclass(frozen=True)
class TensorFieldStats:
    exponent_counts: np.ndarray
    sign_counts: np.ndarray
    mantissa_counts: np.ndarray
    mantissa_bits: int
    num_weights: int
    body_bits: int
    original_bits: int
    dtype_name: str


def _tensor_bit_view(tensor: torch.Tensor) -> tuple[np.ndarray, FloatLayout]:
    layout = FLOAT_LAYOUTS.get(tensor.dtype)
    if layout is None:
        raise TypeError(f"Unsupported dtype for exponent extraction: {tensor.dtype}")

    flat = ensure_cpu_contiguous(tensor.detach()).view(-1)
    return flat.view(layout.bits_dtype).numpy(), layout


def iter_tensor_exponent_chunks(
    tensor: torch.Tensor,
    chunk_size: int = EXPONENT_CHUNK_SIZE,
) -> Iterator[np.ndarray]:
    bits, layout = _tensor_bit_view(tensor)
    exponent_mask = (1 << layout.exponent_bits) - 1
    for start in range(0, bits.size, chunk_size):
        bit_chunk = bits[start : start + chunk_size]
        yield ((bit_chunk >> layout.mantissa_bits) & exponent_mask).astype(np.uint8, copy=False)


def tensor_sign_stream(tensor: torch.Tensor) -> np.ndarray:
    bits, layout = _tensor_bit_view(tensor)
    sign_shift = layout.exponent_bits + layout.mantissa_bits
    return ((bits >> sign_shift) & 1).astype(np.uint8, copy=False)


def tensor_mantissa_stream(tensor: torch.Tensor) -> tuple[np.ndarray, int]:
    bits, layout = _tensor_bit_view(tensor)
    mantissa_mask = (1 << layout.mantissa_bits) - 1
    mantissas = (bits & mantissa_mask).astype(np.uint32, copy=False)
    return mantissas, layout.mantissa_bits


def pack_binary_stream(values: np.ndarray) -> bytes:
    if values.size == 0:
        return b""
    return np.packbits(values.astype(np.uint8, copy=False), bitorder="big").tobytes()


def pack_unsigned_stream(values: np.ndarray, bit_width: int) -> bytes:
    if values.size == 0 or bit_width <= 0:
        return b""
    if bit_width <= 8:
        cast_dtype = ">u1"
        bytes_per_value = 1
    elif bit_width <= 16:
        cast_dtype = ">u2"
        bytes_per_value = 2
    elif bit_width <= 32:
        cast_dtype = ">u4"
        bytes_per_value = 4
    else:
        raise ValueError(f"Unsupported bit width for packing: {bit_width}")

    cast_values = values.astype(cast_dtype, copy=False)
    byte_view = cast_values.view(np.uint8).reshape(-1, bytes_per_value)
    unpacked = np.unpackbits(byte_view, axis=1, bitorder="big")
    trimmed = unpacked[:, -bit_width:].reshape(-1)
    return np.packbits(trimmed, bitorder="big").tobytes()


def tensor_exponent_stats(
    tensor: torch.Tensor,
) -> tuple[np.ndarray, int, int, int, str] | None:
    stats = tensor_field_stats(tensor)
    if stats is None:
        return None
    return (
        stats.exponent_counts,
        stats.num_weights,
        stats.body_bits,
        stats.original_bits,
        stats.dtype_name,
    )


def tensor_field_stats(tensor: torch.Tensor) -> TensorFieldStats | None:
    if tensor.numel() == 0 or not tensor.is_floating_point():
        return None

    try:
        bits, layout = _tensor_bit_view(tensor)
    except TypeError:
        return None

    exponent_mask = (1 << layout.exponent_bits) - 1
    mantissa_mask = (1 << layout.mantissa_bits) - 1
    sign_shift = layout.exponent_bits + layout.mantissa_bits

    exponent_counts = np.bincount(
        (bits >> layout.mantissa_bits) & exponent_mask,
        minlength=256,
    ).astype(np.int64, copy=False)
    sign_counts = np.bincount(
        (bits >> sign_shift) & 1,
        minlength=2,
    ).astype(np.int64, copy=False)
    mantissa_counts = np.bincount(
        bits & mantissa_mask,
        minlength=1 << layout.mantissa_bits,
    ).astype(np.int64, copy=False)
    num_weights = int(tensor.numel())
    body_bits = num_weights * (1 + layout.mantissa_bits)
    original_bits = num_weights * layout.total_bits
    dtype_name = str(tensor.dtype).replace("torch.", "")
    return TensorFieldStats(
        exponent_counts=exponent_counts,
        sign_counts=sign_counts,
        mantissa_counts=mantissa_counts,
        mantissa_bits=layout.mantissa_bits,
        num_weights=num_weights,
        body_bits=body_bits,
        original_bits=original_bits,
        dtype_name=dtype_name,
    )


def tensor_exponent_stream(tensor: torch.Tensor) -> np.ndarray:
    return concat_arrays(list(iter_tensor_exponent_chunks(tensor)))


def summarize_dtypes(dtype_weights: dict[str, int]) -> str:
    ordered = sorted(dtype_weights.items(), key=lambda item: item[1], reverse=True)
    return ", ".join(f"{dtype_name}:{num_weights}" for dtype_name, num_weights in ordered)
