from typing import Iterator

import numpy as np
import torch

from schemas import FloatLayout


FLOAT_LAYOUTS = {
    torch.float16: FloatLayout(total_bits=16, exponent_bits=5, mantissa_bits=10, bits_dtype=torch.uint16),
    torch.bfloat16: FloatLayout(total_bits=16, exponent_bits=8, mantissa_bits=7, bits_dtype=torch.uint16),
    torch.float32: FloatLayout(total_bits=32, exponent_bits=8, mantissa_bits=23, bits_dtype=torch.uint32),
}
EXPONENT_CHUNK_SIZE = 1 << 22


def _tensor_bit_view(tensor: torch.Tensor) -> tuple[np.ndarray, FloatLayout]:
    layout = FLOAT_LAYOUTS.get(tensor.dtype)
    if layout is None:
        raise TypeError(f"Unsupported dtype for exponent extraction: {tensor.dtype}")

    flat = tensor.detach()
    if flat.device.type != "cpu":
        flat = flat.cpu()
    if not flat.is_contiguous():
        flat = flat.contiguous()
    flat = flat.view(-1)
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


def tensor_exponent_stats(
    tensor: torch.Tensor,
) -> tuple[np.ndarray, int, int, int, str] | None:
    if tensor.numel() == 0 or not tensor.is_floating_point():
        return None

    layout = FLOAT_LAYOUTS.get(tensor.dtype)
    if layout is None:
        return None

    counts = np.zeros(256, dtype=np.int64)
    for exponents in iter_tensor_exponent_chunks(tensor):
        counts += np.bincount(exponents, minlength=256).astype(np.int64)
    num_weights = int(tensor.numel())
    body_bits = num_weights * (1 + layout.mantissa_bits)
    original_bits = num_weights * layout.total_bits
    dtype_name = str(tensor.dtype).replace("torch.", "")
    return counts, num_weights, body_bits, original_bits, dtype_name


def tensor_exponent_stream(tensor: torch.Tensor) -> np.ndarray:
    chunks = list(iter_tensor_exponent_chunks(tensor))
    if not chunks:
        return np.array([], dtype=np.uint8)
    if len(chunks) == 1:
        return chunks[0]
    return np.concatenate(chunks)


def summarize_dtypes(dtype_weights: dict[str, int]) -> str:
    ordered = sorted(dtype_weights.items(), key=lambda item: item[1], reverse=True)
    return ", ".join(f"{dtype_name}:{num_weights}" for dtype_name, num_weights in ordered)
