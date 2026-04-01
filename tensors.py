import numpy as np
import torch

from schemas import FloatLayout


FLOAT_LAYOUTS = {
    torch.float16: FloatLayout(total_bits=16, exponent_bits=5, mantissa_bits=10, bits_dtype=torch.uint16),
    torch.bfloat16: FloatLayout(total_bits=16, exponent_bits=8, mantissa_bits=7, bits_dtype=torch.uint16),
    torch.float32: FloatLayout(total_bits=32, exponent_bits=8, mantissa_bits=23, bits_dtype=torch.uint32),
}


def tensor_exponent_stats(
    tensor: torch.Tensor,
) -> tuple[np.ndarray, int, int, int, str] | None:
    if tensor.numel() == 0 or not tensor.is_floating_point():
        return None

    layout = FLOAT_LAYOUTS.get(tensor.dtype)
    if layout is None:
        return None

    exponents = tensor_exponent_stream(tensor)
    counts = np.bincount(exponents.astype(np.int64, copy=False), minlength=256).astype(np.int64)
    num_weights = int(tensor.numel())
    body_bits = num_weights * (1 + layout.mantissa_bits)
    original_bits = num_weights * layout.total_bits
    dtype_name = str(tensor.dtype).replace("torch.", "")
    return counts, num_weights, body_bits, original_bits, dtype_name


def tensor_exponent_stream(tensor: torch.Tensor) -> np.ndarray:
    layout = FLOAT_LAYOUTS.get(tensor.dtype)
    if layout is None:
        raise TypeError(f"Unsupported dtype for exponent extraction: {tensor.dtype}")

    flat = tensor.detach().cpu().contiguous().view(-1)
    bits = flat.view(layout.bits_dtype).to(torch.int64)
    exponent_mask = (1 << layout.exponent_bits) - 1
    exponents = (bits >> layout.mantissa_bits) & exponent_mask
    return exponents.cpu().numpy().astype(np.uint32, copy=False)


def summarize_dtypes(dtype_weights: dict[str, int]) -> str:
    ordered = sorted(dtype_weights.items(), key=lambda item: item[1], reverse=True)
    return ", ".join(f"{dtype_name}:{num_weights}" for dtype_name, num_weights in ordered)
