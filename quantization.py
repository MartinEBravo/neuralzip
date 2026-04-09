from __future__ import annotations

from dataclasses import dataclass

import torch

from tensors import ensure_cpu_contiguous


@dataclass(frozen=True)
class FloatQuantFormat:
    total_bits: int
    exponent_bits: int
    mantissa_bits: int


FLOAT_QUANT_FORMATS = {
    8: FloatQuantFormat(total_bits=8, exponent_bits=4, mantissa_bits=3),   # E4M3
    12: FloatQuantFormat(total_bits=12, exponent_bits=5, mantissa_bits=6),  # E5M6
}
SUPPORTED_PREQUANT_BITS = tuple(sorted(FLOAT_QUANT_FORMATS))


def normalize_prequant_bits(bits: int | None) -> int | None:
    if bits is None or bits == 0:
        return None
    normalized_bits = int(bits)
    if normalized_bits not in FLOAT_QUANT_FORMATS:
        supported = ", ".join(str(value) for value in SUPPORTED_PREQUANT_BITS)
        raise ValueError(
            f"Pre-quantization bits must be 0 or one of: {supported}."
        )
    return normalized_bits


def _quantize_subnormals(
    magnitude: torch.Tensor,
    fmt: FloatQuantFormat,
    min_normal: float,
) -> torch.Tensor:
    min_subnormal = float(2.0 ** (1 - ((1 << (fmt.exponent_bits - 1)) - 1) - fmt.mantissa_bits))
    max_subnormal = float(((1 << fmt.mantissa_bits) - 1) * min_subnormal)
    quantized = torch.round(magnitude / min_subnormal) * min_subnormal
    quantized = torch.clamp(quantized, 0.0, max_subnormal)
    return torch.where(quantized >= min_normal, torch.full_like(quantized, min_normal), quantized)


def _quantize_normals(
    magnitude: torch.Tensor,
    fmt: FloatQuantFormat,
    min_normal_exp: int,
    max_normal_exp: int,
    max_finite: float,
) -> torch.Tensor:
    mantissa, exponent = torch.frexp(torch.clamp(magnitude, 0.0, max_finite))
    significand = mantissa * 2.0
    unbiased_exp = exponent - 1

    fraction = significand - 1.0
    scale = 1 << fmt.mantissa_bits
    quantized_fraction = torch.round(fraction * scale) / scale
    carry = quantized_fraction >= 1.0

    unbiased_exp = unbiased_exp + carry.to(unbiased_exp.dtype)
    quantized_fraction = torch.where(carry, torch.zeros_like(quantized_fraction), quantized_fraction)

    overflow = unbiased_exp > max_normal_exp
    unbiased_exp = torch.clamp(unbiased_exp, min=min_normal_exp, max=max_normal_exp)
    quantized = torch.ldexp(1.0 + quantized_fraction, unbiased_exp)
    return torch.where(overflow, torch.full_like(quantized, max_finite), quantized)


def float_quantize_tensor(tensor: torch.Tensor, bits: int | None) -> torch.Tensor:
    normalized_bits = normalize_prequant_bits(bits)
    if normalized_bits is None:
        return tensor
    if tensor.numel() == 0 or not tensor.is_floating_point():
        return tensor

    fmt = FLOAT_QUANT_FORMATS[normalized_bits]
    working = ensure_cpu_contiguous(tensor.detach())
    if not torch.isfinite(working).all():
        return working.clone()

    compute = working.to(torch.float32)
    magnitude = torch.abs(compute)
    negative = torch.signbit(compute)

    bias = (1 << (fmt.exponent_bits - 1)) - 1
    min_normal_exp = 1 - bias
    max_normal_exp = ((1 << fmt.exponent_bits) - 2) - bias
    min_normal = float(2.0 ** min_normal_exp)
    max_finite = float((2.0 - 2.0 ** (-fmt.mantissa_bits)) * (2.0 ** max_normal_exp))

    quantized_magnitude = torch.zeros_like(magnitude)
    nonzero = magnitude > 0
    if bool(nonzero.any()):
        subnormal_mask = nonzero & (magnitude < min_normal)
        normal_mask = nonzero & ~subnormal_mask

        if bool(subnormal_mask.any()):
            quantized_magnitude[subnormal_mask] = _quantize_subnormals(
                magnitude[subnormal_mask],
                fmt,
                min_normal=min_normal,
            )

        if bool(normal_mask.any()):
            quantized_magnitude[normal_mask] = _quantize_normals(
                magnitude[normal_mask],
                fmt,
                min_normal_exp=min_normal_exp,
                max_normal_exp=max_normal_exp,
                max_finite=max_finite,
            )

    quantized = torch.where(negative, -quantized_magnitude, quantized_magnitude)
    return quantized.to(dtype=working.dtype)
