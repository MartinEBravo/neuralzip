from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class FloatLayout:
    total_bits: int
    exponent_bits: int
    mantissa_bits: int
    bits_dtype: torch.dtype


@dataclass(frozen=True)
class LayerInfo:
    name: str
    exponent_histogram: np.ndarray


@dataclass(frozen=True)
class ScanSummary:
    layer_infos: list[LayerInfo]
    original_bits: int
    payload_bits: int
    zipnn_bits: int
    zipnn_plus_bits: int
    dtype_summary: str
    num_tensors: int
    loader: str
