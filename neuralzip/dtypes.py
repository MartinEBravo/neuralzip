from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class DTypeSpec:
    name: str
    torch_dtype: torch.dtype
    numpy: np.dtype
    element_size: int
    bit_reorder: int
    mantissa_bits: int
    exponent_mask: int
    exponent_stream: int | None = None  # Only a byte stream of pure exponents.
    global_codebooks: tuple[tuple[int, str], ...] = ()

    @property
    def stream_ids(self):
        return range(self.element_size)

    def reorder_bytes(self, raw_bytes, *, restore=False):
        """Move the sign below the exponent, or undo it; layout 0 is identity."""
        if not self.bit_reorder:
            return raw_bytes
        unsigned = self.numpy.type
        exponent_bits = self.exponent_mask.bit_length()
        values = np.frombuffer(raw_bytes, dtype=self.numpy).copy()
        mantissa = values & unsigned((1 << self.mantissa_bits) - 1)
        if restore:
            sign = (values << unsigned(exponent_bits)) & unsigned(
                1 << (self.mantissa_bits + exponent_bits)
            )
            exponent = (values >> unsigned(1)) & unsigned(
                self.exponent_mask << self.mantissa_bits
            )
        else:
            sign = (values >> unsigned(exponent_bits)) & unsigned(
                1 << self.mantissa_bits
            )
            exponent = (values << unsigned(1)) & unsigned(
                self.exponent_mask << (self.mantissa_bits + 1)
            )
        return (sign | exponent | mantissa).astype(self.numpy, copy=False).tobytes()

    def extract_exponents(self, raw_bytes):
        """Extract exponent fields from original words, not from codec streams."""
        values = np.frombuffer(raw_bytes, dtype=self.numpy)
        return (values >> self.mantissa_bits) & self.exponent_mask


bfloat16 = DTypeSpec(
    name="bfloat16",
    torch_dtype=torch.bfloat16,
    numpy=np.dtype("<u2"),
    element_size=2,
    bit_reorder=1,
    mantissa_bits=7,
    exponent_mask=0xFF,
    exponent_stream=1,
)

float16 = DTypeSpec(
    name="float16",
    torch_dtype=torch.float16,
    numpy=np.dtype("<u2"),
    element_size=2,
    bit_reorder=0,
    mantissa_bits=10,
    exponent_mask=0x1F,
)

float32 = DTypeSpec(
    name="float32",
    torch_dtype=torch.float32,
    numpy=np.dtype("<u4"),
    element_size=4,
    bit_reorder=1,
    mantissa_bits=23,
    exponent_mask=0xFF,
    exponent_stream=3,
)

DTYPES = {spec.torch_dtype: spec for spec in (float16, bfloat16, float32)}
