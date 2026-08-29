"""Native BF16 payloads consumed by the CUDA inference path."""

from dataclasses import dataclass, field
from itertools import pairwise

from .. import _huff0_cpp

BYTES_PER_THREAD = 8
THREADS_PER_BLOCK = 512
MAX_CODE_BITS = 12


@dataclass(frozen=True)
class GPUCodebook:
    code_lengths: tuple[int, ...]
    code_values: tuple[int, ...]
    max_bits: int = field(init=False)
    decode_lut: tuple[int, ...] = field(init=False, repr=False)

    def __post_init__(self):
        lengths = tuple(self.code_lengths)
        values = tuple(self.code_values)
        if len(lengths) != 256 or len(values) != 256:
            raise ValueError("A BF16 exponent codebook must contain 256 entries.")
        max_bits = max(lengths)
        if not 1 <= max_bits <= MAX_CODE_BITS or any(length < 0 for length in lengths):
            raise ValueError("The GPU Huffman code length is unsupported.")

        lut = [0] * (1 << max_bits)
        for symbol, (length, value) in enumerate(zip(lengths, values, strict=True)):
            if not length:
                continue
            if not 0 <= value < 1 << length:
                raise ValueError("A Huffman code does not fit its declared length.")
            begin = value << (max_bits - length)
            end = begin + (1 << (max_bits - length))
            entry = (length << 8) | symbol
            for index in range(begin, end):
                if lut[index]:
                    raise ValueError("The Huffman codebook is not prefix-free.")
                lut[index] = entry
        if not all(lut):
            # A zero entry makes the CUDA decoder advance by zero bits forever.
            raise ValueError("The Huffman codebook does not cover every prefix.")

        object.__setattr__(self, "code_lengths", lengths)
        object.__setattr__(self, "code_values", values)
        object.__setattr__(self, "max_bits", max_bits)
        object.__setattr__(self, "decode_lut", tuple(lut))

    @classmethod
    def from_codes(cls, code_lengths, code_values):
        return cls(tuple(code_lengths), tuple(code_values))


@dataclass(frozen=True)
class GPUEncodedExponents:
    bitstream: bytes
    bit_length: int
    symbol_count: int
    packed_gaps: bytes
    output_positions: tuple[int, ...]

    def __post_init__(self):
        bitstream = bytes(self.bitstream)
        gaps = bytes(self.packed_gaps)
        positions = tuple(self.output_positions)
        blocks = (len(bitstream) + BYTES_PER_THREAD * THREADS_PER_BLOCK - 1) // (
            BYTES_PER_THREAD * THREADS_PER_BLOCK
        )
        if len(bitstream) != (self.bit_length + 7) // 8:
            raise ValueError("Exponent bit length does not match its payload.")
        if self.bit_length < 0 or self.symbol_count < 0:
            raise ValueError("Exponent sizes cannot be negative.")
        if len(gaps) != (blocks * THREADS_PER_BLOCK + 1) // 2:
            raise ValueError("Exponent restart metadata has the wrong size.")
        if len(positions) != blocks + 1 or positions[0] != 0:
            raise ValueError("Exponent output positions have the wrong shape.")
        if positions[-1] != self.symbol_count:
            raise ValueError("Exponent output positions have the wrong end offset.")
        if any(left > right for left, right in pairwise(positions)):
            raise ValueError("Exponent output positions must be ordered.")
        object.__setattr__(self, "bitstream", bitstream)
        object.__setattr__(self, "packed_gaps", gaps)
        object.__setattr__(self, "output_positions", positions)

    @property
    def block_count(self):
        block_bytes = BYTES_PER_THREAD * THREADS_PER_BLOCK
        return (len(self.bitstream) + block_bytes - 1) // block_bytes


@dataclass(frozen=True)
class GPUEncodedBF16:
    sign_mantissa: bytes
    exponents: GPUEncodedExponents

    def __post_init__(self):
        data = bytes(self.sign_mantissa)
        if len(data) != self.exponents.symbol_count:
            raise ValueError("BF16 fields contain different element counts.")
        object.__setattr__(self, "sign_mantissa", data)

    @property
    def numel(self):
        return len(self.sign_mantissa)


def encode_bfloat16_native(raw_bfloat16, codebook_header):
    """Encode one BF16 byte stream into the CUDA decoder layout."""

    encoded, sign_mantissa, gaps, positions, bit_length, _max_bits = (
        _huff0_cpp.encode_gpu_bfloat16(
            bytes(raw_bfloat16),
            bytes(codebook_header),
            bytes_per_thread=BYTES_PER_THREAD,
            threads_per_block=THREADS_PER_BLOCK,
        )
    )
    return GPUEncodedBF16(
        sign_mantissa=sign_mantissa,
        exponents=GPUEncodedExponents(
            bitstream=encoded,
            bit_length=int(bit_length),
            symbol_count=len(sign_mantissa),
            packed_gaps=gaps,
            output_positions=tuple(positions),
        ),
    )
