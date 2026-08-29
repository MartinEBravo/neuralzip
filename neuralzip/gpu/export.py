"""Validated Python representation of NeuralZip's native BF16 GPU format.

The GPU representation keeps the sign and mantissa byte verbatim and stores
the eight exponent bits in one continuous, MSB-first Huffman bitstream.  It is
emitted by the required native extension and can be archived or validated on
machines without a CUDA runtime.

The flat decode table uses one unsigned 16-bit entry per ``max_bits`` prefix::

    entry = (number_of_code_bits << 8) | exponent_symbol

An entry of zero is invalid.  CUDA consumers may therefore decode a symbol
with a single lookup and advance by ``entry >> 8`` bits.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from itertools import pairwise

from .. import _huff0_cpp

BF16_ALPHABET_SIZE = 256
DEFAULT_BYTES_PER_THREAD = 8
DEFAULT_THREADS_PER_BLOCK = 512
GAP_BITS = 4
MAX_CODE_BITS = 12


def _as_bytes(value, *, name: str) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, (bytearray, memoryview)):
        return bytes(value)
    raise TypeError(f"{name} must be bytes-like.")


def _as_int_tuple(
    values: Sequence[int] | Iterable[int], *, name: str
) -> tuple[int, ...]:
    try:
        output = tuple(values)
    except TypeError as error:
        raise TypeError(f"{name} must be an iterable of integers.") from error
    if any(isinstance(value, bool) or not isinstance(value, int) for value in output):
        raise TypeError(f"{name} must contain only integers.")
    return output


@dataclass(frozen=True)
class GPUCodebook:
    """A prefix code and its flat uint16 GPU decode lookup table.

    ``code_values`` are right-aligned integers whose most-significant used bit
    is the first bit written to the stream.  A zero length marks an unused
    exponent symbol and requires a zero code value.
    """

    code_lengths: tuple[int, ...]
    code_values: tuple[int, ...]
    max_bits: int = field(init=False)
    decode_lut: tuple[int, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        lengths = _as_int_tuple(self.code_lengths, name="code_lengths")
        values = _as_int_tuple(self.code_values, name="code_values")
        if len(lengths) != BF16_ALPHABET_SIZE:
            raise ValueError(f"code_lengths must have {BF16_ALPHABET_SIZE} entries.")
        if len(values) != BF16_ALPHABET_SIZE:
            raise ValueError(f"code_values must have {BF16_ALPHABET_SIZE} entries.")
        if any(length < 0 or length > MAX_CODE_BITS for length in lengths):
            raise ValueError(f"Code lengths must be in [0, {MAX_CODE_BITS}].")
        if not any(lengths):
            raise ValueError("At least one exponent symbol must have a code.")

        for symbol, (length, value) in enumerate(zip(lengths, values, strict=True)):
            if value < 0:
                raise ValueError(
                    f"Code value for symbol {symbol} must be non-negative."
                )
            if length == 0:
                if value != 0:
                    raise ValueError(
                        f"Unused symbol {symbol} must have a zero code value."
                    )
            elif value >= 1 << length:
                raise ValueError(
                    f"Code value for symbol {symbol} does not fit in {length} bits."
                )

        max_bits = max(lengths)
        lut = [0] * (1 << max_bits)
        for symbol, (length, value) in enumerate(zip(lengths, values, strict=True)):
            if length == 0:
                continue
            entry = (length << 8) | symbol
            suffix_bits = max_bits - length
            begin = value << suffix_bits
            end = begin + (1 << suffix_bits)
            for index in range(begin, end):
                if lut[index] != 0:
                    raise ValueError(
                        "Codebook is not prefix-free: "
                        f"symbol {symbol} overlaps another code."
                    )
                lut[index] = entry
        if any(entry == 0 for entry in lut):
            raise ValueError(
                "Codebook must cover every bit prefix for safe parallel decoding."
            )

        object.__setattr__(self, "code_lengths", lengths)
        object.__setattr__(self, "code_values", values)
        object.__setattr__(self, "max_bits", max_bits)
        object.__setattr__(self, "decode_lut", tuple(lut))

    @classmethod
    def from_codes(
        cls,
        code_lengths: Sequence[int] | Iterable[int],
        code_values: Sequence[int] | Iterable[int],
    ) -> GPUCodebook:
        """Build a validated GPU codebook from exported Huffman codes."""

        return cls(tuple(code_lengths), tuple(code_values))

    @property
    def lut_nbytes(self) -> int:
        return 2 * len(self.decode_lut)


def unpack_restart_gaps(
    packed_gaps: bytes | bytearray | memoryview,
    count: int,
) -> tuple[int, ...]:
    """Unpack exactly ``count`` MSB-first restart-gap values."""

    packed = _as_bytes(packed_gaps, name="packed_gaps")
    if isinstance(count, bool) or not isinstance(count, int):
        raise TypeError("count must be an integer.")
    if count < 0:
        raise ValueError("count must be non-negative.")
    expected_bytes = (count * GAP_BITS + 7) // 8
    if len(packed) != expected_bytes:
        raise ValueError(
            f"packed_gaps has {len(packed)} bytes; expected {expected_bytes}."
        )

    values = []
    bit_offset = 0
    for _ in range(count):
        value = 0
        for _bit in range(GAP_BITS):
            byte = packed[bit_offset // 8]
            value = (value << 1) | ((byte >> (7 - bit_offset % 8)) & 1)
            bit_offset += 1
        values.append(value)

    for offset in range(bit_offset, len(packed) * 8):
        if (packed[offset // 8] >> (7 - offset % 8)) & 1:
            raise ValueError("Unused packed-gap padding bits must be zero.")
    return tuple(values)


@dataclass(frozen=True)
class GPUEncodedExponents:
    """A continuous exponent stream plus CUDA segment restart metadata."""

    bitstream: bytes
    bit_length: int
    symbol_count: int
    packed_gaps: bytes
    output_positions: tuple[int, ...]
    bytes_per_thread: int = DEFAULT_BYTES_PER_THREAD
    threads_per_block: int = DEFAULT_THREADS_PER_BLOCK

    def __post_init__(self) -> None:
        bitstream = _as_bytes(self.bitstream, name="bitstream")
        packed_gaps = _as_bytes(self.packed_gaps, name="packed_gaps")
        positions = _as_int_tuple(self.output_positions, name="output_positions")
        for name, value in (
            ("bit_length", self.bit_length),
            ("symbol_count", self.symbol_count),
            ("bytes_per_thread", self.bytes_per_thread),
            ("threads_per_block", self.threads_per_block),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
        if self.bit_length < 0:
            raise ValueError("bit_length must be non-negative.")
        if self.symbol_count < 0:
            raise ValueError("symbol_count must be non-negative.")
        if self.bytes_per_thread <= 0 or self.threads_per_block <= 0:
            raise ValueError("CUDA segment dimensions must be positive.")
        if self.bytes_per_thread != DEFAULT_BYTES_PER_THREAD:
            raise ValueError(
                f"bytes_per_thread must be {DEFAULT_BYTES_PER_THREAD} for this ABI."
            )
        if self.threads_per_block != DEFAULT_THREADS_PER_BLOCK:
            raise ValueError(
                f"threads_per_block must be {DEFAULT_THREADS_PER_BLOCK} for this ABI."
            )
        if len(bitstream) != (self.bit_length + 7) // 8:
            raise ValueError("bitstream length does not match bit_length.")
        if self.bit_length == 0 and self.symbol_count != 0:
            raise ValueError("A non-empty symbol stream cannot have zero bits.")
        if bitstream and self.bit_length % 8:
            padding_mask = (1 << (8 - self.bit_length % 8)) - 1
            if bitstream[-1] & padding_mask:
                raise ValueError("Unused bitstream padding bits must be zero.")

        block_bytes = self.bytes_per_thread * self.threads_per_block
        blocks = (len(bitstream) + block_bytes - 1) // block_bytes
        gap_count = blocks * self.threads_per_block
        unpack_restart_gaps(packed_gaps, gap_count)
        if len(positions) != blocks + 1:
            raise ValueError(
                "output_positions must contain one entry per block plus an end."
            )
        if not positions or positions[0] != 0:
            raise ValueError("output_positions must start at zero.")
        if positions[-1] != self.symbol_count:
            raise ValueError("output_positions must end at symbol_count.")
        if any(position < 0 or position > self.symbol_count for position in positions):
            raise ValueError("output_positions entries are outside the symbol stream.")
        if any(left > right for left, right in pairwise(positions)):
            raise ValueError("output_positions must be non-decreasing.")

        object.__setattr__(self, "bitstream", bitstream)
        object.__setattr__(self, "packed_gaps", packed_gaps)
        object.__setattr__(self, "output_positions", positions)

    @property
    def block_count(self) -> int:
        block_bytes = self.bytes_per_thread * self.threads_per_block
        return (len(self.bitstream) + block_bytes - 1) // block_bytes


@dataclass(frozen=True)
class GPUEncodedBF16:
    """A BF16 payload ready to copy to a GPU decoder."""

    sign_mantissa: bytes
    exponents: GPUEncodedExponents

    def __post_init__(self) -> None:
        sign_mantissa = _as_bytes(self.sign_mantissa, name="sign_mantissa")
        if not isinstance(self.exponents, GPUEncodedExponents):
            raise TypeError("exponents must be GPUEncodedExponents.")
        if len(sign_mantissa) != self.exponents.symbol_count:
            raise ValueError(
                "sign_mantissa length must equal the exponent symbol count."
            )
        object.__setattr__(self, "sign_mantissa", sign_mantissa)

    @property
    def numel(self) -> int:
        return len(self.sign_mantissa)


def encode_bfloat16_native(
    raw_bfloat16: bytes | bytearray | memoryview,
    codebook_header: bytes | bytearray | memoryview,
    *,
    bytes_per_thread: int = DEFAULT_BYTES_PER_THREAD,
    threads_per_block: int = DEFAULT_THREADS_PER_BLOCK,
) -> GPUEncodedBF16:
    """Use the native Huff0 backend to emit the CUDA inference layout."""

    try:
        native_encoder = _huff0_cpp.encode_gpu_bfloat16
    except AttributeError as error:  # pragma: no cover - mismatched binary install.
        raise RuntimeError(
            "The installed NeuralZip native extension does not provide the BF16 "
            "GPU encoder; reinstall NeuralZip so Python and native code match."
        ) from error
    (
        encoded_exponents,
        sign_mantissa,
        packed_gaps,
        output_positions,
        bit_length,
        _max_bits,
    ) = native_encoder(
        _as_bytes(raw_bfloat16, name="raw_bfloat16"),
        _as_bytes(codebook_header, name="codebook_header"),
        bytes_per_thread=bytes_per_thread,
        threads_per_block=threads_per_block,
    )
    return GPUEncodedBF16(
        sign_mantissa=sign_mantissa,
        exponents=GPUEncodedExponents(
            bitstream=encoded_exponents,
            bit_length=int(bit_length),
            symbol_count=len(sign_mantissa),
            packed_gaps=packed_gaps,
            output_positions=tuple(output_positions),
            bytes_per_thread=bytes_per_thread,
            threads_per_block=threads_per_block,
        ),
    )


__all__ = [
    "DEFAULT_BYTES_PER_THREAD",
    "DEFAULT_THREADS_PER_BLOCK",
    "MAX_CODE_BITS",
    "GPUCodebook",
    "GPUEncodedBF16",
    "GPUEncodedExponents",
    "encode_bfloat16_native",
    "unpack_restart_gaps",
]
