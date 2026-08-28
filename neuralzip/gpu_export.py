"""CPU-side primitives for NeuralZip's BF16 GPU inference format.

The GPU representation keeps the sign and mantissa byte verbatim and stores
the eight exponent bits in one continuous, MSB-first Huffman bitstream.  It is
deliberately independent of a CUDA runtime so archives can be prepared and
validated on machines without a GPU.

The flat decode table uses one unsigned 16-bit entry per ``max_bits`` prefix::

    entry = (number_of_code_bits << 8) | exponent_symbol

An entry of zero is invalid.  CUDA consumers may therefore decode a symbol
with a single lookup and advance by ``entry >> 8`` bits.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from itertools import pairwise

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
class BF16Lanes:
    """The two lossless byte lanes of a little-endian BF16 sequence."""

    exponents: bytes
    sign_mantissa: bytes

    def __post_init__(self) -> None:
        exponents = _as_bytes(self.exponents, name="exponents")
        sign_mantissa = _as_bytes(self.sign_mantissa, name="sign_mantissa")
        if len(exponents) != len(sign_mantissa):
            raise ValueError("BF16 lanes must contain the same number of elements.")
        object.__setattr__(self, "exponents", exponents)
        object.__setattr__(self, "sign_mantissa", sign_mantissa)

    @property
    def numel(self) -> int:
        return len(self.exponents)

    def assemble(self) -> bytes:
        return assemble_bfloat16_lanes(self)


def split_bfloat16_lanes(raw_bfloat16: bytes | bytearray | memoryview) -> BF16Lanes:
    """Split little-endian BF16 words into exponent and sign/mantissa lanes.

    The returned ``sign_mantissa`` byte has the sign in bit 7 and the seven
    mantissa bits in bits 6..0.  This is the same bit reorder used by ZipNN.
    """

    raw = _as_bytes(raw_bfloat16, name="raw_bfloat16")
    if len(raw) % 2:
        raise ValueError("A BF16 byte sequence must have an even length.")

    exponents = bytearray(len(raw) // 2)
    sign_mantissa = bytearray(len(raw) // 2)
    for index in range(len(exponents)):
        low = raw[2 * index]
        high = raw[2 * index + 1]
        exponents[index] = ((high & 0x7F) << 1) | (low >> 7)
        sign_mantissa[index] = (high & 0x80) | (low & 0x7F)
    return BF16Lanes(bytes(exponents), bytes(sign_mantissa))


def assemble_bfloat16_lanes(
    exponents: BF16Lanes | bytes | bytearray | memoryview,
    sign_mantissa: bytes | bytearray | memoryview | None = None,
) -> bytes:
    """Reassemble exact little-endian BF16 words from their two byte lanes."""

    if isinstance(exponents, BF16Lanes):
        if sign_mantissa is not None:
            raise TypeError("sign_mantissa must be omitted when passing BF16Lanes.")
        lanes = exponents
    else:
        if sign_mantissa is None:
            raise TypeError("sign_mantissa is required when passing exponent bytes.")
        lanes = BF16Lanes(exponents, sign_mantissa)

    raw = bytearray(2 * lanes.numel)
    for index, (exponent, other) in enumerate(
        zip(lanes.exponents, lanes.sign_mantissa, strict=True)
    ):
        raw[2 * index] = (other & 0x7F) | ((exponent & 1) << 7)
        raw[2 * index + 1] = (other & 0x80) | (exponent >> 1)
    return bytes(raw)


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

    def lookup(self, prefix: int) -> tuple[int, int]:
        """Return ``(symbol, code_length)`` for one ``max_bits`` prefix."""

        if isinstance(prefix, bool) or not isinstance(prefix, int):
            raise TypeError("prefix must be an integer.")
        if prefix < 0 or prefix >= len(self.decode_lut):
            raise ValueError("prefix is outside the decode table.")
        entry = self.decode_lut[prefix]
        if entry == 0:
            raise ValueError("prefix does not identify a codebook symbol.")
        return entry & 0xFF, entry >> 8


def pack_restart_gaps(gaps: Sequence[int] | Iterable[int]) -> bytes:
    """Pack gap values MSB first, matching ``numpy.packbits`` semantics."""

    values = _as_int_tuple(gaps, name="gaps")
    accumulator = 0
    pending = 0
    packed = bytearray()
    for gap in values:
        if gap < 0 or gap >= 1 << GAP_BITS:
            raise ValueError(f"Each gap must fit in {GAP_BITS} bits.")
        accumulator = (accumulator << GAP_BITS) | gap
        pending += GAP_BITS
        while pending >= 8:
            pending -= 8
            packed.append((accumulator >> pending) & 0xFF)
            accumulator &= (1 << pending) - 1 if pending else 0
    if pending:
        packed.append((accumulator << (8 - pending)) & 0xFF)
    return bytes(packed)


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

    @property
    def gaps(self) -> tuple[int, ...]:
        return unpack_restart_gaps(
            self.packed_gaps,
            self.block_count * self.threads_per_block,
        )


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


def _restart_metadata(
    symbol_starts: Sequence[int],
    *,
    bit_length: int,
    encoded_bytes: int,
    bytes_per_thread: int,
    threads_per_block: int,
) -> tuple[bytes, tuple[int, ...]]:
    block_bytes = bytes_per_thread * threads_per_block
    blocks = (encoded_bytes + block_bytes - 1) // block_bytes
    if blocks == 0:
        return b"", (0,)

    segment_bits = bytes_per_thread * 8
    populated_segments = (encoded_bytes + bytes_per_thread - 1) // bytes_per_thread
    gaps = [0] * (blocks * threads_per_block)
    for segment in range(populated_segments):
        boundary = segment * segment_bits
        symbol_index = bisect_left(symbol_starts, boundary)
        next_boundary = (
            symbol_starts[symbol_index]
            if symbol_index < len(symbol_starts)
            else bit_length
        )
        gap = next_boundary - boundary
        if gap < 0 or gap >= 1 << GAP_BITS:
            raise ValueError(
                f"A code crosses too much of a CUDA segment to fit in {GAP_BITS}-bit gaps."
            )
        gaps[segment] = gap

    block_bits = block_bytes * 8
    positions = tuple(
        bisect_left(symbol_starts, block * block_bits) for block in range(blocks)
    ) + (len(symbol_starts),)
    return pack_restart_gaps(gaps), positions


def encode_exponents(
    exponents: bytes | bytearray | memoryview,
    codebook: GPUCodebook,
    *,
    bytes_per_thread: int = DEFAULT_BYTES_PER_THREAD,
    threads_per_block: int = DEFAULT_THREADS_PER_BLOCK,
) -> GPUEncodedExponents:
    """Huffman-encode exponent bytes into a continuous MSB-first stream."""

    source = _as_bytes(exponents, name="exponents")
    for name, value in (
        ("bytes_per_thread", bytes_per_thread),
        ("threads_per_block", threads_per_block),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer.")
        if value <= 0:
            raise ValueError(f"{name} must be positive.")
    if bytes_per_thread != DEFAULT_BYTES_PER_THREAD:
        raise ValueError(
            f"bytes_per_thread must be {DEFAULT_BYTES_PER_THREAD} for this ABI."
        )
    if threads_per_block != DEFAULT_THREADS_PER_BLOCK:
        raise ValueError(
            f"threads_per_block must be {DEFAULT_THREADS_PER_BLOCK} for this ABI."
        )

    encoded = bytearray()
    accumulator = 0
    pending = 0
    bit_length = 0
    starts = []
    for index, symbol in enumerate(source):
        length = codebook.code_lengths[symbol]
        if length == 0:
            raise ValueError(
                f"Exponent symbol {symbol} at index {index} has no codebook entry."
            )
        starts.append(bit_length)
        accumulator = (accumulator << length) | codebook.code_values[symbol]
        pending += length
        bit_length += length
        while pending >= 8:
            pending -= 8
            encoded.append((accumulator >> pending) & 0xFF)
            accumulator &= (1 << pending) - 1 if pending else 0
    if pending:
        encoded.append((accumulator << (8 - pending)) & 0xFF)

    packed_gaps, output_positions = _restart_metadata(
        starts,
        bit_length=bit_length,
        encoded_bytes=len(encoded),
        bytes_per_thread=bytes_per_thread,
        threads_per_block=threads_per_block,
    )
    return GPUEncodedExponents(
        bitstream=bytes(encoded),
        bit_length=bit_length,
        symbol_count=len(source),
        packed_gaps=packed_gaps,
        output_positions=output_positions,
        bytes_per_thread=bytes_per_thread,
        threads_per_block=threads_per_block,
    )


def encode_bfloat16(
    raw_bfloat16: bytes | bytearray | memoryview,
    codebook: GPUCodebook,
    *,
    bytes_per_thread: int = DEFAULT_BYTES_PER_THREAD,
    threads_per_block: int = DEFAULT_THREADS_PER_BLOCK,
) -> GPUEncodedBF16:
    """Prepare an exact little-endian BF16 byte sequence for GPU decoding."""

    lanes = split_bfloat16_lanes(raw_bfloat16)
    return GPUEncodedBF16(
        sign_mantissa=lanes.sign_mantissa,
        exponents=encode_exponents(
            lanes.exponents,
            codebook,
            bytes_per_thread=bytes_per_thread,
            threads_per_block=threads_per_block,
        ),
    )


def _decode_with_starts(
    encoded: GPUEncodedExponents,
    codebook: GPUCodebook,
) -> tuple[bytes, tuple[int, ...]]:
    padded = encoded.bitstream + bytes((codebook.max_bits + 14) // 8)
    window_bytes = (7 + codebook.max_bits + 7) // 8
    window_bits = 8 * window_bytes
    mask = (1 << codebook.max_bits) - 1
    output = bytearray(encoded.symbol_count)
    starts = []
    bit_offset = 0

    for index in range(encoded.symbol_count):
        starts.append(bit_offset)
        byte_offset, intra_byte = divmod(bit_offset, 8)
        window = int.from_bytes(
            padded[byte_offset : byte_offset + window_bytes],
            "big",
        )
        prefix = (window >> (window_bits - intra_byte - codebook.max_bits)) & mask
        entry = codebook.decode_lut[prefix]
        if entry == 0:
            raise ValueError(f"Invalid Huffman prefix at exponent index {index}.")
        length = entry >> 8
        if bit_offset + length > encoded.bit_length:
            raise ValueError("Exponent bitstream ends in the middle of a Huffman code.")
        output[index] = entry & 0xFF
        bit_offset += length

    if bit_offset != encoded.bit_length:
        raise ValueError(
            "Exponent symbol_count does not consume exactly the encoded bitstream."
        )
    return bytes(output), tuple(starts)


def decode_exponents_reference(
    encoded: GPUEncodedExponents,
    codebook: GPUCodebook,
    *,
    validate_metadata: bool = True,
) -> bytes:
    """Decode an exponent stream on CPU and optionally verify restart metadata."""

    output, starts = _decode_with_starts(encoded, codebook)
    if validate_metadata:
        expected_gaps, expected_positions = _restart_metadata(
            starts,
            bit_length=encoded.bit_length,
            encoded_bytes=len(encoded.bitstream),
            bytes_per_thread=encoded.bytes_per_thread,
            threads_per_block=encoded.threads_per_block,
        )
        if encoded.packed_gaps != expected_gaps:
            raise ValueError("Packed gaps do not match the exponent bitstream.")
        if encoded.output_positions != expected_positions:
            raise ValueError("Output positions do not match the exponent bitstream.")
    return output


def decode_bfloat16_reference(
    encoded: GPUEncodedBF16,
    codebook: GPUCodebook,
    *,
    validate_metadata: bool = True,
) -> bytes:
    """Reference-decode a GPU payload to bitwise-exact little-endian BF16."""

    exponents = decode_exponents_reference(
        encoded.exponents,
        codebook,
        validate_metadata=validate_metadata,
    )
    return assemble_bfloat16_lanes(exponents, encoded.sign_mantissa)


__all__ = [
    "DEFAULT_BYTES_PER_THREAD",
    "DEFAULT_THREADS_PER_BLOCK",
    "MAX_CODE_BITS",
    "BF16Lanes",
    "GPUCodebook",
    "GPUEncodedBF16",
    "GPUEncodedExponents",
    "assemble_bfloat16_lanes",
    "decode_bfloat16_reference",
    "decode_exponents_reference",
    "encode_bfloat16",
    "encode_exponents",
    "pack_restart_gaps",
    "split_bfloat16_lanes",
    "unpack_restart_gaps",
]
