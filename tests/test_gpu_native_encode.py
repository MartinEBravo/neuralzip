import random
from functools import lru_cache

import numpy as np
import pytest

from neuralzip import _huff0_cpp, huff0
from neuralzip.gpu_export import (
    GPUCodebook,
    decode_bfloat16_reference,
    encode_bfloat16,
)


@lru_cache(maxsize=1)
def _variable_codebook():
    sample = (
        bytes((127,)) * 20_000
        + bytes((128,)) * 8_000
        + bytes((126,)) * 4_000
        + bytes((129,)) * 2_000
        + bytes(range(256))
    )
    return huff0.build_codebook(sample, prior_count=1)


def _gpu_codebook(codebook):
    table = huff0.export_code_table(codebook)
    return GPUCodebook.from_codes(table.code_lengths, table.code_values)


def _raw_from_lanes(exponents, sign_mantissa):
    raw = bytearray(2 * len(exponents))
    for index, (exponent, other) in enumerate(
        zip(exponents, sign_mantissa, strict=True)
    ):
        raw[2 * index] = (other & 0x7F) | ((exponent & 1) << 7)
        raw[2 * index + 1] = (other & 0x80) | (exponent >> 1)
    return bytes(raw)


def _assert_matches_reference(
    raw,
    codebook,
    *,
    bytes_per_thread=8,
    threads_per_block=512,
    pass_header=False,
):
    gpu_codebook = _gpu_codebook(codebook)
    reference = encode_bfloat16(
        raw,
        gpu_codebook,
        bytes_per_thread=bytes_per_thread,
        threads_per_block=threads_per_block,
    )
    source = codebook.codebook if pass_header else codebook
    native = huff0.encode_gpu_bfloat16(
        raw,
        source,
        bytes_per_thread=bytes_per_thread,
        threads_per_block=threads_per_block,
    )

    assert isinstance(native, huff0.GPUEncodedBFloat16)
    assert native.encoded_exponents == reference.exponents.bitstream
    assert native.sign_mantissa == reference.sign_mantissa
    assert native.packed_gaps == reference.exponents.packed_gaps
    assert native.output_positions == reference.exponents.output_positions
    assert native.bit_length == reference.exponents.bit_length
    assert native.max_bits == gpu_codebook.max_bits
    assert decode_bfloat16_reference(reference, gpu_codebook) == raw


def test_native_gpu_encoder_matches_all_bfloat16_bit_patterns():
    raw = b"".join(word.to_bytes(2, "little") for word in range(1 << 16))
    _assert_matches_reference(raw, _variable_codebook(), pass_header=True)


@pytest.mark.parametrize(
    "numel",
    [
        0,
        1,
        7,
        8,
        9,
        63,
        64,
        65,
        4_095,
        4_096,
        4_097,
        12_000,
        40_000,
    ],
)
def test_native_gpu_encoder_matches_tail_sizes(numel):
    codebook = _variable_codebook()
    exponents = bytes(index & 0xFF for index in range(numel))
    sign_mantissa = bytes((index * 73 + 19) & 0xFF for index in range(numel))
    raw = _raw_from_lanes(exponents, sign_mantissa)

    _assert_matches_reference(
        raw,
        codebook,
        pass_header=numel % 2 == 0,
    )


def test_native_gpu_encoder_matches_exact_segment_and_block_crossings():
    codebook = _variable_codebook()
    table = huff0.export_code_table(codebook)
    coded = [
        (length, symbol)
        for symbol, length in enumerate(table.code_lengths)
        if length > 0
    ]
    representatives = (min(coded), max(coded))

    for length, symbol in representatives:
        for boundary_bits in (64, 32_768, 65_536):
            center = boundary_bits // length
            for numel in range(max(0, center - 3), center + 4):
                exponents = bytes((symbol,)) * numel
                sign_mantissa = bytes((0xA5,)) * numel
                _assert_matches_reference(
                    _raw_from_lanes(exponents, sign_mantissa),
                    codebook,
                    pass_header=(numel & 1) == 0,
                )


def test_native_gpu_encoder_matches_maximum_restart_gap():
    fibonacci = [1, 1]
    for _ in range(11):
        fibonacci.append(fibonacci[-1] + fibonacci[-2])
    counts = [0] * 256
    counts[:13] = reversed(fibonacci)
    codebook = huff0.build_codebook_from_counts(
        counts,
        prior_count=0,
        max_bits=12,
    )
    table = huff0.export_code_table(codebook, max_bits=12)
    lengths = table.code_lengths
    symbols_by_length = {
        length: next(symbol for symbol, value in enumerate(lengths) if value == length)
        for length in (7, 8, 12)
    }
    exponents = bytes(
        (
            symbols_by_length[7],
            symbols_by_length[8],
            *([symbols_by_length[12]] * 5),
            symbols_by_length[7],
        )
    )
    raw = _raw_from_lanes(exponents, bytes((0xA5,)) * len(exponents))

    native = huff0.encode_gpu_bfloat16(raw, codebook)

    assert native.max_bits == 12
    assert native.packed_gaps[0] == 0x0B
    _assert_matches_reference(raw, codebook)


def test_native_gpu_encoder_matches_random_variable_length_streams():
    rng = random.Random(0xD11)
    codebook = _variable_codebook()

    for iteration in range(30):
        numel = rng.randrange(0, 25_000)
        exponents = bytes(rng.randrange(256) for _ in range(numel))
        sign_mantissa = bytes(rng.randrange(256) for _ in range(numel))
        raw = _raw_from_lanes(exponents, sign_mantissa)
        _assert_matches_reference(
            raw,
            codebook,
            pass_header=iteration % 2 == 0,
        )


def test_native_gpu_encoder_validation_is_strict():
    complete = _variable_codebook()
    sparse = huff0.build_codebook(b"\x7f" * 20 + b"\x80")

    with pytest.raises(ValueError, match="even length"):
        huff0.encode_gpu_bfloat16(b"\x00", complete)
    with pytest.raises(ValueError, match="positive"):
        huff0.encode_gpu_bfloat16(b"", complete, bytes_per_thread=0)
    with pytest.raises(ValueError, match="positive"):
        huff0.encode_gpu_bfloat16(b"", complete, threads_per_block=0)
    with pytest.raises(ValueError, match="must be 8"):
        huff0.encode_gpu_bfloat16(b"", complete, bytes_per_thread=4)
    with pytest.raises(ValueError, match="must be 512"):
        huff0.encode_gpu_bfloat16(b"", complete, threads_per_block=256)
    with pytest.raises(ValueError, match="no codebook entry"):
        huff0.encode_gpu_bfloat16(_raw_from_lanes(b"\x00", b"\x00"), sparse)
    with pytest.raises(ValueError):
        huff0.encode_gpu_bfloat16(b"", complete.codebook[:-1])


def test_native_gpu_encoder_rejects_sizes_beyond_decoder_int32_abi():
    # A read-only strided view advertises the ABI-boundary size without
    # allocating or touching a multi-gigabyte buffer. The native check must run
    # before reading its contents.
    backing = np.zeros(1, dtype=np.uint8)
    advertised = np.lib.stride_tricks.as_strided(
        backing,
        shape=(2 * (1 << 31),),
        strides=(1,),
        writeable=False,
    )

    with pytest.raises(OverflowError, match="int32 CUDA decoder ABI"):
        _huff0_cpp.encode_gpu_bfloat16(advertised, _variable_codebook().codebook)


def test_native_gpu_encoder_snapshots_mutable_inputs():
    codebook = _variable_codebook()
    raw = bytearray(_raw_from_lanes(bytes(range(256)), bytes(reversed(range(256)))))
    header = bytearray(codebook.codebook)

    mutable_result = huff0.encode_gpu_bfloat16(raw, header)
    immutable_result = huff0.encode_gpu_bfloat16(bytes(raw), bytes(header))

    assert mutable_result == immutable_result
