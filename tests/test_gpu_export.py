import unittest
from dataclasses import replace

from neuralzip.gpu_export import (
    BF16Lanes,
    GPUCodebook,
    GPUEncodedBF16,
    assemble_bfloat16_lanes,
    decode_bfloat16_reference,
    decode_exponents_reference,
    encode_bfloat16,
    encode_exponents,
    pack_restart_gaps,
    split_bfloat16_lanes,
    unpack_restart_gaps,
)


def _variable_codebook():
    # Complete code: 0, 10, 110, 111.  High-valued symbols ensure the LUT
    # does not reserve any of the 256 possible BF16 exponent values.
    lengths = [0] * 256
    values = [0] * 256
    for symbol, length, value in (
        (7, 1, 0b0),
        (127, 2, 0b10),
        (240, 3, 0b110),
        (255, 3, 0b111),
    ):
        lengths[symbol] = length
        values[symbol] = value
    return GPUCodebook.from_codes(lengths, values)


def _identity_codebook():
    return GPUCodebook.from_codes([8] * 256, list(range(256)))


def _max_depth_codebook():
    # Complete chain code with lengths 1..11 and two terminal 12-bit codes.
    lengths = [0] * 256
    values = [0] * 256
    for length in range(1, 12):
        lengths[length - 1] = length
        values[length - 1] = (1 << length) - 2
    lengths[11] = lengths[12] = 12
    values[11], values[12] = 0xFFE, 0xFFF
    return GPUCodebook.from_codes(lengths, values)


class GPUExportTests(unittest.TestCase):
    def test_bfloat16_lanes_roundtrip_every_bit_pattern(self):
        raw = b"".join(word.to_bytes(2, "little") for word in range(1 << 16))

        lanes = split_bfloat16_lanes(raw)

        self.assertEqual(lanes.numel, 1 << 16)
        self.assertEqual(lanes.assemble(), raw)
        self.assertEqual(
            assemble_bfloat16_lanes(lanes.exponents, lanes.sign_mantissa),
            raw,
        )

    def test_bfloat16_lane_validation_is_strict(self):
        with self.assertRaisesRegex(ValueError, "even length"):
            split_bfloat16_lanes(b"\x00")
        with self.assertRaisesRegex(ValueError, "same number"):
            BF16Lanes(b"\x00", b"")
        with self.assertRaisesRegex(TypeError, "must be omitted"):
            assemble_bfloat16_lanes(BF16Lanes(b"", b""), b"")

    def test_flat_lut_decodes_all_prefix_completions(self):
        codebook = _variable_codebook()

        self.assertEqual(codebook.max_bits, 3)
        self.assertEqual(codebook.lut_nbytes, 16)
        self.assertEqual(codebook.lookup(0b000), (7, 1))
        self.assertEqual(codebook.lookup(0b011), (7, 1))
        self.assertEqual(codebook.lookup(0b100), (127, 2))
        self.assertEqual(codebook.lookup(0b101), (127, 2))
        self.assertEqual(codebook.lookup(0b110), (240, 3))
        self.assertEqual(codebook.lookup(0b111), (255, 3))
        self.assertEqual(codebook.decode_lut[0b111], (3 << 8) | 255)

    def test_codebook_rejects_invalid_or_overlapping_codes(self):
        with self.assertRaisesRegex(ValueError, "256 entries"):
            GPUCodebook.from_codes([1], [0])

        lengths = [0] * 256
        values = [0] * 256
        lengths[0], values[0] = 1, 0
        lengths[1], values[1] = 2, 0
        with self.assertRaisesRegex(ValueError, "prefix-free"):
            GPUCodebook.from_codes(lengths, values)

        lengths[1], values[1] = 13, 0
        with self.assertRaisesRegex(ValueError, r"\[0, 12\]"):
            GPUCodebook.from_codes(lengths, values)

        lengths = [0] * 256
        values = [0] * 256
        lengths[0] = 1
        with self.assertRaisesRegex(ValueError, "cover every bit prefix"):
            GPUCodebook.from_codes(lengths, values)

    def test_continuous_encoder_is_msb_first_and_roundtrips(self):
        codebook = _variable_codebook()
        source = bytes((7, 127, 240, 255))

        encoded = encode_exponents(source, codebook)

        self.assertEqual(encoded.bit_length, 9)
        self.assertEqual(encoded.bitstream, b"\x5b\x80")
        self.assertEqual(encoded.output_positions, (0, 4))
        self.assertEqual(encoded.gaps[:2], (0, 0))
        self.assertEqual(decode_exponents_reference(encoded, codebook), source)

    def test_four_bit_gap_packing_matches_msb_first_layout(self):
        gaps = (0, 1, 2, 3, 15, 7)
        packed = pack_restart_gaps(gaps)

        expected_bits = "".join(f"{gap:04b}" for gap in gaps).ljust(24, "0")
        self.assertEqual(packed, int(expected_bits, 2).to_bytes(3, "big"))
        self.assertEqual(unpack_restart_gaps(packed, len(gaps)), gaps)

        with self.assertRaisesRegex(ValueError, "fit in 4 bits"):
            pack_restart_gaps((16,))
        with self.assertRaisesRegex(ValueError, "padding bits"):
            unpack_restart_gaps(b"\x01", 1)

    def test_restart_metadata_tracks_crossed_segments_and_blocks(self):
        codebook = _variable_codebook()
        # Three-bit symbols cross both 64-bit thread segments and the 32768-bit
        # CUDA block boundary.  The next symbol begins two and one bits after
        # those respective boundaries.
        source = bytes((255,)) * 12_000

        encoded = encode_exponents(source, codebook)

        self.assertEqual(encoded.gaps[0], 0)
        self.assertEqual(encoded.gaps[1], 2)
        self.assertEqual(encoded.output_positions, (0, 10_923, 12_000))
        self.assertEqual(decode_exponents_reference(encoded, codebook), source)

    def test_four_bits_hold_the_maximum_restart_gap(self):
        codebook = _max_depth_codebook()
        # The first six symbols occupy 63 bits. The following 12-bit symbol
        # crosses bit 64, so the next safe restart is exactly 11 bits later.
        source = bytes((6, 7, 11, 11, 11, 11, 11, 0))

        encoded = encode_exponents(source, codebook)

        self.assertEqual(codebook.max_bits, 12)
        self.assertEqual(encoded.gaps[1], 11)
        self.assertEqual(encoded.packed_gaps[0], 0x0B)
        self.assertEqual(decode_exponents_reference(encoded, codebook), source)

    def test_output_positions_for_byte_aligned_codes(self):
        codebook = _identity_codebook()
        source = bytes(index % 256 for index in range(40_000))

        encoded = encode_exponents(source, codebook)

        self.assertEqual(encoded.bitstream, source)
        self.assertEqual(
            encoded.output_positions,
            tuple(range(0, 36_865, 4_096)) + (40_000,),
        )
        self.assertEqual(decode_exponents_reference(encoded, codebook), source)

    def test_reference_decoder_detects_tampered_restart_metadata(self):
        codebook = _variable_codebook()
        encoded = encode_exponents(bytes((255,)) * 100, codebook)
        gaps = list(encoded.gaps)
        gaps[1] = 0
        damaged = replace(encoded, packed_gaps=pack_restart_gaps(gaps))

        with self.assertRaisesRegex(ValueError, "Packed gaps"):
            decode_exponents_reference(damaged, codebook)
        self.assertEqual(
            decode_exponents_reference(damaged, codebook, validate_metadata=False),
            bytes((255,)) * 100,
        )

    def test_bfloat16_gpu_payload_roundtrip_is_bitwise_exact(self):
        codebook = _identity_codebook()
        words = (0x0000, 0x8000, 0x3F80, 0xBF80, 0x7F80, 0x7FC1, 0xFFFF)
        raw = b"".join(word.to_bytes(2, "little") for word in words)

        encoded = encode_bfloat16(raw, codebook)

        self.assertIsInstance(encoded, GPUEncodedBF16)
        self.assertEqual(encoded.numel, len(words))
        self.assertEqual(decode_bfloat16_reference(encoded, codebook), raw)

    def test_empty_payload_has_canonical_metadata(self):
        codebook = _identity_codebook()

        encoded = encode_exponents(b"", codebook)

        self.assertEqual(encoded.bitstream, b"")
        self.assertEqual(encoded.packed_gaps, b"")
        self.assertEqual(encoded.output_positions, (0,))
        self.assertEqual(decode_exponents_reference(encoded, codebook), b"")

    def test_encoder_rejects_symbols_missing_from_codebook(self):
        with self.assertRaisesRegex(ValueError, "no codebook entry"):
            encode_exponents(b"\x00", _variable_codebook())

    def test_encoder_rejects_a_different_cuda_segment_abi(self):
        with self.assertRaisesRegex(ValueError, "bytes_per_thread must be 8"):
            encode_exponents(b"", _identity_codebook(), bytes_per_thread=4)
        with self.assertRaisesRegex(ValueError, "threads_per_block must be 512"):
            encode_exponents(b"", _identity_codebook(), threads_per_block=256)


if __name__ == "__main__":
    unittest.main()
