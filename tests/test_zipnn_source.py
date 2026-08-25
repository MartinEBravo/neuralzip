import importlib.util
import struct
import sys
import unittest

from neuralzip.zipnn_source import ZipNNSourceCodec, parse_zipnn_streams


def _frame(*, element_size, original_bytes, chunk_exponent, totals):
    header = bytearray(32)
    header[:2] = b"ZN"
    header[5] = 220 if element_size == 4 else 10
    header[6] = 1
    header[8] = 2
    header[14] = chunk_exponent
    header[16:24] = int(original_bytes).to_bytes(8, "little")
    shape = bytes((1, 2, 4, 0))
    chunk_bytes = 1 << chunk_exponent
    chunks = (original_bytes + chunk_bytes - 1) // chunk_bytes
    types = bytes(element_size * chunks)
    entries = []
    for total in totals:
        for chunk in range(chunks):
            entries.append((total * (chunk + 1)) // chunks)
    sizes = b"".join(
        int(value).to_bytes(struct.calcsize("P"), sys.byteorder)
        for value in entries
    )
    data = bytes(sum(totals))
    payload = header + shape + types + sizes + data
    payload[24:32] = len(payload).to_bytes(8, "little")
    return bytes(payload)


class ZipNNSourceParserTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("zipnn"), "zipnn is not installed")
    def test_shared_adapter_is_non_destructive_and_reports_one_time_boundary(self):
        import torch

        source = torch.arange(4096, dtype=torch.bfloat16)
        reference = source.clone()
        codec = ZipNNSourceCodec(
            threads=2,
            chunk_size=256 << 10,
            compression_threshold=0.95,
        )

        encoded = codec.encode_tensor(source, verify=True)

        self.assertTrue(torch.equal(source.view(torch.uint8), reference.view(torch.uint8)))
        self.assertTrue(encoded.roundtrip_ok)
        self.assertGreaterEqual(encoded.input_copy_seconds, 0.0)
        self.assertGreaterEqual(encoded.codec_encode_seconds, 0.0)
        self.assertAlmostEqual(
            encoded.encode_seconds,
            encoded.input_copy_seconds + encoded.codec_encode_seconds,
        )
        self.assertEqual(encoded.layout.exponent_stream.raw_bytes, source.numel())

    def test_official_empty_tensor_golden_frames(self):
        frames = (
            (
                2,
                "5a4e0005040a010102000000000012060000000000000000"
                "2300000000000000010100",
                10,
            ),
            (
                4,
                "5a4e000504dc010102000000000012010000000000000000"
                "2300000000000000010100",
                220,
            ),
        )
        for element_size, encoded, byte_reorder in frames:
            with self.subTest(element_size=element_size):
                layout = parse_zipnn_streams(
                    bytes.fromhex(encoded),
                    element_size=element_size,
                    expected_raw_bytes=0,
                )

                self.assertEqual(layout.archive_bytes, 35)
                self.assertEqual(layout.header_bytes, 35)
                self.assertEqual(layout.chunk_count, 0)
                self.assertEqual(layout.byte_reorder, byte_reorder)
                self.assertTrue(
                    all(stream.compressed_bytes == 0 for stream in layout.streams)
                )

    def test_bf16_stream_accounting(self):
        payload = _frame(
            element_size=2,
            original_bytes=64,
            chunk_exponent=4,
            totals=(11, 13),
        )
        layout = parse_zipnn_streams(
            payload,
            element_size=2,
            expected_raw_bytes=64,
        )

        self.assertEqual(layout.chunk_count, 4)
        self.assertEqual(layout.exponent_stream.raw_bytes, 32)
        self.assertEqual(layout.exponent_stream.payload_bytes, 13)
        self.assertEqual(
            layout.exponent_stream.metadata_bytes,
            4 * (1 + struct.calcsize("P")),
        )

    def test_fp32_stream_accounting(self):
        payload = _frame(
            element_size=4,
            original_bytes=128,
            chunk_exponent=5,
            totals=(5, 7, 9, 17),
        )
        layout = parse_zipnn_streams(payload, element_size=4)

        self.assertEqual(layout.chunk_count, 4)
        self.assertEqual(layout.exponent_stream.stream_id, 3)
        self.assertEqual(layout.exponent_stream.raw_bytes, 32)
        self.assertEqual(layout.exponent_stream.payload_bytes, 17)

    def test_empty_stream_has_no_chunk_metadata(self):
        payload = _frame(
            element_size=2,
            original_bytes=0,
            chunk_exponent=18,
            totals=(0, 0),
        )
        layout = parse_zipnn_streams(payload, element_size=2, expected_raw_bytes=0)

        self.assertEqual(layout.chunk_count, 0)
        self.assertEqual(layout.exponent_stream.raw_bytes, 0)
        self.assertEqual(layout.exponent_stream.payload_bytes, 0)
        self.assertEqual(layout.exponent_stream.metadata_bytes, 0)

    def test_rejects_truncated_and_inconsistent_frames(self):
        payload = _frame(
            element_size=2,
            original_bytes=64,
            chunk_exponent=4,
            totals=(11, 13),
        )
        with self.assertRaises(ValueError):
            parse_zipnn_streams(payload[:-1], element_size=2)
        with self.assertRaises(ValueError):
            parse_zipnn_streams(payload, element_size=2, expected_raw_bytes=32)
        byte_frame = bytearray(payload)
        byte_frame[8] = 1
        with self.assertRaisesRegex(ValueError, "torch or numpy"):
            parse_zipnn_streams(byte_frame, element_size=2)

    @unittest.skipUnless(importlib.util.find_spec("zipnn"), "zipnn is not installed")
    def test_official_zipnn_frames_roundtrip_and_parse(self):
        import torch
        from zipnn import ZipNN

        for dtype in (torch.bfloat16, torch.float32):
            chunk_elements = (256 << 10) // torch.empty((), dtype=dtype).element_size()
            for elements in (0, chunk_elements - 1, chunk_elements, chunk_elements + 1):
                with self.subTest(dtype=dtype, elements=elements):
                    tensor = torch.arange(elements, dtype=dtype)
                    codec = ZipNN(
                        method="HUFFMAN",
                        input_format="torch",
                        threads=2,
                        compression_chunk=256 << 10,
                        compression_threshold=0.95,
                    )
                    payload = codec.compress(tensor.clone())
                    layout = parse_zipnn_streams(
                        payload,
                        element_size=tensor.element_size(),
                        expected_raw_bytes=tensor.numel() * tensor.element_size(),
                    )

                    self.assertEqual(layout.archive_bytes, len(payload))
                    self.assertEqual(layout.exponent_stream.raw_bytes, tensor.numel())
                    restored = codec.decompress(payload)
                    self.assertTrue(
                        torch.equal(tensor.view(torch.uint8), restored.view(torch.uint8))
                    )


if __name__ == "__main__":
    unittest.main()
