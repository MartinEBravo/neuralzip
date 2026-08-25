import unittest

from neuralzip import huff0
from neuralzip.packed_exponents import tokenize_ngrams, topk_vocab


def _raw_from_bit_reordered_streams(*streams):
    """Invert ZipNN's bit reorder for FP16/FP32-like byte streams."""
    raw = bytearray()
    for values in zip(*streams, strict=True):
        raw.extend(values[:-2])
        penultimate, last = values[-2:]
        raw.append((penultimate & 0x7F) | ((last & 1) << 7))
        raw.append(((last >> 1) & 0x7F) | (penultimate & 0x80))
    return bytes(raw)


class Huff0Tests(unittest.TestCase):
    def test_single_block_facade_composes_codebook_encode_and_decode(self):
        source = b"abcde" * 4096
        codebook = huff0.build_codebook(source, prior_count=1)
        encoded = huff0.encode_with_codebook(source, codebook)
        payload = huff0.compress(source, prior_count=1)

        self.assertIsNotNone(encoded)
        self.assertEqual(payload, encoded)
        self.assertEqual(huff0.decompress(payload), source)
        self.assertIsNone(huff0.compress(source, min_saving_bytes=len(source)))
        with self.assertRaisesRegex(ValueError, "min_saving_bytes"):
            huff0.compress(source, min_saving_bytes=-1)

    def test_parallel_local_blocks_roundtrip_and_keep_order(self):
        chunks = [
            bytes([index % 5 for index in range(64 << 10)]),
            bytes([9, 8, 9, 7]) * (16 << 10),
            bytes(range(256)) * 256,
        ]
        payloads = huff0.compress_many(chunks, threads=4)

        self.assertIsNotNone(payloads[0])
        self.assertIsNotNone(payloads[1])
        self.assertIsNone(payloads[2])
        for source, payload in zip(chunks, payloads):
            if payload is not None:
                self.assertEqual(huff0.decompress_block(payload, len(source)), source)

    def test_parallel_shared_codebook_roundtrip(self):
        chunks = [b"abcde" * 4096, b"edcba" * 4096]
        codebook = huff0.build_codebook(b"".join(chunks), prior_count=1)
        payloads = huff0.encode_many_with_codebook(
            chunks,
            codebook,
            threads=4,
            compression_threshold=1.0,
        )
        decodebook = huff0.build_decodebook(codebook.codebook)

        for source, payload in zip(chunks, payloads):
            self.assertIsNotNone(payload)
            restored = huff0.decompress_with_decodebook(
                huff0.Huff0Payload(payload, codebook.codebook, len(source)),
                decodebook,
            )
            self.assertEqual(restored, source)

    def test_output_limit_returns_none(self):
        source = b"abcde" * 4096
        codebook = huff0.build_codebook(source, prior_count=1)
        self.assertIsNone(
            huff0.encode_with_codebook(source, codebook, max_output_bytes=1)
        )

    def test_packed_ngram_roundtrip_in_cpp(self):
        source = (bytes([127, 127, 126, 127, 127, 127, 124, 127]) * 1024) + b"x"
        for n in (3, 8):
            with self.subTest(n=n):
                vocab = topk_vocab(source, n, top_k=16)
                tokens, _, _, _, _ = tokenize_ngrams(source, n, vocab)
                codebook = huff0.build_codebook(tokens, prior_count=1)
                payload = huff0.encode_ngram_with_codebook(
                    source,
                    codebook,
                    vocab,
                    n=n,
                )
                self.assertIsNotNone(payload)
                decodebook = huff0.build_decodebook(codebook.codebook)
                restored = huff0.decompress_ngram_with_decodebook(
                    payload,
                    decodebook,
                    vocab,
                    n=n,
                )
                self.assertEqual(restored, source)

    def test_parallel_packed_ngram_matches_single_and_roundtrips(self):
        chunks = [
            (bytes([127, 127, 126, 127, 127, 127, 124, 127]) * (32 << 10)) + b"x",
            (bytes([126, 126, 127, 126, 124, 126, 127, 126]) * (32 << 10)) + b"yz",
            bytes(range(256)) * 1024,
        ]
        n = 3
        metadata_bytes = 50
        compression_threshold = 1.0
        vocab = topk_vocab(b"".join(chunks[:2]), n, top_k=16)
        token_streams = [tokenize_ngrams(chunk, n, vocab)[0] for chunk in chunks]
        codebook = huff0.build_codebook(b"".join(token_streams), prior_count=1)

        expected = []
        for chunk in chunks:
            payload = huff0.encode_ngram_with_codebook(
                chunk,
                codebook,
                vocab,
                n=n,
                max_output_bytes=max(
                    0,
                    int(len(chunk) * compression_threshold) - metadata_bytes,
                ),
            )
            if (
                payload is not None
                and metadata_bytes
                + len(payload.payload)
                + len(payload.escape_bytes)
                + len(payload.tail)
                >= len(chunk) * compression_threshold
            ):
                payload = None
            expected.append(payload)

        actual = huff0.encode_many_ngram_with_codebook(
            chunks,
            codebook,
            vocab,
            n=n,
            compression_threshold=compression_threshold,
            metadata_bytes=metadata_bytes,
            threads=4,
        )

        self.assertEqual(actual, expected)
        self.assertIsNotNone(actual[0])
        self.assertIsNone(actual[2])
        decodebook = huff0.build_decodebook(codebook.codebook)
        for source, payload in zip(chunks, actual):
            if payload is not None:
                restored = huff0.decompress_ngram_with_decodebook(
                    payload,
                    decodebook,
                    vocab,
                    n=n,
                )
                self.assertEqual(restored, source)

    def test_mixed_packed_matches_contiguous_bit_reordered_tokenization(self):
        def interleave_reordered_streams(first, second):
            raw = bytearray()
            for first_byte, second_byte in zip(first, second, strict=True):
                raw.append((first_byte & 0x7F) | ((second_byte & 1) << 7))
                raw.append(((second_byte >> 1) & 0x7F) | (first_byte & 0x80))
            return bytes(raw)

        for n in (2, 3, 8):
            with self.subTest(n=n):
                packed_stream = bytes([127, 127, 126, 127, 127, 127, 124, 127]) * 8193
                first_stream = bytes(range(256)) * ((len(packed_stream) + 255) // 256)
                first_stream = first_stream[: len(packed_stream)]
                raw = interleave_reordered_streams(first_stream, packed_stream)

                vocab = topk_vocab(packed_stream, n, top_k=16)
                tokens, escapes, tail, hits, misses = tokenize_ngrams(
                    packed_stream, n, vocab
                )
                codebook = huff0.build_codebook(tokens, prior_count=1)
                expected = huff0.encode_ngram_with_codebook_4x(
                    packed_stream,
                    codebook,
                    vocab,
                    n=n,
                    max_output_bytes=len(packed_stream) - 50,
                )
                self.assertIsNotNone(expected)
                self.assertLess(
                    50
                    + len(expected.payload)
                    + len(expected.escape_bytes)
                    + len(expected.tail),
                    len(packed_stream),
                )

                groups = huff0.compress_zipnn_mixed(
                    raw,
                    element_size=2,
                    bit_reorder=1,
                    chunk_size=len(packed_stream),
                    stream_modes=(
                        {"kind": "raw"},
                        {
                            "kind": "packed_huff0",
                            "codebook": codebook,
                            "vocab": vocab,
                            "ngram_n": n,
                            "frame_4x": True,
                        },
                    ),
                    compression_threshold=1.0,
                    threads=2,
                )

                self.assertEqual(
                    groups[0][0],
                    ("raw", first_stream, len(first_stream), None, b"", b"", 1, 0, 0),
                )
                self.assertEqual(
                    groups[1][0],
                    (
                        "packed_huff0",
                        expected.payload,
                        len(packed_stream),
                        len(tokens),
                        escapes,
                        tail,
                        n,
                        hits,
                        misses,
                    ),
                )

                if n == 3:
                    fallback = huff0.compress_zipnn_mixed(
                        raw,
                        element_size=2,
                        bit_reorder=1,
                        chunk_size=len(packed_stream),
                        stream_modes=(
                            {"kind": "raw"},
                            {
                                "kind": "packed_huff0",
                                "codebook": codebook,
                                "vocab": vocab,
                                "ngram_n": n,
                                "frame_4x": True,
                            },
                        ),
                        compression_threshold=0.01,
                        threads=2,
                    )
                    self.assertEqual(fallback[1][0][0], "raw")
                    self.assertEqual(fallback[1][0][1], packed_stream)

    def test_mixed_many_matches_single_calls_for_fp16_fp32_and_keeps_order(self):
        length = 48 << 10
        packed_stream = (
            bytes([127, 127, 126, 127, 127, 127, 124, 127]) * ((length + 7) // 8)
        )[:length]
        fp16_first_stream = (bytes(range(256)) * ((length + 255) // 256))[:length]
        ngram_n = 3
        packed_vocab = topk_vocab(packed_stream, ngram_n, top_k=16)
        packed_tokens = tokenize_ngrams(packed_stream, ngram_n, packed_vocab)[0]
        packed_codebook = huff0.build_codebook(packed_tokens, prior_count=1)
        fp16_modes = (
            {"kind": "raw"},
            {
                "kind": "packed_huff0",
                "codebook": packed_codebook,
                "vocab": packed_vocab,
                "ngram_n": ngram_n,
                "frame_4x": True,
            },
        )
        fp16_spec = (
            _raw_from_bit_reordered_streams(
                fp16_first_stream,
                packed_stream,
            ),
            2,
            1,
            12 << 10,
            fp16_modes,
        )

        fp32_streams = (
            (b"layernorm-feedforward-" * ((length + 21) // 22))[:length],
            (bytes(range(256)) * ((length + 255) // 256))[:length],
            (b"attention-projection-" * ((length + 20) // 21))[:length],
            (b"abcde" * ((length + 4) // 5))[:length],
        )
        shared_codebook = huff0.build_codebook(fp32_streams[3], prior_count=1)
        fp32_modes = (
            {"kind": "local_huff0"},
            {"kind": "raw"},
            {"kind": "local_huff0"},
            {
                "kind": "shared_huff0",
                "codebook": shared_codebook,
                "frame_4x": True,
            },
        )
        fp32_spec = (
            _raw_from_bit_reordered_streams(*fp32_streams),
            4,
            1,
            12 << 10,
            fp32_modes,
        )

        specs = (fp16_spec, fp32_spec)
        common = {
            "min_saving_bytes": 0,
            "threads": 4,
            "compression_threshold": 1.0,
        }
        expected = [
            huff0.compress_zipnn_mixed(
                data,
                element_size=element_size,
                bit_reorder=bit_reorder,
                chunk_size=chunk_size,
                stream_modes=stream_modes,
                **common,
            )
            for data, element_size, bit_reorder, chunk_size, stream_modes in specs
        ]

        actual = huff0.compress_zipnn_mixed_many(specs, **common)

        self.assertEqual(actual, expected)
        self.assertEqual(
            huff0.compress_zipnn_mixed_many(tuple(reversed(specs)), **common),
            list(reversed(expected)),
        )
        emitted_modes = {
            chunk[0]
            for tensor_groups in actual
            for stream_chunks in tensor_groups
            for chunk in stream_chunks
        }
        self.assertTrue(
            {"raw", "local_huff0", "shared_huff0", "packed_huff0"} <= emitted_modes
        )

    def test_mixed_many_matches_single_calls_when_planned_modes_fallback(self):
        length = 8192
        shared_stream = (b"abcde" * ((length + 4) // 5))[:length]
        packed_stream = (
            bytes([127, 127, 126, 127, 127, 127, 124, 127]) * ((length + 7) // 8)
        )[:length]
        ngram_n = 3
        packed_vocab = topk_vocab(packed_stream, ngram_n, top_k=16)
        packed_codebook = huff0.build_codebook(
            tokenize_ngrams(packed_stream, ngram_n, packed_vocab)[0],
            prior_count=1,
        )
        shared_codebook = huff0.build_codebook(shared_stream, prior_count=1)
        stream_modes = (
            {
                "kind": "shared_huff0",
                "codebook": shared_codebook,
                "frame_4x": True,
            },
            {
                "kind": "packed_huff0",
                "codebook": packed_codebook,
                "vocab": packed_vocab,
                "ngram_n": ngram_n,
                "frame_4x": True,
            },
        )
        spec = (
            _raw_from_bit_reordered_streams(shared_stream, packed_stream),
            2,
            1,
            length,
            stream_modes,
        )
        common = {
            "min_saving_bytes": 0,
            "threads": 4,
            "compression_threshold": 0.001,
        }
        expected = huff0.compress_zipnn_mixed(
            spec[0],
            element_size=spec[1],
            bit_reorder=spec[2],
            chunk_size=spec[3],
            stream_modes=spec[4],
            **common,
        )

        actual = huff0.compress_zipnn_mixed_many((spec,), **common)

        self.assertEqual(actual, [expected])
        self.assertTrue(
            all(
                chunk[0] in {"raw", "local_huff0"}
                for stream_chunks in actual[0]
                for chunk in stream_chunks
            )
        )

    def test_mixed_shared_rejection_uses_exact_local_then_raw_fallback(self):
        length = 8192
        local_stream = (b"ab" * ((length + 1) // 2))[:length]
        raw_stream = (bytes(range(256)) * ((length + 255) // 256))[:length]
        mismatched_source = (b"cd" * ((length + 1) // 2))[:length]
        flat_codebook = huff0.build_codebook(mismatched_source, prior_count=1)
        modes = tuple(
            {
                "kind": "shared_huff0",
                "codebook": flat_codebook,
                "frame_4x": True,
            }
            for _ in range(2)
        )
        interleaved = bytearray(length * 2)
        interleaved[0::2] = local_stream
        interleaved[1::2] = raw_stream

        groups = huff0.compress_zipnn_mixed(
            interleaved,
            element_size=2,
            bit_reorder=0,
            chunk_size=length,
            stream_modes=modes,
            compression_threshold=1.0,
            threads=2,
        )
        expected_local = huff0.compress_many([local_stream], threads=1)[0]
        self.assertIsNotNone(expected_local)
        self.assertEqual(
            groups[0][0],
            (
                "local_huff0",
                expected_local,
                length,
                None,
                b"",
                b"",
                1,
                0,
                0,
            ),
        )
        self.assertEqual(
            groups[1][0],
            ("raw", raw_stream, length, None, b"", b"", 1, 0, 0),
        )

    def test_mixed_many_empty_and_releases_buffers_after_late_error(self):
        self.assertEqual(huff0.compress_zipnn_mixed_many((), threads=4), [])

        raw_modes = ({"kind": "raw"}, {"kind": "raw"})
        first_buffer = bytearray(b"\x01\x02" * 256)
        valid_spec = (first_buffer, 2, 0, 64, raw_modes)
        invalid_later_spec = (b"\x00\x01\x02", 2, 0, 64, raw_modes)

        with self.assertRaisesRegex(
            ValueError,
            "data length must be divisible by element_size",
        ):
            huff0.compress_zipnn_mixed_many(
                (valid_spec, invalid_later_spec),
                threads=2,
            )

        # Resizing would raise BufferError if the first native Py_buffer leaked.
        first_buffer.extend(b"\x03\x04")
        valid_spec = (first_buffer, 2, 0, 64, raw_modes)
        expected = huff0.compress_zipnn_mixed(
            first_buffer,
            element_size=2,
            bit_reorder=0,
            chunk_size=64,
            stream_modes=raw_modes,
            threads=2,
        )
        self.assertEqual(
            huff0.compress_zipnn_mixed_many((valid_spec,), threads=2),
            [expected],
        )


if __name__ == "__main__":
    unittest.main()
