import struct
import unittest

from neuralzip import huff0
from neuralzip.packed_exponents import tokenize_ngrams, topk_vocab


class Huff0PrecomputedFourStreamTests(unittest.TestCase):
    def test_bounded_prebuilt_frame_keeps_four_stream_workspace_slack(self):
        # Regression: constraining Huff0's destination buffer to the final
        # threshold before encoding rejected this valid 12-byte frame.
        source = bytes.fromhex("00010001000100010001000100010000")
        regular = huff0.compress_many([source], threads=1)[0]
        codebook = huff0.build_prebuilt_block_codebook(source)

        self.assertIsNotNone(codebook)
        self.assertEqual(len(regular), 12)
        self.assertEqual(
            huff0.encode_prebuilt_block(
                source,
                codebook,
                compression_threshold=1.0,
            ),
            regular,
        )
        unlimited = huff0.encode_with_codebook_4x(source, codebook)
        self.assertIsNotNone(unlimited)
        self.assertEqual(codebook.codebook + unlimited.payload, regular)

    def test_prebuilt_local_frames_match_regular_local_blocks(self):
        chunks = [
            bytes([1, 2, 1, 2, 1, 3]) * (16 << 10),
            bytes([9, 8, 9, 8, 7, 8]) * (16 << 10),
        ]
        codebooks = [huff0.build_prebuilt_block_codebook(chunk) for chunk in chunks]
        self.assertTrue(all(codebook is not None for codebook in codebooks))

        frames = huff0.encode_prebuilt_blocks(
            chunks,
            codebooks,
            compression_threshold=1.0,
            threads=4,
        )
        regular = huff0.compress_many(chunks, threads=4)

        self.assertEqual(frames, regular)
        self.assertEqual(
            huff0.encode_prebuilt_block(
                chunks[0],
                codebooks[0],
                compression_threshold=1.0,
            ),
            regular[0],
        )
        for source, frame in zip(chunks, frames):
            self.assertEqual(huff0.decompress_block(frame, len(source)), source)

        expected_native_bytes = len(codebooks[0].codebook) + 257 * struct.calcsize("P")
        self.assertEqual(
            huff0.codebook_memory_bytes(codebooks[0]),
            expected_native_bytes,
        )
        self.assertIsNone(
            huff0.build_prebuilt_block_codebook(bytes(range(256)) * 256)
        )

    def test_single_and_parallel_precomputed_four_stream_roundtrip(self):
        chunks = [
            b"abcde" * (16 << 10),
            b"edcba" * (16 << 10),
        ]
        codebook = huff0.build_codebook(b"".join(chunks), prior_count=1)
        decodebook = huff0.build_decodebook(codebook.codebook)

        singles = [
            huff0.encode_with_codebook_4x(chunk, codebook) for chunk in chunks
        ]
        batched = huff0.encode_many_with_codebook_4x(
            chunks,
            codebook,
            compression_threshold=1.0,
            threads=4,
        )

        for source, single, batch_payload in zip(chunks, singles, batched):
            self.assertIsNotNone(single)
            self.assertEqual(batch_payload, single.payload)
            self.assertEqual(
                huff0.decompress_with_decodebook_4x(single, decodebook),
                source,
            )

        # Existing one-stream APIs remain unchanged and independently decodable.
        legacy = huff0.encode_with_codebook(chunks[0], codebook)
        self.assertIsNotNone(legacy)
        self.assertEqual(
            huff0.decompress_with_decodebook(legacy, decodebook),
            chunks[0],
        )

    def test_shared_batch_rejects_exact_threshold_equality(self):
        source = b"abcde" * 4096
        codebook = huff0.build_codebook(source, prior_count=1)

        for frame_4x in (False, True):
            with self.subTest(frame_4x=frame_4x):
                single_encoder = (
                    huff0.encode_with_codebook_4x
                    if frame_4x
                    else huff0.encode_with_codebook
                )
                batch_encoder = (
                    huff0.encode_many_with_codebook_4x
                    if frame_4x
                    else huff0.encode_many_with_codebook
                )
                payload = single_encoder(source, codebook)
                self.assertIsNotNone(payload)
                metadata_bytes = len(source) - len(payload.payload)
                self.assertGreater(metadata_bytes, 0)

                # Direct mixed encoding accepts only a strict improvement.  A
                # shared frame whose payload plus metadata lands exactly on
                # the threshold must therefore be rejected by both batch APIs.
                self.assertEqual(
                    batch_encoder(
                        [source],
                        codebook,
                        compression_threshold=1.0,
                        metadata_bytes=metadata_bytes,
                        threads=2,
                    ),
                    [None],
                )

    def test_pretokenized_batch_honors_each_exact_payload_limit(self):
        token_chunks = [
            bytes([1, 2, 1, 2]) * (16 << 10),
            bytes([3, 4, 3, 4]) * (16 << 10),
        ]
        codebook = huff0.build_codebook(b"".join(token_chunks), prior_count=1)
        decodebook = huff0.build_decodebook(codebook.codebook)

        payloads = huff0.encode_many_pretokenized_with_codebook_4x(
            token_chunks,
            codebook,
            [-1, 1],
            threads=4,
        )

        self.assertIsNotNone(payloads[0])
        self.assertIsNone(payloads[1])
        token_payload = huff0.Huff0Payload(
            payloads[0],
            codebook.codebook,
            len(token_chunks[0]),
        )
        self.assertEqual(
            huff0.decompress_with_decodebook_4x(token_payload, decodebook),
            token_chunks[0],
        )
        with self.assertRaises(ValueError):
            huff0.encode_many_pretokenized_with_codebook_4x(
                token_chunks,
                codebook,
                [-1],
            )

    def test_packed_ngram_four_stream_roundtrip(self):
        chunks = [
            bytes([127, 127, 126, 127, 127, 127, 124, 127]) * (8 << 10),
            bytes([126, 126, 127, 126, 124, 126, 127, 126]) * (8 << 10),
        ]
        n = 3
        vocab = topk_vocab(b"".join(chunks), n, top_k=16)
        token_chunks = [tokenize_ngrams(chunk, n, vocab)[0] for chunk in chunks]
        codebook = huff0.build_codebook(b"".join(token_chunks), prior_count=1)
        decodebook = huff0.build_decodebook(codebook.codebook)

        payloads = huff0.encode_many_ngram_with_codebook_4x(
            chunks,
            codebook,
            vocab,
            n=n,
            compression_threshold=1.0,
            threads=4,
        )

        for source, payload in zip(chunks, payloads):
            self.assertIsNotNone(payload)
            self.assertEqual(
                huff0.decompress_ngram_with_decodebook_4x(
                    payload,
                    decodebook,
                    vocab,
                    n=n,
                ),
                source,
            )


if __name__ == "__main__":
    unittest.main()
