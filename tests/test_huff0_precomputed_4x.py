import unittest

from neuralzip import huff0
from neuralzip.packed_exponents import tokenize_ngrams, topk_vocab


class Huff0PrecomputedFourStreamTests(unittest.TestCase):
    def test_single_and_parallel_precomputed_four_stream_roundtrip(self):
        chunks = [
            b"abcde" * (16 << 10),
            b"edcba" * (16 << 10),
        ]
        codebook = huff0.build_codebook(b"".join(chunks), prior_count=1)
        decodebook = huff0.build_decodebook(codebook.codebook)

        singles = [huff0.encode_with_codebook_4x(chunk, codebook) for chunk in chunks]
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
