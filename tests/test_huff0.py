import unittest

from neuralzip import huff0
from neuralzip.packed_exponents import tokenize_ngrams, topk_vocab


class Huff0Tests(unittest.TestCase):
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
        n = 3
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


if __name__ == "__main__":
    unittest.main()
