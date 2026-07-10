import unittest

from neuralzip import huff0
from neuralzip.packed_exponents import tokenize_ngrams, topk_vocab


class Huff0Tests(unittest.TestCase):
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
