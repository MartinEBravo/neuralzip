import unittest

from neuralzip.packed_exponents import (
    restore_2d_stream,
    restore_ngrams,
    tokenize_ngrams,
    topk_vocab,
    transpose_2d_stream,
)


class PackedExponentTests(unittest.TestCase):
    def test_ngram_escape_roundtrip(self):
        source = bytes([127, 127, 126, 127, 127, 127, 130, 129, 5])
        for n in (2, 3, 4, 8):
            with self.subTest(n=n):
                vocab = topk_vocab(source[:6], n, top_k=2)
                tokens, escapes, tail, _, _ = tokenize_ngrams(source, n, vocab)
                restored = restore_ngrams(tokens, n, vocab, escapes, tail)
                self.assertEqual(restored, source)

    def test_transpose_roundtrip(self):
        source = bytes(range(12))
        transposed, layout = transpose_2d_stream(source, (3, 4))
        self.assertEqual(layout, "transpose2d")
        self.assertEqual(restore_2d_stream(transposed, (3, 4), layout), source)

    def test_empty_vocabulary_uses_escapes(self):
        source = b"abcdefg"
        tokens, escapes, tail, hits, misses = tokenize_ngrams(source, 3, b"")
        self.assertEqual((hits, misses), (0, 2))
        self.assertEqual(restore_ngrams(tokens, 3, b"", escapes, tail), source)

    def test_ngram8_vocabulary_hit_and_escape_roundtrip(self):
        source = b"abcdefgh" + b"ijklmnop" + b"z"
        vocab = topk_vocab(source[:8], 8, top_k=1)
        tokens, escapes, tail, hits, misses = tokenize_ngrams(source, 8, vocab)
        self.assertEqual((hits, misses), (1, 1))
        self.assertEqual(len(vocab), 8)
        self.assertEqual(restore_ngrams(tokens, 8, vocab, escapes, tail), source)


if __name__ == "__main__":
    unittest.main()
