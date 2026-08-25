import unittest

import torch

from neuralzip.ablations import zipnn_exponent_stream
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

    def test_ablation_transpose_is_byte_identical_to_legacy_layout(self):
        tensor = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            dtype=torch.bfloat16,
        )
        original = zipnn_exponent_stream(tensor, layout="original")
        expected = (
            torch.frombuffer(bytearray(original), dtype=torch.uint8)
            .reshape(tuple(tensor.shape))
            .t()
            .contiguous()
            .reshape(-1)
            .numpy()
            .tobytes()
        )

        self.assertEqual(
            zipnn_exponent_stream(tensor, layout="transpose2d"),
            expected,
        )

    def test_ablation_transpose_preserves_non_matrix_stream(self):
        tensor = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
        self.assertEqual(
            zipnn_exponent_stream(tensor, layout="transpose2d"),
            zipnn_exponent_stream(tensor, layout="original"),
        )

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
