import unittest

import numpy as np

from neuralzip import huffman


class HuffmanTests(unittest.TestCase):
    def test_codebook_from_counts_matches_codebook_from_bytes(self):
        source = bytes([0, 0, 0, 1, 1, 2, 4, 4, 4, 4]) * 128
        counts = np.bincount(np.frombuffer(source, dtype=np.uint8), minlength=256)

        direct = huffman.build_codebook(source, prior_count=1)
        counted = huffman.build_codebook_from_counts(counts, prior_count=1)

        self.assertEqual(direct.code_lengths, counted.code_lengths)
        encoded = huffman.encode_with_codebook(
            source,
            counted,
            min_saving_bytes=0,
        )
        self.assertIsNotNone(encoded)
        restored = huffman.decompress(encoded)
        self.assertEqual(restored, source)

    def test_codebook_from_counts_validates_alphabet(self):
        with self.assertRaises(ValueError):
            huffman.build_codebook_from_counts([1] * 255)
        with self.assertRaises(OverflowError):
            huffman.build_codebook_from_counts([-1] + [0] * 255)


if __name__ == "__main__":
    unittest.main()
