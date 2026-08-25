import unittest

import math
import numpy as np

from neuralzip.ablations import jensen_shannon
from neuralzip.metrics import jensen_shannon_distance, shannon_entropy_from_counts
from neuralzip.neuralzip_codec import _jensen_shannon


class MetricsTests(unittest.TestCase):
    def test_jensen_shannon_compatibility_aliases_share_one_implementation(self):
        self.assertIs(jensen_shannon, jensen_shannon_distance)
        self.assertIs(_jensen_shannon, jensen_shannon_distance)

    def test_jensen_shannon_characterization_values_are_unchanged(self):
        cases = (
            ((1.0, 0.0), (1.0, 0.0), 0.0),
            ((1.0, 0.0), (0.0, 1.0), 1.0),
            ((0.5, 0.5), (1.0, 0.0), 0.5579230452841438),
            ((0.0, 0.0), (0.0, 0.0), 0.0),
        )
        for left, right, expected in cases:
            with self.subTest(left=left, right=right):
                self.assertEqual(jensen_shannon_distance(left, right), expected)

    def test_jensen_shannon_clamps_roundoff_below_zero(self):
        left = np.array([20_000_000, 80_000_000], dtype=np.float64)
        right = np.array([20_000_000, 80_000_001], dtype=np.float64)
        distance = jensen_shannon_distance(left / left.sum(), right / right.sum())

        self.assertTrue(math.isfinite(distance))
        self.assertEqual(distance, 0.0)

    def test_entropy_from_counts_preserves_zero_and_known_distributions(self):
        self.assertEqual(shannon_entropy_from_counts(()), 0.0)
        self.assertEqual(shannon_entropy_from_counts((0, 0, 0)), 0.0)
        self.assertEqual(shannon_entropy_from_counts((2, 2)), 1.0)
        self.assertEqual(shannon_entropy_from_counts((4, 0)), 0.0)


if __name__ == "__main__":
    unittest.main()
