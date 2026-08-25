"""Shared numerical metrics used by compression and experiment pipelines."""

import numpy as np


def jensen_shannon_distance(left, right):
    """Return the base-2 Jensen--Shannon distance between two distributions.

    Inputs are intentionally not normalized here.  This preserves the behavior
    of the historical codec and ablation helpers, whose callers already pass
    probability distributions.
    """
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    middle = 0.5 * (left + right)
    left_mask = left > 0
    right_mask = right > 0
    divergence = 0.5 * np.sum(
        left[left_mask] * np.log2(left[left_mask] / middle[left_mask])
    )
    divergence += 0.5 * np.sum(
        right[right_mask] * np.log2(right[right_mask] / middle[right_mask])
    )
    return float(np.sqrt(max(float(divergence), 0.0)))


# Descriptive alias for experiment code that historically used this shorter
# name.  Both names intentionally refer to the same implementation.
jensen_shannon = jensen_shannon_distance


def shannon_entropy_from_counts(counts):
    """Return zero-order base-2 entropy for non-negative symbol counts."""

    counts = np.asarray(counts, dtype=np.float64)
    total = float(counts.sum())
    if total == 0.0:
        return 0.0
    probabilities = counts[counts > 0] / total
    return float(-(probabilities * np.log2(probabilities)).sum())
