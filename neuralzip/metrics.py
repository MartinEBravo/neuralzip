"""Numerical metrics used by NeuralZip clustering."""

import numpy as np

from .utils import split_fields


def exponent_distribution(tensor):
    """Return the normalized 8-bit exponent histogram of a floating tensor."""

    _, exponent, _ = split_fields(tensor.detach().cpu().contiguous())
    stream = exponent.reshape(-1).numpy()
    counts = np.bincount(stream, minlength=256).astype(np.float64)
    total = counts.sum()
    return counts / total if total else counts


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
