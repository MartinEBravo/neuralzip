import numpy as np

from neuralzip.ablations import first_order_entropy, zero_order_entropy


def test_first_order_entropy_detects_alternating_sequence():
    values = np.array([0, 1] * 64, dtype=np.uint8)

    assert zero_order_entropy(values) == 1.0
    assert first_order_entropy(values) == 0.0


def test_first_order_entropy_matches_zero_order_for_uniform_transitions():
    values = np.tile(np.array([0, 0, 1, 1], dtype=np.uint8), 64)

    assert np.isclose(zero_order_entropy(values), 1.0)
    assert first_order_entropy(values) > 0.99


def test_first_order_entropy_handles_short_sequences():
    assert first_order_entropy(np.array([], dtype=np.uint8)) == 0.0
    assert first_order_entropy(np.array([7], dtype=np.uint8)) == 0.0
