import torch

from neuralzip.tensors import bitwise_equal


def test_bitwise_equal_requires_shape_dtype_and_payload_identity():
    value = torch.tensor([1.0, 2.0], dtype=torch.float32)
    assert bitwise_equal(value, value.clone())
    assert not bitwise_equal(value, value.reshape(2, 1))
    assert not bitwise_equal(value, value.view(torch.int32))
    changed = value.clone()
    changed[0] = 3.0
    assert not bitwise_equal(value, changed)
