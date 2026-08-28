import torch

from neuralzip.dtypes import dtype_name


def test_dtype_name_is_canonical():
    assert dtype_name(torch.float32) == "float32"
    assert dtype_name("torch.bfloat16") == "bfloat16"
