import pytest
import torch

from neuralzip.dtypes import dtype_name, resolve_torch_dtype


def test_dtype_name_is_canonical():
    assert dtype_name(torch.float32) == "float32"
    assert dtype_name("torch.bfloat16") == "bfloat16"


@pytest.mark.parametrize(
    ("value", "expected"),
    (
        (None, "auto"),
        ("auto", "auto"),
        ("AUTO", "auto"),
        ("fp16", torch.float16),
        ("torch.float16", torch.float16),
        ("bf16", torch.bfloat16),
        ("f32", torch.float32),
        (torch.float32, torch.float32),
    ),
)
def test_resolve_torch_dtype_preserves_supported_aliases(value, expected):
    assert resolve_torch_dtype(value) == expected


def test_resolve_torch_dtype_rejects_unknown_values():
    with pytest.raises(ValueError, match="Unsupported dtype"):
        resolve_torch_dtype("float64")
