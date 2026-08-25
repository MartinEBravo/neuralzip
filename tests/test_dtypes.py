import pytest
import torch

from neuralzip.dtypes import dtype_name, resolve_torch_dtype
from neuralzip.model import resolve_torch_dtype as model_resolve_torch_dtype
from neuralzip.zipnn_codec import _dtype_name


def test_dtype_name_has_one_compatibility_implementation():
    assert _dtype_name is dtype_name
    assert dtype_name(torch.float32) == "float32"
    assert dtype_name("torch.bfloat16") == "bfloat16"


@pytest.mark.parametrize(
    ("value", "expected"),
    (
        (None, "auto"),
        ("auto", "auto"),
        ("fp16", torch.float16),
        ("torch.float16", torch.float16),
        ("bf16", torch.bfloat16),
        ("f32", torch.float32),
        (torch.float32, torch.float32),
    ),
)
def test_resolve_torch_dtype_preserves_supported_aliases(value, expected):
    assert resolve_torch_dtype(value) == expected
    assert model_resolve_torch_dtype(value) == expected


def test_resolve_torch_dtype_rejects_unknown_values():
    with pytest.raises(ValueError, match="Unsupported dtype"):
        resolve_torch_dtype("float64")
