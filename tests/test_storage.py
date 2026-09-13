import pytest
import torch

from neuralzip.setup import NeuralZipSetup
from neuralzip.storage import decode_storage, encode_storage


@pytest.mark.parametrize(
    "dtype,chunk_size,expected_chunks",
    [
        (torch.bfloat16, 9, [4, 4, 3]),
        (torch.float16, 9, [4, 4, 3]),
        (torch.float32, 9, [2, 2, 2, 2, 2, 1]),
        (torch.float32, 3, [1] * 11),
    ],
)
def test_chunk_size_roundtrip(dtype, chunk_size, expected_chunks):
    original = torch.arange(11, dtype=torch.float32).to(dtype)
    named_tensors = {"weight": original}
    setup = NeuralZipSetup(
        chunk_size=chunk_size,
        enable_shared_codebooks=False,
        enable_packed_codebooks=False,
        threads=1,
    ).build(named_tensors)

    blob, metadata, _, _ = encode_storage(tuple(named_tensors.items()), setup)

    for group in metadata["tensors"][0]["groups"]:
        assert [chunk["symbols"] for chunk in group] == expected_chunks
    decoded = decode_storage(blob, metadata)["weight"]
    assert decoded.dtype == original.dtype
    assert decoded.shape == original.shape
    assert torch.equal(decoded.view(torch.uint8), original.view(torch.uint8))
