from __future__ import annotations

from dataclasses import replace
from types import ModuleType
from unittest.mock import patch

import pytest
import torch
from torch import nn

from examples.experiments import dfloat11_in_memory as helper


class _ToyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.left = nn.Linear(4, 3, bias=False, dtype=torch.bfloat16)
        self.right = nn.Linear(3, 2, bias=False, dtype=torch.bfloat16)


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_ToyBlock(), _ToyBlock()])


class _FakeCodec:
    pass


def _fake_api() -> helper._OfficialDFloat11API:
    utils = ModuleType("fake_dfloat11_utils")
    utils.tqdm = lambda iterable, *args, **kwargs: iterable

    def get_codec(weight):
        assert weight.dtype == torch.bfloat16
        return _FakeCodec(), {0: int(weight.numel())}

    def get_32bit_codec(counter):
        assert counter
        return _FakeCodec(), counter, {0: (1, 0)}

    def get_luts(table):
        assert table
        return torch.zeros((2, 256), dtype=torch.uint8)

    def encode_weights(weights, codec, bytes_per_thread, threads_per_block):
        assert isinstance(codec, _FakeCodec)
        assert bytes_per_thread == 8
        assert threads_per_block == 512
        numels = [weight.numel() for weight in weights]
        numel = sum(numels)
        encoded = torch.zeros((7,), dtype=torch.uint8)
        sign_mantissa = torch.zeros((numel,), dtype=torch.uint8)
        output_positions = torch.tensor([0, numel], dtype=torch.uint32)
        gaps = torch.zeros((5,), dtype=torch.uint8)
        split_positions = torch.cumsum(torch.tensor(numels, dtype=torch.int64), 0)[:-1]
        return encoded, sign_mantissa, output_positions, gaps, split_positions

    def get_hook(threads_per_block, bytes_per_thread):
        assert tuple(threads_per_block) == (512,)
        assert bytes_per_thread == 8

        def hook(module, inputs):
            return None

        return hook

    return helper._OfficialDFloat11API(
        get_codec=get_codec,
        get_32bit_codec=get_32bit_codec,
        get_luts=get_luts,
        encode_weights=encode_weights,
        get_hook=get_hook,
        utils_module=utils,
        bytes_per_thread=8,
        threads_per_block=(512,),
        version="0.5.0-test",
    )


def _encode(model: nn.Module) -> helper.DFloat11InMemoryEncoding:
    with patch.object(helper, "_load_official_api", return_value=_fake_api()):
        return helper.encode_dfloat11_in_memory(
            model,
            {r"layers\.\d+": ["left", "right"]},
        )


def test_encode_uses_exact_exponent_ratio_buffers_without_mutating_model():
    model = _ToyModel()
    original_weights = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }

    encoding = _encode(model)

    assert encoding.block_names == ("layers.0", "layers.1")
    assert encoding.dfloat11_version == "0.5.0-test"
    assert encoding.raw_exponent_bytes == 36
    assert encoding.exponent_stream_bytes == 14
    assert encoding.exponent_metadata_bytes == 26
    assert encoding.exponent_codebook_bytes == 1024
    assert encoding.exponent_compressed_bytes == 1064
    assert encoding.exponent_ratio == pytest.approx(1064 / 36)
    assert encoding.sign_mantissa_bytes == 36
    assert encoding.gpu_payload_bytes == 1100
    assert encoding.split_positions_host_bytes == 16
    assert encoding.encode_seconds >= 0.0
    assert set(encoding.weight_names) == set(original_weights)
    assert set(dict(model.named_parameters())) == set(original_weights)
    for name, expected in original_weights.items():
        assert torch.equal(model.get_parameter(name), expected)


def test_install_matches_official_runtime_buffer_and_injection_layout():
    model = _ToyModel()
    encoding = _encode(model)

    with patch.object(helper, "_load_official_api", return_value=_fake_api()):
        returned = helper.install_dfloat11_runtime(model, encoding)

    assert returned is model
    assert not tuple(model.named_parameters())
    for block_result, block in zip(encoding.blocks, model.layers, strict=True):
        assert block.weight_injection_modules == [block.left, block.right]
        assert block.split_positions == [12]
        assert block.shared_mem_size == 512 * 4 + 4 + block_result.numel * 2
        assert block.output_positions.dtype == torch.uint8
        assert block.output_positions.numel() == block_result.output_positions.nbytes
        assert torch.equal(block.encoded_exponent, block_result.encoded_exponent)
        assert torch.equal(block.sign_mantissa, block_result.sign_mantissa)
        assert torch.equal(block.gaps, block_result.gaps)
        assert torch.equal(block.luts, block_result.luts)
        assert len(block._forward_pre_hooks) == 1


def test_install_validates_all_targets_before_mutating_any_weight():
    model = _ToyModel()
    encoding = _encode(model)
    damaged_first = replace(
        encoding.blocks[0],
        weight_shapes=((99, 99), *encoding.blocks[0].weight_shapes[1:]),
    )
    damaged = replace(encoding, blocks=(damaged_first, *encoding.blocks[1:]))

    with (
        patch.object(helper, "_load_official_api", return_value=_fake_api()),
        pytest.raises(RuntimeError, match="changed shape"),
    ):
        helper.install_dfloat11_runtime(model, damaged)

    assert len(tuple(model.named_parameters())) == 4
    assert all(not hasattr(block, "encoded_exponent") for block in model.layers)


def test_encode_rejects_a_non_bfloat16_target():
    model = _ToyModel()
    model.layers[1].right = nn.Linear(3, 2, bias=False, dtype=torch.float32)

    with (
        patch.object(helper, "_load_official_api", return_value=_fake_api()),
        pytest.raises(TypeError, match="must be bfloat16"),
    ):
        helper.encode_dfloat11_in_memory(
            model,
            {r"layers\.\d+": ["left", "right"]},
        )


def test_stable_cuda_memory_rejects_cpu_devices():
    with pytest.raises(ValueError, match="CUDA device"):
        helper.stable_cuda_allocated_bytes("cpu")
