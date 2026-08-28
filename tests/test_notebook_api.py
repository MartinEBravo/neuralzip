"""Executable contract for the public API shown in ``compress.ipynb``."""

import gc
import weakref
from collections.abc import Mapping

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from neuralzip import Compressor


def _tiny_gpt2(*, seed):
    torch.manual_seed(seed)
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=32,
            n_positions=8,
            n_ctx=8,
            n_embd=8,
            n_layer=1,
            n_head=2,
            bos_token_id=0,
            eos_token_id=1,
        )
    )
    return model.to(dtype=torch.bfloat16).eval()


def _parameter_snapshot(model):
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }


def _assert_lossless_tensor(expected, actual):
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert torch.equal(
        actual.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
        expected.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
    )


def _assert_reconstructed_gpt2(expected_parameters, restored):
    assert isinstance(restored, GPT2LMHeadModel)
    actual_parameters = dict(restored.named_parameters())
    assert tuple(actual_parameters) == tuple(expected_parameters)
    for name, expected in expected_parameters.items():
        _assert_lossless_tensor(expected, actual_parameters[name])


class _BufferedModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(4, 4, bias=False)
        self.normalization = torch.nn.BatchNorm1d(4)
        self.register_buffer(
            "runtime_scale",
            torch.arange(4, dtype=torch.bfloat16),
            persistent=False,
        )

    def forward(self, inputs):
        return self.normalization(self.projection(inputs))


def test_notebook_neuralzip_fit_encode_decode_reconstructs_hf_model():
    model = _tiny_gpt2(seed=1)
    model.transformer.wte.weight.requires_grad_(False)
    expected = _parameter_snapshot(model)
    source_parameters = tuple(
        weakref.ref(parameter) for parameter in model.parameters()
    )
    neuralzip = Compressor("neuralzip", keep_payloads=True)

    setup = neuralzip.fit(model)
    encoded = neuralzip.encode(model, setup=setup)

    # Neither the reusable setup nor the encoded artifact may keep the source
    # model's parameter objects alive.  Decoding must work from compressed data.
    del model
    gc.collect()
    assert all(reference() is None for reference in source_parameters)

    decoded = neuralzip.decode(encoded)
    _assert_reconstructed_gpt2(expected, decoded)
    assert decoded.transformer.wte.weight.requires_grad is False
    assert decoded.lm_head.weight.requires_grad is False


def test_notebook_zipnn_encode_decode_reconstructs_hf_model():
    pytest.importorskip("zipnn", reason="official ZipNN package is optional")
    model = _tiny_gpt2(seed=2)
    expected = _parameter_snapshot(model)
    zipnn = Compressor("zipnn", keep_payloads=True)

    encoded = zipnn.encode(model)
    decoded = zipnn.decode(encoded)

    _assert_reconstructed_gpt2(expected, decoded)


def test_neuralzip_setup_maps_a_related_target_and_round_trips_a_mapping():
    pattern = torch.tensor([1.0, 2.0, 1.0, 2.0], dtype=torch.bfloat16).repeat(2048)
    source = {
        "transformer.block.weight": pattern,
        "transformer.head.weight": pattern.flip(0).contiguous(),
    }
    target = {name: (tensor + 1).contiguous() for name, tensor in source.items()}
    expected = {name: tensor.clone() for name, tensor in target.items()}
    neuralzip = Compressor("neuralzip")

    setup = neuralzip.fit(source)
    encoded = neuralzip.encode(target, setup=setup)
    decoded = neuralzip.decode(encoded)

    assert isinstance(decoded, Mapping)
    assert tuple(decoded) == tuple(expected)
    for name, tensor in expected.items():
        _assert_lossless_tensor(tensor, decoded[name])


def test_saved_setup_path_reuses_codebooks_without_the_source_model(tmp_path):
    source = _tiny_gpt2(seed=3)
    source_parameters = tuple(
        weakref.ref(parameter) for parameter in source.parameters()
    )
    fitted = Compressor("neuralzip")
    fitted.fit(source)
    setup_path = fitted.save_setup(tmp_path / "gpt2.setup")

    del source, fitted
    gc.collect()
    assert all(reference() is None for reference in source_parameters)

    target = _tiny_gpt2(seed=4)
    expected = _parameter_snapshot(target)
    neuralzip = Compressor("neuralzip", setup=setup_path)
    encoded = neuralzip.encode(target)
    decoded = neuralzip.decode(encoded)

    _assert_reconstructed_gpt2(expected, decoded)


def test_model_roundtrip_preserves_persistent_buffers_and_complete_state():
    torch.manual_seed(8)
    model = _BufferedModule().train()
    model.normalization.eval()
    model.normalization.running_mean.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    model.normalization.running_var.copy_(torch.tensor([2.0, 3.0, 4.0, 5.0]))
    model.normalization.num_batches_tracked.fill_(17)
    model.projection.weight.requires_grad_(False)
    expected = {
        name: tensor.detach().clone() for name, tensor in model.state_dict().items()
    }
    expected_runtime_scale = model.runtime_scale.detach().clone()
    sample = torch.randn(3, 4)
    expected_output = model(sample)
    compressor = Compressor("neuralzip")

    setup = compressor.fit(model)
    encoded = compressor.encode(model, setup=setup)
    restored = compressor.decode(encoded, model_factory=_BufferedModule)

    assert restored.training is True
    assert restored.normalization.training is False
    assert restored.projection.weight.requires_grad is False
    for name, tensor in restored.state_dict().items():
        _assert_lossless_tensor(expected[name], tensor)
    _assert_lossless_tensor(expected_runtime_scale, restored.runtime_scale)
    assert torch.equal(restored(sample), expected_output)


def test_neuralzip_mapping_round_trip_supports_scalar_and_empty_tensors():
    source = {
        "weight": torch.arange(4096, dtype=torch.float16),
        "scale": torch.tensor(0.5, dtype=torch.bfloat16),
        "empty": torch.empty(0, 3, dtype=torch.float32),
    }
    compressor = Compressor("neuralzip")

    setup = compressor.fit(source)
    encoded = compressor.encode(source, setup=setup)
    restored = compressor.decode(encoded)

    for name, tensor in source.items():
        _assert_lossless_tensor(tensor, restored[name])
