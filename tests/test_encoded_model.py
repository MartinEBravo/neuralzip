import gc
import math
import weakref
from dataclasses import dataclass

import pytest
import torch

from neuralzip.encoded_model import (
    EncodedModel,
    _StoredTensor,
    decode_model,
    make_encoded_model,
)


@dataclass(frozen=True)
class FakePayload:
    name: str
    token: int


class FakeDecoder:
    def __init__(self, tensors):
        self.tensors = tensors
        self.codebooks = None

    def reset_decompression(self):
        pass

    def load_codebooks(self, codebooks):
        self.codebooks = tuple(codebooks)

    def decompress_tensors(self, payloads):
        return tuple(self.tensors[payload.token].clone() for payload in payloads)


class SmallModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(3, 2, bias=False)


class TwoLayerModule(torch.nn.Module):
    def __init__(self, *, second_shape=(2, 2)):
        super().__init__()
        self.first = torch.nn.Linear(2, 2, bias=False)
        self.second = torch.nn.Parameter(torch.empty(second_shape))


class AliasModule(torch.nn.Module):
    def __init__(self, *, tied):
        super().__init__()
        self.first = torch.nn.Linear(2, 2, bias=False)
        self.second = torch.nn.Linear(2, 2, bias=False)
        if tied:
            self.second.weight = self.first.weight


def test_encoded_mapping_does_not_retain_source_weights():
    source = {"weight": torch.arange(6, dtype=torch.float16)}
    reference = weakref.ref(source["weight"])
    encoded = make_encoded_model(
        source,
        method="neuralzip",
        names=("weight",),
        payloads=(FakePayload("weight", 0),),
        codebooks=(b"table",),
        raw_bytes=12,
        compressed_bytes=8,
        encode_seconds=0.25,
        mapping_seconds=0.1,
    )
    expected = torch.arange(6, dtype=torch.float16)

    del source
    gc.collect()
    assert reference() is None
    restored = decode_model(encoded, FakeDecoder({0: expected}))

    assert torch.equal(restored["weight"], expected)
    assert encoded.compression_ratio == pytest.approx(2 / 3)
    assert encoded.total_seconds == pytest.approx(0.35)
    assert encoded.encoder_path == "compress_tensor"


def test_huggingface_reconstruction_preserves_dtypes_and_ties():
    transformers = pytest.importorskip("transformers")
    source = (
        transformers.GPT2LMHeadModel(
            transformers.GPT2Config(
                vocab_size=19,
                n_positions=8,
                n_embd=8,
                n_layer=1,
                n_head=2,
            )
        )
        .to(dtype=torch.bfloat16)
        .eval()
    )
    tensors = {
        name: parameter.detach().clone()
        for name, parameter in source.named_parameters()
    }
    payloads = tuple(FakePayload(name, index) for index, name in enumerate(tensors))
    encoded = make_encoded_model(
        source,
        method="neuralzip",
        names=tensors,
        payloads=payloads,
        raw_bytes=sum(value.nbytes for value in tensors.values()),
        compressed_bytes=123,
        encode_seconds=0.1,
    )

    restored = decode_model(
        encoded,
        FakeDecoder(dict(enumerate(tensors.values()))),
    )

    assert type(restored) is transformers.GPT2LMHeadModel
    assert restored.training is False
    for name, expected in tensors.items():
        actual = dict(restored.named_parameters())[name]
        assert actual.dtype is expected.dtype
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    assert (
        restored.transformer.wte.weight.data_ptr() == restored.lm_head.weight.data_ptr()
    )


def test_generic_module_requires_a_destination_or_factory():
    source = SmallModule().to(dtype=torch.float16)
    tensor = source.projection.weight.detach().clone()
    encoded = make_encoded_model(
        source,
        method="neuralzip",
        names=("projection.weight",),
        payloads=(FakePayload("projection.weight", 0),),
        raw_bytes=tensor.nbytes,
        compressed_bytes=8,
        encode_seconds=0.2,
    )
    decoder = FakeDecoder({0: tensor})

    with pytest.raises(TypeError, match="requires destination= or model_factory="):
        decode_model(encoded, decoder)
    restored = decode_model(encoded, decoder, model_factory=SmallModule)
    assert restored.projection.weight.dtype is torch.float16
    assert torch.equal(restored.projection.weight.detach(), tensor)


def test_encoded_value_rejects_raw_tensor_payloads():
    with pytest.raises(TypeError, match="cannot retain source weights"):
        EncodedModel(
            method="neuralzip",
            names=("weight",),
            payloads=(torch.ones(1),),
            source_kind="mapping",
            raw_bytes=4,
            compressed_bytes=4,
            encode_seconds=0.1,
        )


def test_encoded_repr_is_not_polluted_by_payload_or_codebook_bytes():
    encoded = make_encoded_model(
        {"weight": torch.ones(4, dtype=torch.float16)},
        method="neuralzip",
        names=("weight",),
        payloads=(FakePayload("weight", 0),),
        codebooks=(b"secret-codebook" * 100,),
        raw_bytes=8,
        compressed_bytes=4,
        encode_seconds=0.125,
    )

    rendered = repr(encoded)

    assert rendered == (
        "EncodedModel(method='neuralzip', tensors=1, "
        "compression_ratio=0.5000, encode_seconds=0.1250)"
    )
    assert "secret-codebook" not in rendered


def test_nonempty_encoding_of_an_empty_tensor_has_infinite_ratio():
    encoded = EncodedModel(
        method="zipnn",
        names=("empty",),
        payloads=(b"frame",),
        source_kind="mapping",
        raw_bytes=0,
        compressed_bytes=5,
        encode_seconds=0.1,
    )

    assert math.isinf(encoded.compression_ratio)


def test_destination_is_unchanged_when_late_state_validation_fails():
    source = TwoLayerModule(second_shape=(3, 2))
    tensors = {
        name: parameter.detach().clone()
        for name, parameter in source.named_parameters()
    }
    encoded = make_encoded_model(
        source,
        method="neuralzip",
        names=tuple(tensors),
        payloads=tuple(FakePayload(name, index) for index, name in enumerate(tensors)),
        raw_bytes=sum(tensor.nbytes for tensor in tensors.values()),
        compressed_bytes=12,
        encode_seconds=0.1,
        requires_grad=tuple(
            parameter.requires_grad for parameter in source.parameters()
        ),
    )
    destination = TwoLayerModule(second_shape=(2, 2))
    before = {
        name: parameter.detach().clone()
        for name, parameter in destination.named_parameters()
    }

    with pytest.raises(ValueError, match="Shape mismatch"):
        decode_model(
            encoded,
            FakeDecoder(dict(enumerate(tensors.values()))),
            destination=destination,
        )

    for name, parameter in destination.named_parameters():
        assert torch.equal(parameter.detach(), before[name])


def test_empty_auxiliary_tensor_round_trips_without_a_buffer_view_error():
    stored = _StoredTensor.from_tensor(
        "empty",
        torch.empty(0, dtype=torch.int64),
        kind="buffer",
    )

    restored = stored.tensor()

    assert restored.dtype is torch.int64
    assert restored.shape == (0,)


def test_source_parameter_aliases_are_recreated_in_an_untied_destination():
    source = AliasModule(tied=True)
    tensor = source.first.weight.detach().clone()
    encoded = make_encoded_model(
        source,
        method="neuralzip",
        names=("first.weight",),
        payloads=(FakePayload("first.weight", 0),),
        raw_bytes=tensor.nbytes,
        compressed_bytes=8,
        encode_seconds=0.1,
        requires_grad=(True,),
    )

    restored = decode_model(
        encoded,
        FakeDecoder({0: tensor}),
        model_factory=lambda: AliasModule(tied=False),
    )

    assert restored.first.weight is restored.second.weight
    assert torch.equal(restored.first.weight.detach(), tensor)


def test_untied_source_parameters_are_not_left_tied_by_the_destination():
    source = AliasModule(tied=False)
    tensors = {
        name: parameter.detach().clone()
        for name, parameter in source.named_parameters()
    }
    encoded = make_encoded_model(
        source,
        method="neuralzip",
        names=tuple(tensors),
        payloads=tuple(FakePayload(name, index) for index, name in enumerate(tensors)),
        raw_bytes=sum(tensor.nbytes for tensor in tensors.values()),
        compressed_bytes=16,
        encode_seconds=0.1,
        requires_grad=(True, True),
    )

    restored = decode_model(
        encoded,
        FakeDecoder(dict(enumerate(tensors.values()))),
        model_factory=lambda: AliasModule(tied=True),
    )

    assert restored.first.weight is not restored.second.weight
    assert torch.equal(restored.first.weight.detach(), tensors["first.weight"])
    assert torch.equal(restored.second.weight.detach(), tensors["second.weight"])
