"""Weight-free encoded model values returned by ``NeuralZip.encode``."""

from collections.abc import Mapping
from dataclasses import dataclass, field

import torch

from .codec_primitives import _tensor_from_raw_bytes, _tensor_raw_bytes


@dataclass(frozen=True, slots=True)
class _StoredTensor:
    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    data: bytes = field(repr=False)
    requires_grad: bool = False

    @classmethod
    def from_tensor(cls, name, tensor):
        return cls(
            name=str(name),
            shape=tuple(tensor.shape),
            dtype=tensor.dtype,
            data=_tensor_raw_bytes(tensor),
            requires_grad=bool(tensor.requires_grad),
        )

    def tensor(self):
        return _tensor_from_raw_bytes(self.data, self.dtype, self.shape)


@dataclass(frozen=True, slots=True, repr=False)
class EncodedModel:
    """Compressed parameter payloads plus minimal reconstruction metadata."""

    names: tuple[str, ...]
    payloads: tuple = field(repr=False)
    raw_bytes: int
    compressed_bytes: int
    encode_seconds: float
    source_kind: str
    codebooks: tuple = field(default=(), repr=False)
    mapping_seconds: float = 0.0
    requires_grad: tuple[bool, ...] = ()
    auxiliary_state: tuple[_StoredTensor, ...] = field(default=(), repr=False)
    state_aliases: tuple[tuple[str, str], ...] = ()
    model_class: type[torch.nn.Module] | None = field(default=None, repr=False)
    config_class: type | None = field(default=None, repr=False)
    config: dict | None = field(default=None, repr=False)
    module_training: tuple[tuple[str, bool], ...] = ()

    @property
    def compression_ratio(self):
        return self.compressed_bytes / self.raw_bytes

    @property
    def total_seconds(self):
        return self.mapping_seconds + self.encode_seconds

    def __repr__(self):
        return (
            f"EncodedModel(tensors={len(self.names)}, "
            f"compression_ratio={self.compression_ratio:.4f}, "
            f"encode_seconds={self.encode_seconds:.4f})"
        )


def _source_metadata(source, names):
    if isinstance(source, Mapping):
        return {
            "source_kind": "mapping",
        }

    parameters = tuple(source.named_parameters(remove_duplicate=False))
    buffers = tuple(source.named_buffers(remove_duplicate=False))
    auxiliary = []
    state_aliases = []
    encoded_names = set(names)
    for items in (parameters, buffers):
        groups = {}
        for name, tensor in items:
            groups.setdefault(id(tensor), []).append((name, tensor))
        for group in groups.values():
            encoded_group = [name for name, _ in group if name in encoded_names]
            canonical = encoded_group[0] if encoded_group else group[0][0]
            if not encoded_group:
                auxiliary.append(_StoredTensor.from_tensor(canonical, group[0][1]))
            state_aliases.extend(
                (name, canonical) for name, _ in group if name != canonical
            )

    module_training = tuple(
        (str(name), bool(module.training))
        for name, module in source.named_modules(remove_duplicate=False)
    )

    from transformers import PreTrainedModel

    metadata = {
        "source_kind": "module",
        "model_class": type(source),
        "module_training": module_training,
        "auxiliary_state": tuple(auxiliary),
        "state_aliases": tuple(state_aliases),
    }
    if isinstance(source, PreTrainedModel):
        metadata.update(
            source_kind="huggingface",
            config_class=type(source.config),
            config=source.config.to_dict(),
        )
    return metadata


def make_encoded_model(
    source,
    *,
    names,
    payloads,
    raw_bytes,
    compressed_bytes,
    encode_seconds,
    codebooks=(),
    mapping_seconds=0.0,
    requires_grad=None,
):
    """Build an EncodedModel without retaining ``source`` or its weights."""

    names = tuple(names)
    requires_grad = (
        tuple(requires_grad) if requires_grad is not None else (False,) * len(names)
    )
    source_metadata = _source_metadata(source, names)
    auxiliary_bytes = sum(
        len(value.data) for value in source_metadata.get("auxiliary_state", ())
    )
    return EncodedModel(
        names=names,
        payloads=tuple(payloads),
        codebooks=tuple(codebooks),
        raw_bytes=raw_bytes + auxiliary_bytes,
        compressed_bytes=compressed_bytes + auxiliary_bytes,
        encode_seconds=encode_seconds,
        mapping_seconds=mapping_seconds,
        requires_grad=requires_grad,
        **source_metadata,
    )


def _decode_payloads(encoded, codec):
    codec.load_codebooks(encoded.codebooks)
    decoded = tuple(codec.decompress_tensors(encoded.payloads))

    if len(decoded) != len(encoded.names):
        raise RuntimeError("Decoder returned a different number of tensors.")
    return {
        name: tensor.detach().contiguous()
        for name, tensor in zip(encoded.names, decoded, strict=True)
    }


def _set_state_tensor(module, name, tensor):
    parent_name, _, child_name = name.rpartition(".")
    parent = module.get_submodule(parent_name) if parent_name else module
    setattr(parent, child_name, tensor)


def _install_state(
    module,
    tensors,
    requires_grad,
    state_aliases,
):
    values = dict(tensors)
    for alias, canonical in state_aliases:
        values[alias] = values[canonical]

    destination_state = module.state_dict(keep_vars=True)
    persistent = {
        name: tensor.to(destination_state[name].device)
        for name, tensor in values.items()
        if name in destination_state
    }
    module.load_state_dict(persistent, strict=True, assign=True)

    buffers = dict(module.named_buffers(remove_duplicate=False))
    for name, tensor in values.items():
        if name not in persistent:
            old = buffers[name]
            _set_state_tensor(module, name, tensor.to(old.device).contiguous())

    parameters = dict(module.named_parameters(remove_duplicate=False))
    buffers = dict(module.named_buffers(remove_duplicate=False))
    for alias, canonical in state_aliases:
        tensor = parameters.get(canonical)
        if tensor is None:
            tensor = buffers[canonical]
        _set_state_tensor(module, alias, tensor)

    for name, requires in requires_grad.items():
        tensor = parameters.get(name)
        if tensor is None:
            tensor = buffers[name]
        tensor.requires_grad_(requires)


def _reconstruct(
    encoded,
    tensors,
    *,
    destination=None,
    model_factory=None,
):
    if encoded.source_kind == "mapping":
        return {
            name: tensor.requires_grad_(encoded.requires_grad[index])
            for index, (name, tensor) in enumerate(tensors.items())
        }
    if destination is None and model_factory is not None:
        destination = model_factory()
    elif destination is None and encoded.source_kind == "huggingface":
        destination = encoded.model_class(
            encoded.config_class.from_dict(encoded.config)
        )
    elif destination is None:
        raise TypeError(
            "Generic nn.Module reconstruction requires destination= or model_factory=."
        )
    state = dict(tensors)
    gradients = dict(zip(encoded.names, encoded.requires_grad, strict=True))
    for stored in encoded.auxiliary_state:
        state[stored.name] = stored.tensor()
        gradients[stored.name] = stored.requires_grad
    _install_state(
        destination,
        state,
        gradients,
        encoded.state_aliases,
    )
    for name, training in encoded.module_training:
        (
            destination if not name else destination.get_submodule(name)
        ).training = training
    return destination


def decode_model(encoded, codec, *, destination=None, model_factory=None):
    """Decode payloads and reconstruct their original input category."""

    return _reconstruct(
        encoded,
        _decode_payloads(encoded, codec),
        destination=destination,
        model_factory=model_factory,
    )


__all__ = ["EncodedModel"]
