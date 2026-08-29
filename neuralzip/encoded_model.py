"""Weight-free encoded model values returned by ``NeuralZip.encode``."""

import copy
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import torch

from .codec_primitives import _tensor_from_raw_bytes, _tensor_raw_bytes


@dataclass(frozen=True, slots=True)
class _StoredTensor:
    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    data: bytes = field(repr=False)
    kind: str = "buffer"
    requires_grad: bool = False

    @classmethod
    def from_tensor(cls, name, tensor, *, kind):
        return cls(
            name=str(name),
            shape=tuple(tensor.shape),
            dtype=tensor.dtype,
            data=_tensor_raw_bytes(tensor),
            kind=str(kind),
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
    tensor_kinds: tuple[str, ...] = ()
    auxiliary_state: tuple[_StoredTensor, ...] = field(default=(), repr=False)
    state_aliases: tuple[tuple[str, str], ...] = ()
    model_class: type[torch.nn.Module] | None = field(default=None, repr=False)
    config_class: type | None = field(default=None, repr=False)
    config: dict | None = field(default=None, repr=False)
    module_training: tuple[tuple[str, bool], ...] = ()

    @property
    def compression_ratio(self):
        if self.raw_bytes:
            return self.compressed_bytes / self.raw_bytes
        return 0.0 if not self.compressed_bytes else math.inf

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
            "tensor_kinds": ("mapping",) * len(names),
        }
    if not isinstance(source, torch.nn.Module):
        raise TypeError("Expected a tensor mapping or torch.nn.Module source.")

    parameters = tuple(source.named_parameters(remove_duplicate=False))
    parameter_by_name = dict(parameters)
    buffers = tuple(source.named_buffers(remove_duplicate=False))
    buffer_by_name = dict(buffers)
    tensor_kinds = []
    for name in names:
        if name in parameter_by_name:
            tensor_kinds.append("parameter")
        elif name in buffer_by_name:
            tensor_kinds.append("buffer")
        else:
            raise ValueError(f"Encoded model state has no tensor named {name!r}.")

    auxiliary = []
    state_aliases = []
    encoded_names = set(names)
    for kind, items in (("parameter", parameters), ("buffer", buffers)):
        groups = {}
        for name, tensor in items:
            groups.setdefault(id(tensor), []).append((name, tensor))
        for group in groups.values():
            encoded_group = [name for name, _ in group if name in encoded_names]
            if len(encoded_group) > 1:
                raise ValueError(
                    "Aliased source tensors must be encoded only once: "
                    + ", ".join(encoded_group)
                )
            canonical = encoded_group[0] if encoded_group else group[0][0]
            if not encoded_group:
                auxiliary.append(
                    _StoredTensor.from_tensor(canonical, group[0][1], kind=kind)
                )
            state_aliases.extend(
                (name, canonical) for name, _ in group if name != canonical
            )

    named_modules = source.named_modules(remove_duplicate=False)
    module_training = tuple(
        (str(name), bool(module.training)) for name, module in named_modules
    )

    from transformers import PreTrainedModel

    metadata = {
        "source_kind": "module",
        "model_class": type(source),
        "module_training": module_training,
        "tensor_kinds": tuple(tensor_kinds),
        "auxiliary_state": tuple(auxiliary),
        "state_aliases": tuple(state_aliases),
    }
    if isinstance(source, PreTrainedModel):
        metadata.update(
            source_kind="huggingface",
            config_class=type(source.config),
            config=copy.deepcopy(source.config.to_dict()),
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
        raw_bytes=int(raw_bytes) + auxiliary_bytes,
        compressed_bytes=int(compressed_bytes) + auxiliary_bytes,
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
    tensor_kinds,
    requires_grad,
    state_aliases,
):
    tables = {
        "parameter": dict(module.named_parameters(remove_duplicate=False)),
        "buffer": dict(module.named_buffers(remove_duplicate=False)),
    }
    aliases_by_canonical = {}
    for alias, canonical in state_aliases:
        aliases_by_canonical.setdefault(canonical, []).append(alias)

    expected_names = {
        "parameter": {
            name for name, kind in tensor_kinds.items() if kind == "parameter"
        },
        "buffer": {name for name, kind in tensor_kinds.items() if kind == "buffer"},
    }
    for alias, canonical in state_aliases:
        expected_names[tensor_kinds[canonical]].add(alias)
    for kind, table in tables.items():
        observed = set(table)
        expected = expected_names[kind]
        if observed != expected:
            missing = sorted(expected - observed)
            unexpected = sorted(observed - expected)
            raise ValueError(
                f"Destination {kind} schema differs from the encoded model: "
                f"missing={missing!r}, unexpected={unexpected!r}."
            )

    planned = []
    for name, tensor in tensors.items():
        kind = tensor_kinds[name]
        members = (name, *aliases_by_canonical.get(name, ()))
        old_tensors = tuple(tables[kind][member] for member in members)
        for member, old in zip(members, old_tensors, strict=True):
            if tuple(tensor.shape) != tuple(old.shape):
                raise ValueError(f"Shape mismatch for {kind} {member!r}.")
        devices = {old.device for old in old_tensors}
        if len(devices) != 1:
            raise ValueError(
                f"Aliased destination {kind} tensors for {name!r} span devices."
            )
        restored = tensor.detach().to(device=old_tensors[0].device).contiguous()
        if kind == "parameter":
            restored = torch.nn.Parameter(
                restored,
                requires_grad=bool(requires_grad[name]),
            )
        else:
            restored.requires_grad_(bool(requires_grad[name]))
        planned.append((members, restored))

    # Commit only after the complete schema, every shape, and every alias group
    # has been validated. Assigning the same replacement to a group's members
    # recreates source tying even if the destination constructor did not.
    for members, tensor in planned:
        for member in members:
            _set_state_tensor(module, member, tensor)


def _module_training_targets(module, states):
    targets = []
    for name, training in states:
        try:
            target = module if not name else module.get_submodule(name)
        except AttributeError as error:
            raise ValueError(f"Destination has no module named {name!r}.") from error
        targets.append((target, training))
    return targets


def _reconstruct(
    encoded,
    tensors,
    *,
    destination=None,
    model_factory: Callable[[], torch.nn.Module] | None = None,
):
    if encoded.source_kind == "mapping":
        if destination is not None or model_factory is not None:
            raise ValueError("Mapping reconstruction does not accept a destination.")
        return {
            name: tensor.requires_grad_(encoded.requires_grad[index])
            for index, (name, tensor) in enumerate(tensors.items())
        }
    if destination is not None and model_factory is not None:
        raise ValueError("Pass destination or model_factory, not both.")

    if destination is None and model_factory is not None:
        destination = model_factory()
    elif destination is None and encoded.source_kind == "huggingface":
        config = encoded.config_class.from_dict(copy.deepcopy(encoded.config))
        destination = encoded.model_class(config)
    elif destination is None:
        raise TypeError(
            "Generic nn.Module reconstruction requires destination= or model_factory=."
        )
    if not isinstance(destination, torch.nn.Module):
        raise TypeError("The reconstruction destination must be a torch.nn.Module.")
    if encoded.model_class is not None and type(destination) is not encoded.model_class:
        raise TypeError(
            "Destination model class differs from the encoded source: "
            f"expected {encoded.model_class.__name__}, got "
            f"{type(destination).__name__}."
        )

    training_targets = _module_training_targets(destination, encoded.module_training)

    state = dict(tensors)
    kinds = dict(zip(encoded.names, encoded.tensor_kinds, strict=True))
    gradients = dict(zip(encoded.names, encoded.requires_grad, strict=True))
    for stored in encoded.auxiliary_state:
        state[stored.name] = stored.tensor()
        kinds[stored.name] = stored.kind
        gradients[stored.name] = stored.requires_grad
    _install_state(
        destination,
        state,
        kinds,
        gradients,
        encoded.state_aliases,
    )
    for module, training in training_targets:
        module.training = training
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
