"""Weight-free encoded model values returned by ``NeuralZip.encode``."""

import copy
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import torch


@dataclass(frozen=True, slots=True)
class _StoredTensor:
    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    data: bytes = field(repr=False)
    kind: str = "buffer"
    requires_grad: bool = False

    def __post_init__(self):
        shape = tuple(int(dimension) for dimension in self.shape)
        if any(dimension < 0 for dimension in shape):
            raise ValueError("Stored tensor dimensions cannot be negative.")
        if not isinstance(self.data, bytes):
            raise TypeError("Stored tensor data must be immutable bytes.")
        if self.kind not in {"parameter", "buffer"}:
            raise ValueError(f"Unsupported stored tensor kind {self.kind!r}.")
        element_size = torch.empty((), dtype=self.dtype).element_size()
        expected_bytes = math.prod(shape) * element_size
        if len(self.data) != expected_bytes:
            raise ValueError(
                f"Stored tensor byte length mismatch: {len(self.data)} != "
                f"{expected_bytes}."
            )
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "requires_grad", bool(self.requires_grad))

    @classmethod
    def from_tensor(cls, name, tensor, *, kind):
        cpu = tensor.detach().cpu().contiguous()
        return cls(
            name=str(name),
            shape=tuple(cpu.shape),
            dtype=cpu.dtype,
            data=cpu.reshape(-1).view(torch.uint8).numpy().tobytes(),
            kind=str(kind),
            requires_grad=bool(tensor.requires_grad),
        )

    def tensor(self):
        if not self.data:
            return torch.empty(self.shape, dtype=self.dtype)
        raw = torch.frombuffer(bytearray(self.data), dtype=torch.uint8)
        return raw.view(self.dtype).reshape(self.shape)


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

    def __post_init__(self):
        names = tuple(str(name) for name in self.names)
        payloads = tuple(self.payloads)
        codebooks = tuple(self.codebooks)
        if not names or len(names) != len(payloads):
            raise ValueError("names and payloads must have the same non-zero length.")
        if len(set(names)) != len(names):
            raise ValueError("Encoded tensor names must be unique.")
        requires_grad = tuple(bool(value) for value in self.requires_grad)
        if requires_grad and len(requires_grad) != len(names):
            raise ValueError("requires_grad must match the encoded tensor count.")
        if not requires_grad:
            requires_grad = (False,) * len(names)
        tensor_kinds = tuple(str(value) for value in self.tensor_kinds)
        if not tensor_kinds:
            default_kind = "mapping" if self.source_kind == "mapping" else "parameter"
            tensor_kinds = (default_kind,) * len(names)
        if len(tensor_kinds) != len(names) or any(
            value not in {"mapping", "parameter", "buffer"} for value in tensor_kinds
        ):
            raise ValueError("tensor_kinds must describe every encoded tensor.")
        auxiliary_state = tuple(self.auxiliary_state)
        if any(not isinstance(value, _StoredTensor) for value in auxiliary_state):
            raise TypeError("auxiliary_state contains an unsupported value.")
        stored_names = names + tuple(value.name for value in auxiliary_state)
        if len(set(stored_names)) != len(stored_names):
            raise ValueError("Stored tensor names must be unique.")
        state_aliases = tuple(
            (str(alias), str(canonical)) for alias, canonical in self.state_aliases
        )
        alias_names = tuple(alias for alias, _ in state_aliases)
        if len(set(alias_names)) != len(alias_names):
            raise ValueError("State alias names must be unique.")
        if set(alias_names).intersection(stored_names):
            raise ValueError("A state alias cannot also store its own tensor.")
        missing_canonicals = {
            canonical for _, canonical in state_aliases if canonical not in stored_names
        }
        if missing_canonicals:
            raise ValueError(
                "State aliases reference missing canonical tensors: "
                + ", ".join(sorted(missing_canonicals))
            )
        module_training = tuple(
            (str(name), bool(training)) for name, training in self.module_training
        )
        if len({name for name, _ in module_training}) != len(module_training):
            raise ValueError("Module training-state names must be unique.")
        for name, payload in zip(names, payloads, strict=True):
            if torch.is_tensor(payload) or isinstance(payload, torch.nn.Module):
                raise TypeError("Encoded payloads cannot retain source weights.")
            if str(getattr(payload, "name", name)) != name:
                raise ValueError(f"Payload name does not match {name!r}.")
        if self.source_kind not in {"mapping", "huggingface", "module"}:
            raise ValueError(f"Unsupported source kind {self.source_kind!r}.")
        if self.source_kind == "huggingface" and (
            self.model_class is None or self.config_class is None or self.config is None
        ):
            raise ValueError("Hugging Face results require model and config metadata.")
        if self.source_kind == "mapping" and self.model_class is not None:
            raise ValueError("Mapping results cannot carry a model class.")
        if self.source_kind == "mapping" and (
            auxiliary_state or state_aliases or module_training
        ):
            raise ValueError("Mapping results cannot carry module state metadata.")

        numeric = {
            "raw_bytes": int(self.raw_bytes),
            "compressed_bytes": int(self.compressed_bytes),
            "encode_seconds": float(self.encode_seconds),
            "mapping_seconds": float(self.mapping_seconds),
        }
        if any(value < 0 for value in numeric.values()):
            raise ValueError("Encoded sizes and times cannot be negative.")
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "payloads", payloads)
        object.__setattr__(self, "codebooks", codebooks)
        object.__setattr__(self, "requires_grad", requires_grad)
        object.__setattr__(self, "tensor_kinds", tensor_kinds)
        object.__setattr__(self, "auxiliary_state", auxiliary_state)
        object.__setattr__(self, "state_aliases", state_aliases)
        object.__setattr__(self, "module_training", module_training)
        object.__setattr__(self, "config", copy.deepcopy(self.config))
        for name, value in numeric.items():
            object.__setattr__(self, name, value)

    @property
    def compression_ratio(self):
        if self.raw_bytes:
            return self.compressed_bytes / self.raw_bytes
        return 0.0 if not self.compressed_bytes else math.inf

    @property
    def total_seconds(self):
        return self.mapping_seconds + self.encode_seconds

    def prepare_inference(
        self,
        model,
        *,
        adapter=None,
        include_embeddings=False,
        device=None,
        block_devices=None,
        decoder=None,
        cpu_offload=False,
        cpu_offload_blocks=None,
        pin_memory=True,
    ):
        """Convert this encoded model to the GPU layout and attach its runtime.

        ``encode()`` owns model compression. This method consumes those encoded
        payloads, converts their sharing-only BF16 streams to the CUDA decoder
        layout, and uses ``model`` only as the executable module skeleton.
        """

        if self.source_kind == "mapping":
            raise ValueError("prepare_inference() requires an encoded torch.nn.Module.")
        from .gpu.inference import (
            NeuralZipRuntime,
            export_encoded_model_for_gpu,
        )

        exported = export_encoded_model_for_gpu(
            model,
            self,
            adapter=adapter,
            include_embeddings=include_embeddings,
        )
        NeuralZipRuntime.attach(
            model,
            exported,
            device=device,
            block_devices=block_devices,
            decoder=decoder,
            cpu_offload=cpu_offload,
            cpu_offload_blocks=cpu_offload_blocks,
            pin_memory=pin_memory,
        )
        return model

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

    parameters = _all_named_parameters(source)
    parameter_by_name = dict(parameters)
    buffers = _all_named_buffers(source)
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

    try:
        from transformers import PreTrainedModel
    except ImportError:  # pragma: no cover - transformers is a project dependency.
        PreTrainedModel = ()
    if isinstance(source, PreTrainedModel):
        to_dict = getattr(source.config, "to_dict", None)
        if not callable(to_dict):
            raise TypeError("The Hugging Face config must implement to_dict().")
        return {
            "source_kind": "huggingface",
            "model_class": type(source),
            "config_class": type(source.config),
            "config": copy.deepcopy(to_dict()),
            "module_training": module_training,
            "tensor_kinds": tuple(tensor_kinds),
            "auxiliary_state": tuple(auxiliary),
            "state_aliases": tuple(state_aliases),
        }
    return {
        "source_kind": "module",
        "model_class": type(source),
        "module_training": module_training,
        "tensor_kinds": tuple(tensor_kinds),
        "auxiliary_state": tuple(auxiliary),
        "state_aliases": tuple(state_aliases),
    }


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


def _decode_payloads(encoded, decoder):
    reset = getattr(decoder, "reset_decompression", None)
    if callable(reset):
        reset()
    load_codebooks = getattr(decoder, "load_codebooks", None)
    if encoded.codebooks and callable(load_codebooks):
        load_codebooks(encoded.codebooks)
    decode_many = getattr(decoder, "decompress_tensors", None)
    if callable(decode_many):
        decoded = tuple(decode_many(encoded.payloads))
    else:
        decode_one = getattr(decoder, "decompress_tensor", None)
        if not callable(decode_one):
            raise TypeError("Decoder must implement decompress_tensor(s).")
        decoded = tuple(decode_one(payload) for payload in encoded.payloads)

    if len(decoded) != len(encoded.names):
        raise RuntimeError("Decoder returned a different number of tensors.")
    return {
        name: tensor.detach().contiguous()
        for name, tensor in zip(encoded.names, decoded, strict=True)
    }


def _all_named_parameters(module):
    return tuple(module.named_parameters(remove_duplicate=False))


def _all_named_buffers(module):
    return tuple(module.named_buffers(remove_duplicate=False))


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
    parameters = _all_named_parameters(module)
    buffers = _all_named_buffers(module)
    tables = {
        "parameter": dict(parameters),
        "buffer": dict(buffers),
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
        from_dict = getattr(encoded.config_class, "from_dict", None)
        config_data = copy.deepcopy(encoded.config)
        config = (
            from_dict(config_data)
            if callable(from_dict)
            else encoded.config_class(**config_data)
        )
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


def decode_model(encoded, decoder, *, destination=None, model_factory=None):
    """Decode payloads and reconstruct their original input category."""

    return _reconstruct(
        encoded,
        _decode_payloads(encoded, decoder),
        destination=destination,
        model_factory=model_factory,
    )


__all__ = ["EncodedModel"]
