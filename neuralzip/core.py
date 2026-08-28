"""Stateless public operations for fitting, encoding, and inference."""

import time
from collections.abc import Mapping
from os import PathLike

import torch

from ._setup import load_setup as load_setup_state
from ._setup import materialize_setup, setup_from_codec, validate_setup
from ._setup import save_setup as save_setup_state
from .encoded_model import EncodedModel, decode_model, make_encoded_model
from .neuralzip_codec import NeuralZipCodec
from .operations import (
    DecodeStats,
    EncodeStats,
    encode_tensor_batch,
    prepare_neuralzip_codec,
)
from .serialization import serialized_size


def _named_tensors(value, prefix=""):
    if torch.is_tensor(value):
        yield prefix, value
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = str(key) if not prefix else f"{prefix}.{key}"
            yield from _named_tensors(child, name)
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            name = str(index) if not prefix else f"{prefix}.{index}"
            yield from _named_tensors(child, name)


def _model_source(model):
    if isinstance(model, (Mapping, torch.nn.Module)):
        return model
    wrapped = getattr(model, "model", None)
    if isinstance(wrapped, (Mapping, torch.nn.Module)):
        return wrapped
    return None


def _floating_tensors(model):
    """Return floating parameters or tensor leaves, excluding module buffers."""

    source = _model_source(model)
    if isinstance(source, Mapping):
        items = _named_tensors(source)
    elif isinstance(source, torch.nn.Module):
        items = source.named_parameters()
    else:
        try:
            items = iter(model)
        except TypeError as error:
            raise TypeError(
                "Expected a tensor mapping, torch.nn.Module, or adapter exposing "
                "one through .model."
            ) from error
    tensors = [
        (str(name), tensor)
        for name, tensor in items
        if torch.is_tensor(tensor) and tensor.is_floating_point()
    ]
    if not tensors:
        raise ValueError("No floating tensors were found.")
    return tensors


def _model_tensors(model):
    """Include persistent floating buffers for reconstructable model encodes."""

    source = _model_source(model)
    if not isinstance(source, torch.nn.Module):
        return _floating_tensors(model)

    tensors = []
    seen = set()
    for name, tensor in source.named_parameters():
        if tensor.is_floating_point() and id(tensor) not in seen:
            tensors.append((str(name), tensor))
            seen.add(id(tensor))

    persistent = set(source.state_dict())
    try:
        buffers = source.named_buffers(remove_duplicate=False)
    except TypeError:  # pragma: no cover - compatibility with older PyTorch.
        buffers = source.named_buffers()
    for name, tensor in buffers:
        if name in persistent and tensor.is_floating_point() and id(tensor) not in seen:
            tensors.append((str(name), tensor))
            seen.add(id(tensor))
    if not tensors:
        raise ValueError("No floating tensors were found.")
    return tensors


def _encoded_source(model, named_tensors):
    source = _model_source(model)
    return source if source is not None else dict(named_tensors)


def _load_pretrained_model(
    source,
    *,
    model_class=None,
    model_kwargs=None,
    require_bfloat16=False,
):
    if model_class is None:
        from transformers import AutoModelForCausalLM

        model_class = AutoModelForCausalLM
    loader = getattr(model_class, "from_pretrained", None)
    if not callable(loader):
        raise TypeError("model_class must provide from_pretrained().")

    options = dict(model_kwargs or {})
    if "device_map" in options:
        raise ValueError(
            "Source weights must be loaded on CPU; pass deployment device_map "
            "to NeuralZip.from_pretrained()."
        )
    if "dtype" not in options and "torch_dtype" not in options:
        options["dtype"] = torch.bfloat16
    requested_dtype = options.get("dtype", options.get("torch_dtype"))
    if require_bfloat16 and requested_dtype not in {torch.bfloat16, "bfloat16"}:
        raise ValueError(
            "NeuralZip GPU inference requires BF16 source weights; omit dtype "
            "or pass dtype=torch.bfloat16."
        )
    model = loader(source, **options)
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model_class.from_pretrained() must return torch.nn.Module.")
    if require_bfloat16 and not any(
        parameter.dtype == torch.bfloat16 for parameter in model.parameters()
    ):
        raise ValueError("The loaded model has no BF16 parameters to compress.")
    return model.eval()


def _fit_options(options):
    selected = dict(options)
    selected.setdefault("ngram_values", ())
    selected.setdefault("exponent_layout", "original")
    return selected


class NeuralZip:
    """Single, stateless entry point for NeuralZip operations."""

    def __new__(cls, *args, **kwargs):
        del args, kwargs
        raise TypeError(
            "NeuralZip is not instantiated; call NeuralZip.fit(), encode(), "
            "decode(), or from_pretrained() directly."
        )

    @staticmethod
    def fit(
        model_or_id,
        *,
        model_class=None,
        model_kwargs=None,
        **setup_options,
    ):
        """Fit and return an immutable setup from a model or model identifier."""

        is_identifier = isinstance(model_or_id, (str, PathLike))
        if is_identifier:
            model = _load_pretrained_model(
                model_or_id,
                model_class=model_class,
                model_kwargs=model_kwargs,
            )
        else:
            if model_class is not None or model_kwargs is not None:
                raise ValueError(
                    "model_class and model_kwargs apply only when fitting a model "
                    "identifier."
                )
            model = model_or_id

        named_tensors = _model_tensors(model)
        codec = NeuralZipCodec(**_fit_options(setup_options))
        report = prepare_neuralzip_codec(codec, named_tensors)
        validate_setup(codec)
        return setup_from_codec(codec, report)

    @staticmethod
    def save_setup(setup, path):
        """Persist a trusted setup as one atomic file."""

        return save_setup_state(setup, path)

    @staticmethod
    def load_setup(path):
        """Load a setup file produced by a trusted NeuralZip process."""

        return load_setup_state(path)

    @staticmethod
    def map_target(model, *, setup):
        """Validate and report how a target maps to a reusable setup."""

        codec = materialize_setup(setup)
        return dict(codec.prepare_direct_transfer_target(_model_tensors(model)))

    @staticmethod
    def encode(model, *, setup, threads=0, return_stats=False):
        """Encode a model or tensor mapping under an explicit setup."""

        if not isinstance(return_stats, bool):
            raise TypeError("return_stats must be a bool.")
        codec = materialize_setup(setup, threads=threads)
        named_tensors = _model_tensors(model)
        source = _encoded_source(model, named_tensors)

        mapping_started = time.perf_counter()
        codec.prepare_direct_transfer_target(named_tensors)
        mapping_seconds = time.perf_counter() - mapping_started
        batch = encode_tensor_batch(codec, named_tensors)
        codebooks = codec.export_codebooks()
        encoded = make_encoded_model(
            source,
            names=tuple(name for name, _ in named_tensors),
            payloads=batch.payloads,
            codebooks=codebooks,
            raw_bytes=batch.raw_bytes,
            compressed_bytes=serialized_size(codebooks, batch.payloads),
            encode_seconds=batch.encode_seconds,
            mapping_seconds=mapping_seconds,
            requires_grad=tuple(tensor.requires_grad for _, tensor in named_tensors),
        )
        if not return_stats:
            return encoded
        return encoded, EncodeStats(
            raw_bytes=encoded.raw_bytes,
            compressed_bytes=encoded.compressed_bytes,
            compression_ratio=encoded.compression_ratio,
            mapping_seconds=encoded.mapping_seconds,
            encode_seconds=encoded.encode_seconds,
            total_seconds=encoded.total_seconds,
        )

    @staticmethod
    def decode(
        encoded,
        *,
        destination=None,
        model_factory=None,
        threads=0,
        return_stats=False,
    ):
        """Decode an encoded value without requiring its original setup."""

        if not isinstance(encoded, EncodedModel):
            raise TypeError("encoded must be an EncodedModel returned by encode().")
        if not isinstance(return_stats, bool):
            raise TypeError("return_stats must be a bool.")
        started = time.perf_counter()
        decoded = decode_model(
            encoded,
            NeuralZipCodec(threads=threads),
            destination=destination,
            model_factory=model_factory,
        )
        stats = DecodeStats(time.perf_counter() - started)
        return (decoded, stats) if return_stats else decoded

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *,
        setup=None,
        model_class=None,
        device=None,
        device_map=None,
        max_memory=None,
        offload_folder=None,
        execution_device=None,
        adapter=None,
        include_embeddings=False,
        decoder=None,
        cpu_offload=False,
        cpu_offload_blocks=None,
        pin_memory=True,
        threads=0,
        setup_options=None,
        **model_kwargs,
    ):
        """Load, compress, and return a native model ready for inference."""

        threads = int(threads)
        if threads < 0:
            raise ValueError("threads must be non-negative.")
        if setup is not None and setup_options:
            raise ValueError("setup_options cannot modify an explicit setup.")
        selected = None if setup is None else load_setup_state(setup)
        model = _load_pretrained_model(
            pretrained_model_name_or_path,
            model_class=model_class,
            model_kwargs=model_kwargs,
            require_bfloat16=True,
        )
        if selected is None:
            options = dict(setup_options or {})
            if tuple(options.get("ngram_values", ())):
                raise ValueError(
                    "GPU inference requires a sharing-only setup; ngram_values "
                    "must be empty."
                )
            if options.get("enable_shared_codebooks", True) is not True:
                raise ValueError(
                    "GPU inference requires enable_shared_codebooks=True."
                )
            options["ngram_values"] = ()
            options["enable_shared_codebooks"] = True
            options.setdefault("exponent_layout", "original")
            options["threads"] = int(threads)
            selected = cls.fit(model, **options)

        from .gpu.pretrained import attach_gpu_model

        model = attach_gpu_model(
            model,
            selected,
            threads=threads,
            device=device,
            device_map=device_map,
            max_memory=max_memory,
            offload_folder=offload_folder,
            execution_device=execution_device,
            adapter=adapter,
            include_embeddings=include_embeddings,
            decoder=decoder,
            cpu_offload=cpu_offload,
            cpu_offload_blocks=cpu_offload_blocks,
            pin_memory=pin_memory,
        )
        model.neuralzip_setup = selected
        return model

    @staticmethod
    def save_pretrained(model, directory, **kwargs):
        """Persist an attached NeuralZip model without restoring BF16 weights."""

        from .gpu.pretrained import save_gpu_pretrained

        return save_gpu_pretrained(model, directory=directory, **kwargs)

    @staticmethod
    def load_pretrained(pretrained_model_name_or_path, **kwargs):
        """Load a previously persisted NeuralZip inference checkpoint."""

        from .gpu.pretrained import load_gpu_pretrained

        return load_gpu_pretrained(pretrained_model_name_or_path, **kwargs)
