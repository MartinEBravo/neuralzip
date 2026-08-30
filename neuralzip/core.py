import time
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import chain
from os import PathLike
import torch
from neuralzip._setup import SetupReport, materialize_setup, setup_from_codec
from neuralzip._setup import load_setup as load_setup_state
from neuralzip._setup import save_setup as save_setup_state
from neuralzip.encoded_model import decode_model, make_encoded_model
from neuralzip.neuralzip_codec import NeuralZipCodec
from neuralzip.payloads import serialized_size


@dataclass(frozen=True)
class DecodeStats:
    decode_seconds: float


@dataclass(frozen=True)
class EncodeStats:
    raw_bytes: int
    compressed_bytes: int
    compression_ratio: float
    mapping_seconds: float
    encode_seconds: float
    total_seconds: float


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


def _model_tensors(model):
    if isinstance(model, Mapping):
        items = _named_tensors(model)
    elif isinstance(model, torch.nn.Module):
        persistent = set(model.state_dict())
        items = chain(
            model.named_parameters(),
            (
                (name, tensor)
                for name, tensor in model.named_buffers()
                if name in persistent
            ),
        )
    else:
        raise TypeError("Expected a tensor mapping or torch.nn.Module.")
    tensors = [
        (str(name), tensor)
        for name, tensor in items
        if torch.is_tensor(tensor) and tensor.is_floating_point()
    ]
    if not tensors:
        raise ValueError("No floating tensors were found.")
    if len({name for name, _tensor in tensors}) != len(tensors):
        raise ValueError("Tensor names must be unique after flattening the input.")
    return tensors


def _load_pretrained_model(source, *, model_class=None, model_kwargs=None):
    if model_class is None:
        from transformers import AutoModelForCausalLM

        model_class = AutoModelForCausalLM
    options = dict(model_kwargs or {})
    if "dtype" not in options and "torch_dtype" not in options:
        options["dtype"] = torch.bfloat16
    return model_class.from_pretrained(source, **options).eval()


class NeuralZip:
    """Single, stateless entry point for NeuralZip operations."""

    @staticmethod
    def fit(model_or_id, *, model_class=None, model_kwargs=None, **setup_options):
        """Fit and return an immutable setup from a model or model identifier."""
        if isinstance(model_or_id, (str, PathLike)):
            model = _load_pretrained_model(
                model_or_id,
                model_class=model_class,
                model_kwargs=model_kwargs,
            )
        else:
            model = model_or_id

        named_tensors = _model_tensors(model)
        setup_options.setdefault("ngram_values", ())
        codec = NeuralZipCodec(**setup_options)
        started = time.perf_counter()
        codec.prepare_clusters(named_tensors)
        clustering_seconds = time.perf_counter() - started
        started = time.perf_counter()
        codec.build_codebooks_and_select_modes(named_tensors)
        report = SetupReport(
            clustering_seconds=clustering_seconds,
            codebook_setup_seconds=time.perf_counter() - started,
        )
        return setup_from_codec(codec, report)

    @staticmethod
    def save_setup(setup, path):
        """Persist a trusted setup in one file."""
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
        codec = materialize_setup(setup, threads=threads)
        named_tensors = _model_tensors(model)

        mapping_started = time.perf_counter()
        codec.prepare_direct_transfer_target(named_tensors)
        mapping_seconds = time.perf_counter() - mapping_started
        encode_started = time.perf_counter()
        payloads = tuple(codec.compress_tensors(named_tensors))
        encode_seconds = time.perf_counter() - encode_started
        codebooks = codec.export_codebooks()
        encoded = make_encoded_model(
            model,
            names=tuple(name for name, _ in named_tensors),
            payloads=payloads,
            codebooks=codebooks,
            raw_bytes=sum(int(payload.raw_bytes) for payload in payloads),
            compressed_bytes=serialized_size(codebooks, payloads),
            encode_seconds=encode_seconds,
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
        include_embeddings=False,
        threads=0,
        setup_options=None,
        **model_kwargs,
    ):
        """Load, compress, and return a native model ready for inference."""
        threads = int(threads)
        selected = None if setup is None else load_setup_state(setup)
        model = _load_pretrained_model(
            pretrained_model_name_or_path,
            model_class=model_class,
            model_kwargs=model_kwargs,
        )
        if selected is None:
            options = dict(setup_options or {})
            options.update(
                ngram_values=(),
                enable_shared_codebooks=True,
                threads=threads,
            )
            selected = cls.fit(model, **options)

        from neuralzip.gpu.pretrained import attach_gpu_model

        model = attach_gpu_model(
            model,
            selected,
            threads=threads,
            device=device,
            include_embeddings=include_embeddings,
        )
        model.neuralzip_setup = selected
        return model

    @staticmethod
    def save_pretrained(model, directory, *, tokenizer=None):
        """Persist an attached NeuralZip model without restoring BF16 weights."""
        from neuralzip.gpu.pretrained import save_gpu_pretrained

        return save_gpu_pretrained(model, directory=directory, tokenizer=tokenizer)

    @staticmethod
    def load_pretrained(
        pretrained_model_name_or_path,
        *,
        device=None,
        revision="main",
        token=None,
        cache_dir=None,
        local_files_only=False,
    ):
        """Load a previously persisted NeuralZip inference checkpoint."""
        from neuralzip.gpu.pretrained import load_gpu_pretrained

        return load_gpu_pretrained(
            pretrained_model_name_or_path,
            device=device,
            revision=revision,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
