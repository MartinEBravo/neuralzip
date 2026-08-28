import time
from collections.abc import Mapping
from contextlib import contextmanager
from functools import wraps
from os import PathLike
from pathlib import Path
from threading import RLock

import torch

from ._setup import (
    codebook_digest,
    load_setup,
    materialize_setup,
    setup_from_codec,
    validate_setup,
)
from ._setup import save_setup as save_setup_state
from .encoded_model import EncodedModel, decode_model, make_encoded_model
from .neuralzip_codec import NeuralZipCodec
from .operations import (
    DecodeStats,
    EncodeStats,
    decode_tensor_batch,
    encode_tensor_batch,
    prepare_neuralzip_codec,
)
from .serialization import serialized_size


def _named_input_tensors(model):
    """Return floating-tensor candidates without truncating nested modules."""

    source = model
    if not isinstance(source, (Mapping, torch.nn.Module)):
        source = getattr(model, "model", None)
    if isinstance(source, Mapping):
        return _named_tensors(source)
    if isinstance(source, torch.nn.Module):
        return source.named_parameters()
    raise TypeError(
        "Expected a tensor mapping, torch.nn.Module, or adapter exposing one "
        "through .model."
    )


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


def _synchronized(method):
    """Serialize operations that read or replace a NeuralZip instance's state."""

    @wraps(method)
    def locked(self, *args, **kwargs):
        with self._state_lock:
            return method(self, *args, **kwargs)

    return locked


class NeuralZip:
    """Losslessly encode model weights with a reusable statistical setup."""

    def __init__(
        self,
        *,
        min_saving_bytes=None,
        chunk_size=None,
        compression_threshold=None,
        threads=0,
        cluster_map=None,
        cluster_js_threshold=0.05,
        max_codebook_sample_bytes=8 << 20,
        prior_count=1,
        exponent_prior_count=1,
        ngram_values=(2, 3, 4),
        top_k=254,
        exponent_layout=None,
        packed_source_budget_ratio=0.30,
        packed_time_budget_ratio=1.0,
        speed_ratio_tolerance=0.03,
        enable_shared_codebooks=True,
        direct_batch_bytes=None,
        setup=None,
    ):
        self.threads = int(threads)
        self._state_lock = RLock()
        self._setup_path = None
        self._setup_report = None
        self._setup_ready = False
        self._reuse_setup = False
        self._active_setup = None
        self._last_mapping = None
        self._last_mapping_signature = None
        explicit_runtime_options = {
            name
            for name, value in (
                ("min_saving_bytes", min_saving_bytes),
                ("chunk_size", chunk_size),
                ("compression_threshold", compression_threshold),
                ("direct_batch_bytes", direct_batch_bytes),
            )
            if value is not None
        }
        if min_saving_bytes is None:
            min_saving_bytes = 0
        if chunk_size is None:
            chunk_size = 256 << 10
        if compression_threshold is None:
            compression_threshold = 0.95
        if direct_batch_bytes is None:
            direct_batch_bytes = 128 << 20
        codec_kwargs = {
            "min_saving_bytes": min_saving_bytes,
            "chunk_size": chunk_size,
            "compression_threshold": compression_threshold,
            "threads": threads,
        }
        codec_kwargs["prior_count"] = prior_count
        codec_kwargs["exponent_prior_count"] = exponent_prior_count
        codec_kwargs["cluster_js_threshold"] = cluster_js_threshold
        codec_kwargs["max_codebook_sample_bytes"] = max_codebook_sample_bytes
        codec_kwargs["top_k"] = top_k
        codec_kwargs["exponent_layout"] = exponent_layout
        codec_kwargs["packed_source_budget_ratio"] = packed_source_budget_ratio
        codec_kwargs["packed_time_budget_ratio"] = packed_time_budget_ratio
        codec_kwargs["speed_ratio_tolerance"] = speed_ratio_tolerance
        codec_kwargs["enable_shared_codebooks"] = enable_shared_codebooks
        codec_kwargs["direct_batch_bytes"] = direct_batch_bytes
        codec_kwargs["cluster_map"] = dict(cluster_map or {})
        codec_kwargs["ngram_values"] = tuple(ngram_values)
        self._neuralzip_codec_options = dict(codec_kwargs)
        self._explicit_runtime_options = explicit_runtime_options

        # Constructing the template validates every option even when a
        # persisted setup supplies the statistical state.
        self.codec = NeuralZipCodec(**self._neuralzip_codec_options)
        if setup is not None:
            self._active_setup = load_setup(setup)
            self.codec = materialize_setup(self._active_setup)
            self._apply_runtime_options(self.codec)
            self._setup_path = (
                Path(setup) if isinstance(setup, (str, PathLike)) else None
            )
            self._setup_report = self._active_setup.setup_report
            self._setup_ready = True
            self._reuse_setup = True

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
        neuralzip_kwargs=None,
        **model_kwargs,
    ):
        """Load, compress, and return a native model ready for inference.

        ``setup=None`` fits a sharing-only setup on the source model. Passing a
        setup object or trusted setup path reuses it without fitting again.
        Model-loading options are forwarded to the selected Hugging Face auto
        class; NeuralZip options belong in ``neuralzip_kwargs``.
        """

        if model_class is None:
            from transformers import AutoModelForCausalLM

            model_class = AutoModelForCausalLM
        loader = getattr(model_class, "from_pretrained", None)
        if not callable(loader):
            raise TypeError("model_class must provide from_pretrained().")
        if "device_map" in model_kwargs:
            raise ValueError(
                "Pass deployment device_map to NeuralZip.from_pretrained(), not "
                "to the Hugging Face loader. Source weights are loaded on CPU."
            )
        if "dtype" not in model_kwargs and "torch_dtype" not in model_kwargs:
            model_kwargs["dtype"] = torch.bfloat16

        options = dict(neuralzip_kwargs or {})
        if "setup" in options:
            raise ValueError("Pass setup directly, not through neuralzip_kwargs.")
        if setup is None:
            requested_ngrams = tuple(options.get("ngram_values", ()))
            if requested_ngrams:
                raise ValueError(
                    "GPU inference currently requires a sharing-only setup; "
                    "ngram_values must be empty."
                )
            options["ngram_values"] = ()
            options.setdefault("exponent_layout", "original")

        model = loader(pretrained_model_name_or_path, **model_kwargs).eval()
        neuralzip = cls(setup=setup, **options)
        if setup is None:
            neuralzip.fit(model)

        from .gpu.pretrained import attach_gpu_model

        model = attach_gpu_model(
            model,
            neuralzip,
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
        model.neuralzip_setup = neuralzip.setup
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

    @staticmethod
    def _floating_tensors(model):
        wrapped = getattr(model, "model", None)
        if isinstance(model, (Mapping, torch.nn.Module)) or isinstance(
            wrapped, (Mapping, torch.nn.Module)
        ):
            items = _named_input_tensors(model)
        else:
            items = iter(model)
        tensors = [
            (str(name), tensor)
            for name, tensor in items
            if torch.is_tensor(tensor) and tensor.is_floating_point()
        ]
        if not tensors:
            raise ValueError("No floating tensors were found.")
        return tensors

    @staticmethod
    def _model_tensors(model):
        """Include persistent floating buffers for reconstructable model encodes."""

        source = model
        if not isinstance(source, (Mapping, torch.nn.Module)):
            source = getattr(model, "model", None)
        if not isinstance(source, torch.nn.Module):
            return NeuralZip._floating_tensors(model)

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
            if (
                name in persistent
                and tensor.is_floating_point()
                and id(tensor) not in seen
            ):
                tensors.append((str(name), tensor))
                seen.add(id(tensor))
        if not tensors:
            raise ValueError("No floating tensors were found.")
        return tensors

    @property
    def has_setup(self):
        # The state flags retain compatibility with manually constructed
        # codecs used by the low-level GPU export API and its tests.
        return bool(
            self._active_setup is not None or (self._setup_ready and self._reuse_setup)
        )

    @property
    def setup(self):
        """The instance's active reusable setup, if one is available."""

        return self._active_setup

    @property
    def setup_report(self):
        return self._setup_report

    @property
    def setup_path(self):
        return self._setup_path

    @property
    def codebooks(self):
        return self.codec.export_codebooks() if self._setup_ready else ()

    @property
    def setup_digest(self):
        if not self.has_setup:
            raise RuntimeError(
                "Call fit() or provide setup= before reading setup_digest."
            )
        return codebook_digest(self.codebooks)

    @property
    def selected_modes(self):
        return self.codec.selected_modes if self._setup_ready else {}

    @property
    def cluster_count(self):
        return int(self.codec.cluster_count) if self._setup_ready else 0

    @property
    def last_mapping(self):
        return None if self._last_mapping is None else dict(self._last_mapping)

    def _apply_runtime_options(self, codec):
        codec.threads = self.threads
        for name in self._explicit_runtime_options:
            setattr(codec, name, self._neuralzip_codec_options[name])

    def _setup_state(self):
        codec = self.codec
        transfer = None
        if isinstance(codec, NeuralZipCodec):
            transfer = (
                dict(codec._transfer_mode_by_tensor_stream),
                dict(codec._transfer_cluster_by_name),
                bool(codec._transfer_active),
            )
        return (
            codec,
            self._active_setup,
            self._setup_report,
            self._setup_path,
            self._setup_ready,
            self._reuse_setup,
            self._last_mapping,
            self._last_mapping_signature,
            transfer,
        )

    def _restore_setup_state(self, state):
        (
            self.codec,
            self._active_setup,
            self._setup_report,
            self._setup_path,
            self._setup_ready,
            self._reuse_setup,
            self._last_mapping,
            self._last_mapping_signature,
            transfer,
        ) = state
        if transfer is not None:
            modes, clusters, active = transfer
            self.codec._transfer_mode_by_tensor_stream = modes
            self.codec._transfer_cluster_by_name = clusters
            self.codec._transfer_active = active

    @contextmanager
    def _setup_transaction(self, setup=None):
        state = self._setup_state()
        try:
            self._activate_setup(setup)
            yield
        except BaseException:
            self._restore_setup_state(state)
            raise

    @contextmanager
    def _target_mapping_transaction(self, model, *, include_buffers=True):
        """Map one target and roll back the instance if its consumer fails."""

        state = self._setup_state()
        try:
            tensor_source = (
                self._model_tensors if include_buffers else self._floating_tensors
            )
            named_tensors = tensor_source(model)
            mapping = self.codec.prepare_direct_transfer_target(named_tensors)
            self._last_mapping = dict(mapping)
            self._last_mapping_signature = self._target_signature(named_tensors)
            yield dict(mapping)
        except BaseException:
            self._restore_setup_state(state)
            raise

    def _activate_setup(self, setup=None):
        selected = self._active_setup if setup is None else load_setup(setup)
        if selected is None:
            raise RuntimeError("Call fit() or provide setup= before encoding.")

        if self._active_setup is None or selected != self._active_setup:
            codec = materialize_setup(selected)
            self._apply_runtime_options(codec)
            self.codec = codec
            self._active_setup = selected
            self._setup_report = selected.setup_report
            self._setup_path = (
                Path(setup) if isinstance(setup, (str, PathLike)) else None
            )
            self._last_mapping = None
            self._last_mapping_signature = None
        self._setup_ready = True
        self._reuse_setup = True
        return selected

    @_synchronized
    def fit(self, model):
        """Learn a reusable setup and return it as an opaque data value.

        Fitting is transactional: a failed fit leaves any previously active
        setup untouched.
        """

        named_tensors = self._model_tensors(model)
        candidate = NeuralZipCodec(**self._neuralzip_codec_options)
        report = prepare_neuralzip_codec(candidate, named_tensors)
        validate_setup(candidate)
        setup = setup_from_codec(candidate, report)

        self.codec = candidate
        self._active_setup = setup
        self._setup_report = report
        self._setup_ready = True
        self._reuse_setup = True
        self._setup_path = None
        self._last_mapping = None
        self._last_mapping_signature = None
        return setup

    @_synchronized
    def save_setup(self, path, *, setup=None):
        """Persist an explicit setup or the one most recently fitted/loaded."""

        selected = self._active_setup if setup is None else load_setup(setup)
        if selected is None:
            raise RuntimeError("Call fit() before save_setup().")
        saved = save_setup_state(selected, path)
        if selected == self._active_setup:
            self._setup_path = saved
        return saved

    @_synchronized
    def map_target(self, model, *, setup=None):
        """Validate and map a target to the fitted or loaded setup."""

        named_tensors = self._model_tensors(model)
        with self._setup_transaction(setup):
            mapping = self.codec.prepare_direct_transfer_target(named_tensors)
            self._last_mapping = dict(mapping)
            self._last_mapping_signature = self._target_signature(named_tensors)
            return dict(mapping)

    @staticmethod
    def _target_signature(named_tensors):
        return tuple(
            (
                str(name),
                str(tensor.dtype),
                tuple(int(dimension) for dimension in tensor.shape),
            )
            for name, tensor in named_tensors
        )

    @staticmethod
    def _encoded_source(model, named_tensors):
        if isinstance(model, (Mapping, torch.nn.Module)):
            return model
        wrapped = getattr(model, "model", None)
        if isinstance(wrapped, (Mapping, torch.nn.Module)):
            return wrapped
        return dict(named_tensors)

    def _encode_neuralzip_model(self, source, names, named_tensors):
        mapping_seconds = 0.0
        signature = self._target_signature(named_tensors)
        if signature != self._last_mapping_signature:
            mapping_started = time.perf_counter()
            self._last_mapping = dict(
                self.codec.prepare_direct_transfer_target(named_tensors)
            )
            mapping_seconds = time.perf_counter() - mapping_started
            self._last_mapping_signature = signature
        batch = encode_tensor_batch(self.codec, named_tensors)
        codebooks = self.codec.export_codebooks()
        return make_encoded_model(
            source,
            names=names,
            payloads=batch.payloads,
            codebooks=codebooks,
            raw_bytes=batch.raw_bytes,
            compressed_bytes=serialized_size(codebooks, batch.payloads),
            encode_seconds=batch.encode_seconds,
            mapping_seconds=mapping_seconds,
            used_batch=batch.used_batch,
            requires_grad=tuple(tensor.requires_grad for _, tensor in named_tensors),
        )

    @_synchronized
    def encode(self, model, *, setup=None, return_stats=False):
        """Encode a model or tensor mapping through one common public API.

        NeuralZip accepts the data-only value returned by :meth:`fit` (or
        a trusted setup path). Set ``return_stats=True`` to also receive a
        stable timing and size summary.
        """

        if not isinstance(return_stats, bool):
            raise TypeError("return_stats must be a bool.")
        named_tensors = self._model_tensors(model)
        source = self._encoded_source(model, named_tensors)
        names = tuple(name for name, _ in named_tensors)
        with self._setup_transaction(setup):
            encoded = self._encode_neuralzip_model(source, names, named_tensors)

        if not return_stats:
            return encoded
        stats = EncodeStats(
            raw_bytes=encoded.raw_bytes,
            compressed_bytes=encoded.compressed_bytes,
            compression_ratio=encoded.compression_ratio,
            setup_seconds=encoded.setup_seconds,
            mapping_seconds=encoded.mapping_seconds,
            encode_seconds=encoded.encode_seconds,
            total_seconds=encoded.total_seconds,
        )
        return encoded, stats

    @_synchronized
    def decode(
        self,
        encoded,
        *,
        destination=None,
        model_factory=None,
        return_stats=False,
    ):
        """Decode a result and optionally return end-to-end timing stats."""

        if not isinstance(return_stats, bool):
            raise TypeError("return_stats must be a bool.")
        decode_started = time.perf_counter()

        if isinstance(encoded, EncodedModel):
            decoder = NeuralZipCodec(threads=self.threads)
            decoded = decode_model(
                encoded,
                decoder,
                destination=destination,
                model_factory=model_factory,
            )
            stats = DecodeStats(time.perf_counter() - decode_started)
            return (decoded, stats) if return_stats else decoded

        if not self.has_setup:
            raise RuntimeError("Call fit() or provide setup= before decode().")
        payloads = getattr(encoded, "payloads", encoded)
        decoder = NeuralZipCodec(threads=self.threads)
        decoder.load_codebooks(self.codebooks)
        tensors, _ = decode_tensor_batch(decoder, tuple(payloads))
        stats = DecodeStats(time.perf_counter() - decode_started)
        return (tensors, stats) if return_stats else tensors
