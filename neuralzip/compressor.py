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
from .codec_protocol import (
    DecodeStats,
    EncodeStats,
    decode_tensor_batch,
    encode_tensor_batch,
    prepare_neuralzip_codec,
)
from .encoded_model import EncodedModel, decode_model, make_encoded_model
from .neuralzip_codec import NeuralZipCodec
from .serialization import (
    serialized_codebooks_size,
    serialized_size,
)
from .tensors import bitwise_equal
from .zipnn_codec import ZipNNCodec
from .zipnn_source import ZipNNSourceCodec

_ZIPNN_MODEL_IO_ERROR = (
    "zipnn produces opaque per-tensor benchmark frames; model "
    "save/load/decompress is supported only for NeuralZip and zipnn-local archives."
)
_ZIPNN_RESULT_METHOD_KEYS = {"zipnn", "zipnn_src"}


def _method_key(method):
    return str(method).strip().lower().replace("-", "_").replace(" ", "_")


def _normalize_method(method):
    method_key = _method_key(method)
    methods = {"neuralzip", "zipnn", "zipnn_local"}
    if method_key not in methods:
        raise ValueError(
            f"Unsupported compression method {method!r}. Expected one of: "
            "neuralzip, zipnn, zipnn-local."
        )
    return method_key


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


def _raw_tensor_bytes(tensor):
    return int(tensor.numel() * tensor.element_size())


def _synchronized(method):
    """Serialize operations that read or replace a Compressor's codec state."""

    @wraps(method)
    def locked(self, *args, **kwargs):
        with self._state_lock:
            return method(self, *args, **kwargs)

    return locked


class Compressor:
    """Compress floating parameters from modules or tensor leaves from mappings."""

    def __init__(
        self,
        method="neuralzip",
        *,
        verify=True,
        keep_payloads=False,
        min_saving_bytes=None,
        chunk_size=None,
        compression_threshold=None,
        threads=0,
        zipnn_method="HUFFMAN",
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
        self.method = _normalize_method(method)
        self.verify = bool(verify)
        self.keep_payloads = bool(keep_payloads)
        self.zipnn_method = str(zipnn_method).upper()
        self.threads = int(threads)
        self._state_lock = RLock()
        self._setup_path = None
        self._setup_report = None
        self._setup_ready = False
        self._reuse_setup = False
        self._active_setup = None
        self._last_mapping = None
        self._last_mapping_signature = None
        if setup is not None and self.method != "neuralzip":
            raise ValueError("setup is supported only for method='neuralzip'.")
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
        if self.method == "neuralzip":
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
        elif self.method == "zipnn_local":
            self.codec = ZipNNCodec(**codec_kwargs)
        else:
            self.codec = ZipNNSourceCodec(
                method=self.zipnn_method,
                threads=self.threads,
                compression_threshold=compression_threshold,
                chunk_size=chunk_size,
            )

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
            return Compressor._floating_tensors(model)

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

    def _require_neuralzip(self, operation):
        if self.method != "neuralzip":
            raise ValueError(f"{operation} is supported only for method='neuralzip'.")

    @property
    def has_setup(self):
        # The state flags retain compatibility with manually constructed
        # codecs used by the low-level GPU export API and its tests.
        return bool(
            self._active_setup is not None or (self._setup_ready and self._reuse_setup)
        )

    @property
    def setup(self):
        """The compressor's default reusable setup, if one is active."""

        return self._active_setup

    @property
    def setup_report(self):
        return self._setup_report

    @property
    def setup_path(self):
        return self._setup_path

    @property
    def codebooks(self):
        self._require_neuralzip("codebooks")
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
        self._require_neuralzip("selected_modes")
        return self.codec.selected_modes if self._setup_ready else {}

    @property
    def cluster_count(self):
        self._require_neuralzip("cluster_count")
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
        """Map one target and roll back the compressor if its consumer fails."""

        self._require_neuralzip("target mapping")
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
        self._require_neuralzip("setup")
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

        self._require_neuralzip("fit")
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

        self._require_neuralzip("save_setup")
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

        self._require_neuralzip("map_target")
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
            method="neuralzip",
            names=names,
            payloads=batch.payloads,
            codebooks=codebooks,
            raw_bytes=batch.raw_bytes,
            compressed_bytes=serialized_size("neuralzip", codebooks, batch.payloads),
            encode_seconds=batch.encode_seconds,
            mapping_seconds=mapping_seconds,
            used_batch=batch.used_batch,
            requires_grad=tuple(tensor.requires_grad for _, tensor in named_tensors),
        )

    @_synchronized
    def encode(self, model, *, setup=None, return_stats=False):
        """Encode a model or tensor mapping through one common public API.

        NeuralZip accepts the data-only value returned by :meth:`fit` (or a
        trusted setup path). Official ZipNN is one-shot and does not accept a
        setup. Set ``return_stats=True`` to also receive a stable timing and
        size summary.
        """

        if not isinstance(return_stats, bool):
            raise TypeError("return_stats must be a bool.")
        named_tensors = self._model_tensors(model)
        source = self._encoded_source(model, named_tensors)
        names = tuple(name for name, _ in named_tensors)
        if self.method == "neuralzip":
            with self._setup_transaction(setup):
                encoded = self._encode_neuralzip_model(
                    source, names, named_tensors
                )

        elif setup is not None:
            raise ValueError("setup is supported only for method='neuralzip'.")

        elif self.method == "zipnn_local":
            batch = encode_tensor_batch(self.codec, named_tensors)
            encoded = make_encoded_model(
                source,
                method="zipnn-local",
                names=names,
                payloads=batch.payloads,
                raw_bytes=batch.raw_bytes,
                compressed_bytes=serialized_size("zipnn-local", (), batch.payloads),
                encode_seconds=batch.encode_seconds,
                used_batch=batch.used_batch,
                requires_grad=tuple(
                    tensor.requires_grad for _, tensor in named_tensors
                ),
            )
        elif self.method == "zipnn":
            encode_started = time.perf_counter()
            encoded_tensors = tuple(
                self.codec.encode_tensor(tensor, verify=False)
                for _, tensor in named_tensors
            )
            encode_seconds = time.perf_counter() - encode_started
            encoded = make_encoded_model(
                source,
                method="zipnn",
                names=names,
                payloads=tuple(value.payload for value in encoded_tensors),
                raw_bytes=sum(value.raw_bytes for value in encoded_tensors),
                compressed_bytes=sum(len(value.payload) for value in encoded_tensors),
                encode_seconds=encode_seconds,
                requires_grad=tuple(
                    tensor.requires_grad for _, tensor in named_tensors
                ),
            )
        else:
            raise AssertionError(f"Unhandled compression method {self.method!r}.")

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
            if encoded.method == "neuralzip":
                decoder = NeuralZipCodec(threads=self.threads)
            elif encoded.method == "zipnn_local":
                decoder = ZipNNCodec(threads=self.threads)
            else:
                decoder = (
                    self.codec
                    if self.method == "zipnn"
                    else ZipNNSourceCodec(
                        method=self.zipnn_method,
                        threads=self.threads,
                        compression_threshold=0.95,
                        chunk_size=256 << 10,
                    )
                )
            decoded = decode_model(
                encoded,
                decoder,
                destination=destination,
                model_factory=model_factory,
            )
            stats = DecodeStats(time.perf_counter() - decode_started)
            return (decoded, stats) if return_stats else decoded

        self._require_neuralzip("decode")
        if not self.has_setup:
            raise RuntimeError("Call fit() or provide setup= before decode().")
        payloads = getattr(encoded, "payloads", encoded)
        decoder = NeuralZipCodec(threads=self.threads)
        decoder.load_codebooks(self.codebooks)
        tensors, _ = decode_tensor_batch(decoder, tuple(payloads))
        stats = DecodeStats(time.perf_counter() - decode_started)
        return (tensors, stats) if return_stats else tensors

    @_synchronized
    def compress(self, model):
        if self.method == "zipnn_local":
            return self._codec_compression(
                model,
                method_name="zipnn-local",
                method_detail="dtype-byte-reorder-huffman",
            )
        if self.method == "neuralzip":
            try:
                if self._reuse_setup:
                    with self._setup_transaction():
                        return self._codec_compression(
                            model,
                            method_name="neuralzip",
                            method_detail=(
                                "js-clustered-shared-huffman-packed-exponents"
                            ),
                        )
                return self._codec_compression(
                    model,
                    method_name="neuralzip",
                    method_detail=("js-clustered-shared-huffman-packed-exponents"),
                )
            except BaseException:
                if not self._reuse_setup:
                    self.codec = NeuralZipCodec(**self._neuralzip_codec_options)
                    self._setup_ready = False
                    self._setup_report = None
                    self._last_mapping = None
                    self._last_mapping_signature = None
                raise
        if self.method == "zipnn":
            return self._zipnn(model)
        raise AssertionError(f"Unhandled compression method {self.method!r}.")

    def decompress(self, compressed_model):
        result_is_zipnn = (
            _method_key(compressed_model.get("method")) in _ZIPNN_RESULT_METHOD_KEYS
        )
        if self.method == "zipnn" or result_is_zipnn:
            raise NotImplementedError(_ZIPNN_MODEL_IO_ERROR)

        method_key = _method_key(compressed_model.get("method"))
        decoder = (
            ZipNNCodec(threads=self.threads)
            if method_key == "zipnn_local"
            else NeuralZipCodec(threads=self.threads)
        )
        if hasattr(decoder, "load_codebooks"):
            header = compressed_model.get("header") or {}
            codebooks = compressed_model.get("codebooks") or header.get("codebooks")
            decoder.load_codebooks(codebooks)

        payloads = []
        for tensor_result in compressed_model.get("tensor_results", []):
            payload = tensor_result.get("payload")
            if payload is None:
                raise ValueError(
                    "Cannot decompress this result because tensor payloads were not "
                    "kept. Run Compressor(..., keep_payloads=True)."
                )
            payloads.append(payload)
        tensors, _ = decode_tensor_batch(decoder, payloads)
        return tensors

    @staticmethod
    def save(compressed_model, path):
        if _method_key(compressed_model.get("method")) in _ZIPNN_RESULT_METHOD_KEYS:
            raise NotImplementedError(_ZIPNN_MODEL_IO_ERROR)
        from .archive import dump

        dump(compressed_model, path)

    @staticmethod
    def load(path):
        from .archive import load

        return load(path)

    def decompress_file(self, path):
        if self.method == "zipnn":
            raise NotImplementedError(_ZIPNN_MODEL_IO_ERROR)
        return self.decompress(self.load(path))

    def _codec_compression(self, model, *, method_name, method_detail):
        tensor_results = []
        total_raw_bytes = 0
        total_payload_bytes = 0
        total_decode_seconds = 0.0
        total_huffman_chunks = 0
        total_raw_chunks = 0
        total_zero_chunks = 0
        total_local_huff0_chunks = 0
        total_packed_ngram_chunks = 0
        total_ngram_hits = 0
        total_ngram_escapes = 0
        clustering_seconds = 0.0
        codebook_setup_seconds = 0.0
        codebook_metadata_bytes = 0
        codebooks = ()
        roundtrip_ok = True if self.verify else None

        named_tensors = self._floating_tensors(model)
        mapping_seconds = 0.0
        mapped_tensors = None
        setup_reused = False

        if self.method == "neuralzip":
            if self._reuse_setup:
                setup_reused = True
                mapping_started = time.perf_counter()
                self._last_mapping = dict(
                    self.codec.prepare_direct_transfer_target(named_tensors)
                )
                self._last_mapping_signature = self._target_signature(named_tensors)
                mapping_seconds = time.perf_counter() - mapping_started
                mapped_tensors = int(self._last_mapping["matched_tensors"])
            else:
                # Preserve Compressor's one-shot behavior: unless fit() or
                # setup= made reuse explicit, every call learns its own setup.
                self.codec.reset_compression()
                self.codec.reset_decompression()
                setup_report = prepare_neuralzip_codec(self.codec, named_tensors)
                validate_setup(self.codec)
                self._setup_report = setup_report
                self._setup_ready = True
                clustering_seconds = setup_report.clustering_seconds
                codebook_setup_seconds = setup_report.codebook_setup_seconds
                self._last_mapping = None
                self._last_mapping_signature = None
            codebooks = self.codec.export_codebooks()
            codebook_metadata_bytes = serialized_codebooks_size(codebooks)
        else:
            if hasattr(self.codec, "reset_compression"):
                self.codec.reset_compression()
            if hasattr(self.codec, "reset_decompression"):
                self.codec.reset_decompression()
        setup_seconds = clustering_seconds + codebook_setup_seconds

        encoded_batch = encode_tensor_batch(self.codec, named_tensors)
        compressed_payloads = list(encoded_batch.payloads)
        total_encode_seconds = encoded_batch.encode_seconds

        decoded_tensors = None
        if self.verify:
            decoded_tensors, total_decode_seconds = decode_tensor_batch(
                self.codec,
                encoded_batch.payloads,
            )

        for tensor_index, (name, tensor) in enumerate(named_tensors):
            raw_bytes = _raw_tensor_bytes(tensor)
            total_raw_bytes += raw_bytes

            compressed = encoded_batch.payloads[tensor_index]
            total_payload_bytes += compressed.payload_bytes
            total_huffman_chunks += compressed.huffman_chunks
            total_raw_chunks += compressed.raw_chunks
            total_zero_chunks += compressed.zero_chunks
            total_local_huff0_chunks += getattr(compressed, "local_huff0_chunks", 0)
            total_packed_ngram_chunks += getattr(compressed, "packed_ngram_chunks", 0)
            total_ngram_hits += getattr(compressed, "ngram_hits", 0)
            total_ngram_escapes += getattr(compressed, "ngram_escapes", 0)
            if self.verify:
                decompressed = decoded_tensors[name]
                original = tensor.detach().cpu().contiguous()
                tensor_roundtrip_ok = bitwise_equal(original, decompressed)
                roundtrip_ok = bool(roundtrip_ok and tensor_roundtrip_ok)

            result = {"name": name, "raw_bytes": raw_bytes}
            if self.keep_payloads:
                result["payload"] = compressed
            tensor_results.append(result)

        total_compressed_bytes = serialized_size(
            method_name,
            codebooks,
            compressed_payloads,
        )
        total_metadata_bytes = total_compressed_bytes - total_payload_bytes
        model_compression_ratio = (
            total_compressed_bytes / total_raw_bytes if total_raw_bytes else 0.0
        )
        return {
            "method": method_name,
            "zipnn_method": method_detail,
            "num_tensors": len(tensor_results),
            "raw_bytes": total_raw_bytes,
            "payload_bytes": total_payload_bytes,
            "metadata_bytes": total_metadata_bytes,
            "codebook_metadata_bytes": codebook_metadata_bytes,
            "compressed_bytes": total_compressed_bytes,
            "compression_ratio": model_compression_ratio,
            "huffman_chunks": total_huffman_chunks,
            "raw_chunks": total_raw_chunks,
            "zero_chunks": total_zero_chunks,
            "shared_huff0_chunks": sum(
                getattr(payload, "shared_huff0_chunks", 0)
                for payload in compressed_payloads
            ),
            "local_huff0_chunks": total_local_huff0_chunks,
            "packed_ngram_chunks": total_packed_ngram_chunks,
            "ngram_hits": total_ngram_hits,
            "ngram_escapes": total_ngram_escapes,
            "clusters": getattr(self.codec, "cluster_count", 0),
            "selected_modes": getattr(self.codec, "selected_modes", {}),
            "clustering_seconds": clustering_seconds,
            "codebook_setup_seconds": codebook_setup_seconds,
            "setup_seconds": setup_seconds,
            "setup_seconds_once": (
                self._setup_report.setup_seconds
                if self._setup_report is not None
                else None
            ),
            "setup_reused": setup_reused,
            "mapping_seconds": mapping_seconds,
            "mapped_tensors": mapped_tensors,
            "encode_seconds": total_encode_seconds,
            "total_seconds": setup_seconds + mapping_seconds + total_encode_seconds,
            "decode_seconds": total_decode_seconds if self.verify else None,
            "roundtrip_ok": roundtrip_ok,
            "codebooks": codebooks,
            "tensor_results": tensor_results,
        }

    def _zipnn(self, model):
        tensor_results = []
        total_raw_bytes = 0
        total_compressed_bytes = 0
        total_encode_seconds = 0.0
        total_codec_encode_seconds = 0.0
        total_input_copy_seconds = 0.0
        total_decode_seconds = 0.0
        roundtrip_ok = True if self.verify else None
        codec = self.codec
        encoded_rows = []

        for name, tensor in _named_input_tensors(model):
            if not torch.is_tensor(tensor) or not tensor.is_floating_point():
                continue

            original = tensor.detach().cpu().contiguous()
            raw_bytes = _raw_tensor_bytes(original)
            total_raw_bytes += raw_bytes

            encoded = codec.encode_tensor(original, verify=False)
            payload = encoded.payload
            input_copy_seconds = encoded.input_copy_seconds
            codec_encode_seconds = encoded.codec_encode_seconds
            encode_seconds = encoded.encode_seconds

            compressed_bytes = len(payload)
            total_compressed_bytes += compressed_bytes
            total_encode_seconds += encode_seconds
            total_codec_encode_seconds += codec_encode_seconds
            total_input_copy_seconds += input_copy_seconds

            encoded_rows.append((name, original, encoded))

            result = {"name": name, "raw_bytes": raw_bytes}
            if self.keep_payloads:
                result["payload"] = payload
            tensor_results.append(result)

        if self.verify:
            decode_started = time.perf_counter()
            decoded_tensors = {
                name: codec.decode(encoded.payload) for name, _, encoded in encoded_rows
            }
            total_decode_seconds = time.perf_counter() - decode_started
            for name, original, _ in encoded_rows:
                roundtrip_ok = bool(
                    roundtrip_ok and bitwise_equal(original, decoded_tensors[name])
                )

        model_compression_ratio = (
            total_compressed_bytes / total_raw_bytes if total_raw_bytes else 0.0
        )
        return {
            "method": "zipnn",
            "zipnn_method": self.zipnn_method,
            "num_tensors": len(tensor_results),
            "raw_bytes": total_raw_bytes,
            "payload_bytes": total_compressed_bytes,
            "metadata_bytes": 0,
            "compressed_bytes": total_compressed_bytes,
            "compression_ratio": model_compression_ratio,
            "huffman_chunks": None,
            "raw_chunks": None,
            "zero_chunks": None,
            "setup_seconds": 0.0,
            "encode_seconds": total_encode_seconds,
            "total_seconds": total_encode_seconds,
            "codec_encode_seconds": total_codec_encode_seconds,
            "input_copy_seconds": total_input_copy_seconds,
            "decode_seconds": total_decode_seconds if self.verify else None,
            "roundtrip_ok": roundtrip_ok,
            "tensor_results": tensor_results,
        }
