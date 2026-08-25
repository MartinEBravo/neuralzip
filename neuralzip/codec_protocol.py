"""Shared, versioned protocol primitives for checkpoint experiments.

This module deliberately contains no model- or task-specific policy.  It
defines the timing and byte-accounting boundary used by experiment runners so
that two frontends cannot silently benchmark different NeuralZip paths.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import time

from .archive import (
    serialized_codebooks_size,
    serialized_group_size,
    serialized_tensor_size,
)
from .tensors import bitwise_equal


CODEC_PROTOCOL_VERSION = 1


@dataclass(frozen=True)
class CodecProtocolSpec:
    version: int = CODEC_PROTOCOL_VERSION
    prefer_batch: bool = True
    timer_scope: str = "codec encode call only"
    size_scope: str = "serialized tensor bytes; setup metadata reported separately"


DEFAULT_CODEC_PROTOCOL = CodecProtocolSpec()


def is_current_codec_protocol(value):
    """Accept integer-like CSV values while rejecting missing/legacy rows."""

    try:
        return float(value) == float(CODEC_PROTOCOL_VERSION)
    except (TypeError, ValueError, OverflowError):
        return False


@dataclass(frozen=True)
class SetupReport:
    precompute_seconds: float
    codebook_setup_seconds: float

    @property
    def total_setup_seconds(self):
        return self.precompute_seconds + self.codebook_setup_seconds


@dataclass(frozen=True)
class EncodedTensorBatch:
    payloads: tuple
    raw_bytes: int
    compressed_bytes: int
    encode_seconds: float
    huffman_chunks: int
    raw_chunks: int
    shared_codebook_refs: int
    used_batch: bool
    materialization_active: bool
    materialization_bytes: int
    materialization_hits: int

    @property
    def encoder_path(self):
        return "compress_tensors" if self.used_batch else "compress_tensor"


@dataclass(frozen=True)
class ExponentSizeReport:
    """Exact serialized accounting for the exponent stream of a payload set."""

    raw_bytes: int
    group_bytes: int
    codebook_bytes: int
    referenced_codebooks: int

    @property
    def compressed_bytes(self):
        return self.group_bytes + self.codebook_bytes

    @property
    def ratio_pct(self):
        return 100.0 * self.compressed_bytes / self.raw_bytes if self.raw_bytes else 0.0


def named_tensor_items(tensors):
    """Normalize a mapping or iterable into an immutable named-tensor batch."""

    if isinstance(tensors, Mapping):
        return tuple(tensors.items())
    return tuple(tensors)


def encode_tensor_batch(codec, tensors, *, protocol=DEFAULT_CODEC_PROTOCOL):
    """Encode one named tensor batch under a stable timing/size contract."""

    named_tensors = named_tensor_items(tensors)
    batch_encoder = getattr(codec, "compress_tensors", None)
    used_batch = bool(protocol.prefer_batch and callable(batch_encoder))
    cache_stats = getattr(codec, "prepacked_cache_stats", {}) or {}
    cache_hits_before = int(cache_stats.get("cache_hits", 0))
    materialization_bytes = int(getattr(codec, "prepacked_cache_bytes", 0))

    started = time.perf_counter()
    if used_batch:
        payloads = tuple(batch_encoder(named_tensors))
    else:
        payloads = tuple(
            codec.compress_tensor(name, tensor) for name, tensor in named_tensors
        )
    encode_seconds = time.perf_counter() - started
    cache_stats = getattr(codec, "prepacked_cache_stats", {}) or {}
    materialization_hits = max(
        0,
        int(cache_stats.get("cache_hits", 0)) - cache_hits_before,
    )

    if len(payloads) != len(named_tensors):
        raise RuntimeError("Codec returned a different number of tensor payloads.")
    expected_names = tuple(str(name) for name, _ in named_tensors)
    payload_names = tuple(str(payload.name) for payload in payloads)
    if payload_names != expected_names:
        raise RuntimeError(
            "Codec changed tensor names or order: "
            f"expected {expected_names!r}, observed {payload_names!r}."
        )

    raw_bytes = sum(int(payload.raw_bytes) for payload in payloads)
    compressed_bytes = sum(serialized_tensor_size(payload) for payload in payloads)
    return EncodedTensorBatch(
        payloads=payloads,
        raw_bytes=raw_bytes,
        compressed_bytes=int(compressed_bytes),
        encode_seconds=float(encode_seconds),
        huffman_chunks=sum(int(payload.huffman_chunks) for payload in payloads),
        raw_chunks=sum(int(payload.raw_chunks) for payload in payloads),
        shared_codebook_refs=sum(
            int(getattr(payload, "shared_codebook_refs", 0))
            for payload in payloads
        ),
        used_batch=used_batch,
        materialization_active=materialization_hits > 0,
        materialization_bytes=materialization_bytes,
        materialization_hits=materialization_hits,
    )


def prepare_neuralzip_codec(codec, named_tensors):
    """Fit clustering and codebooks while keeping both setup costs separate."""

    named_tensors = named_tensor_items(named_tensors)
    started = time.perf_counter()
    codec.prepare_clusters(named_tensors)
    precompute_seconds = time.perf_counter() - started

    started = time.perf_counter()
    codec.build_codebooks_and_select_modes(named_tensors)
    codebook_setup_seconds = time.perf_counter() - started
    return SetupReport(
        precompute_seconds=float(precompute_seconds),
        codebook_setup_seconds=float(codebook_setup_seconds),
    )


def decode_tensor_batch(codec, payloads):
    """Decode payloads and return the exact decode-call wall time."""

    started = time.perf_counter()
    tensors = {
        payload.name: codec.decompress_tensor(payload)
        for payload in payloads
    }
    return tensors, float(time.perf_counter() - started)


def payload_exponent_sizes(payload):
    """Return raw and serialized bytes for a tensor's exponent stream."""

    if not payload.groups:
        return 0, 0
    exponent_group = payload.groups[int(payload.element_size) - 1]
    return (
        int(exponent_group.symbol_count),
        int(serialized_group_size(exponent_group)),
    )


def measure_exponent_streams(payloads, codebooks=(), *, codebook_scope="referenced"):
    """Measure exponent groups and only the codebooks they reference.

    Tensor-global archive headers are intentionally excluded.  This is the
    common exponent-only scope used throughout the experiment tables.
    ``codebook_scope="stream"`` includes every configured codebook for the
    observed exponent dtype/stream pairs, including a codebook whose chunks
    all fell back to raw encoding.
    """

    if codebook_scope not in {"referenced", "stream"}:
        raise ValueError("codebook_scope must be 'referenced' or 'stream'.")

    codebooks = tuple(codebooks)
    raw_bytes = 0
    group_bytes = 0
    referenced_ids = set()
    stream_keys = set()
    for payload in payloads:
        if not payload.groups:
            continue
        stream_id = int(payload.element_size) - 1
        group = payload.groups[stream_id]
        raw_bytes += int(group.symbol_count)
        group_bytes += int(serialized_group_size(group))
        stream_keys.add((str(payload.dtype), stream_id))
        referenced_ids.update(
            int(chunk.codebook_id)
            for chunk in group.chunks
            if chunk.codebook_id is not None
        )

    supplied_ids = {int(codebook.codebook_id) for codebook in codebooks}
    missing_ids = referenced_ids - supplied_ids
    if missing_ids:
        raise ValueError(
            "Exponent payloads reference missing codebooks: "
            + ", ".join(str(value) for value in sorted(missing_ids))
        )

    if codebook_scope == "referenced":
        included = tuple(
            codebook
            for codebook in codebooks
            if int(codebook.codebook_id) in referenced_ids
        )
    else:
        included = tuple(
            codebook
            for codebook in codebooks
            if (str(codebook.dtype), int(codebook.stream_id)) in stream_keys
        )
    return ExponentSizeReport(
        raw_bytes=int(raw_bytes),
        group_bytes=int(group_bytes),
        codebook_bytes=int(serialized_codebooks_size(included)),
        referenced_codebooks=len(referenced_ids),
    )


def measure_compressor_exponents(result, *, dtypes=None, codebook_scope="stream"):
    """Adapt a ``Compressor.compress`` result to exponent accounting."""

    accepted = None if dtypes is None else {str(dtype) for dtype in dtypes}
    payloads = tuple(
        row.get("payload")
        for row in result.get("tensor_results", ())
        if row.get("payload") is not None
        and (
            accepted is None
            or str(row["payload"].dtype) in accepted
        )
    )
    return measure_exponent_streams(
        payloads,
        result.get("codebooks") or (),
        codebook_scope=codebook_scope,
    )


def assert_tensor_roundtrip(expected, decoded, *, method, checkpoint):
    """Require a bitwise-identical named tensor mapping."""

    if expected.keys() != decoded.keys():
        raise AssertionError(
            f"{method} changed the tensor names for checkpoint {checkpoint}."
        )
    for name, tensor in expected.items():
        restored = decoded[name]
        if not bitwise_equal(tensor, restored):
            raise AssertionError(
                f"{method} failed bitwise roundtrip for {checkpoint}:{name}."
            )
