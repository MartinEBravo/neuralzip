"""Shared direct-encoding measurements for fixed BF16 checkpoints."""

from __future__ import annotations

import gc
import time
from collections import Counter

import torch

from neuralzip import Compressor
from neuralzip.codec_protocol import measure_exponent_streams
from neuralzip.serialization import serialized_codebooks_size
from neuralzip.tensors import bitwise_equal
from neuralzip.zipnn_source import ZipNNSourceCodec, parse_zipnn_exponent_stream

try:
    from .model_loading import (
        DEFAULT_CHUNK_SIZE,
        DEFAULT_COMPRESSION_THRESHOLD,
        MappingModel,
        iter_float_tensors,
    )
except ImportError:  # Direct execution from the examples directory.
    from model_loading import (
        DEFAULT_CHUNK_SIZE,
        DEFAULT_COMPRESSION_THRESHOLD,
        MappingModel,
        iter_float_tensors,
    )


def zipnn_src_pass(
    model: MappingModel,
    *,
    threads: int,
    verify: bool,
) -> dict[str, object]:
    total_encode = 0.0
    total_codec = 0.0
    total_copy = 0.0
    exponent_raw = 0
    exponent_data = 0
    exponent_metadata = 0
    tensors = 0
    chunks = 0
    roundtrip_ok = True
    codec = ZipNNSourceCodec(
        method="HUFFMAN",
        threads=threads,
        compression_threshold=DEFAULT_COMPRESSION_THRESHOLD,
        chunk_size=DEFAULT_CHUNK_SIZE,
    )

    for name, tensor in iter_float_tensors(model):
        original = tensor.detach().cpu().contiguous()
        if original.dtype != torch.bfloat16:
            raise TypeError(f"{name} is {original.dtype}; expected BF16.")
        encoded = codec.encode_tensor(original, verify=verify)
        layout, exponent = parse_zipnn_exponent_stream(
            encoded.payload,
            element_size=2,
            expected_raw_bytes=encoded.raw_bytes,
            expected_byte_reorder=10,
            expected_bit_reorder=1,
            expected_chunk_size=DEFAULT_CHUNK_SIZE,
        )

        total_copy += encoded.input_copy_seconds
        total_codec += encoded.codec_encode_seconds
        total_encode += encoded.encode_seconds
        exponent_raw += int(original.numel())
        exponent_data += exponent.payload_bytes
        exponent_metadata += exponent.metadata_bytes
        chunks += layout.chunk_count
        tensors += 1

        if verify and not encoded.roundtrip_ok:
            roundtrip_ok = False
            if encoded.roundtrip_ok is False:
                raise RuntimeError(f"ZipNN src round-trip failed for tensor {name}.")

    exponent_compressed = exponent_data + exponent_metadata
    return {
        "encode_seconds": total_encode,
        "codec_encode_seconds": total_codec,
        "input_copy_seconds": total_copy,
        "setup_seconds": 0.0,
        "exponent_raw_bytes": exponent_raw,
        "exponent_data_bytes": exponent_data,
        "exponent_metadata_bytes": exponent_metadata,
        "exponent_compressed_bytes": exponent_compressed,
        "exponent_ratio_pct": (
            100.0 * exponent_compressed / exponent_raw if exponent_raw else None
        ),
        "tensor_count": tensors,
        "chunk_count": chunks,
        "roundtrip_ok": bool(roundtrip_ok),
    }


def selected_mode_counts(codec) -> dict[str, int]:
    selected = codec.selected_modes
    if all(isinstance(value, int) for value in selected.values()):
        return {str(key): int(value) for key, value in sorted(selected.items())}
    counts = Counter()
    for mode in selected.values():
        kind = mode.get("kind", "unknown") if isinstance(mode, dict) else mode
        counts[str(kind)] += 1
    return dict(sorted(counts.items()))


def exponent_metrics(codec, payloads) -> dict[str, object]:
    payloads = tuple(
        payload for payload in payloads if str(payload.dtype) == "bfloat16"
    )
    codebooks = tuple(codec.export_codebooks())
    sizes = measure_exponent_streams(
        payloads,
        codebooks,
        codebook_scope="stream",
    )
    data_bytes = 0
    local_chunks = 0
    shared_chunks = 0
    packed_chunks = 0
    raw_chunks = 0
    zero_chunks = 0
    hits = 0
    escapes = 0

    for payload in payloads:
        if not payload.groups:
            continue
        group = payload.groups[int(payload.element_size) - 1]
        data_bytes += int(group.payload_bytes)
        local_chunks += int(group.local_huff0_chunks)
        shared_chunks += int(group.shared_huff0_chunks)
        packed_chunks += int(group.packed_ngram_chunks)
        raw_chunks += int(group.raw_chunks)
        zero_chunks += int(group.zero_chunks)
        hits += int(group.ngram_hits)
        escapes += int(group.ngram_escapes)

    # The archive serializer stores all codebook definitions together.  Report
    # both the exact total and the exponent-only definition bytes used by the
    # thesis tables.
    all_codebook_bytes = int(serialized_codebooks_size(codebooks))
    group_metadata_bytes = sizes.group_bytes - data_bytes
    return {
        "exponent_raw_bytes": sizes.raw_bytes,
        "exponent_data_bytes": data_bytes,
        "exponent_group_metadata_bytes": group_metadata_bytes,
        "exponent_codebook_metadata_bytes": sizes.codebook_bytes,
        "all_codebook_metadata_bytes": all_codebook_bytes,
        "exponent_compressed_bytes": sizes.compressed_bytes,
        "exponent_ratio_pct": sizes.ratio_pct if sizes.raw_bytes else None,
        "local_huff0_chunks": local_chunks,
        "shared_huff0_chunks": shared_chunks,
        "packed_ngram_chunks": packed_chunks,
        "raw_chunks": raw_chunks,
        "zero_chunks": zero_chunks,
        "ngram_hits": hits,
        "ngram_escapes": escapes,
    }


def prepare_neuralzip(
    model,
    *,
    threads: int,
    verify_warmup: bool = True,
    direct_batch_bytes: int | None = None,
):
    compressor = Compressor(
        "neuralzip",
        verify=False,
        keep_payloads=False,
        threads=threads,
        chunk_size=DEFAULT_CHUNK_SIZE,
        compression_threshold=DEFAULT_COMPRESSION_THRESHOLD,
    )
    codec = compressor.codec
    if direct_batch_bytes is not None:
        if direct_batch_bytes <= 0:
            raise ValueError("direct_batch_bytes must be positive")
        codec.direct_batch_bytes = int(direct_batch_bytes)
    parameters = list(iter_float_tensors(model))
    codec.reset_compression()
    codec.reset_decompression()

    started = time.perf_counter()
    codec.prepare_clusters(parameters)
    clustering_seconds = time.perf_counter() - started
    started = time.perf_counter()
    codec.build_codebooks_and_select_modes(parameters)
    codebook_setup_seconds = time.perf_counter() - started

    started = time.perf_counter()
    warmup_payloads = tuple(codec.compress_tensors(parameters))
    warmup_seconds = time.perf_counter() - started
    if len(warmup_payloads) != len(parameters):
        raise RuntimeError("NeuralZip warm-up returned the wrong tensor count")
    if verify_warmup:
        for (tensor_name, tensor), payload in zip(
            parameters, warmup_payloads, strict=True
        ):
            restored = codec.decompress_tensor(payload)
            if not bitwise_equal(tensor, restored):
                raise RuntimeError(f"NeuralZip round trip failed for {tensor_name}")

    warmup_metrics = exponent_metrics(codec, warmup_payloads)
    del warmup_payloads
    if verify_warmup:
        del restored
    gc.collect()
    return {
        "compressor": compressor,
        "parameters": parameters,
        "clustering_seconds": clustering_seconds,
        "codebook_setup_seconds": codebook_setup_seconds,
        "setup_seconds": clustering_seconds + codebook_setup_seconds,
        "warmup_seconds": warmup_seconds,
        "verified_warmup": bool(verify_warmup),
        "warmup_metrics": warmup_metrics,
        "selected_mode_streams": selected_mode_counts(codec),
    }


def encode_neuralzip(prepared) -> dict[str, object]:
    codec = prepared["compressor"].codec
    started = time.perf_counter()
    payloads = tuple(codec.compress_tensors(prepared["parameters"]))
    encode_seconds = time.perf_counter() - started
    metrics = exponent_metrics(codec, payloads)
    del payloads
    gc.collect()
    return {
        "encode_seconds": encode_seconds,
        "codec_encode_seconds": encode_seconds,
        "input_copy_seconds": 0.0,
        **metrics,
    }
