"""Shared compression protocol for the final checkpoint trajectories.

Vision and language checkpoints deliberately share this implementation so
their timing and byte-accounting boundaries cannot drift independently.
"""

from __future__ import annotations

import gc
import hashlib
import struct
import time
from pathlib import Path
from typing import Callable, Iterable, Mapping

import torch

from neuralzip import Compressor, NeuralZipPlan
from neuralzip.archive import serialized_codebooks_size
from neuralzip.codec_protocol import (
    CODEC_PROTOCOL_VERSION,
    encode_tensor_batch,
    is_current_codec_protocol,
    measure_exponent_streams,
)
from neuralzip.tensors import bitwise_equal
from neuralzip.zipnn_source import parse_zipnn_exponent_stream
from neuralzip.experiment_io import read_csv


BASE_RESULT_FIELDS = (
    "model",
    "model_label",
    "epoch",
    "compressor",
    "codec_protocol_version",
    "encoder_path",
    "checkpoint_path",
    "checkpoint_source",
    "dtype",
    "tensor_count",
    "nonfloating_tensor_count",
    "raw_bytes",
    "compressed_bytes",
    "full_ratio_pct",
    "exponent_raw_bytes",
    "exponent_compressed_bytes",
    "exponent_ratio_pct",
    "encode_seconds",
    "codec_encode_seconds",
    "input_copy_seconds",
    "setup_seconds_once",
    "mapping_seconds",
    "mapped_tensors",
    "roundtrip_exact",
    "input_unchanged",
    "threads",
    "warmup_runs",
    "measured_repetitions",
    "ratio_scope",
    "status",
)

ENCODER_PATHS = {"NeuralZip": "compress_tensors", "ZipNN src": "zipnn-src"}


def is_current_codec_result(row) -> bool:
    compressor = str(row.get("compressor", ""))
    return (
        is_current_codec_protocol(row.get("codec_protocol_version"))
        and row.get("encoder_path") == ENCODER_PATHS.get(compressor)
    )


def completed_checkpoint_methods(
    result_path: Path,
    slug: str,
    epoch: int,
    *,
    identities: Mapping[str, object] | None = None,
) -> set[str]:
    """Return verified current-protocol measurements for one checkpoint."""

    required = dict(identities or {})
    return {
        str(row["compressor"])
        for row in read_csv(result_path)
        if row.get("model") == slug
        and int(row.get("epoch", -1)) == int(epoch)
        and row.get("status") == "ok"
        and row.get("roundtrip_exact", "").lower() == "true"
        and row.get("input_unchanged", "").lower() == "true"
        and all(row.get(key) == str(value) for key, value in required.items())
        and is_current_codec_result(row)
    }


def tensor_digest(named_tensors: Iterable[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(named_tensors, key=lambda item: item[0]):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(struct.pack("<Q", value.ndim))
        for dimension in value.shape:
            digest.update(struct.pack("<Q", int(dimension)))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def nonfloating_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        str(name): tensor.detach().cpu().contiguous()
        for name, tensor in state.items()
        if torch.is_tensor(tensor) and not tensor.is_floating_point()
    }


def floating_state(
    state: Mapping[str, torch.Tensor],
    *,
    error_prefix: str,
) -> dict[str, torch.Tensor]:
    tensors = {
        str(name): tensor.detach().cpu().contiguous()
        for name, tensor in state.items()
        if torch.is_tensor(tensor) and tensor.is_floating_point() and tensor.numel()
    }
    invalid = {
        name: tensor.dtype
        for name, tensor in tensors.items()
        if tensor.dtype != torch.float32
    }
    if invalid:
        preview = ", ".join(
            f"{name}={dtype}" for name, dtype in list(invalid.items())[:8]
        )
        raise TypeError(f"{error_prefix} require exact FP32 tensors; got {preview}.")
    return tensors


def exponent_metrics(payloads, codebooks=()) -> dict[str, int | float]:
    report = measure_exponent_streams(payloads, codebooks)
    return {
        "exponent_raw_bytes": report.raw_bytes,
        "exponent_compressed_bytes": report.compressed_bytes,
        "exponent_group_bytes": report.group_bytes,
        "exponent_codebook_bytes": report.codebook_bytes,
        "exponent_ratio_pct": report.ratio_pct,
    }


def zipnn_src_exponent_size(
    payload: bytes,
    tensor: torch.Tensor,
    chunk_size: int,
) -> tuple[int, int]:
    """Return exponent bytes under the same group-level scope as NeuralZip."""

    if tensor.dtype != torch.float32:
        raise TypeError(f"Expected FP32 ZipNN payload, got {tensor.dtype}.")
    _, exponent = parse_zipnn_exponent_stream(
        payload,
        element_size=tensor.element_size(),
        expected_raw_bytes=tensor.numel() * tensor.element_size(),
        expected_byte_reorder=220,
        expected_bit_reorder=1,
        expected_chunk_size=chunk_size,
    )
    return int(exponent.raw_bytes), int(exponent.compressed_bytes)


def _as_plan(value) -> NeuralZipPlan:
    return value if isinstance(value, NeuralZipPlan) else NeuralZipPlan(codec=value)


def setup_neuralzip(
    named_tensors: list[tuple[str, torch.Tensor]],
    *,
    threads: int,
) -> tuple[NeuralZipPlan, dict]:
    plan = NeuralZipPlan.fit(
        named_tensors,
        threads=threads,
        compression_threshold=0.95,
    )
    report = plan.setup_report
    if report is None:
        raise RuntimeError("Fitted NeuralZip plan did not report setup timing.")
    codebooks = plan.codebooks
    return plan, {
        "clustering_seconds": report.clustering_seconds,
        "codebook_seconds": report.codebook_setup_seconds,
        "setup_seconds": report.setup_seconds,
        "clusters": int(plan.codec.cluster_count),
        "codebooks": len(codebooks),
        "codebook_digest": plan.digest,
        "selected_modes": plan.codec.selected_modes,
        "epoch": 0,
        "scope": "one frozen plan per training trajectory",
    }


def _encode_batch(plan, named_tensors):
    plan = _as_plan(plan)
    return encode_tensor_batch(plan.codec, tuple(named_tensors))


def verify_neuralzip(plan, named_tensors, payloads) -> bool:
    plan = _as_plan(plan)
    expected = dict(named_tensors)
    if len(payloads) != len(expected):
        return False
    restored = plan.decode(payloads)
    return all(
        name in restored and bitwise_equal(tensor, restored[name])
        for name, tensor in expected.items()
    )


def benchmark_neuralzip(
    plan,
    named_tensors: list[tuple[str, torch.Tensor]],
    *,
    epoch: int,
    setup: Mapping,
) -> dict:
    plan = _as_plan(plan)
    before = tensor_digest(named_tensors)
    mapping_started = time.perf_counter()
    transfer = plan.map_target(
        named_tensors,
        strategy="structural",
        require_full_match=True,
    )
    mapping_seconds = time.perf_counter() - mapping_started
    if int(transfer["matched_tensors"]) != len(named_tensors):
        raise RuntimeError(f"NeuralZip failed full checkpoint mapping at epoch {epoch}.")
    if plan.digest != str(setup["codebook_digest"]):
        raise RuntimeError("NeuralZip changed the frozen epoch-0 codebooks.")

    warmup = _encode_batch(plan, named_tensors)
    del warmup
    gc.collect()
    encoded = _encode_batch(plan, named_tensors)
    if not encoded.used_batch:
        raise RuntimeError("NeuralZip checkpoint encoding must use compress_tensors.")
    metrics = exponent_metrics(encoded.payloads, plan.codebooks)
    roundtrip = verify_neuralzip(plan, named_tensors, encoded.payloads)
    unchanged = tensor_digest(named_tensors) == before
    if not roundtrip or not unchanged:
        raise RuntimeError(
            f"NeuralZip verification failed at epoch {epoch}: "
            f"roundtrip={roundtrip}, input_unchanged={unchanged}."
        )
    compressed_bytes = encoded.compressed_bytes + int(
        serialized_codebooks_size(plan.codebooks)
    )
    output = {
        "codec_protocol_version": CODEC_PROTOCOL_VERSION,
        "encoder_path": encoded.encoder_path,
        "raw_bytes": encoded.raw_bytes,
        "compressed_bytes": compressed_bytes,
        "full_ratio_pct": (
            100.0 * compressed_bytes / encoded.raw_bytes
            if encoded.raw_bytes
            else 0.0
        ),
        "encode_seconds": encoded.encode_seconds,
        "codec_encode_seconds": encoded.encode_seconds,
        "input_copy_seconds": 0.0,
        "setup_seconds_once": float(setup["setup_seconds"]),
        "mapping_seconds": mapping_seconds,
        "mapped_tensors": int(transfer["matched_tensors"]),
        "roundtrip_exact": roundtrip,
        "input_unchanged": unchanged,
        **metrics,
    }
    del encoded
    gc.collect()
    return output


def benchmark_zipnn_src(
    tensors: Mapping[str, torch.Tensor],
    *,
    threads: int,
    chunk_size: int = 256 << 10,
) -> dict:
    named_tensors = list(tensors.items())
    before = tensor_digest(named_tensors)
    warmup = Compressor(
        "zipnn-src",
        verify=False,
        keep_payloads=False,
        threads=threads,
        zipnn_method="HUFFMAN",
        chunk_size=chunk_size,
        compression_threshold=0.95,
    ).compress(tensors)
    del warmup
    gc.collect()

    result = Compressor(
        "zipnn-src",
        verify=True,
        keep_payloads=True,
        threads=threads,
        zipnn_method="HUFFMAN",
        chunk_size=chunk_size,
        compression_threshold=0.95,
    ).compress(tensors)
    exponent_raw_bytes = 0
    exponent_compressed_bytes = 0
    for tensor_result in result["tensor_results"]:
        name = str(tensor_result["name"])
        raw, compressed = zipnn_src_exponent_size(
            tensor_result["payload"],
            tensors[name],
            chunk_size,
        )
        exponent_raw_bytes += raw
        exponent_compressed_bytes += compressed
    unchanged = tensor_digest(named_tensors) == before
    roundtrip = result["roundtrip_ok"] is True
    if not unchanged or not roundtrip:
        raise RuntimeError(
            "ZipNN src must receive a dedicated clone and preserve the checkpoint: "
            f"roundtrip={roundtrip}, input_unchanged={unchanged}."
        )
    output = {
        "codec_protocol_version": CODEC_PROTOCOL_VERSION,
        "encoder_path": "zipnn-src",
        "raw_bytes": int(result["raw_bytes"]),
        "compressed_bytes": int(result["compressed_bytes"]),
        "full_ratio_pct": 100.0 * float(result["compression_ratio"]),
        "exponent_raw_bytes": exponent_raw_bytes,
        "exponent_compressed_bytes": exponent_compressed_bytes,
        "exponent_ratio_pct": (
            100.0 * exponent_compressed_bytes / exponent_raw_bytes
            if exponent_raw_bytes
            else 0.0
        ),
        "encode_seconds": float(result["encode_seconds"]),
        "codec_encode_seconds": float(
            result.get("codec_encode_seconds", result["encode_seconds"])
        ),
        "input_copy_seconds": float(result.get("input_copy_seconds", 0.0)),
        "setup_seconds_once": 0.0,
        "mapping_seconds": 0.0,
        "mapped_tensors": len(named_tensors),
        "roundtrip_exact": roundtrip,
        "input_unchanged": unchanged,
    }
    del result
    gc.collect()
    return output


def benchmark_checkpoint_codecs(
    *,
    slug: str,
    epoch: int,
    plan,
    setup: Mapping,
    tensors: Mapping[str, torch.Tensor],
    common_row: Mapping,
    completed: set[str],
    threads: int,
    upsert_result: Callable[[Mapping], None],
    transform_metrics: Callable[[Mapping], Mapping] = dict,
) -> None:
    """Benchmark and persist both final checkpoint codecs in stable order."""

    named_tensors = list(tensors.items())
    methods = (
        (
            "NeuralZip",
            lambda: benchmark_neuralzip(
                plan,
                named_tensors,
                epoch=epoch,
                setup=setup,
            ),
        ),
        ("ZipNN src", lambda: benchmark_zipnn_src(tensors, threads=threads)),
    )
    for compressor, benchmark in methods:
        if compressor in completed:
            continue
        metrics = dict(transform_metrics(benchmark()))
        upsert_result({**common_row, "compressor": compressor, **metrics})
        print(
            f"[{slug}] epoch={epoch:03d} {compressor} "
            f"ratio={metrics['exponent_ratio_pct']:.3f}% "
            f"encode={metrics['encode_seconds']:.6f}s",
            flush=True,
        )
