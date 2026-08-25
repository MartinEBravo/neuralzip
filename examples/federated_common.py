"""Shared lossless-codec adapters and reporting helpers for federated suites.

This module contains only protocol-neutral mechanics shared by the vision and
language runners.  Dataset partitioning, model state layout, training, and the
timing boundaries around the federated protocol remain owned by each runner.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, MutableMapping, Sequence

import torch

from neuralzip.archive import serialized_tensor_size
from neuralzip.experiment_io import atomic_write_csv, atomic_write_json
from neuralzip.neuralzip_codec import NeuralZipCodec
from neuralzip.tensors import bitwise_equal
from neuralzip.zipnn_source import parse_zipnn_streams


COMPRESSOR_SRC = "ZipNN src"
COMPRESSOR_NEURALZIP = "NeuralZip"
COMPRESSORS = (COMPRESSOR_SRC, COMPRESSOR_NEURALZIP)
DTYPE_NAME = "float32"

_CUMULATIVE_FIELDS = (
    "exponent_raw_bytes",
    "exponent_compressed_bytes",
    "archive_bytes",
    "raw_floating_bytes",
)


class SuspendAutomaticGC:
    """Keep incidental cyclic collection outside the measured encode region."""

    def __enter__(self):
        self._was_enabled = gc.isenabled()
        if self._was_enabled:
            gc.disable()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._was_enabled:
            gc.enable()


@dataclass
class EncodedState:
    payloads: Mapping[str, object]
    encode_seconds: float
    exponent_raw_bytes: int
    exponent_compressed_bytes: int
    archive_bytes: int
    raw_floating_bytes: int


class ZipNNSrcStateCodec:
    """Non-destructive adapter around the official ZipNN source package."""

    def __init__(self, *, threads: int, chunk_size: int, threshold: float):
        try:
            from zipnn import ZipNN
        except ImportError as error:
            raise ImportError(
                "The official zipnn package is required: pip install zipnn."
            ) from error

        self._settings = {
            "method": "HUFFMAN",
            "input_format": "torch",
            "threads": int(threads),
            "compression_threshold": float(threshold),
            "compression_chunk": int(chunk_size),
        }
        self._encoder = ZipNN(**self._settings)
        self._decoder = ZipNN(**self._settings)

    @classmethod
    def _fp32_exponent_sizes(cls, payload: bytes) -> tuple[int, int]:
        """Return exact raw/compressed bytes for ZipNN's FP32 exponent group.

        ZipNN source writes, after its tensor header, one compression-type byte
        and one native ``size_t`` cumulative length per chunk and byte group,
        followed by the group payloads.  FP32 bit reordering places the
        exponent/sign stream in byte group 3.  We charge that group's own
        metadata and payload, matching the per-stream accounting used by
        NeuralZip while excluding the tensor-global header.
        """

        layout = parse_zipnn_streams(payload, element_size=4)
        if layout.byte_reorder != 220 or layout.bit_reorder != 1:
            raise ValueError(
                "Expected the official ZipNN FP32 byte/bit-reorder layout."
            )
        exponent_stream = layout.exponent_stream
        return exponent_stream.raw_bytes, exponent_stream.compressed_bytes

    def encode(self, tensors: Mapping[str, torch.Tensor]) -> EncodedState:
        payloads = {}
        exponent_raw_bytes = 0
        exponent_compressed_bytes = 0
        archive_bytes = 0
        raw_floating_bytes = 0

        encode_seconds = 0.0
        for name, tensor in tensors.items():
            if tensor.dtype != torch.float32:
                raise TypeError(f"{name} is {tensor.dtype}; FP32 is required.")
            # ZipNN source reorders the input buffer in place.  The clone is a
            # required part of a non-destructive encode and is timed.
            with SuspendAutomaticGC():
                started = time.perf_counter()
                compressor_input = tensor.detach().cpu().contiguous().clone()
                if tensor.numel() and compressor_input.data_ptr() == tensor.data_ptr():
                    raise AssertionError(
                        "ZipNN source input clone aliases the model state."
                    )
                payload = self._encoder.compress(compressor_input)
                encode_seconds += time.perf_counter() - started

            # Archive bookkeeping and exponent-ratio parsing are deliberately
            # outside the encoding timer, matching the NeuralZip adapter.
            payloads[name] = payload
            raw_bytes, compressed_bytes = self._fp32_exponent_sizes(payload)
            exponent_raw_bytes += raw_bytes
            exponent_compressed_bytes += compressed_bytes
            archive_bytes += len(payload)
            raw_floating_bytes += int(tensor.numel()) * tensor.element_size()

        return EncodedState(
            payloads=payloads,
            encode_seconds=encode_seconds,
            exponent_raw_bytes=exponent_raw_bytes,
            exponent_compressed_bytes=exponent_compressed_bytes,
            archive_bytes=archive_bytes,
            raw_floating_bytes=raw_floating_bytes,
        )

    def decode(self, encoded: EncodedState) -> dict[str, torch.Tensor]:
        return {
            name: self._decoder.decompress(payload).detach().cpu().contiguous()
            for name, payload in encoded.payloads.items()
        }


class NeuralZipStateCodec:
    """NeuralZip adapter with one frozen initial setup."""

    def __init__(self, *, threads: int, chunk_size: int, threshold: float):
        self.codec = NeuralZipCodec(
            threads=int(threads),
            chunk_size=int(chunk_size),
            compression_threshold=float(threshold),
        )
        self.setup_seconds = 0.0
        self.setup_bytes = 0

    def prepare_once(self, tensors: Mapping[str, torch.Tensor]) -> None:
        named_tensors = list(tensors.items())
        started = time.perf_counter()
        self.codec.prepare_clusters(named_tensors)
        self.codec.build_codebooks_and_select_modes(named_tensors)
        self.setup_seconds = time.perf_counter() - started
        self.setup_bytes = int(self.codec.codebook_metadata_bytes())

    def encode(self, tensors: Mapping[str, torch.Tensor]) -> EncodedState:
        for name, tensor in tensors.items():
            if tensor.dtype != torch.float32:
                raise TypeError(f"{name} is {tensor.dtype}; FP32 is required.")

        with SuspendAutomaticGC():
            started = time.perf_counter()
            payload_list = self.codec.compress_tensors(list(tensors.items()))
            encode_seconds = time.perf_counter() - started
        payloads = {payload.name: payload for payload in payload_list}
        if payloads.keys() != tensors.keys():
            raise RuntimeError("NeuralZip changed the floating-state tensor names.")

        exponent_raw_bytes = 0
        exponent_compressed_bytes = 0
        archive_bytes = 0
        raw_floating_bytes = 0
        for payload in payload_list:
            exponent_group = payload.groups[int(payload.element_size) - 1]
            exponent_raw_bytes += int(exponent_group.symbol_count)
            exponent_compressed_bytes += int(exponent_group.compressed_bytes)
            archive_bytes += int(serialized_tensor_size(payload))
            raw_floating_bytes += int(payload.raw_bytes)

        return EncodedState(
            payloads=payloads,
            encode_seconds=encode_seconds,
            exponent_raw_bytes=exponent_raw_bytes,
            exponent_compressed_bytes=exponent_compressed_bytes,
            archive_bytes=archive_bytes,
            raw_floating_bytes=raw_floating_bytes,
        )

    def decode(self, encoded: EncodedState) -> dict[str, torch.Tensor]:
        return {
            name: self.codec.decompress_tensor(payload).detach().cpu().contiguous()
            for name, payload in encoded.payloads.items()
        }


class FederatedCodecProtocol:
    """One shared, verified codec protocol for every federated runner.

    Task-specific runners still own their models, data, local training, state
    layout, and FedAvg implementation.  This object owns the behavior which
    must remain identical across tasks: codec setup, alternating measurement
    order, non-destructive encoding, bitwise round-trip checks, and selection
    of the NeuralZip-decoded canonical trajectory.
    """

    def __init__(self, *, threads: int, chunk_size: int, threshold: float):
        self.codecs = {
            COMPRESSOR_SRC: ZipNNSrcStateCodec(
                threads=threads,
                chunk_size=chunk_size,
                threshold=threshold,
            ),
            COMPRESSOR_NEURALZIP: NeuralZipStateCodec(
                threads=threads,
                chunk_size=chunk_size,
                threshold=threshold,
            ),
        }
        self.setup_seconds = {name: 0.0 for name in COMPRESSORS}
        self.setup_bytes = {name: 0 for name in COMPRESSORS}

    def prepare_once(self, tensors: Mapping[str, torch.Tensor]) -> None:
        neuralzip = self.codecs[COMPRESSOR_NEURALZIP]
        if not isinstance(neuralzip, NeuralZipStateCodec):
            raise TypeError("NeuralZip codec adapter has an unexpected type.")
        neuralzip.prepare_once(tensors)
        self.setup_seconds[COMPRESSOR_NEURALZIP] = neuralzip.setup_seconds
        self.setup_bytes[COMPRESSOR_NEURALZIP] = neuralzip.setup_bytes

    @staticmethod
    def encoding_order(
        round_index: int,
        *,
        client_index: int | None = None,
    ) -> tuple[str, str]:
        order = (
            COMPRESSORS
            if int(round_index) % 2
            else tuple(reversed(COMPRESSORS))
        )
        if client_index is not None and int(client_index) % 2:
            order = tuple(reversed(order))
        return order

    def encode_verified(
        self,
        tensors: Mapping[str, torch.Tensor],
        *,
        order: Sequence[str],
        context: str,
        phase: str,
        retain_payloads: bool = False,
    ) -> tuple[dict[str, EncodedState], dict[str, torch.Tensor]]:
        """Encode with both codecs and return NeuralZip's verified decode."""

        reference = _clone_tensor_map(tensors)
        measurements: dict[str, EncodedState] = {}
        decoded: dict[str, dict[str, torch.Tensor]] = {}
        for name in order:
            if name not in self.codecs:
                raise ValueError(f"Unknown federated compressor: {name}.")
            encoded = self.codecs[name].encode(tensors)
            _assert_same_tensors(
                reference,
                tensors,
                context=f"{context} {name} {phase} input immutability",
            )
            decoded[name] = self.codecs[name].decode(encoded)
            _assert_same_tensors(
                tensors,
                decoded[name],
                context=f"{context} {name} {phase}",
            )
            measurements[name] = (
                encoded if retain_payloads else replace(encoded, payloads={})
            )

        missing = set(COMPRESSORS).difference(decoded)
        if missing:
            raise ValueError(
                "Federated codec order omitted compressors: "
                + ", ".join(sorted(missing))
            )
        _assert_same_tensors(
            decoded[COMPRESSOR_SRC],
            decoded[COMPRESSOR_NEURALZIP],
            context=f"{context} cross-codec {phase}",
        )
        return measurements, decoded[COMPRESSOR_NEURALZIP]

    def warmup(self, tensors: Mapping[str, torch.Tensor], *, context: str) -> None:
        self.encode_verified(
            tensors,
            order=COMPRESSORS,
            context=context,
            phase="warmup",
        )


def _assert_same_tensors(
    expected: Mapping[str, torch.Tensor],
    observed: Mapping[str, torch.Tensor],
    *,
    context: str,
) -> None:
    if expected.keys() != observed.keys():
        raise AssertionError(f"Tensor names changed during {context}.")
    for name, tensor in expected.items():
        candidate = observed[name]
        if not bitwise_equal(tensor, candidate):
            raise AssertionError(f"Bitwise mismatch during {context}: {name}.")


def _clone_tensor_map(
    tensors: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Create an untimed immutable reference for codec input checks."""

    return {
        name: tensor.detach().cpu().contiguous().clone()
        for name, tensor in tensors.items()
    }


def _write_csv_atomic(path: Path, rows: list[dict[str, object]]) -> None:
    atomic_write_csv(
        path,
        rows[0].keys(),
        rows,
        lineterminator="\n",
    )


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    atomic_write_json(path, payload)


def _encoded_metrics(
    downlink: EncodedState,
    uplinks: list[EncodedState],
    *,
    clients: int,
) -> dict[str, float | int]:
    exponent_raw = downlink.exponent_raw_bytes * int(clients) + sum(
        encoded.exponent_raw_bytes for encoded in uplinks
    )
    exponent_compressed = downlink.exponent_compressed_bytes * int(clients) + sum(
        encoded.exponent_compressed_bytes for encoded in uplinks
    )
    archive_bytes = downlink.archive_bytes * int(clients) + sum(
        encoded.archive_bytes for encoded in uplinks
    )
    raw_floating_bytes = downlink.raw_floating_bytes * int(clients) + sum(
        encoded.raw_floating_bytes for encoded in uplinks
    )
    critical_seconds = downlink.encode_seconds + max(
        (encoded.encode_seconds for encoded in uplinks), default=0.0
    )
    return {
        "encoding_critical_seconds": critical_seconds,
        "exponent_raw_bytes": exponent_raw,
        "exponent_compressed_bytes": exponent_compressed,
        "exponent_ratio_pct": (
            100.0 * exponent_compressed / exponent_raw if exponent_raw else 0.0
        ),
        "archive_bytes": archive_bytes,
        "raw_floating_bytes": raw_floating_bytes,
    }


def new_cumulative_metrics() -> dict[str, dict[str, float | int]]:
    """Create the stable accumulator schema shared by both final suites."""

    return {
        name: {
            "encoding_seconds": 0.0,
            **{field: 0 for field in _CUMULATIVE_FIELDS},
        }
        for name in COMPRESSORS
    }


def build_round_rows(
    *,
    prefix: Mapping[str, object],
    suffix: Mapping[str, object],
    downlinks: Mapping[str, EncodedState],
    uplinks: Mapping[str, list[EncodedState]],
    clients: int,
    cumulative: MutableMapping[str, MutableMapping[str, float | int]],
    setup_seconds: Mapping[str, float],
    setup_bytes: Mapping[str, int],
) -> list[dict[str, object]]:
    """Accumulate one round and render its two codec rows consistently."""

    rows = []
    for name in COMPRESSORS:
        metrics = _encoded_metrics(downlinks[name], uplinks[name], clients=clients)
        totals = cumulative[name]
        totals["encoding_seconds"] = float(totals["encoding_seconds"]) + float(
            metrics["encoding_critical_seconds"]
        )
        for field in _CUMULATIVE_FIELDS:
            totals[field] = int(totals[field]) + int(metrics[field])
        rows.append(
            {
                **prefix,
                "compressor": name,
                "setup_seconds_once": setup_seconds[name],
                "setup_bytes_once": setup_bytes[name],
                "downlink_encode_seconds": downlinks[name].encode_seconds,
                "uplink_encode_max_seconds": max(
                    encoded.encode_seconds for encoded in uplinks[name]
                ),
                "encoding_critical_seconds": metrics["encoding_critical_seconds"],
                "encoding_critical_cumulative_seconds": totals[
                    "encoding_seconds"
                ],
                "exponent_raw_bytes": metrics["exponent_raw_bytes"],
                "exponent_compressed_bytes": metrics[
                    "exponent_compressed_bytes"
                ],
                "exponent_ratio_pct": metrics["exponent_ratio_pct"],
                "archive_bytes": metrics["archive_bytes"],
                "raw_floating_bytes": metrics["raw_floating_bytes"],
                **suffix,
            }
        )
    return rows


def build_summary_rows(
    *,
    prefix: Mapping[str, object],
    dimensions: Mapping[str, object],
    clients: int,
    cumulative: Mapping[str, Mapping[str, float | int]],
    setup_seconds: Mapping[str, float],
    setup_bytes: Mapping[str, int],
) -> list[dict[str, object]]:
    """Render cumulative rows with identical setup amortization accounting."""

    rows = []
    for name in COMPRESSORS:
        totals = cumulative[name]
        raw_exponent = int(totals["exponent_raw_bytes"])
        compressed_exponent = int(totals["exponent_compressed_bytes"])
        setup_communication_bytes = int(setup_bytes[name]) * int(clients)
        compressed_with_setup = compressed_exponent + setup_communication_bytes
        rows.append(
            {
                **prefix,
                "compressor": name,
                **dimensions,
                "setup_seconds_once": setup_seconds[name],
                "setup_bytes_once": setup_bytes[name],
                "encoding_critical_seconds": totals["encoding_seconds"],
                "encoding_plus_setup_seconds": (
                    float(totals["encoding_seconds"]) + setup_seconds[name]
                ),
                "exponent_raw_bytes": raw_exponent,
                "exponent_compressed_bytes": compressed_exponent,
                "exponent_ratio_pct": (
                    100.0 * compressed_exponent / raw_exponent
                    if raw_exponent
                    else 0.0
                ),
                "setup_communication_bytes": setup_communication_bytes,
                "exponent_compressed_plus_setup_bytes": compressed_with_setup,
                "exponent_ratio_with_setup_pct": (
                    100.0 * compressed_with_setup / raw_exponent
                    if raw_exponent
                    else 0.0
                ),
                "archive_bytes": totals["archive_bytes"],
                "raw_floating_bytes": totals["raw_floating_bytes"],
                "roundtrip_ok": True,
            }
        )
    return rows
