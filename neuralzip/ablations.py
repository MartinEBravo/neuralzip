import heapq
import math
import time

import numpy as np
import torch

from . import turborc
from .neuralzip_codec import NeuralZipCodec
from .packed_exponents import transpose_2d_stream
from .serialization import RAW_CHUNK_METADATA_BYTES
from .byte_group_codec import ByteGroupCodec, _byte_group_dtype_config, _byte_group_reordered_bytes


def named_floating_tensors(model):
    module = model.model if hasattr(model, "model") else model
    if isinstance(module, dict):
        for name, tensor in module.items():
            if torch.is_tensor(tensor) and tensor.is_floating_point():
                yield name, tensor
        return

    for name, tensor in module.named_parameters():
        if torch.is_tensor(tensor) and tensor.is_floating_point():
            yield name, tensor


def exponent_stream(tensor, *, layout="original"):
    tensor = tensor.detach().cpu().contiguous()
    config = _byte_group_dtype_config(tensor.dtype)
    reordered = _byte_group_reordered_bytes(tensor, config)
    element_size = int(config["element_size"])
    stream = reordered[element_size - 1 :: element_size]
    if layout == "transpose2d":
        stream, _ = transpose_2d_stream(
            stream,
            tuple(int(dimension) for dimension in tensor.shape),
        )
        return stream
    if layout != "original":
        raise ValueError("layout must be 'original' or 'transpose2d'.")
    return stream


def _iter_chunks(data, chunk_size):
    for start in range(0, len(data), int(chunk_size)):
        yield data[start : start + int(chunk_size)]


def _huffman_lengths_from_counts(counts):
    counts = {
        int(symbol): int(count)
        for symbol, count in enumerate(np.asarray(counts, dtype=np.int64))
        if int(count) > 0
    }
    if not counts:
        return {}
    if len(counts) == 1:
        symbol = next(iter(counts))
        return {symbol: 1}

    order = 0
    heap = []
    for symbol, count in counts.items():
        heap.append((count, order, {symbol: 0}))
        order += 1
    heapq.heapify(heap)

    while len(heap) > 1:
        left_count, _, left_depths = heapq.heappop(heap)
        right_count, _, right_depths = heapq.heappop(heap)
        merged = {
            symbol: depth + 1
            for depths in (left_depths, right_depths)
            for symbol, depth in depths.items()
        }
        heapq.heappush(heap, (left_count + right_count, order, merged))
        order += 1
    return heap[0][2]


def _huffman_bits_from_counts(counts):
    counts = np.asarray(counts, dtype=np.int64)
    lengths = _huffman_lengths_from_counts(counts)
    return int(sum(int(counts[symbol]) * length for symbol, length in lengths.items()))


def _delta_stream(stream):
    values = np.frombuffer(bytes(stream), dtype=np.uint8)
    if values.size == 0:
        return b""
    delta = np.empty_like(values)
    delta[0] = values[0]
    if values.size > 1:
        delta[1:] = (values[1:].astype(np.int16) - values[:-1].astype(np.int16)) & 0xFF
    return delta.tobytes()


def delta_exponent_huffman_cost(
    model,
    *,
    layout="original",
    chunk_size=256 << 10,
):
    raw_bytes = 0
    compressed_bytes = 0
    codec = ByteGroupCodec(
        chunk_size=chunk_size,
        compression_threshold=1.0,
    )

    stream_specs = []
    for _, tensor in named_floating_tensors(model):
        stream = exponent_stream(tensor, layout=layout)
        raw_bytes += len(stream)
        stream_specs.append((_delta_stream(stream), "delta", chunk_size))

    for group in codec._compress_streams(stream_specs):
        compressed_bytes += sum(chunk.compressed_bytes for chunk in group.chunks)

    return {
        "raw_bytes": raw_bytes,
        "compressed_bytes": compressed_bytes,
        "exponent_ratio_pct": (
            100 * compressed_bytes / raw_bytes if raw_bytes else 0.0
        ),
    }


def _clustered_exponent_streams(model, *, cluster_js_threshold, layout):
    tensors = list(named_floating_tensors(model))
    codec = NeuralZipCodec(
        cluster_js_threshold=float(cluster_js_threshold),
    )
    codec._prepare_clusters(tensors)
    streams_by_cluster = {}
    raw_bytes = 0
    for name, tensor in tensors:
        stream = exponent_stream(tensor, layout=layout)
        streams_by_cluster.setdefault(codec.cluster_id(name), []).append(stream)
        raw_bytes += len(stream)
    return streams_by_cluster, raw_bytes


def order1_huffman_cluster_cost(
    model,
    *,
    cluster_js_threshold=0.05,
    layout="original",
):
    streams_by_cluster, raw_bytes = _clustered_exponent_streams(
        model,
        cluster_js_threshold=cluster_js_threshold,
        layout=layout,
    )
    payload_bits = 0
    first_symbol_bytes = 0
    codebook_bytes = 0
    stream_headers = 0

    for streams in streams_by_cluster.values():
        counts = np.zeros((256, 256), dtype=np.int64)
        for stream in streams:
            values = np.frombuffer(bytes(stream), dtype=np.uint8)
            if values.size == 0:
                continue
            first_symbol_bytes += 1
            stream_headers += RAW_CHUNK_METADATA_BYTES
            if values.size > 1:
                np.add.at(counts, (values[:-1], values[1:]), 1)

        for row in counts:
            if int(row.sum()) == 0:
                continue
            payload_bits += _huffman_bits_from_counts(row)
            codebook_bytes += 1 + 256

    codec_header_bytes = 8 + len(b"order1-huffman") + 8 + len(layout)
    cluster_header_bytes = len(streams_by_cluster) * (8 + 8)
    payload_bytes = math.ceil(payload_bits / 8) + first_symbol_bytes
    metadata_bytes = (
        codec_header_bytes + cluster_header_bytes + stream_headers + codebook_bytes
    )
    compressed_bytes = payload_bytes + metadata_bytes

    return {
        "raw_bytes": raw_bytes,
        "compressed_bytes": compressed_bytes,
        "exponent_ratio_pct": (
            100 * compressed_bytes / raw_bytes if raw_bytes else 0.0
        ),
    }


def selected_turborc_range_cost(
    model,
    *,
    cluster_js_threshold=0.05,
    layout="transpose2d",
    chunk_size=256 << 10,
    method="rccs",
):
    method = turborc._normalize_method(method)
    method_label = f"turborc-{method}"
    chunk_header_bytes = RAW_CHUNK_METADATA_BYTES
    codec_header_bytes = 8 + len(method_label.encode("utf-8")) + 8 + len(layout)

    streams_by_cluster, raw_bytes = _clustered_exponent_streams(
        model,
        cluster_js_threshold=cluster_js_threshold,
        layout=layout,
    )
    start = time.perf_counter()
    total_bytes = codec_header_bytes
    for streams in streams_by_cluster.values():
        for stream in streams:
            for chunk in _iter_chunks(stream, chunk_size):
                raw_cost = chunk_header_bytes + len(chunk)
                encoded = turborc.compress(
                    chunk,
                    method=method,
                    min_saving_bytes=0,
                )
                encoded_cost = (
                    raw_cost
                    if encoded is None
                    else chunk_header_bytes + len(encoded.payload)
                )
                total_bytes += min(raw_cost, encoded_cost)

    return {
        "raw_bytes": raw_bytes,
        "compressed_bytes": total_bytes,
        "exponent_ratio_pct": (100 * total_bytes / raw_bytes if raw_bytes else 0.0),
        "encode_seconds": time.perf_counter() - start,
    }
