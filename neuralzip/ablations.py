import re
import math
import zlib
import lzma

import numpy as np
import pandas as pd
import torch

from . import huffman
from .neuralzip_codec import NeuralZipCodec
from .utils import split_fields
from .zipnn_codec import _zipnn_dtype_config, _zipnn_reordered_bytes


TRANSFORMER_WEIGHT_PATTERNS = (
    "attn.c_attn.weight",
    "attn.c_proj.weight",
    "mlp.c_fc.weight",
    "mlp.c_proj.weight",
)


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


def transformer_weight_subset(model, limit=16):
    tensors = {}
    for name, tensor in named_floating_tensors(model):
        if any(pattern in name for pattern in TRANSFORMER_WEIGHT_PATTERNS):
            tensors[name] = tensor
        if len(tensors) >= int(limit):
            break
    return tensors


def layer_signature(name):
    parts = []
    for part in str(name).split("."):
        parts.append(re.sub(r"\d+", "#", part))
    return ".".join(parts)


def exponent_stream(tensor):
    _, exponent, _ = split_fields(tensor.detach().cpu().contiguous())
    return exponent.reshape(-1).to(torch.uint8).numpy()


def zipnn_exponent_stream(tensor, *, layout="original"):
    tensor = tensor.detach().cpu().contiguous()
    config = _zipnn_dtype_config(tensor.dtype)
    reordered = _zipnn_reordered_bytes(tensor, config)
    element_size = int(config["element_size"])
    stream = reordered[element_size - 1 :: element_size]
    if layout == "transpose2d":
        return _transpose_2d_stream(tensor, stream)
    if layout != "original":
        raise ValueError("layout must be 'original' or 'transpose2d'.")
    return stream


def _transpose_2d_stream(tensor, stream):
    if tensor.ndim != 2 or tensor.numel() != len(stream):
        return stream
    array = (
        torch.frombuffer(bytearray(stream), dtype=torch.uint8)
        .reshape(tuple(tensor.shape))
        .t()
        .contiguous()
        .reshape(-1)
    )
    return array.numpy().tobytes()


def zero_order_entropy(values):
    if len(values) == 0:
        return 0.0
    _, counts = np.unique(values, return_counts=True)
    probabilities = counts.astype(np.float64) / float(counts.sum())
    return float(-(probabilities * np.log2(probabilities)).sum())


def exponent_entropy_by_signature(model, max_tensors=None):
    table, _ = exponent_histograms(model, max_tensors=max_tensors)
    if table.empty:
        return table

    return (
        table.groupby("signature", as_index=False)
        .agg(
            tensors=("name", "count"),
            total_exponents=("num_exponents", "sum"),
            mean_h0_bits=("entropy_bits", "mean"),
            median_h0_bits=("entropy_bits", "median"),
            min_h0_bits=("entropy_bits", "min"),
            max_h0_bits=("entropy_bits", "max"),
            mean_dominant_exponent_mass=("dominant_exponent_mass", "mean"),
        )
        .sort_values("total_exponents", ascending=False)
    )


def exponent_histograms(model, max_tensors=None):
    rows = []
    histograms = []
    for tensor_index, (name, tensor) in enumerate(named_floating_tensors(model)):
        if max_tensors is not None and tensor_index >= int(max_tensors):
            break
        stream = exponent_stream(tensor)
        if stream.size == 0:
            continue
        counts = np.bincount(stream, minlength=256).astype(np.float64)
        distribution = counts / counts.sum()
        rows.append(
            {
                "name": name,
                "signature": layer_signature(name),
                "num_exponents": int(stream.size),
                "entropy_bits": zero_order_entropy(stream),
                "dominant_exponent": int(np.argmax(counts)),
                "dominant_exponent_mass": float(counts.max() / counts.sum()),
            }
        )
        histograms.append(distribution)
    return pd.DataFrame(rows), np.array(histograms)


def jensen_shannon(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    middle = 0.5 * (left + right)
    mask_left = left > 0
    mask_right = right > 0
    divergence = 0.0
    divergence += 0.5 * np.sum(left[mask_left] * np.log2(left[mask_left] / middle[mask_left]))
    divergence += 0.5 * np.sum(right[mask_right] * np.log2(right[mask_right] / middle[mask_right]))
    return float(np.sqrt(divergence))


def heterogeneity_table(model, max_tensors=None):
    table, histograms = exponent_histograms(model, max_tensors=max_tensors)
    rows = []
    for i in range(len(table)):
        for j in range(i + 1, len(table)):
            rows.append(
                {
                    "left": table.iloc[i]["name"],
                    "right": table.iloc[j]["name"],
                    "left_signature": table.iloc[i]["signature"],
                    "right_signature": table.iloc[j]["signature"],
                    "same_signature": bool(
                        table.iloc[i]["signature"] == table.iloc[j]["signature"]
                    ),
                    "js_distance": jensen_shannon(histograms[i], histograms[j]),
                }
            )
    pairs = pd.DataFrame(rows)
    if pairs.empty:
        summary = pd.DataFrame()
    else:
        summary = (
            pairs.groupby("same_signature", as_index=False)
            .agg(
                pairs=("js_distance", "size"),
                mean_js=("js_distance", "mean"),
                median_js=("js_distance", "median"),
                min_js=("js_distance", "min"),
                max_js=("js_distance", "max"),
            )
            .sort_values("same_signature", ascending=False)
        )
    return table, pairs, summary


def _iter_chunks(data, chunk_size):
    for start in range(0, len(data), int(chunk_size)):
        yield data[start : start + int(chunk_size)]


def zipnn_exponent_huffman_cost(
    model,
    *,
    layout="original",
    chunk_size=256 << 10,
):
    rows = []
    raw_bytes = 0
    compressed_bytes = 0
    chunks = 0
    for name, tensor in named_floating_tensors(model):
        stream = zipnn_exponent_stream(tensor, layout=layout)
        raw_bytes += len(stream)
        for chunk in _iter_chunks(stream, chunk_size):
            chunks += 1
            encoded = huffman.compress(chunk, min_saving_bytes=0)
            raw_cost = 1 + 8 + 8 + len(chunk)
            if encoded is None:
                compressed_bytes += raw_cost
                continue
            encoded_cost = 1 + 8 + 8 + len(encoded.code_lengths) + len(encoded.payload)
            compressed_bytes += min(raw_cost, encoded_cost)

    rows.append(
        {
            "method": "ZipNN local Huffman",
            "layout": layout,
            "raw_bytes": raw_bytes,
            "compressed_bytes": compressed_bytes,
            "compressed_pct": 100 * compressed_bytes / raw_bytes if raw_bytes else 0.0,
            "chunks": chunks,
        }
    )
    return pd.DataFrame(rows)


def global_exponent_huffman_cost(
    model,
    *,
    layout="original",
    chunk_size=256 << 10,
    compression_threshold=1.0,
    prior_count=1,
):
    tensor_streams = [
        (name, zipnn_exponent_stream(tensor, layout=layout))
        for name, tensor in named_floating_tensors(model)
    ]
    chunks = []
    raw_bytes = 0
    for name, stream in tensor_streams:
        raw_bytes += len(stream)
        for chunk_index, chunk in enumerate(_iter_chunks(stream, chunk_size)):
            if chunk:
                chunks.append((name, chunk_index, chunk))

    if not chunks:
        return pd.DataFrame(
            [
                {
                    "method": "Global Huffman",
                    "layout": layout,
                    "raw_bytes": 0,
                    "compressed_bytes": 0,
                    "compressed_pct": 0.0,
                    "payload_bytes": 0,
                    "codebook_bytes": 0,
                    "metadata_bytes": 0,
                    "chunks": 0,
                    "huffman_chunks": 0,
                    "raw_chunks": 0,
                    "representative_tensor": None,
                    "representative_chunk": None,
                    "prior_count": int(prior_count),
                }
            ]
        )

    representative_name, representative_chunk, representative = max(
        ((name, chunk_index, chunk) for name, chunk_index, chunk in chunks),
        key=lambda item: (len(item[2]), -item[1], item[0]),
    )
    codebook = huffman.build_codebook(
        representative,
        prior_count=int(prior_count),
    )
    codebook_bytes = len(codebook.code_lengths)
    chunk_header_bytes = 1 + 8 + 8
    codebook_header_bytes = 8 + len("global".encode("utf-8")) + 8 + len("exponent")

    compressed_bytes = codebook_header_bytes + codebook_bytes
    payload_bytes = 0
    huffman_chunks = 0
    raw_chunks = 0
    tensor_rows = []

    for name, chunk_index, chunk in chunks:
        raw_cost = chunk_header_bytes + len(chunk)
        max_payload_bytes = max(
            0,
            int(len(chunk) * float(compression_threshold)) - chunk_header_bytes,
        )
        encoded = huffman.encode_with_codebook(
            chunk,
            codebook,
            min_saving_bytes=0,
            max_output_bytes=max_payload_bytes,
        )
        if encoded is None:
            compressed_bytes += raw_cost
            payload_bytes += len(chunk)
            raw_chunks += 1
            tensor_rows.append(
                {
                    "name": name,
                    "chunk": int(chunk_index),
                    "mode": "raw",
                    "raw_bytes": len(chunk),
                    "compressed_bytes": raw_cost,
                }
            )
            continue

        encoded_cost = chunk_header_bytes + len(encoded.payload)
        compressed_bytes += encoded_cost
        payload_bytes += len(encoded.payload)
        huffman_chunks += 1
        tensor_rows.append(
            {
                "name": name,
                "chunk": int(chunk_index),
                "mode": "global_huffman",
                "raw_bytes": len(chunk),
                "compressed_bytes": encoded_cost,
            }
        )

    metadata_bytes = compressed_bytes - payload_bytes
    return pd.DataFrame(
        [
            {
                "method": "Global Huffman",
                "layout": layout,
                "raw_bytes": raw_bytes,
                "compressed_bytes": compressed_bytes,
                "compressed_pct": 100 * compressed_bytes / raw_bytes if raw_bytes else 0.0,
                "payload_bytes": payload_bytes,
                "codebook_bytes": codebook_bytes,
                "metadata_bytes": metadata_bytes,
                "chunks": len(chunks),
                "huffman_chunks": huffman_chunks,
                "raw_chunks": raw_chunks,
                "representative_tensor": representative_name,
                "representative_chunk": int(representative_chunk),
                "prior_count": int(prior_count),
                "tensor_results": tensor_rows,
            }
        ]
    )


def _clustered_zipnn_exponent_streams(model, *, cluster_js_threshold, layout):
    tensors = list(named_floating_tensors(model))
    codec = NeuralZipCodec(
        clustering="js-distance",
        cluster_js_threshold=float(cluster_js_threshold),
    )
    codec._prepare_clusters(tensors)
    streams_by_cluster = {}
    tensor_streams = []
    raw_bytes = 0
    for name, tensor in tensors:
        stream = zipnn_exponent_stream(tensor, layout=layout)
        cluster_id = codec.cluster_id(name)
        streams_by_cluster.setdefault(cluster_id, []).append(stream)
        tensor_streams.append((cluster_id, stream))
        raw_bytes += len(stream)
    return streams_by_cluster, tensor_streams, raw_bytes


def _ngram_counts_and_codebook(data, n):
    n = int(n)
    complete_length = (len(data) // n) * n
    if complete_length == 0:
        return 0, 0, 0.0, b"", len(data)
    values = np.frombuffer(data[:complete_length], dtype=np.dtype(f"V{n}"))
    symbols, counts = np.unique(values, return_counts=True)
    token_count = int(counts.sum())
    probabilities = counts.astype(np.float64) / token_count
    entropy = float(-(probabilities * np.log2(probabilities)).sum())
    codebook = bytearray()
    codebook.append(n)
    codebook.extend(int(len(symbols)).to_bytes(4, "little"))
    codebook.extend(symbols.tobytes())
    codebook.extend(counts.astype("<u4", copy=False).tobytes())
    return int(len(symbols)), token_count, entropy, bytes(codebook), len(data) - complete_length


def _compressed_codebook_size(codebook, compressor):
    if compressor == "raw":
        return len(codebook)
    if compressor == "zlib":
        return len(zlib.compress(codebook, 9))
    if compressor == "lzma":
        return len(lzma.compress(codebook, preset=9))
    raise ValueError("compressor must be 'raw', 'zlib', or 'lzma'.")


def _aligned_ngram_training_data(streams, n):
    n = int(n)
    parts = []
    tail_bytes = 0
    for stream in streams:
        complete_length = (len(stream) // n) * n
        if complete_length:
            parts.append(stream[:complete_length])
        tail_bytes += len(stream) - complete_length
    return b"".join(parts), tail_bytes


def ngram_codebook_boundary(
    model,
    *,
    cluster_js_threshold=0.2,
    n_values=(4, 5, 6, 8),
    layout="transpose2d",
    codebook_compressors=("raw", "zlib", "lzma"),
):
    streams_by_cluster, _, raw_bytes = _clustered_zipnn_exponent_streams(
        model,
        cluster_js_threshold=cluster_js_threshold,
        layout=layout,
    )
    rows = []
    for n in n_values:
        cluster_stats = []
        payload_bits = 0.0
        tail_bytes = 0
        total_unique = 0
        codebooks = []
        for cluster_id, streams in streams_by_cluster.items():
            data, stream_tail = _aligned_ngram_training_data(streams, n)
            if not data:
                tail_bytes += stream_tail
                continue
            (
                unique,
                token_count,
                entropy,
                codebook,
                aligned_tail,
            ) = _ngram_counts_and_codebook(data, n)
            payload_bits += token_count * entropy
            tail_bytes += stream_tail + aligned_tail
            total_unique += unique
            codebooks.append(codebook)
            cluster_stats.append((cluster_id, unique, token_count))

        chunk_headers = len(streams_by_cluster) * (1 + 8 + 8 + 8 + 8 + 8)
        payload_bytes = math.ceil(payload_bits / 8) + tail_bytes + chunk_headers
        for compressor in codebook_compressors:
            codebook_bytes = sum(
                _compressed_codebook_size(codebook, compressor)
                for codebook in codebooks
            )
            compressed_bytes = payload_bytes + codebook_bytes
            rows.append(
                {
                    "method": "clustered n-gram arithmetic",
                    "cluster_js_threshold": float(cluster_js_threshold),
                    "layout": layout,
                    "n": int(n),
                    "codebook_compressor": compressor,
                    "clusters": len(streams_by_cluster),
                    "unique_ngrams": total_unique,
                    "raw_bytes": raw_bytes,
                    "payload_bytes": payload_bytes,
                    "codebook_bytes": codebook_bytes,
                    "compressed_bytes": compressed_bytes,
                    "payload_pct": 100 * payload_bytes / raw_bytes,
                    "codebook_pct": 100 * codebook_bytes / raw_bytes,
                    "compressed_pct": 100 * compressed_bytes / raw_bytes,
                }
            )
    return pd.DataFrame(rows)


def selected_ngram_codebook_cost(
    model,
    *,
    cluster_js_thresholds=(0.2, 0.1, 0.05, 0.03, 0.02),
    n_values=(2, 3, 4),
    layout="transpose2d",
):
    rows = []
    for threshold in cluster_js_thresholds:
        streams_by_cluster, _, raw_bytes = _clustered_zipnn_exponent_streams(
            model,
            cluster_js_threshold=threshold,
            layout=layout,
        )
        total_bytes = 0
        codebook_bytes = 0
        selected = {}
        encoded_tensors = 0
        for cluster_id, streams in streams_by_cluster.items():
            candidates = [
                ("zipnn-local", _zipnn_local_stream_cost(streams), 0, 0)
            ]
            for n in n_values:
                data, _ = _aligned_ngram_training_data(streams, n)
                if not data:
                    continue
                try:
                    codebook = huffman.build_sparse_ngram_arithmetic_codebook(
                        data,
                        n=int(n),
                        max_total=1 << 26,
                    )
                except ValueError:
                    continue
                payload_cost = 0
                encoded_count = 0
                for stream in streams:
                    encoded = huffman.sparse_ngram_arithmetic_encode_with_codebook(
                        stream,
                        codebook,
                        min_saving_bytes=0,
                    )
                    if encoded is None:
                        payload_cost += 1 + 8 + 8 + len(stream)
                        continue
                    payload_cost += (
                        1
                        + 8
                        + 8
                        + 8
                        + 8
                        + 8
                        + len(encoded.payload)
                        + len(encoded.tail)
                    )
                    encoded_count += 1
                header = (
                    8
                    + len(str(cluster_id).encode("utf-8"))
                    + 8
                    + len("bfloat16")
                    + 1
                    + 1
                    + 8
                )
                candidates.append(
                    (
                        f"n={int(n)}",
                        payload_cost + len(codebook.payload) + header,
                        len(codebook.payload),
                        encoded_count,
                    )
                )
            mode, cost, mode_codebook_bytes, mode_encoded = min(
                candidates,
                key=lambda candidate: candidate[1],
            )
            total_bytes += cost
            codebook_bytes += mode_codebook_bytes
            encoded_tensors += mode_encoded
            selected[mode] = selected.get(mode, 0) + 1

        rows.append(
            {
                "cluster_js_threshold": float(threshold),
                "layout": layout,
                "clusters": len(streams_by_cluster),
                "raw_bytes": raw_bytes,
                "compressed_bytes": total_bytes,
                "compressed_pct": 100 * total_bytes / raw_bytes,
                "codebook_bytes": codebook_bytes,
                "encoded_tensors": encoded_tensors,
                "selected_modes": selected,
            }
        )
    return pd.DataFrame(rows)


def _zipnn_local_stream_cost(streams):
    total = 0
    for stream in streams:
        encoded = huffman.compress(stream, min_saving_bytes=0)
        raw_cost = 1 + 8 + 8 + len(stream)
        if encoded is None:
            total += raw_cost
            continue
        encoded_cost = 1 + 8 + 8 + len(encoded.code_lengths) + len(encoded.payload)
        total += min(raw_cost, encoded_cost)
    return total
