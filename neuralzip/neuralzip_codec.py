import re

import numpy as np
import torch

from . import huffman
from .utils import split_fields
from .zipnn_codec import (
    DTYPE_BY_NAME,
    ByteGroupPayload,
    CodebookPayload,
    StreamChunk,
    TensorPayload,
    ZipNNCodec,
    _dtype_name,
    _tensor_from_raw_bytes,
    _zipnn_dtype_config,
    _zipnn_reordered_bytes,
    _zipnn_restored_bytes,
)


def _layer_signature(name):
    parts = str(name).split(".")
    return ".".join(re.sub(r"\d+", "#", part) for part in parts)


def _jensen_shannon(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    middle = 0.5 * (left + right)
    left_mask = left > 0
    right_mask = right > 0
    divergence = 0.0
    divergence += 0.5 * np.sum(
        left[left_mask] * np.log2(left[left_mask] / middle[left_mask])
    )
    divergence += 0.5 * np.sum(
        right[right_mask] * np.log2(right[right_mask] / middle[right_mask])
    )
    return float(np.sqrt(divergence))


def _exponent_distribution(tensor):
    _, exponent, _ = split_fields(tensor.detach().cpu().contiguous())
    stream = exponent.reshape(-1).to(torch.int64).numpy()
    counts = np.bincount(stream, minlength=256).astype(np.float64)
    total = counts.sum()
    if total == 0:
        return counts
    return counts / total


class NeuralZipCodec(ZipNNCodec):
    """ZipNN byte grouping with JS-clustered shared Huffman codebooks."""

    def __init__(
        self,
        *,
        min_saving_bytes=0,
        chunk_size=256 << 10,
        compression_threshold=0.95,
        prior_count=0,
        exponent_prior_count=0,
        cluster_map=None,
        clustering="js-distance",
        cluster_js_threshold=0.05,
        max_codebook_sample_bytes=8 << 20,
        test_cluster_compressibility=True,
    ):
        super().__init__(
            min_saving_bytes=min_saving_bytes,
            chunk_size=chunk_size,
            compression_threshold=compression_threshold,
        )
        self.prior_count = int(prior_count)
        self.exponent_prior_count = int(exponent_prior_count)
        self.cluster_map = dict(cluster_map or {})
        self.clustering = str(clustering)
        self.cluster_js_threshold = float(cluster_js_threshold)
        self.max_codebook_sample_bytes = int(max_codebook_sample_bytes)
        self.test_cluster_compressibility = bool(test_cluster_compressibility)
        if self.prior_count < 0:
            raise ValueError("prior_count must be non-negative.")
        if self.exponent_prior_count < 0:
            raise ValueError("exponent_prior_count must be non-negative.")
        if self.cluster_js_threshold < 0:
            raise ValueError("cluster_js_threshold must be non-negative.")
        if self.max_codebook_sample_bytes <= 0:
            raise ValueError("max_codebook_sample_bytes must be positive.")
        self.reset_compression()
        self.reset_decompression()

    def reset_compression(self):
        self._codebook_by_key = {}
        self._codebook_payloads = []
        self._codebook_payload_by_id = {}
        self._raw_codebook_keys = set()
        self._computed_cluster_map = {}
        self._next_codebook_id = 0

    def reset_decompression(self):
        self._decode_codebooks = {}
        self._codebook_payload_by_id = {}

    def export_codebooks(self):
        return tuple(self._codebook_payloads)

    def load_codebooks(self, codebooks):
        self._codebook_payload_by_id = {
            int(codebook.codebook_id): codebook for codebook in codebooks or ()
        }
        self._decode_codebooks = {
            int(codebook.codebook_id): huffman.build_decodebook(codebook.code_lengths)
            for codebook in codebooks or ()
        }

    def codebook_metadata_bytes(self):
        return sum(codebook.metadata_bytes for codebook in self._codebook_payloads)

    def _append_codebook_payload(
        self,
        *,
        codebook_id,
        cluster_id,
        stream_id,
        dtype,
        code_lengths,
    ):
        payload = CodebookPayload(
            codebook_id=codebook_id,
            cluster_id=cluster_id,
            stream_id=stream_id,
            dtype=dtype,
            code_lengths=code_lengths,
        )
        self._codebook_payloads.append(payload)
        self._codebook_payload_by_id[int(codebook_id)] = payload
        return payload

    @property
    def raw_codebook_keys(self):
        return len(self._raw_codebook_keys)

    @property
    def cluster_count(self):
        return len(set(self._computed_cluster_map.values()) | set(self.cluster_map.values()))

    def cluster_id(self, name):
        if name in self.cluster_map:
            return str(self.cluster_map[name])
        if name in self._computed_cluster_map:
            return str(self._computed_cluster_map[name])
        return str(_layer_signature(name))

    def _prepare_clusters(self, named_tensors):
        clustering = self.clustering.lower().replace("_", "-")
        if clustering in {"architecture-aware", "architecture", "layer-signature"}:
            self._computed_cluster_map = {
                name: _layer_signature(name)
                for name, tensor in named_tensors
                if name not in self.cluster_map
                and torch.is_tensor(tensor)
                and tensor.is_floating_point()
            }
            return

        if clustering not in {
            "js-distance",
            "jensen-shannon",
            "jensen-shannon-distance",
        }:
            raise ValueError(
                "Unsupported clustering mode "
                f"{self.clustering!r}. Expected 'js-distance' or "
                "'architecture-aware'."
            )

        clusters = []
        cluster_map = {}
        for name, tensor in named_tensors:
            if name in self.cluster_map:
                continue
            if not torch.is_tensor(tensor) or not tensor.is_floating_point():
                continue

            distribution = _exponent_distribution(tensor)
            weight = int(tensor.numel())
            best_index = None
            best_distance = float("inf")
            for index, cluster in enumerate(clusters):
                distance = _jensen_shannon(distribution, cluster["centroid"])
                if distance < best_distance:
                    best_index = index
                    best_distance = distance

            if best_index is None or best_distance > self.cluster_js_threshold:
                cluster_id = f"js_{len(clusters):03d}"
                clusters.append(
                    {
                        "cluster_id": cluster_id,
                        "centroid": distribution,
                        "weight": weight,
                    }
                )
                cluster_map[name] = cluster_id
                continue

            cluster = clusters[best_index]
            total_weight = cluster["weight"] + weight
            cluster["centroid"] = (
                cluster["centroid"] * cluster["weight"] + distribution * weight
            ) / total_weight
            cluster["weight"] = total_weight
            cluster_map[name] = cluster["cluster_id"]

        self._computed_cluster_map = cluster_map

    def precompute_codebooks(self, named_tensors):
        self._prepare_clusters(named_tensors)
        samples_by_key = {}
        sample_sizes = {}
        prior_by_key = {}

        for name, tensor in named_tensors:
            if not torch.is_tensor(tensor) or not tensor.is_floating_point():
                continue

            original = tensor.detach().cpu().contiguous()
            config = _zipnn_dtype_config(original.dtype)
            reordered_bytes = _zipnn_reordered_bytes(original, config)
            element_size = int(config["element_size"])

            for byte_offset in range(int(config["num_buffers"])):
                prior_count = (
                    self.exponent_prior_count
                    if int(byte_offset) == element_size - 1
                    else self.prior_count
                )
                key = self._codebook_key(name, original, byte_offset)
                prior_by_key[key] = prior_count
                current_size = sample_sizes.get(key, 0)
                remaining = self.max_codebook_sample_bytes - current_size
                if remaining <= 0:
                    continue

                sample = reordered_bytes[
                    byte_offset : byte_offset + remaining * element_size : element_size
                ]
                if not sample:
                    continue
                samples_by_key.setdefault(key, bytearray()).extend(sample)
                sample_sizes[key] = current_size + len(sample)

        for key, sample in samples_by_key.items():
            if key in self._codebook_by_key:
                continue
            codebook = huffman.build_codebook(
                bytes(sample),
                prior_count=prior_by_key[key],
            )
            codebook_id = self._next_codebook_id
            self._codebook_by_key[key] = (codebook_id, codebook)
            cluster_id, dtype, stream_id = key
            self._append_codebook_payload(
                codebook_id=codebook_id,
                cluster_id=cluster_id,
                stream_id=stream_id,
                dtype=dtype,
                code_lengths=codebook.code_lengths,
            )
            self._next_codebook_id += 1
            if self.test_cluster_compressibility:
                header_size = 1 + 8 + 8 + 8 + 1
                max_payload_bytes = max(
                    0,
                    int(len(sample) * self.compression_threshold) - header_size,
                )
                encoded_sample = huffman.encode_with_codebook(
                    bytes(sample),
                    codebook,
                    min_saving_bytes=self.min_saving_bytes,
                    max_output_bytes=max_payload_bytes,
                )
                if encoded_sample is None:
                    self._raw_codebook_keys.add(key)
        self.load_codebooks(self._codebook_payloads)

    def compress_tensor(self, name, tensor):
        if not torch.is_tensor(tensor) or not tensor.is_floating_point():
            raise TypeError("NeuralZipCodec only compresses floating tensors.")

        original = tensor.detach().cpu().contiguous()
        config = _zipnn_dtype_config(original.dtype)
        return self._compress_neuralzip(name, original, config)

    def decompress_tensor(self, compressed):
        if compressed.dtype not in DTYPE_BY_NAME:
            raise ValueError(f"Unsupported dtype in payload: {compressed.dtype!r}")

        if compressed.strategy in {"neuralzip", "byte_groups"}:
            return self._decompress_neuralzip(compressed)
        if compressed.strategy == "delta":
            raise ValueError("Delta payloads require decompress_delta_tensor().")
        raise ValueError(f"Unsupported NeuralZip strategy: {compressed.strategy!r}")

    def _codebook_key(self, name, tensor, byte_offset):
        return (
            self.cluster_id(name),
            _dtype_name(tensor.dtype),
            int(byte_offset),
        )

    def _ensure_codebook(self, key, chunk, prior_count):
        codebook = self._codebook_by_key.get(key)
        if codebook is None:
            codebook = huffman.build_codebook(
                chunk,
                prior_count=prior_count,
            )
            codebook_id = self._next_codebook_id
            self._codebook_by_key[key] = (codebook_id, codebook)
            cluster_id, dtype, stream_id = key
            self._append_codebook_payload(
                codebook_id=codebook_id,
                cluster_id=cluster_id,
                stream_id=stream_id,
                dtype=dtype,
                code_lengths=codebook.code_lengths,
            )
            self._decode_codebooks[codebook_id] = huffman.build_decodebook(
                codebook.code_lengths
            )
            self._next_codebook_id += 1
        return self._codebook_by_key[key]

    def _encode_shared_huffman_chunk(self, key, chunk, prior_count):
        codebook_id, codebook = self._ensure_codebook(key, chunk, prior_count)
        header_size = 1 + 8 + 8 + 8 + 1
        max_payload_bytes = max(
            0,
            int(len(chunk) * self.compression_threshold) - header_size,
        )
        try:
            encoded = huffman.encode_with_codebook(
                chunk,
                codebook,
                min_saving_bytes=self.min_saving_bytes,
                max_output_bytes=max_payload_bytes,
            )
        except ValueError as error:
            if "missing" not in str(error):
                raise
            encoded = None
        if encoded is None:
            return None
        candidate = StreamChunk(
            mode="shared_huffman",
            symbol_count=encoded.symbol_count,
            payload=encoded.payload,
            code_lengths=b"",
            codebook_id=codebook_id,
        )
        if candidate.compressed_bytes >= len(chunk) * self.compression_threshold:
            return None
        return candidate

    def _compress_shared_stream(self, name, tensor, stream, stream_kind, byte_offset):
        chunk_size = self._stream_chunk_size(tensor.element_size())
        is_exponent_stream = int(byte_offset) == int(tensor.element_size()) - 1
        prior_count = (
            self.exponent_prior_count
            if is_exponent_stream
            else self.prior_count
        )
        key = self._codebook_key(name, tensor, byte_offset)
        chunks = []

        for start in range(0, len(stream), chunk_size):
            chunk = stream[start : start + chunk_size]
            if not chunk:
                continue

            if key not in self._raw_codebook_keys:
                shared = self._encode_shared_huffman_chunk(key, chunk, prior_count)
                if shared is not None:
                    chunks.append(shared)
                    continue

            chunks.append(
                StreamChunk(
                    mode="raw",
                    symbol_count=len(chunk),
                    payload=chunk,
                )
            )

        return ByteGroupPayload(
            stream_kind=stream_kind,
            symbol_count=len(stream),
            chunks=tuple(chunks),
        )

    def _decompress_shared_stream(self, group):
        output = bytearray()
        for chunk in group.chunks:
            if chunk.mode == "raw":
                output.extend(chunk.payload)
            elif chunk.mode == "zero":
                output.extend(b"\x00" * chunk.symbol_count)
            elif chunk.mode == "huffman":
                output.extend(
                    huffman.decompress(
                        huffman.HuffmanPayload(
                            payload=chunk.payload,
                            code_lengths=chunk.code_lengths,
                            symbol_count=chunk.symbol_count,
                        )
                    )
                )
            elif chunk.mode == "shared_huffman":
                if chunk.codebook_id is None:
                    raise ValueError("Shared Huffman chunk is missing a codebook id.")
                if chunk.code_lengths:
                    self._decode_codebooks[chunk.codebook_id] = huffman.build_decodebook(
                        chunk.code_lengths
                    )
                decodebook = self._decode_codebooks.get(chunk.codebook_id)
                if decodebook is None:
                    raise ValueError(
                        f"Missing shared Huffman codebook {chunk.codebook_id}."
                    )
                output.extend(
                    huffman.decompress_with_decodebook(
                        huffman.HuffmanPayload(
                            payload=chunk.payload,
                            code_lengths=decodebook.code_lengths,
                            symbol_count=chunk.symbol_count,
                        ),
                        decodebook,
                    )
                )
            else:
                raise ValueError(f"Unsupported stream chunk mode: {chunk.mode!r}")

        if len(output) != group.symbol_count:
            raise ValueError(
                f"Stream decoded to {len(output)} symbols; "
                f"expected {group.symbol_count}."
            )
        return bytes(output)

    def _compress_neuralzip(self, name, tensor, config):
        reordered_bytes = _zipnn_reordered_bytes(tensor, config)
        element_size = int(config["element_size"])
        groups = []

        for byte_offset in range(int(config["num_buffers"])):
            stream = reordered_bytes[byte_offset::element_size]
            groups.append(
                self._compress_shared_stream(
                    name,
                    tensor,
                    stream,
                    "byte_group",
                    byte_offset,
                )
            )

        return TensorPayload(
            name=name,
            shape=tuple(int(dim) for dim in tensor.shape),
            dtype=_dtype_name(tensor.dtype),
            element_size=element_size,
            numel=int(tensor.numel()),
            strategy="neuralzip",
            groups=tuple(groups),
            byte_reorder=int(config["byte_reorder"]),
            bit_reorder=int(config["bit_reorder"]),
            num_buffers=int(config["num_buffers"]),
            cluster_id=self.cluster_id(name),
        )

    def _decompress_neuralzip(self, compressed):
        dtype = DTYPE_BY_NAME[compressed.dtype]
        config = _zipnn_dtype_config(dtype)
        if compressed.element_size != int(config["element_size"]):
            raise ValueError(
                f"Payload element size {compressed.element_size} does not match "
                f"dtype {compressed.dtype}."
            )
        if len(compressed.groups) != int(config["num_buffers"]):
            raise ValueError(
                f"Payload has {len(compressed.groups)} byte groups; "
                f"expected {config['num_buffers']}."
            )

        raw = bytearray(compressed.raw_bytes)
        for byte_offset, group in enumerate(compressed.groups):
            stream = self._decompress_shared_stream(group)
            expected_symbols = len(raw[byte_offset::compressed.element_size])
            if len(stream) != expected_symbols:
                raise ValueError(
                    f"Byte group {byte_offset} has {len(stream)} symbols; "
                    f"expected {expected_symbols}."
                )
            raw[byte_offset::compressed.element_size] = stream

        restored = _zipnn_restored_bytes(bytes(raw), dtype, config)
        return _tensor_from_raw_bytes(restored, dtype, compressed.shape)
