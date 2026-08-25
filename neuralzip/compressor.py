import time
from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch

from .archive import (
    VERSION as ARCHIVE_VERSION,
    serialized_codebooks_size,
    serialized_group_size,
    serialized_size,
    serialized_tensor_size,
)
from .codec_protocol import encode_tensor_batch, prepare_neuralzip_codec
from .dtypes import dtype_name
from .neuralzip_codec import NeuralZipCodec
from .tensors import bitwise_equal
from .zipnn_codec import ZipNNCodec
from .zipnn_source import ZipNNSourceCodec

if TYPE_CHECKING:
    from .model import HFModel


def _normalize_method(method):
    method_key = str(method).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "zipnn": "zipnn_local",
        "zipnn_local": "zipnn_local",
        "zipnn_original": "zipnn_src",
        "zipnn_src": "zipnn_src",
        "zipnn_source": "zipnn_src",
        "official_zipnn": "zipnn_src",
        "neuralzip": "neuralzip",
        "neural_zip": "neuralzip",
    }
    if method_key not in aliases:
        choices = ", ".join(sorted(set(aliases.values())))
        raise ValueError(
            f"Unsupported compression method {method!r}. Expected one of: {choices}."
        )
    return aliases[method_key]


def _model_parameters(model):
    module = model.model if hasattr(model, "model") else model
    if isinstance(module, Mapping):
        return _named_tensors(module)
    if not hasattr(module, "named_parameters"):
        raise TypeError("Expected an HFModel or a torch module with named_parameters().")
    return module.named_parameters()


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


def _stream_summaries(tensor_payloads):
    streams = {}
    for compressed in tensor_payloads:
        for stream_id, group in enumerate(compressed.groups):
            raw_bytes = int(group.symbol_count)
            entry = streams.setdefault(
                int(stream_id),
                {
                    "stream_id": int(stream_id),
                    "raw_bytes": 0,
                    "payload_bytes": 0,
                    "metadata_bytes": 0,
                    "compressed_bytes": 0,
                    "local_huff0_chunks": 0,
                    "shared_huff0_chunks": 0,
                    "packed_ngram_chunks": 0,
                    "raw_chunks": 0,
                    "zero_chunks": 0,
                    "ngram_hits": 0,
                    "ngram_escapes": 0,
                },
            )
            serialized_group_bytes = serialized_group_size(group)
            entry["raw_bytes"] += raw_bytes
            entry["payload_bytes"] += group.payload_bytes
            entry["metadata_bytes"] += serialized_group_bytes - group.payload_bytes
            entry["compressed_bytes"] += serialized_group_bytes
            entry["local_huff0_chunks"] += group.local_huff0_chunks
            entry["shared_huff0_chunks"] += group.shared_huff0_chunks
            entry["packed_ngram_chunks"] += group.packed_ngram_chunks
            entry["raw_chunks"] += group.raw_chunks
            entry["zero_chunks"] += group.zero_chunks
            entry["ngram_hits"] += group.ngram_hits
            entry["ngram_escapes"] += group.ngram_escapes

    for entry in streams.values():
        raw_bytes = entry["raw_bytes"]
        entry["compression_ratio"] = (
            entry["compressed_bytes"] / raw_bytes if raw_bytes else 0.0
        )

    return tuple(streams[index] for index in sorted(streams))


class Compressor:
    def __init__(
        self,
        method="neuralzip",
        *,
        verify=True,
        keep_payloads=False,
        min_saving_bytes=0,
        chunk_size=256 << 10,
        compression_threshold=None,
        threads=0,
        zipnn_method="HUFFMAN",
        cluster_map=None,
        clustering="js-distance",
        cluster_js_threshold=0.05,
        max_codebook_sample_bytes=8 << 20,
        ngram_values=(2, 3, 4),
        top_k=254,
        exponent_layout="original",
        packed_source_budget_ratio=0.30,
        packed_time_budget_ratio=1.0,
        speed_ratio_tolerance=0.03,
        enable_shared_codebooks=True,
        direct_batch_bytes=128 << 20,
    ):
        self.method = _normalize_method(method)
        self.verify = bool(verify)
        self.keep_payloads = bool(keep_payloads)
        self.zipnn_method = str(zipnn_method).upper()
        self.threads = int(threads)
        self.cluster_js_threshold = float(cluster_js_threshold)
        if compression_threshold is None:
            compression_threshold = 0.95
        codec_cls = NeuralZipCodec if self.method == "neuralzip" else ZipNNCodec
        codec_kwargs = {
            "min_saving_bytes": min_saving_bytes,
            "chunk_size": chunk_size,
            "compression_threshold": compression_threshold,
            "threads": threads,
        }
        if self.method == "neuralzip":
            codec_kwargs["cluster_map"] = cluster_map
            codec_kwargs["clustering"] = clustering
            codec_kwargs["cluster_js_threshold"] = cluster_js_threshold
            codec_kwargs["max_codebook_sample_bytes"] = max_codebook_sample_bytes
            codec_kwargs["ngram_values"] = ngram_values
            codec_kwargs["top_k"] = top_k
            codec_kwargs["exponent_layout"] = exponent_layout
            codec_kwargs["packed_source_budget_ratio"] = (
                packed_source_budget_ratio
            )
            codec_kwargs["packed_time_budget_ratio"] = packed_time_budget_ratio
            codec_kwargs["speed_ratio_tolerance"] = speed_ratio_tolerance
            codec_kwargs["enable_shared_codebooks"] = enable_shared_codebooks
            codec_kwargs["direct_batch_bytes"] = direct_batch_bytes
        self.codec = codec_cls(**codec_kwargs)

    def compress(self, model: "HFModel"):
        if self.method == "zipnn_local":
            return self._codec_compression(
                model,
                method_name="zipnn-local",
                method_detail="dtype-byte-reorder-huffman",
            )
        if self.method == "neuralzip":
            return self._codec_compression(
                model,
                method_name="neuralzip",
                method_detail=(
                    "js-clustered-shared-local-huffman-packed-exponents"
                ),
            )
        if self.method == "zipnn_src":
            return self._zipnn_src(model)
        raise AssertionError(f"Unhandled compression method {self.method!r}.")

    def decompress(self, compressed_model):
        if hasattr(self.codec, "reset_decompression"):
            self.codec.reset_decompression()
        if hasattr(self.codec, "load_codebooks"):
            header = compressed_model.get("header") or {}
            codebooks = compressed_model.get("codebooks") or header.get("codebooks")
            self.codec.load_codebooks(codebooks)

        tensors = {}
        for tensor_result in compressed_model.get("tensor_results", []):
            payload = tensor_result.get("payload")
            if payload is None:
                raise ValueError(
                    "Cannot decompress this result because tensor payloads were not "
                    "kept. Run Compressor(..., keep_payloads=True)."
                )
            tensors[payload.name] = self.codec.decompress_tensor(payload)
        return tensors

    @staticmethod
    def save(compressed_model, path):
        from .archive import dump

        dump(compressed_model, path)

    @staticmethod
    def load(path):
        from .archive import load

        return load(path)

    def decompress_file(self, path):
        return self.decompress(self.load(path))

    def _codec_compression(self, model: "HFModel", *, method_name, method_detail):
        tensor_results = []
        total_raw_bytes = 0
        total_payload_bytes = 0
        total_encode_seconds = 0.0
        total_decode_seconds = 0.0
        total_huffman_chunks = 0
        total_raw_chunks = 0
        total_zero_chunks = 0
        total_shared_codebook_refs = 0
        total_local_huff0_chunks = 0
        total_packed_ngram_chunks = 0
        total_ngram_hits = 0
        total_ngram_escapes = 0
        clustering_seconds = 0.0
        codebook_setup_seconds = 0.0
        codebook_metadata_bytes = 0
        codebooks = ()
        compressed_payloads = []
        roundtrip_ok = True if self.verify else None

        parameters = [
            (name, parameter)
            for name, parameter in _model_parameters(model)
            if torch.is_tensor(parameter) and parameter.is_floating_point()
        ]

        if hasattr(self.codec, "reset_compression"):
            self.codec.reset_compression()
        if hasattr(self.codec, "reset_decompression"):
            self.codec.reset_decompression()
        if (
            hasattr(self.codec, "prepare_clusters")
            and hasattr(self.codec, "build_codebooks_and_select_modes")
        ):
            setup_report = prepare_neuralzip_codec(
                self.codec,
                parameters,
            )
            clustering_seconds = setup_report.clustering_seconds
            codebook_setup_seconds = setup_report.codebook_setup_seconds
            codebooks = self.codec.export_codebooks()
            codebook_metadata_bytes = serialized_codebooks_size(codebooks)
        setup_seconds = clustering_seconds + codebook_setup_seconds

        encoded_batch = encode_tensor_batch(self.codec, parameters)
        compressed_payloads = list(encoded_batch.payloads)
        total_encode_seconds = encoded_batch.encode_seconds

        for tensor_index, (name, parameter) in enumerate(parameters):
            raw_bytes = _raw_tensor_bytes(parameter)
            total_raw_bytes += raw_bytes

            compressed = encoded_batch.payloads[tensor_index]
            encode_seconds = encoded_batch.tensor_encode_seconds[tensor_index]

            compressed_bytes = serialized_tensor_size(compressed)
            total_payload_bytes += compressed.payload_bytes
            total_huffman_chunks += compressed.huffman_chunks
            total_raw_chunks += compressed.raw_chunks
            total_zero_chunks += compressed.zero_chunks
            total_shared_codebook_refs += getattr(compressed, "shared_codebook_refs", 0)
            total_local_huff0_chunks += getattr(
                compressed, "local_huff0_chunks", 0
            )
            total_packed_ngram_chunks += getattr(
                compressed, "packed_ngram_chunks", 0
            )
            total_ngram_hits += getattr(compressed, "ngram_hits", 0)
            total_ngram_escapes += getattr(compressed, "ngram_escapes", 0)
            tensor_roundtrip_ok = None
            decode_seconds = None
            if self.verify:
                start = time.perf_counter()
                decompressed = self.codec.decompress_tensor(compressed)
                decode_seconds = time.perf_counter() - start
                original = parameter.detach().cpu().contiguous()
                tensor_roundtrip_ok = bitwise_equal(original, decompressed)
                roundtrip_ok = bool(roundtrip_ok and tensor_roundtrip_ok)
                total_decode_seconds += decode_seconds

            result = {
                "name": name,
                "shape": compressed.shape,
                "dtype": compressed.dtype,
                "raw_bytes": raw_bytes,
                "strategy": compressed.strategy,
                "payload_bytes": compressed.payload_bytes,
                "metadata_bytes": compressed_bytes - compressed.payload_bytes,
                "compressed_bytes": compressed_bytes,
                "compression_ratio": compressed_bytes / raw_bytes if raw_bytes else 0.0,
                "huffman_chunks": compressed.huffman_chunks,
                "raw_chunks": compressed.raw_chunks,
                "zero_chunks": compressed.zero_chunks,
                "shared_codebook_refs": getattr(compressed, "shared_codebook_refs", 0),
                "local_huff0_chunks": getattr(
                    compressed, "local_huff0_chunks", 0
                ),
                "packed_ngram_chunks": getattr(
                    compressed, "packed_ngram_chunks", 0
                ),
                "ngram_hits": getattr(compressed, "ngram_hits", 0),
                "ngram_escapes": getattr(compressed, "ngram_escapes", 0),
                "encode_seconds": encode_seconds,
            }
            if self.verify:
                result["decode_seconds"] = decode_seconds
                result["roundtrip_ok"] = tensor_roundtrip_ok
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
        stream_results = _stream_summaries(compressed_payloads)
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
            "shared_codebook_refs": total_shared_codebook_refs,
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
            "selection_stats": getattr(self.codec, "selection_stats", {}),
            "selection_diagnostics": getattr(
                self.codec,
                "selection_diagnostics",
                (),
            ),
            "clustering_seconds": clustering_seconds,
            "codebook_setup_seconds": codebook_setup_seconds,
            "setup_seconds": setup_seconds,
            "setup_memory_bytes": getattr(
                self.codec,
                "setup_memory_bytes",
                0,
            ),
            "encode_seconds": total_encode_seconds,
            "total_seconds": setup_seconds + total_encode_seconds,
            "decode_seconds": total_decode_seconds if self.verify else None,
            "roundtrip_ok": roundtrip_ok,
            "header": {
                "archive_version": ARCHIVE_VERSION,
                "codebooks": codebooks,
                "chunk_size": getattr(self.codec, "chunk_size", None),
                "clustering": getattr(self.codec, "clustering", None),
                "cluster_js_threshold": getattr(
                    self.codec, "cluster_js_threshold", None
                ),
                "packed_time_budget_ratio": getattr(
                    self.codec, "packed_time_budget_ratio", None
                ),
                "speed_ratio_tolerance": getattr(
                    self.codec, "speed_ratio_tolerance", None
                ),
                "clusters": getattr(self.codec, "cluster_count", 0),
                "selected_modes": getattr(self.codec, "selected_modes", {}),
                "selection_stats": getattr(self.codec, "selection_stats", {}),
            },
            "codebooks": codebooks,
            "stream_results": stream_results,
            "tensor_results": tensor_results,
        }

    def _zipnn_src(self, model: "HFModel"):
        tensor_results = []
        total_raw_bytes = 0
        total_compressed_bytes = 0
        total_encode_seconds = 0.0
        total_codec_encode_seconds = 0.0
        total_input_copy_seconds = 0.0
        total_decode_seconds = 0.0
        roundtrip_ok = True if self.verify else None
        codec = ZipNNSourceCodec(
            method=self.zipnn_method,
            threads=self.threads,
            compression_threshold=self.codec.compression_threshold,
            chunk_size=self.codec.chunk_size,
        )

        for name, parameter in _model_parameters(model):
            if not torch.is_tensor(parameter) or not parameter.is_floating_point():
                continue

            original = parameter.detach().cpu().contiguous()
            raw_bytes = _raw_tensor_bytes(original)
            total_raw_bytes += raw_bytes

            encoded = codec.encode_tensor(original, verify=self.verify)
            payload = encoded.payload
            input_copy_seconds = encoded.input_copy_seconds
            codec_encode_seconds = encoded.codec_encode_seconds
            encode_seconds = encoded.encode_seconds

            compressed_bytes = len(payload)
            total_compressed_bytes += compressed_bytes
            total_encode_seconds += encode_seconds
            total_codec_encode_seconds += codec_encode_seconds
            total_input_copy_seconds += input_copy_seconds

            tensor_roundtrip_ok = encoded.roundtrip_ok
            decode_seconds = encoded.decode_seconds
            if self.verify:
                roundtrip_ok = bool(roundtrip_ok and tensor_roundtrip_ok)
                total_decode_seconds += decode_seconds

            result = {
                "name": name,
                "shape": tuple(int(dim) for dim in original.shape),
                "dtype": dtype_name(original.dtype),
                "raw_bytes": raw_bytes,
                "strategy": "zipnn-src",
                "payload_bytes": compressed_bytes,
                "metadata_bytes": 0,
                "compressed_bytes": compressed_bytes,
                "compression_ratio": compressed_bytes / raw_bytes if raw_bytes else 0.0,
                "encode_seconds": encode_seconds,
                "codec_encode_seconds": codec_encode_seconds,
                "input_copy_seconds": input_copy_seconds,
            }
            if self.verify:
                result["decode_seconds"] = decode_seconds
                result["roundtrip_ok"] = tensor_roundtrip_ok
            if self.keep_payloads:
                result["payload"] = payload
            tensor_results.append(result)

        model_compression_ratio = (
            total_compressed_bytes / total_raw_bytes if total_raw_bytes else 0.0
        )
        return {
            "method": "zipnn-src",
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
