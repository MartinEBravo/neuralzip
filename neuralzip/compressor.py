import time
from collections.abc import Mapping

import torch

from .codec_protocol import (
    decode_tensor_batch,
    encode_tensor_batch,
    prepare_neuralzip_codec,
)
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


class Compressor:
    """Compress floating parameters from modules or tensor leaves from mappings."""

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
        cluster_js_threshold=0.05,
        max_codebook_sample_bytes=8 << 20,
        ngram_values=(2, 3, 4),
        top_k=254,
        exponent_layout=None,
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
        if compression_threshold is None:
            compression_threshold = 0.95
        codec_kwargs = {
            "min_saving_bytes": min_saving_bytes,
            "chunk_size": chunk_size,
            "compression_threshold": compression_threshold,
            "threads": threads,
        }
        if self.method == "neuralzip":
            codec_kwargs["cluster_map"] = cluster_map
            codec_kwargs["cluster_js_threshold"] = cluster_js_threshold
            codec_kwargs["max_codebook_sample_bytes"] = max_codebook_sample_bytes
            codec_kwargs["ngram_values"] = ngram_values
            codec_kwargs["top_k"] = top_k
            codec_kwargs["exponent_layout"] = exponent_layout
            codec_kwargs["packed_source_budget_ratio"] = packed_source_budget_ratio
            codec_kwargs["packed_time_budget_ratio"] = packed_time_budget_ratio
            codec_kwargs["speed_ratio_tolerance"] = speed_ratio_tolerance
            codec_kwargs["enable_shared_codebooks"] = enable_shared_codebooks
            codec_kwargs["direct_batch_bytes"] = direct_batch_bytes
            self.codec = NeuralZipCodec(**codec_kwargs)
        elif self.method == "zipnn_local":
            self.codec = ZipNNCodec(**codec_kwargs)
        else:
            self.codec = ZipNNSourceCodec(
                method=self.zipnn_method,
                threads=self.threads,
                compression_threshold=compression_threshold,
                chunk_size=chunk_size,
            )

    def compress(self, model):
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
                method_detail=("js-clustered-shared-huffman-packed-exponents"),
            )
        if self.method == "zipnn":
            return self._zipnn(model)
        raise AssertionError(f"Unhandled compression method {self.method!r}.")

    def decompress(self, compressed_model):
        result_is_zipnn = (
            _method_key(compressed_model.get("method")) in _ZIPNN_RESULT_METHOD_KEYS
        )
        if self.method == "zipnn" or result_is_zipnn:
            raise NotImplementedError(_ZIPNN_MODEL_IO_ERROR)

        if hasattr(self.codec, "reset_decompression"):
            self.codec.reset_decompression()
        if hasattr(self.codec, "load_codebooks"):
            header = compressed_model.get("header") or {}
            codebooks = compressed_model.get("codebooks") or header.get("codebooks")
            self.codec.load_codebooks(codebooks)

        payloads = []
        for tensor_result in compressed_model.get("tensor_results", []):
            payload = tensor_result.get("payload")
            if payload is None:
                raise ValueError(
                    "Cannot decompress this result because tensor payloads were not "
                    "kept. Run Compressor(..., keep_payloads=True)."
                )
            payloads.append(payload)
        tensors, _ = decode_tensor_batch(self.codec, payloads)
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

        named_tensors = [
            (name, tensor)
            for name, tensor in _named_input_tensors(model)
            if torch.is_tensor(tensor) and tensor.is_floating_point()
        ]

        if hasattr(self.codec, "reset_compression"):
            self.codec.reset_compression()
        if hasattr(self.codec, "reset_decompression"):
            self.codec.reset_decompression()
        if hasattr(self.codec, "prepare_clusters") and hasattr(
            self.codec, "build_codebooks_and_select_modes"
        ):
            setup_report = prepare_neuralzip_codec(
                self.codec,
                named_tensors,
            )
            clustering_seconds = setup_report.clustering_seconds
            codebook_setup_seconds = setup_report.codebook_setup_seconds
            codebooks = self.codec.export_codebooks()
            codebook_metadata_bytes = serialized_codebooks_size(codebooks)
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
            "encode_seconds": total_encode_seconds,
            "total_seconds": setup_seconds + total_encode_seconds,
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
