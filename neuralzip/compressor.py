import time
from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch

from .neuralzip_codec import NeuralZipCodec
from .zipnn_codec import ZipNNCodec, compression_ratio

if TYPE_CHECKING:
    from .model import HFModel


def _normalize_method(method):
    method_key = str(method).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "zipnn": "zipnn",
        "global_huffman_zipnn": "zipnn",
        "neuralzip": "neuralzip",
        "neural_zip": "neuralzip",
        "zipnn_src": "zipnn_src",
        "zipnn_source": "zipnn_src",
        "official_zipnn": "zipnn_src",
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


def _bitwise_equal(left, right):
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    return torch.equal(
        left.detach().cpu().contiguous().view(torch.uint8),
        right.detach().cpu().contiguous().view(torch.uint8),
    )


class Compressor:
    def __init__(
        self,
        method="zipnn",
        *,
        verify=True,
        keep_payloads=False,
        min_saving_bytes=0,
        chunk_size=256 << 10,
        compression_threshold=0.95,
        zipnn_method="HUFFMAN",
    ):
        self.method = _normalize_method(method)
        self.verify = bool(verify)
        self.keep_payloads = bool(keep_payloads)
        self.zipnn_method = str(zipnn_method).upper()
        codec_cls = NeuralZipCodec if self.method == "neuralzip" else ZipNNCodec
        self.codec = codec_cls(
            min_saving_bytes=min_saving_bytes,
            chunk_size=chunk_size,
            compression_threshold=compression_threshold,
        )

    def compress(self, model: "HFModel"):
        if self.method == "zipnn":
            return self._codec_compression(
                model,
                method_name="zipnn",
                method_detail="dtype-byte-reorder-huffman",
            )
        if self.method == "neuralzip":
            return self._codec_compression(
                model,
                method_name="neuralzip",
                method_detail="neuralzip-clone",
            )
        if self.method == "zipnn_src":
            return self._zipnn_src(model)
        raise AssertionError(f"Unhandled compression method {self.method!r}.")

    def decompress(self, compressed_model):
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

    def compress_delta(self, base_model, target_model):
        base_tensors = dict(_model_parameters(base_model))
        target_tensors = dict(_model_parameters(target_model))

        tensor_results = []
        total_raw_bytes = 0
        total_compressed_bytes = 0
        total_payload_bytes = 0
        total_metadata_bytes = 0
        total_encode_seconds = 0.0
        total_decode_seconds = 0.0
        total_huffman_chunks = 0
        total_raw_chunks = 0
        total_zero_chunks = 0
        roundtrip_ok = True if self.verify else None

        for name, target in target_tensors.items():
            base = base_tensors.get(name)
            if base is None:
                continue
            if not torch.is_tensor(target) or not target.is_floating_point():
                continue
            if base.shape != target.shape or base.dtype != target.dtype:
                continue

            raw_bytes = _raw_tensor_bytes(target)
            total_raw_bytes += raw_bytes

            start = time.perf_counter()
            compressed = self.codec.compress_delta_tensor(name, base, target)
            encode_seconds = time.perf_counter() - start

            compressed_bytes = compressed.compressed_bytes
            total_compressed_bytes += compressed_bytes
            total_payload_bytes += compressed.payload_bytes
            total_metadata_bytes += compressed_bytes - compressed.payload_bytes
            total_encode_seconds += encode_seconds
            total_huffman_chunks += compressed.huffman_chunks
            total_raw_chunks += compressed.raw_chunks
            total_zero_chunks += compressed.zero_chunks

            tensor_roundtrip_ok = None
            decode_seconds = None
            if self.verify:
                start = time.perf_counter()
                restored = self.codec.decompress_delta_tensor(compressed, base)
                decode_seconds = time.perf_counter() - start
                tensor_roundtrip_ok = _bitwise_equal(target, restored)
                roundtrip_ok = bool(roundtrip_ok and tensor_roundtrip_ok)
                total_decode_seconds += decode_seconds

            result = {
                "name": name,
                "shape": compressed.shape,
                "dtype": compressed.dtype,
                "numel": compressed.numel,
                "raw_bytes": raw_bytes,
                "strategy": compressed.strategy,
                "payload_bytes": compressed.payload_bytes,
                "metadata_bytes": compressed_bytes - compressed.payload_bytes,
                "compressed_bytes": compressed_bytes,
                "compression_ratio": compression_ratio(compressed),
                "huffman_chunks": compressed.huffman_chunks,
                "raw_chunks": compressed.raw_chunks,
                "zero_chunks": compressed.zero_chunks,
                "encode_seconds": encode_seconds,
            }
            if self.verify:
                result["decode_seconds"] = decode_seconds
                result["roundtrip_ok"] = tensor_roundtrip_ok
            if self.keep_payloads:
                result["payload"] = compressed
            tensor_results.append(result)

        compression = (
            total_compressed_bytes / total_raw_bytes if total_raw_bytes else 0.0
        )
        return {
            "method": f"{self.method}_delta",
            "zipnn_method": "xor-delta-byte-huffman",
            "num_tensors": len(tensor_results),
            "raw_bytes": total_raw_bytes,
            "payload_bytes": total_payload_bytes,
            "metadata_bytes": total_metadata_bytes,
            "compressed_bytes": total_compressed_bytes,
            "compression_ratio": compression,
            "space_saving": 1.0 - compression if total_raw_bytes else 0.0,
            "huffman_chunks": total_huffman_chunks,
            "raw_chunks": total_raw_chunks,
            "zero_chunks": total_zero_chunks,
            "encode_seconds": total_encode_seconds,
            "decode_seconds": total_decode_seconds if self.verify else None,
            "roundtrip_ok": roundtrip_ok,
            "tensor_results": tensor_results,
        }

    def decompress_delta(self, compressed_delta, base_model):
        base_tensors = dict(_model_parameters(base_model))
        tensors = {}
        for tensor_result in compressed_delta.get("tensor_results", []):
            payload = tensor_result.get("payload")
            if payload is None:
                raise ValueError(
                    "Cannot decompress this delta because tensor payloads were not "
                    "kept. Run Compressor(..., keep_payloads=True)."
                )
            base = base_tensors[payload.name]
            tensors[payload.name] = self.codec.decompress_delta_tensor(payload, base)
        return tensors

    def _codec_compression(self, model: "HFModel", *, method_name, method_detail):
        tensor_results = []
        total_raw_bytes = 0
        total_compressed_bytes = 0
        total_payload_bytes = 0
        total_metadata_bytes = 0
        total_numel = 0
        total_encode_seconds = 0.0
        total_decode_seconds = 0.0
        total_huffman_groups = 0
        total_raw_groups = 0
        total_zero_groups = 0
        total_huffman_chunks = 0
        total_raw_chunks = 0
        total_zero_chunks = 0
        strategy_counts = {}
        roundtrip_ok = True if self.verify else None

        for name, parameter in _model_parameters(model):
            if not torch.is_tensor(parameter) or not parameter.is_floating_point():
                continue

            raw_bytes = _raw_tensor_bytes(parameter)
            total_raw_bytes += raw_bytes
            total_numel += int(parameter.numel())

            start = time.perf_counter()
            compressed = self.codec.compress_tensor(name, parameter)
            encode_seconds = time.perf_counter() - start

            compressed_bytes = compressed.compressed_bytes
            total_compressed_bytes += compressed_bytes
            total_payload_bytes += compressed.payload_bytes
            total_metadata_bytes += compressed_bytes - compressed.payload_bytes
            total_encode_seconds += encode_seconds
            total_huffman_groups += compressed.huffman_groups
            total_raw_groups += compressed.raw_groups
            total_zero_groups += compressed.zero_groups
            total_huffman_chunks += compressed.huffman_chunks
            total_raw_chunks += compressed.raw_chunks
            total_zero_chunks += compressed.zero_chunks
            strategy_counts[compressed.strategy] = (
                strategy_counts.get(compressed.strategy, 0) + 1
            )

            tensor_roundtrip_ok = None
            decode_seconds = None
            if self.verify:
                start = time.perf_counter()
                decompressed = self.codec.decompress_tensor(compressed)
                decode_seconds = time.perf_counter() - start
                original = parameter.detach().cpu().contiguous()
                tensor_roundtrip_ok = _bitwise_equal(original, decompressed)
                roundtrip_ok = bool(roundtrip_ok and tensor_roundtrip_ok)
                total_decode_seconds += decode_seconds

            result = {
                "name": name,
                "shape": compressed.shape,
                "dtype": compressed.dtype,
                "numel": compressed.numel,
                "raw_bytes": raw_bytes,
                "strategy": compressed.strategy,
                "payload_bytes": compressed.payload_bytes,
                "metadata_bytes": compressed_bytes - compressed.payload_bytes,
                "compressed_bytes": compressed_bytes,
                "compression_ratio": compression_ratio(compressed),
                "huffman_groups": compressed.huffman_groups,
                "raw_groups": compressed.raw_groups,
                "zero_groups": compressed.zero_groups,
                "huffman_chunks": compressed.huffman_chunks,
                "raw_chunks": compressed.raw_chunks,
                "zero_chunks": compressed.zero_chunks,
                "encode_seconds": encode_seconds,
            }
            if self.verify:
                result["decode_seconds"] = decode_seconds
                result["roundtrip_ok"] = tensor_roundtrip_ok
            if self.keep_payloads:
                result["payload"] = compressed
            tensor_results.append(result)

        model_compression_ratio = (
            total_compressed_bytes / total_raw_bytes if total_raw_bytes else 0.0
        )
        return {
            "method": method_name,
            "zipnn_method": method_detail,
            "input_format": "torch",
            "model_id": getattr(model, "model_name", None),
            "num_tensors": len(tensor_results),
            "numel": total_numel,
            "raw_bytes": total_raw_bytes,
            "payload_bytes": total_payload_bytes,
            "metadata_bytes": total_metadata_bytes,
            "compressed_bytes": total_compressed_bytes,
            "compression_ratio": model_compression_ratio,
            "space_saving": (
                1.0 - model_compression_ratio if total_raw_bytes else 0.0
            ),
            "strategy_counts": strategy_counts,
            "huffman_groups": total_huffman_groups,
            "raw_groups": total_raw_groups,
            "zero_groups": total_zero_groups,
            "huffman_chunks": total_huffman_chunks,
            "raw_chunks": total_raw_chunks,
            "zero_chunks": total_zero_chunks,
            "encode_seconds": total_encode_seconds,
            "decode_seconds": total_decode_seconds if self.verify else None,
            "roundtrip_ok": roundtrip_ok,
            "tensor_results": tensor_results,
        }

    def _zipnn_src(self, model: "HFModel"):
        try:
            from zipnn import ZipNN
        except ImportError as exc:
            raise ImportError(
                "Compressor('zipnn-src') requires the official zipnn package. "
                "Install it with `pip install zipnn` or `pip install -e \".[experiments]\"`."
            ) from exc

        tensor_results = []
        total_raw_bytes = 0
        total_compressed_bytes = 0
        total_numel = 0
        total_encode_seconds = 0.0
        total_decode_seconds = 0.0
        roundtrip_ok = True if self.verify else None

        for name, parameter in _model_parameters(model):
            if not torch.is_tensor(parameter) or not parameter.is_floating_point():
                continue

            original = parameter.detach().cpu().contiguous().clone()
            raw_bytes = _raw_tensor_bytes(original)
            total_raw_bytes += raw_bytes
            total_numel += int(original.numel())

            compressor = ZipNN(method=self.zipnn_method, input_format="torch")
            start = time.perf_counter()
            payload = compressor.compress(original.clone())
            encode_seconds = time.perf_counter() - start

            compressed_bytes = len(payload)
            total_compressed_bytes += compressed_bytes
            total_encode_seconds += encode_seconds

            tensor_roundtrip_ok = None
            decode_seconds = None
            if self.verify:
                decompressor = ZipNN(method=self.zipnn_method, input_format="torch")
                start = time.perf_counter()
                decompressed = decompressor.decompress(payload)
                decode_seconds = time.perf_counter() - start
                tensor_roundtrip_ok = _bitwise_equal(original, decompressed)
                roundtrip_ok = bool(roundtrip_ok and tensor_roundtrip_ok)
                total_decode_seconds += decode_seconds

            result = {
                "name": name,
                "shape": tuple(int(dim) for dim in original.shape),
                "dtype": str(original.dtype).replace("torch.", ""),
                "numel": int(original.numel()),
                "raw_bytes": raw_bytes,
                "strategy": "zipnn-src",
                "payload_bytes": compressed_bytes,
                "metadata_bytes": 0,
                "compressed_bytes": compressed_bytes,
                "compression_ratio": compressed_bytes / raw_bytes if raw_bytes else 0.0,
                "encode_seconds": encode_seconds,
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
            "input_format": "torch",
            "model_id": getattr(model, "model_name", None),
            "num_tensors": len(tensor_results),
            "numel": total_numel,
            "raw_bytes": total_raw_bytes,
            "payload_bytes": total_compressed_bytes,
            "metadata_bytes": 0,
            "compressed_bytes": total_compressed_bytes,
            "compression_ratio": model_compression_ratio,
            "space_saving": (
                1.0 - model_compression_ratio if total_raw_bytes else 0.0
            ),
            "strategy_counts": {"zipnn-src": len(tensor_results)},
            "huffman_groups": None,
            "raw_groups": None,
            "zero_groups": None,
            "huffman_chunks": None,
            "raw_chunks": None,
            "zero_chunks": None,
            "encode_seconds": total_encode_seconds,
            "decode_seconds": total_decode_seconds if self.verify else None,
            "roundtrip_ok": roundtrip_ok,
            "tensor_results": tensor_results,
        }
