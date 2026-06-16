import time
from typing import TYPE_CHECKING

import torch

from .zipnn_codec import HomemadeZipNNCodec, compression_ratio

if TYPE_CHECKING:
    from .model import HFModel


def _normalize_method(method):
    method_key = str(method).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "zipnn": "zipnn",
        "global_huffman_zipnn": "zipnn",
        "homemade_zipnn": "zipnn",
        "zipnn_casera": "zipnn",
    }
    if method_key not in aliases:
        choices = ", ".join(sorted(set(aliases.values())))
        raise ValueError(
            f"Unsupported compression method {method!r}. Expected one of: {choices}."
        )
    return aliases[method_key]


def _model_parameters(model):
    module = model.model if hasattr(model, "model") else model
    if not hasattr(module, "named_parameters"):
        raise TypeError("Expected an HFModel or a torch module with named_parameters().")
    return module.named_parameters()


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
        min_saving_bytes=16,
    ):
        self.method = _normalize_method(method)
        self.verify = bool(verify)
        self.keep_payloads = bool(keep_payloads)
        self.codec = HomemadeZipNNCodec(min_saving_bytes=min_saving_bytes)

    def compress(self, model: "HFModel"):
        if self.method == "zipnn":
            return self._zipnn(model)
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

    def _zipnn(self, model: "HFModel"):
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
                "payload_bytes": compressed.payload_bytes,
                "metadata_bytes": compressed_bytes - compressed.payload_bytes,
                "compressed_bytes": compressed_bytes,
                "compression_ratio": compression_ratio(compressed),
                "huffman_groups": compressed.huffman_groups,
                "raw_groups": compressed.raw_groups,
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
            "method": "zipnn",
            "zipnn_method": "homemade-bytegroup-huffman",
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
            "huffman_groups": total_huffman_groups,
            "raw_groups": total_raw_groups,
            "encode_seconds": total_encode_seconds,
            "decode_seconds": total_decode_seconds if self.verify else None,
            "roundtrip_ok": roundtrip_ok,
            "tensor_results": tensor_results,
        }
