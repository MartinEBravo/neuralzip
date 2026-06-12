import time
from typing import TYPE_CHECKING

import torch

from .formats import FLOAT_FORMATS

if TYPE_CHECKING:
    from .model import HFModel


def _normalize_method(method):
    method_key = str(method).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "zipnn": "zipnn",
        "global_huffman_zipnn": "zipnn",
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
    if not hasattr(module, "named_parameters"):
        raise TypeError("Expected an HFModel or a torch module with named_parameters().")
    return module.named_parameters()


def _raw_tensor_bytes(tensor):
    return int(tensor.numel() * tensor.element_size())


def _bitwise_equal(left, right):
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if left.dtype in FLOAT_FORMATS:
        view_dtype = FLOAT_FORMATS[left.dtype]["view_dtype"]
        return torch.equal(
            left.detach().cpu().contiguous().view(view_dtype),
            right.detach().cpu().contiguous().view(view_dtype),
        )
    return torch.equal(left.detach().cpu(), right.detach().cpu())


class Compressor:
    def __init__(
        self,
        method,
        *,
        zipnn_method="HUFFMAN",
        verify=False,
        keep_payloads=False,
        **zipnn_kwargs,
    ):
        self.method = _normalize_method(method)
        self.zipnn_method = zipnn_method
        self.verify = verify
        self.keep_payloads = keep_payloads
        self.zipnn_kwargs = zipnn_kwargs

    def compress(self, model):
        if self.method == "zipnn":
            return self._zipnn(model)
        if self.method == "neuralzip":
            return self._neuralzip(model)
        raise AssertionError(f"Unhandled compression method {self.method!r}.")

    def _make_zipnn_codec(self):
        try:
            from zipnn import ZipNN
        except ImportError as exc:
            raise ImportError(
                "ZipNN compression requires the optional `zipnn` package. "
                'Install the experiment extras with `pip install -e ".[experiments]"`.'
            ) from exc

        return ZipNN(
            method=self.zipnn_method,
            input_format="torch",
            **self.zipnn_kwargs,
        )

    def _zipnn(self, model: "HFModel"):
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

            raw_bytes = _raw_tensor_bytes(parameter)
            total_raw_bytes += raw_bytes
            total_numel += int(parameter.numel())

            if parameter.numel() == 0:
                result = {
                    "name": name,
                    "shape": tuple(int(dim) for dim in parameter.shape),
                    "dtype": str(parameter.dtype).replace("torch.", ""),
                    "numel": 0,
                    "raw_bytes": 0,
                    "compressed_bytes": 0,
                    "compression_ratio": 0.0,
                    "encode_seconds": 0.0,
                }
                if self.verify:
                    result["decode_seconds"] = 0.0
                    result["roundtrip_ok"] = True
                tensor_results.append(result)
                continue

            original = parameter.detach().cpu().contiguous()

            # ZipNN mutates float32 and bfloat16 tensor views during compression.
            # Compress a clone so the loaded checkpoint remains unchanged.
            tensor_for_zipnn = original.clone()
            codec = self._make_zipnn_codec()

            start = time.perf_counter()
            compressed = codec.compress(tensor_for_zipnn)
            encode_seconds = time.perf_counter() - start

            compressed_bytes = len(compressed)
            total_compressed_bytes += compressed_bytes
            total_encode_seconds += encode_seconds

            result = {
                "name": name,
                "shape": tuple(int(dim) for dim in parameter.shape),
                "dtype": str(parameter.dtype).replace("torch.", ""),
                "numel": int(parameter.numel()),
                "raw_bytes": raw_bytes,
                "compressed_bytes": compressed_bytes,
                "compression_ratio": (
                    compressed_bytes / raw_bytes if raw_bytes else 0.0
                ),
                "encode_seconds": encode_seconds,
            }

            if self.verify:
                start = time.perf_counter()
                decompressed = codec.decompress(compressed)
                decode_seconds = time.perf_counter() - start
                is_exact = _bitwise_equal(original, decompressed)
                result["decode_seconds"] = decode_seconds
                result["roundtrip_ok"] = is_exact
                total_decode_seconds += decode_seconds
                roundtrip_ok = bool(roundtrip_ok and is_exact)

            if self.keep_payloads:
                result["payload"] = compressed

            tensor_results.append(result)

        compression_ratio = (
            total_compressed_bytes / total_raw_bytes
            if total_raw_bytes
            else 0.0
        )
        return {
            "method": "zipnn",
            "zipnn_method": self.zipnn_method,
            "input_format": "torch",
            "model_id": getattr(model, "model_name", None),
            "num_tensors": len(tensor_results),
            "numel": total_numel,
            "raw_bytes": total_raw_bytes,
            "compressed_bytes": total_compressed_bytes,
            "compression_ratio": compression_ratio,
            "space_saving": 1.0 - compression_ratio if total_raw_bytes else 0.0,
            "encode_seconds": total_encode_seconds,
            "decode_seconds": total_decode_seconds if self.verify else None,
            "roundtrip_ok": roundtrip_ok,
            "tensor_results": tensor_results,
        }

    def _neuralzip(self, model: "HFModel"):
        raise NotImplementedError(
            "NeuralZip compression is not implemented in Compressor yet."
        )
