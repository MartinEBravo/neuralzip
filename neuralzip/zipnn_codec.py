import numpy as np
import torch

from . import huff0, huffman
from .payloads import ByteGroupPayload, StreamChunk, TensorPayload


DTYPE_BY_NAME = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

ZIPNN_DTYPE_CONFIG = {
    torch.float32: {
        "element_size": 4,
        "num_buffers": 4,
        "byte_reorder": 220,
        "bit_reorder": 1,
    },
    torch.bfloat16: {
        "element_size": 2,
        "num_buffers": 2,
        "byte_reorder": 10,
        "bit_reorder": 1,
    },
    torch.float16: {
        "element_size": 2,
        "num_buffers": 2,
        "byte_reorder": 10,
        "bit_reorder": 0,
    },
}


def _dtype_name(dtype):
    return str(dtype).replace("torch.", "")


def _tensor_raw_bytes(tensor):
    tensor = tensor.detach().cpu().contiguous()
    return tensor.view(torch.uint8).reshape(-1).numpy().tobytes()


def _tensor_from_raw_bytes(raw_bytes, dtype, shape):
    raw = torch.frombuffer(bytearray(raw_bytes), dtype=torch.uint8)
    return raw.view(dtype).reshape(shape)


def _zipnn_dtype_config(dtype):
    config = ZIPNN_DTYPE_CONFIG.get(dtype)
    if config is None:
        raise TypeError(
            "ZipNN supports torch.float32, torch.bfloat16, and torch.float16 "
            f"tensors; got {dtype}."
        )
    return config


def _reorder_float32(raw_bytes):
    values = np.frombuffer(raw_bytes, dtype=np.dtype("<u4")).copy()
    sign = (values >> np.uint32(8)) & np.uint32(0x00800000)
    exponent = (values << np.uint32(1)) & np.uint32(0xFF000000)
    mantissa = values & np.uint32(0x007FFFFF)
    return (exponent | sign | mantissa).astype(np.dtype("<u4"), copy=False).tobytes()


def _restore_float32(raw_bytes):
    values = np.frombuffer(raw_bytes, dtype=np.dtype("<u4")).copy()
    sign = (values << np.uint32(8)) & np.uint32(0x80000000)
    exponent = (values >> np.uint32(1)) & np.uint32(0x7F800000)
    mantissa = values & np.uint32(0x007FFFFF)
    return (sign | exponent | mantissa).astype(np.dtype("<u4"), copy=False).tobytes()


def _reorder_bfloat16(raw_bytes):
    values = np.frombuffer(raw_bytes, dtype=np.dtype("<u2")).copy()
    sign = (values >> np.uint16(8)) & np.uint16(0x0080)
    exponent = (values << np.uint16(1)) & np.uint16(0xFF00)
    mantissa = values & np.uint16(0x007F)
    return (exponent | sign | mantissa).astype(np.dtype("<u2"), copy=False).tobytes()


def _restore_bfloat16(raw_bytes):
    values = np.frombuffer(raw_bytes, dtype=np.dtype("<u2")).copy()
    sign = (values << np.uint16(8)) & np.uint16(0x8000)
    exponent = (values >> np.uint16(1)) & np.uint16(0x7F80)
    mantissa = values & np.uint16(0x007F)
    return (sign | exponent | mantissa).astype(np.dtype("<u2"), copy=False).tobytes()


def _zipnn_reordered_bytes(tensor, config):
    raw_bytes = _tensor_raw_bytes(tensor)
    if not config["bit_reorder"]:
        return raw_bytes
    if tensor.dtype == torch.float32:
        return _reorder_float32(raw_bytes)
    if tensor.dtype == torch.bfloat16:
        return _reorder_bfloat16(raw_bytes)
    return raw_bytes


def _zipnn_restored_bytes(raw_bytes, dtype, config):
    if not config["bit_reorder"]:
        return raw_bytes
    if dtype == torch.float32:
        return _restore_float32(raw_bytes)
    if dtype == torch.bfloat16:
        return _restore_bfloat16(raw_bytes)
    return raw_bytes


class ZipNNCodec:
    """ZipNN-style dtype-aware byte grouping with Huffman and raw fallback."""

    def __init__(
        self,
        *,
        min_saving_bytes=0,
        chunk_size=256 << 10,
        compression_threshold=0.95,
        threads=0,
    ):
        self.min_saving_bytes = int(min_saving_bytes)
        self.chunk_size = int(chunk_size)
        self.compression_threshold = float(compression_threshold)
        self.threads = int(threads)
        if self.min_saving_bytes < 0:
            raise ValueError("min_saving_bytes must be non-negative.")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        if not 0 < self.compression_threshold <= 1:
            raise ValueError("compression_threshold must be in (0, 1].")
        if self.threads < 0:
            raise ValueError("threads must be non-negative.")

    def compress_tensor(self, name, tensor):
        if not torch.is_tensor(tensor) or not tensor.is_floating_point():
            raise TypeError("ZipNNCodec only compresses floating tensors.")

        # The local native backend is read-only. Avoid charging ZipNN for a
        # private tensor clone when the direct NeuralZip backend consumes the
        # same immutable contiguous input buffer.
        original = tensor.detach().cpu().contiguous()
        config = _zipnn_dtype_config(original.dtype)
        return self._compress_zipnn(name, original, config)

    def decompress_tensor(self, compressed):
        if compressed.dtype not in DTYPE_BY_NAME:
            raise ValueError(f"Unsupported dtype in payload: {compressed.dtype!r}")

        if compressed.strategy in {"zipnn", "byte_groups"}:
            return self._decompress_zipnn(compressed)
        if compressed.strategy == "delta":
            raise ValueError("Delta payloads require decompress_delta_tensor().")
        raise ValueError(f"Unsupported ZipNN strategy: {compressed.strategy!r}")

    def compress_delta_tensor(self, name, base_tensor, target_tensor):
        if not torch.is_tensor(base_tensor) or not torch.is_tensor(target_tensor):
            raise TypeError("Delta compression expects torch tensors.")
        if (
            base_tensor.shape != target_tensor.shape
            or base_tensor.dtype != target_tensor.dtype
        ):
            raise ValueError("Delta tensors must have matching shape and dtype.")
        if not target_tensor.is_floating_point():
            raise TypeError("ZipNN delta compression only supports floating tensors.")

        base = base_tensor.detach().cpu().contiguous()
        target = target_tensor.detach().cpu().contiguous()
        base_bytes = np.frombuffer(_tensor_raw_bytes(base), dtype=np.uint8)
        target_bytes = np.frombuffer(_tensor_raw_bytes(target), dtype=np.uint8)
        delta = np.bitwise_xor(base_bytes, target_bytes).tobytes()
        group = self._compress_stream(delta, "delta")

        return TensorPayload(
            name=name,
            shape=tuple(int(dim) for dim in target.shape),
            dtype=_dtype_name(target.dtype),
            element_size=int(target.element_size()),
            numel=int(target.numel()),
            strategy="delta",
            groups=(group,),
        )

    def decompress_delta_tensor(self, compressed, base_tensor):
        if compressed.strategy != "delta":
            raise ValueError("Expected a delta payload.")
        if len(compressed.groups) != 1:
            raise ValueError("Delta payload must contain exactly one stream.")
        if not torch.is_tensor(base_tensor):
            raise TypeError("base_tensor must be a torch tensor.")
        if tuple(int(dim) for dim in base_tensor.shape) != compressed.shape:
            raise ValueError("Base tensor shape does not match delta payload.")

        dtype = DTYPE_BY_NAME[compressed.dtype]
        if base_tensor.dtype != dtype:
            raise ValueError("Base tensor dtype does not match delta payload.")

        delta = self._decompress_stream(compressed.groups[0])
        if len(delta) != compressed.raw_bytes:
            raise ValueError(
                f"Delta stream has {len(delta)} bytes; expected {compressed.raw_bytes}."
            )

        base = base_tensor.detach().cpu().contiguous()
        base_bytes = np.frombuffer(_tensor_raw_bytes(base), dtype=np.uint8)
        delta_bytes = np.frombuffer(delta, dtype=np.uint8)
        target_bytes = np.bitwise_xor(base_bytes, delta_bytes).tobytes()
        return _tensor_from_raw_bytes(target_bytes, dtype, compressed.shape)

    def _stream_chunk_size(self, element_size):
        return max(1, self.chunk_size // int(element_size))

    def _compress_stream(self, stream, stream_kind, *, chunk_size=None):
        return self._compress_streams(((stream, stream_kind, chunk_size),))[0]

    def _compress_local_huff0_chunks(self, chunks):
        chunks = tuple(bytes(chunk) for chunk in chunks)
        encoded_chunks = huff0.compress_many(
            chunks,
            min_saving_bytes=self.min_saving_bytes,
            threads=self.threads,
        )
        output = []
        for chunk, encoded in zip(chunks, encoded_chunks, strict=True):
            if (
                encoded is not None
                and len(encoded) < len(chunk) * self.compression_threshold
            ):
                output.append(
                    StreamChunk(
                        mode="local_huff0",
                        symbol_count=len(chunk),
                        payload=encoded,
                    )
                )
            else:
                output.append(
                    StreamChunk(
                        mode="raw",
                        symbol_count=len(chunk),
                        payload=chunk,
                    )
                )
        return tuple(output)

    def _compress_streams(self, stream_specs):
        prepared = []
        flat_chunks = []
        max_block_size = huff0.max_block_size()

        for stream, stream_kind, chunk_size in stream_specs:
            if chunk_size is None:
                chunk_size = self.chunk_size
            chunk_size = min(int(chunk_size), max_block_size)
            stream_chunks = []
            for start in range(0, len(stream), chunk_size):
                chunk = stream[start : start + chunk_size]
                if chunk:
                    stream_chunks.append(chunk)
            stream_chunks = tuple(stream_chunks)
            prepared.append((stream_kind, len(stream), stream_chunks))
            flat_chunks.extend(stream_chunks)

        encoded_chunks = self._compress_local_huff0_chunks(flat_chunks)
        groups = []
        offset = 0
        for stream_kind, symbol_count, stream_chunks in prepared:
            chunks = []
            for _ in stream_chunks:
                encoded = encoded_chunks[offset]
                offset += 1
                chunks.append(encoded)
            groups.append(
                ByteGroupPayload(
                    stream_kind=stream_kind,
                    symbol_count=symbol_count,
                    chunks=tuple(chunks),
                )
            )
        return tuple(groups)

    def _decompress_stream(self, group):
        output = bytearray()
        for chunk in group.chunks:
            if chunk.mode == "zero":
                output.extend(b"\x00" * chunk.symbol_count)
            elif chunk.mode == "raw":
                output.extend(chunk.payload)
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
            elif chunk.mode == "local_huff0":
                output.extend(huff0.decompress_block(chunk.payload, chunk.symbol_count))
            else:
                raise ValueError(f"Unsupported stream chunk mode: {chunk.mode!r}")

        if len(output) != group.symbol_count:
            raise ValueError(
                f"Stream decoded to {len(output)} symbols; "
                f"expected {group.symbol_count}."
            )
        return bytes(output)

    def _compress_zipnn(self, name, tensor, config):
        element_size = int(config["element_size"])
        raw = tensor.view(torch.uint8).reshape(-1).numpy()
        native_groups = huff0.compress_zipnn(
            raw,
            element_size=element_size,
            bit_reorder=int(config["bit_reorder"]),
            chunk_size=self._stream_chunk_size(element_size),
            min_saving_bytes=self.min_saving_bytes,
            threads=self.threads,
            compression_threshold=self.compression_threshold,
        )
        groups = tuple(
            ByteGroupPayload(
                stream_kind="byte_group",
                symbol_count=int(tensor.numel()),
                chunks=tuple(
                    StreamChunk(
                        mode="local_huff0" if compressed else "raw",
                        symbol_count=symbol_count,
                        payload=payload,
                    )
                    for compressed, payload, symbol_count in native_chunks
                ),
            )
            for native_chunks in native_groups
        )

        return TensorPayload(
            name=name,
            shape=tuple(int(dim) for dim in tensor.shape),
            dtype=_dtype_name(tensor.dtype),
            element_size=element_size,
            numel=int(tensor.numel()),
            strategy="zipnn",
            groups=groups,
            byte_reorder=int(config["byte_reorder"]),
            bit_reorder=int(config["bit_reorder"]),
            num_buffers=int(config["num_buffers"]),
        )

    def _decompress_zipnn(self, compressed):
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
            stream = self._decompress_stream(group)
            expected_symbols = len(raw[byte_offset::compressed.element_size])
            if len(stream) != expected_symbols:
                raise ValueError(
                    f"Byte group {byte_offset} has {len(stream)} symbols; "
                    f"expected {expected_symbols}."
                )
            raw[byte_offset::compressed.element_size] = stream

        restored = _zipnn_restored_bytes(bytes(raw), dtype, config)
        return _tensor_from_raw_bytes(restored, dtype, compressed.shape)
