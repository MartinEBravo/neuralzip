from dataclasses import dataclass

import torch

from . import huffman


DTYPE_BY_NAME = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
}


def _dtype_name(dtype):
    return str(dtype).replace("torch.", "")


def _raw_tensor_bytes(tensor):
    return int(tensor.numel() * tensor.element_size())


def _tensor_byte_storage(tensor):
    tensor = tensor.detach().cpu().contiguous()
    return tensor.view(torch.uint8).reshape(-1)


def _byte_group(raw_bytes, element_size, byte_offset):
    return raw_bytes[byte_offset::element_size].contiguous().numpy().tobytes()


@dataclass(frozen=True)
class ByteGroupPayload:
    mode: str
    symbol_count: int
    payload: bytes
    code_lengths: bytes = b""

    @property
    def payload_bytes(self):
        return len(self.payload)

    @property
    def metadata_bytes(self):
        # mode + symbol_count + payload length, plus a canonical code-length table
        # when Huffman is used.
        base_header = 1 + 8 + 8
        if self.mode == "huffman":
            return base_header + len(self.code_lengths)
        return base_header

    @property
    def compressed_bytes(self):
        return self.metadata_bytes + self.payload_bytes


@dataclass(frozen=True)
class TensorPayload:
    name: str
    shape: tuple
    dtype: str
    element_size: int
    numel: int
    groups: tuple

    @property
    def raw_bytes(self):
        return self.numel * self.element_size

    @property
    def metadata_bytes(self):
        # name length + dtype id + rank + dimensions + element size.
        return len(self.name.encode("utf-8")) + 1 + 1 + 8 * len(self.shape) + 8

    @property
    def payload_bytes(self):
        return sum(group.payload_bytes for group in self.groups)

    @property
    def compressed_bytes(self):
        return self.metadata_bytes + sum(group.compressed_bytes for group in self.groups)

    @property
    def huffman_groups(self):
        return sum(group.mode == "huffman" for group in self.groups)

    @property
    def raw_groups(self):
        return sum(group.mode == "raw" for group in self.groups)


class HomemadeZipNNCodec:
    """Small ZipNN-inspired byte-group codec.

    The real ZipNN implementation has a native payload format and additional
    engineering around chunks. This codec keeps the core idea needed for local
    experiments: split tensor bytes by byte position, Huffman-code profitable
    streams, and store the rest raw.
    """

    def __init__(self, *, min_saving_bytes=16):
        self.min_saving_bytes = int(min_saving_bytes)

    def compress_tensor(self, name, tensor):
        if not torch.is_tensor(tensor) or not tensor.is_floating_point():
            raise TypeError("HomemadeZipNNCodec only compresses floating tensors.")

        original = tensor.detach().cpu().contiguous()
        raw_bytes = _tensor_byte_storage(original)
        element_size = int(original.element_size())
        groups = []

        for byte_offset in range(element_size):
            stream = _byte_group(raw_bytes, element_size, byte_offset)
            encoded = huffman.compress(
                stream,
                min_saving_bytes=self.min_saving_bytes,
            )
            if encoded is None:
                groups.append(
                    ByteGroupPayload(
                        mode="raw",
                        symbol_count=len(stream),
                        payload=stream,
                    )
                )
                continue

            groups.append(
                ByteGroupPayload(
                    mode="huffman",
                    symbol_count=encoded.symbol_count,
                    payload=encoded.payload,
                    code_lengths=encoded.code_lengths,
                )
            )

        return TensorPayload(
            name=name,
            shape=tuple(int(dim) for dim in original.shape),
            dtype=_dtype_name(original.dtype),
            element_size=element_size,
            numel=int(original.numel()),
            groups=tuple(groups),
        )

    def decompress_tensor(self, compressed):
        if compressed.dtype not in DTYPE_BY_NAME:
            raise ValueError(f"Unsupported dtype in payload: {compressed.dtype!r}")
        if len(compressed.groups) != compressed.element_size:
            raise ValueError(
                f"Payload has {len(compressed.groups)} byte groups; "
                f"expected {compressed.element_size}."
            )

        dtype = DTYPE_BY_NAME[compressed.dtype]
        dtype_element_size = torch.empty((), dtype=dtype).element_size()
        if dtype_element_size != compressed.element_size:
            raise ValueError(
                f"Payload element size {compressed.element_size} does not match "
                f"dtype {compressed.dtype} element size {dtype_element_size}."
            )
        raw = torch.empty(compressed.raw_bytes, dtype=torch.uint8)

        for byte_offset, group in enumerate(compressed.groups):
            if group.mode == "raw":
                stream = group.payload
            elif group.mode == "huffman":
                stream = huffman.decompress(
                    huffman.HuffmanPayload(
                        payload=group.payload,
                        code_lengths=group.code_lengths,
                        symbol_count=group.symbol_count,
                    )
                )
            else:
                raise ValueError(f"Unsupported byte-group mode: {group.mode!r}")

            if len(stream) != compressed.numel:
                raise ValueError(
                    f"Byte group {byte_offset} has {len(stream)} symbols; "
                    f"expected {compressed.numel}."
                )
            raw[byte_offset::compressed.element_size] = torch.frombuffer(
                bytearray(stream),
                dtype=torch.uint8,
            )

        return raw.view(dtype).reshape(compressed.shape)


def compression_ratio(compressed):
    return compressed.compressed_bytes / compressed.raw_bytes if compressed.raw_bytes else 0.0
