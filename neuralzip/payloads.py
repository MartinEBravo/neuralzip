"""Immutable payload records and their canonical byte accounting."""

import struct
from dataclasses import dataclass

_SUPPORTED_CHUNK_MODES = {
    "raw",
    "zero",
    "shared_huff0",
    "packed_huff0",
    "local_huff0",
}


def _struct_size(fmt):
    return struct.calcsize("<" + fmt)


RAW_CHUNK_METADATA_BYTES = _struct_size("BQ") + _struct_size("Q")
SHARED_CHUNK_METADATA_BYTES = RAW_CHUNK_METADATA_BYTES + _struct_size("q")
PACKED_CHUNK_METADATA_BYTES = (
    RAW_CHUNK_METADATA_BYTES + _struct_size("qQB") + 2 * _struct_size("Q")
)


def _chunk_metadata_bytes(mode):
    if mode not in _SUPPORTED_CHUNK_MODES:
        raise ValueError(f"Unsupported archive chunk mode {mode!r}.")
    if mode == "shared_huff0":
        return SHARED_CHUNK_METADATA_BYTES
    if mode == "packed_huff0":
        return PACKED_CHUNK_METADATA_BYTES
    return RAW_CHUNK_METADATA_BYTES


def _bytes_size(value):
    return _struct_size("Q") + len(value)


def _text_size(value):
    return _struct_size("I") + len(str(value).encode("utf-8"))


@dataclass(frozen=True)
class CodebookPayload:
    codebook_id: int
    cluster_id: str
    stream_id: int
    dtype: str
    code_lengths: bytes = b""
    codec: str = "huff0"
    mode: str = "shared_huff0"
    codebook: bytes = b""
    ngram_n: int = 1
    vocab: bytes = b""

    @property
    def metadata_bytes(self):
        return _codebook_size(self)


@dataclass(frozen=True)
class StreamChunk:
    mode: str
    symbol_count: int
    payload: bytes = b""
    codebook_id: int | None = None
    token_count: int | None = None
    escape_payload: bytes = b""
    tail_payload: bytes = b""
    ngram_n: int = 1
    ngram_hits: int = 0
    ngram_escapes: int = 0

    @property
    def payload_bytes(self):
        return len(self.payload) + len(self.escape_payload) + len(self.tail_payload)

    @property
    def metadata_bytes(self):
        return _chunk_metadata_bytes(self.mode)

    @property
    def compressed_bytes(self):
        return self.metadata_bytes + self.payload_bytes


@dataclass(frozen=True)
class ByteGroupPayload:
    stream_kind: str
    symbol_count: int
    chunks: tuple

    @property
    def payload_bytes(self):
        return sum(chunk.payload_bytes for chunk in self.chunks)

    @property
    def metadata_bytes(self):
        return _group_size(self) - self.payload_bytes

    @property
    def compressed_bytes(self):
        return _group_size(self)


@dataclass(frozen=True)
class TensorPayload:
    name: str
    shape: tuple
    dtype: str
    element_size: int
    numel: int
    strategy: str
    groups: tuple
    byte_reorder: int = 0
    bit_reorder: int = 0
    num_buffers: int = 0
    remainder_payload: bytes = b""
    cluster_id: str | None = None
    exponent_layout: str = "original"

    @property
    def raw_bytes(self):
        return self.numel * self.element_size

    @property
    def payload_bytes(self):
        return sum(group.payload_bytes for group in self.groups)

    @property
    def compressed_bytes(self):
        return _tensor_size(self)


def _group_size(group):
    chunks = sum(
        _chunk_metadata_bytes(chunk.mode)
        + len(chunk.payload)
        + len(chunk.escape_payload)
        + len(chunk.tail_payload)
        for chunk in group.chunks
    )
    return _text_size(group.stream_kind) + _struct_size("QQ") + chunks


def _codebook_size(codebook):
    return (
        _struct_size("q")
        + _text_size(codebook.cluster_id)
        + _struct_size("i")
        + _text_size(codebook.dtype)
        + _text_size(codebook.codec)
        + _text_size(codebook.mode)
        + _struct_size("B")
        + _bytes_size(codebook.code_lengths)
        + _bytes_size(codebook.codebook)
        + _bytes_size(codebook.vocab)
    )


def _tensor_size(tensor):
    return (
        _text_size(tensor.name)
        + _text_size(tensor.dtype)
        + _text_size(tensor.strategy)
        + _text_size(tensor.cluster_id or "")
        + _text_size(tensor.exponent_layout)
        + _struct_size("QQiiiI")
        + len(tensor.shape) * _struct_size("q")
        + _bytes_size(tensor.remainder_payload)
        + _struct_size("Q")
        + sum(_group_size(group) for group in tensor.groups)
    )


def serialized_size(codebooks, tensors):
    """Return the complete encoded size under NeuralZip's canonical layout."""

    return (
        len(b"NZP1")
        + _struct_size("H")
        + _text_size("neuralzip")
        + _struct_size("Q")
        + sum(_codebook_size(codebook) for codebook in codebooks)
        + _struct_size("Q")
        + sum(_tensor_size(tensor) for tensor in tensors)
    )
