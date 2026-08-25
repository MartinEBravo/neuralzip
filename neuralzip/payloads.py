from dataclasses import dataclass

from .serialization import (
    serialized_codebook_size as _serialized_codebook_size,
)
from .serialization import (
    serialized_group_size as _serialized_group_size,
)
from .serialization import (
    serialized_tensor_size as _serialized_tensor_size,
)
from .serialization import (
    stream_chunk_metadata_bytes as _stream_chunk_metadata_bytes,
)


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
        return _serialized_codebook_size(self)


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
        return _stream_chunk_metadata_bytes(self.mode)

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
        return _serialized_group_size(self) - self.payload_bytes

    @property
    def compressed_bytes(self):
        return _serialized_group_size(self)

    @property
    def huffman_chunks(self):
        modes = {"local_huff0", "shared_huff0", "packed_huff0"}
        return sum(chunk.mode in modes for chunk in self.chunks)

    @property
    def local_huff0_chunks(self):
        return sum(chunk.mode == "local_huff0" for chunk in self.chunks)

    @property
    def shared_huff0_chunks(self):
        return sum(chunk.mode == "shared_huff0" for chunk in self.chunks)

    @property
    def raw_chunks(self):
        return sum(chunk.mode == "raw" for chunk in self.chunks)

    @property
    def zero_chunks(self):
        return sum(chunk.mode == "zero" for chunk in self.chunks)

    @property
    def packed_ngram_chunks(self):
        return sum(chunk.mode == "packed_huff0" for chunk in self.chunks)

    @property
    def ngram_hits(self):
        return sum(int(chunk.ngram_hits) for chunk in self.chunks)

    @property
    def ngram_escapes(self):
        return sum(int(chunk.ngram_escapes) for chunk in self.chunks)


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
        return _serialized_tensor_size(self)

    @property
    def huffman_chunks(self):
        return sum(group.huffman_chunks for group in self.groups)

    @property
    def raw_chunks(self):
        return sum(group.raw_chunks for group in self.groups)

    @property
    def zero_chunks(self):
        return sum(group.zero_chunks for group in self.groups)

    @property
    def local_huff0_chunks(self):
        return sum(getattr(group, "local_huff0_chunks", 0) for group in self.groups)

    @property
    def shared_huff0_chunks(self):
        return sum(getattr(group, "shared_huff0_chunks", 0) for group in self.groups)

    @property
    def packed_ngram_chunks(self):
        return sum(getattr(group, "packed_ngram_chunks", 0) for group in self.groups)

    @property
    def ngram_hits(self):
        return sum(getattr(group, "ngram_hits", 0) for group in self.groups)

    @property
    def ngram_escapes(self):
        return sum(getattr(group, "ngram_escapes", 0) for group in self.groups)
