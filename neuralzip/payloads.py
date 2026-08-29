"""Immutable payload records and their canonical byte accounting."""

from dataclasses import dataclass

# Canonical layout uses standard little-endian widths: B=1, I/i=4, Q/q=8.
RAW_CHUNK_METADATA_BYTES = 17
SHARED_CHUNK_METADATA_BYTES = 25
PACKED_CHUNK_METADATA_BYTES = 41
CHUNK_METADATA_BYTES = {
    "raw": RAW_CHUNK_METADATA_BYTES,
    "shared_huff0": SHARED_CHUNK_METADATA_BYTES,
    "packed_huff0": PACKED_CHUNK_METADATA_BYTES,
}

def _bytes_size(value):
    return 8 + len(value)

def _text_size(value):
    return 4 + len(str(value).encode("utf-8"))

@dataclass(frozen=True)
class CodebookPayload:
    codebook_id: int
    stream_id: int
    dtype: str
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
    escape_payload: bytes = b""
    tail_payload: bytes = b""

    @property
    def payload_bytes(self):
        return len(self.payload) + len(self.escape_payload) + len(self.tail_payload)

    @property
    def metadata_bytes(self):
        return CHUNK_METADATA_BYTES[self.mode]

    @property
    def compressed_bytes(self):
        return self.metadata_bytes + self.payload_bytes

@dataclass(frozen=True)
class ByteGroupPayload:
    chunks: tuple

@dataclass(frozen=True)
class TensorPayload:
    name: str
    shape: tuple
    dtype: str
    element_size: int
    numel: int
    groups: tuple
    bit_reorder: int = 0

    @property
    def raw_bytes(self):
        return self.numel * self.element_size

def _group_size(group):
    return 8 + sum(chunk.compressed_bytes for chunk in group.chunks)

def _codebook_size(codebook):
    return (
        8
        + 4
        + _text_size(codebook.dtype)
        + _text_size(codebook.mode)
        + 1
        + _bytes_size(codebook.codebook)
        + _bytes_size(codebook.vocab)
    )

def _tensor_size(tensor):
    return (
        _text_size(tensor.name)
        + _text_size(tensor.dtype)
        + 24
        + len(tensor.shape) * 8
        + 8
        + sum(_group_size(group) for group in tensor.groups)
    )

def serialized_size(codebooks, tensors):
    """Return the complete encoded size under NeuralZip's canonical layout."""

    return (
        4
        + 2
        + _text_size("neuralzip")
        + 8
        + sum(_codebook_size(codebook) for codebook in codebooks)
        + 8
        + sum(_tensor_size(tensor) for tensor in tensors)
    )
