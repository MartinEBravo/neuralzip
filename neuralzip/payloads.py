from dataclasses import dataclass

from neuralzip.dtypes import DTYPES

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


def storage_bytes(codebooks, tensors, blob_bytes):
    """Count the blob and canonical metadata, not the surrounding file format."""
    specs = {spec.name: spec for spec in DTYPES.values()}
    raw_bytes = exponent_raw_bytes = exponent_compressed_bytes = 0
    exponent_defined = True
    exponent_streams = set()
    structural_bytes = 4 + 2 + _text_size("neuralzip") + 8 + 8
    structural_bytes += sum(
        value.metadata_bytes - len(value.codebook) - len(value.vocab)
        for value in codebooks
    )
    for tensor in tensors:
        spec = specs[tensor["dtype"]]
        numel = tensor["numel"]
        raw_bytes += numel * tensor["element_size"]
        if numel and spec.exponent_stream is None:
            exponent_defined = False
        if spec.exponent_stream is not None:
            exponent_raw_bytes += numel
            exponent_streams.add((spec.name, spec.exponent_stream))
        structural_bytes += (
            _text_size(tensor["name"])
            + _text_size(tensor["dtype"])
            + 24
            + 8 * len(tensor["shape"])
            + 8
        )
        for stream, group in enumerate(tensor["groups"]):
            metadata_bytes = sum(CHUNK_METADATA_BYTES[chunk["mode"]] for chunk in group)
            structural_bytes += 8 + metadata_bytes
            if stream == spec.exponent_stream:
                exponent_compressed_bytes += metadata_bytes + sum(
                    chunk["payload"][1] + chunk["escapes"][1] + chunk["tail"][1]
                    for chunk in group
                )
    exponent_compressed_bytes += sum(
        value.metadata_bytes
        for value in codebooks
        if (value.dtype, value.stream_id) in exponent_streams
    )
    return {
        "raw_bytes": int(raw_bytes),
        "compressed_bytes": int(blob_bytes + structural_bytes),
        "exponent_raw_bytes": int(exponent_raw_bytes) if exponent_defined else None,
        "exponent_compressed_bytes": (
            int(exponent_compressed_bytes) if exponent_defined else None
        ),
    }
