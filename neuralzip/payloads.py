from dataclasses import dataclass

from neuralzip.dtypes import DTYPES

RAW_CHUNK_METADATA_BYTES = 17
SHARED_CHUNK_METADATA_BYTES = 25
CHUNK_METADATA_BYTES = {
    "raw": RAW_CHUNK_METADATA_BYTES,
    "shared_huff0": SHARED_CHUNK_METADATA_BYTES,
    "mixed_huffman": SHARED_CHUNK_METADATA_BYTES,
}


@dataclass(frozen=True)
class NeuralZipStats:
    setup_seconds: float
    encoding_seconds: float
    raw_bytes: int
    compressed_bytes: int
    exponent_raw_bytes: int | None
    exponent_compressed_bytes: int | None
    representation: str = "storage"
    decoding_seconds: float = 0.0
    runtime_encoding_seconds: float = 0.0
    stream_raw_bytes: int | None = None
    stream_compressed_bytes: int | None = None

    @property
    def compression_ratio(self):
        return self.compressed_bytes / self.raw_bytes if self.raw_bytes else 1.0

    @property
    def exponent_ratio(self):
        if self.exponent_raw_bytes is None or self.exponent_compressed_bytes is None:
            return None
        return (
            self.exponent_compressed_bytes / self.exponent_raw_bytes
            if self.exponent_raw_bytes
            else 1.0
        )

    @property
    def stream_ratio(self):
        """Selected GPU lanes plus their codebooks; excludes resident lanes."""
        if not self.stream_raw_bytes or self.stream_compressed_bytes is None:
            return None
        return self.stream_compressed_bytes / self.stream_raw_bytes

    @property
    def total_seconds(self):
        return self.setup_seconds + self.encoding_seconds


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
    codebook: bytes | memoryview = b""
    vocab: bytes | memoryview = b""

    @property
    def metadata_bytes(self):
        return _codebook_size(self)


@dataclass(frozen=True)
class StreamChunk:
    mode: str
    symbol_count: int
    payload: bytes | memoryview = b""
    codebook_id: int | None = None

    @property
    def payload_bytes(self):
        return len(self.payload)

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
                    chunk["payload"][1]
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
