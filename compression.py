from __future__ import annotations

from dataclasses import dataclass
from itertools import islice

import numpy as np

try:
    from bitarray import bitarray
    from bitarray.util import canonical_decode, canonical_huffman
except ImportError as exc:  # pragma: no cover - dependency error path
    raise RuntimeError(
        "Real ZipNN codecs require the 'bitarray' package. Install it with: pip install bitarray"
    ) from exc

try:
    import constriction
except ImportError as exc:  # pragma: no cover - dependency error path
    raise RuntimeError(
        "NeuralZip arithmetic coding requires the 'constriction' package. Install it with: pip install constriction"
    ) from exc


# ---- Varint and word-level primitives ----

def encode_varint(value: int) -> bytes:
    if value < 0:
        raise ValueError("Varints only support non-negative integers.")

    encoded = bytearray()
    current = value
    while current >= 0x80:
        encoded.append((current & 0x7F) | 0x80)
        current >>= 7
    encoded.append(current)
    return bytes(encoded)


def decode_varint(blob: bytes, offset: int) -> tuple[int, int]:
    shift = 0
    value = 0
    current_offset = offset
    while True:
        if current_offset >= len(blob):
            raise ValueError("Unexpected end of archive while reading varint.")
        byte = blob[current_offset]
        current_offset += 1
        value |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            return value, current_offset
        shift += 7
        if shift > 63:
            raise ValueError("Varint is too large.")


def encode_varint_sequence(values: list[int] | tuple[int, ...]) -> bytes:
    return b"".join(encode_varint(v) for v in values)


def decode_varint_sequence(blob: bytes, offset: int, count: int) -> tuple[list[int], int]:
    result: list[int] = []
    for _ in range(count):
        value, offset = decode_varint(blob, offset)
        result.append(value)
    return result, offset


def uint32_words_to_bytes(words: np.ndarray) -> bytes:
    return words.astype("<u4", copy=False).tobytes()


def bytes_to_uint32_words(data: bytes) -> np.ndarray:
    if len(data) % 4 != 0:
        raise ValueError("Compressed range-coder payload is not aligned to uint32 words.")
    return np.frombuffer(data, dtype="<u4").astype(np.uint32, copy=False)


# ---- Data classes ----

ZIPNN_MAGIC = b"ZNN1"
NEURALZIP_MAGIC = b"NRL2"


@dataclass(frozen=True)
class CanonicalCodebook:
    count_by_length: tuple[int, ...]
    symbols: tuple[int, ...]


class _CompressedBlobMixin:
    blob: bytes

    @property
    def compressed_bits(self) -> int:
        return len(self.blob) * 8


@dataclass(frozen=True)
class ZipNNArchive(_CompressedBlobMixin):
    block_symbol_counts: tuple[int, ...]
    codebook: CanonicalCodebook
    body: bytes
    blob: bytes

    @classmethod
    def build(
        cls,
        block_symbol_counts: list[int],
        codebook: CanonicalCodebook,
        body: bytes,
    ) -> "ZipNNArchive":
        blob = serialize_zipnn_archive(block_symbol_counts, codebook, body)
        return cls(
            block_symbol_counts=tuple(block_symbol_counts),
            codebook=codebook,
            body=body,
            blob=blob,
        )

    @classmethod
    def from_bytes(cls, blob: bytes) -> "ZipNNArchive":
        block_symbol_counts, codebook, body = deserialize_zipnn_archive(blob)
        return cls(
            block_symbol_counts=tuple(block_symbol_counts),
            codebook=codebook,
            body=body,
            blob=blob,
        )


@dataclass(frozen=True)
class NeuralZipCluster:
    symbol_counts: np.ndarray


@dataclass(frozen=True)
class NeuralZipBlock:
    exponent_count: int
    cluster_idx: int
    body: bytes


@dataclass(frozen=True)
class NeuralZipArchive(_CompressedBlobMixin):
    blocks: tuple[NeuralZipBlock, ...]
    clusters: tuple[NeuralZipCluster, ...]
    blob: bytes

    @classmethod
    def build(
        cls,
        blocks: list[NeuralZipBlock],
        clusters: list[NeuralZipCluster],
    ) -> "NeuralZipArchive":
        blob = serialize_neuralzip_archive(blocks, clusters)
        return cls(blocks=tuple(blocks), clusters=tuple(clusters), blob=blob)

    @classmethod
    def from_bytes(cls, blob: bytes) -> "NeuralZipArchive":
        blocks, clusters = deserialize_neuralzip_archive(blob)
        return cls(blocks=tuple(blocks), clusters=tuple(clusters), blob=blob)


# ---- Model building ----

def build_canonical_codebook_from_counts(
    counts: np.ndarray,
) -> tuple[dict[int, bitarray], CanonicalCodebook]:
    frequency_map = {
        symbol: int(count)
        for symbol, count in enumerate(counts.tolist())
        if int(count) > 0
    }
    if not frequency_map:
        raise ValueError("Cannot build a Huffman codebook from an empty frequency map.")

    code, count_by_length, symbols = canonical_huffman(frequency_map)
    return code, CanonicalCodebook(
        count_by_length=tuple(int(count) for count in count_by_length),
        symbols=tuple(int(symbol) for symbol in symbols),
    )


def build_arithmetic_model_from_counts(counts: np.ndarray):
    if int(counts.sum()) <= 0:
        raise ValueError("Cannot build an arithmetic model from empty counts.")
    probabilities = counts.astype(np.float64, copy=False)
    return constriction.stream.model.Categorical(probabilities, perfect=False)


# ---- Huffman (codebook) encoding/decoding ----

def encode_symbols_with_codebook(code: dict[int, bitarray], symbols: np.ndarray) -> bytes:
    if symbols.size == 0:
        return b""
    bits = bitarray(endian="big")
    bits.encode(code, symbols)
    return bits.tobytes()


def decode_symbols_with_codebook(
    body: bytes,
    symbol_count: int,
    codebook: CanonicalCodebook,
) -> np.ndarray:
    if symbol_count == 0:
        return np.array([], dtype=np.uint8)
    bits = bitarray(endian="big")
    bits.frombytes(body)
    decoded = np.fromiter(
        islice(
            canonical_decode(bits, codebook.count_by_length, codebook.symbols),
            symbol_count,
        ),
        dtype=np.uint8,
        count=symbol_count,
    )
    if decoded.size != symbol_count:
        raise ValueError("Decoded Huffman stream is shorter than expected.")
    return decoded


# ---- Arithmetic coding encoding/decoding ----

def encode_symbols_with_arithmetic_model(symbols: np.ndarray, model) -> bytes:
    if symbols.size == 0:
        return b""
    encoder = constriction.stream.queue.RangeEncoder()
    encoder.encode(symbols.astype(np.int32, copy=False), model)
    return uint32_words_to_bytes(encoder.get_compressed())


def decode_symbols_with_arithmetic_model(
    body: bytes,
    symbol_count: int,
    counts: np.ndarray,
    model=None,
    output_dtype=np.int32,
) -> np.ndarray:
    if symbol_count == 0:
        return np.array([], dtype=output_dtype)

    vocabulary = np.flatnonzero(counts)
    if vocabulary.size == 1:
        return np.full(symbol_count, int(vocabulary[0]), dtype=output_dtype)

    if model is None:
        model = build_arithmetic_model_from_counts(counts)
    decoder = constriction.stream.queue.RangeDecoder(bytes_to_uint32_words(body))
    decoded = decoder.decode(model, symbol_count)
    return decoded.astype(output_dtype, copy=False)


# ---- Archive-level stream decoding ----

def decode_zipnn_archive_exponent_streams(blob: bytes) -> list[np.ndarray]:
    archive = ZipNNArchive.from_bytes(blob)
    total_symbols = sum(archive.block_symbol_counts)
    all_symbols = decode_symbols_with_codebook(archive.body, total_symbols, archive.codebook)

    streams: list[np.ndarray] = []
    offset = 0
    for block_symbol_count in archive.block_symbol_counts:
        next_offset = offset + block_symbol_count
        streams.append(all_symbols[offset:next_offset].copy())
        offset = next_offset

    if offset != all_symbols.size:
        raise ValueError("Decoded ZipNN archive has trailing symbols.")
    return streams


def decode_neuralzip_archive_exponent_streams(blob: bytes) -> list[np.ndarray]:
    archive = NeuralZipArchive.from_bytes(blob)
    models = [build_arithmetic_model_from_counts(c.symbol_counts) for c in archive.clusters]
    return [
        decode_symbols_with_arithmetic_model(
            block.body,
            block.exponent_count,
            archive.clusters[block.cluster_idx].symbol_counts,
            models[block.cluster_idx],
            output_dtype=np.uint8,
        )
        for block in archive.blocks
    ]


# ---- Codebook serialization ----

def serialize_codebook(codebook: CanonicalCodebook) -> bytes:
    return b"".join([
        encode_varint(len(codebook.count_by_length)),
        encode_varint_sequence(codebook.count_by_length),
        encode_varint(len(codebook.symbols)),
        bytes(codebook.symbols),
    ])


def deserialize_codebook(blob: bytes, offset: int) -> tuple[CanonicalCodebook, int]:
    num_lengths, offset = decode_varint(blob, offset)
    count_by_length, offset = decode_varint_sequence(blob, offset, num_lengths)
    symbol_count, offset = decode_varint(blob, offset)
    symbols = tuple(int(symbol) for symbol in blob[offset : offset + symbol_count])
    offset += symbol_count
    return CanonicalCodebook(tuple(count_by_length), symbols), offset


# ---- Symbol count serialization ----

def serialize_symbol_counts(counts: np.ndarray) -> bytes:
    nonzero = [(symbol, int(count)) for symbol, count in enumerate(counts.tolist()) if int(count) > 0]
    parts = [encode_varint(len(nonzero))]
    for symbol, count in nonzero:
        parts.append(bytes([symbol]))
        parts.append(encode_varint(count))
    return b"".join(parts)


def deserialize_symbol_counts(blob: bytes, offset: int) -> tuple[np.ndarray, int]:
    counts = np.zeros(256, dtype=np.int64)
    num_nonzero, offset = decode_varint(blob, offset)
    for _ in range(num_nonzero):
        if offset >= len(blob):
            raise ValueError("Unexpected end of archive while reading NeuralZip symbol counts.")
        symbol = int(blob[offset])
        offset += 1
        count, offset = decode_varint(blob, offset)
        counts[symbol] = count
    return counts, offset


# ---- Archive serialization helpers ----

def _validate_magic(blob: bytes, magic: bytes, name: str) -> int:
    if not blob.startswith(magic):
        raise ValueError(f"Invalid {name} archive header.")
    return len(magic)


def _extract_body(blob: bytes, offset: int) -> tuple[bytes, int]:
    body_len, offset = decode_varint(blob, offset)
    body = blob[offset : offset + body_len]
    return body, offset + body_len


def _check_fully_consumed(offset: int, blob_len: int, name: str) -> None:
    if offset != blob_len:
        raise ValueError(f"{name} archive has trailing bytes.")


# ---- ZipNN archive serialization ----

def serialize_zipnn_archive(
    block_symbol_counts: list[int],
    codebook: CanonicalCodebook,
    body: bytes,
) -> bytes:
    return b"".join([
        ZIPNN_MAGIC,
        encode_varint(len(block_symbol_counts)),
        encode_varint_sequence(block_symbol_counts),
        serialize_codebook(codebook),
        encode_varint(len(body)),
        body,
    ])


def deserialize_zipnn_archive(blob: bytes) -> tuple[list[int], CanonicalCodebook, bytes]:
    offset = _validate_magic(blob, ZIPNN_MAGIC, "ZipNN")
    num_blocks, offset = decode_varint(blob, offset)
    block_symbol_counts, offset = decode_varint_sequence(blob, offset, num_blocks)
    codebook, offset = deserialize_codebook(blob, offset)
    body, offset = _extract_body(blob, offset)
    _check_fully_consumed(offset, len(blob), "ZipNN")
    return block_symbol_counts, codebook, body


# ---- NeuralZip archive serialization ----

def _serialize_neuralzip_block(block: NeuralZipBlock) -> bytes:
    return b"".join([
        encode_varint(block.cluster_idx),
        encode_varint(block.exponent_count),
        encode_varint(len(block.body)),
        block.body,
    ])


def serialize_neuralzip_archive(
    blocks: list[NeuralZipBlock],
    clusters: list[NeuralZipCluster],
) -> bytes:
    parts = [
        NEURALZIP_MAGIC,
        encode_varint(len(clusters)),
        *(serialize_symbol_counts(c.symbol_counts) for c in clusters),
        encode_varint(len(blocks)),
        *(_serialize_neuralzip_block(b) for b in blocks),
    ]
    return b"".join(parts)


def deserialize_neuralzip_archive(
    blob: bytes,
) -> tuple[list[NeuralZipBlock], list[NeuralZipCluster]]:
    offset = _validate_magic(blob, NEURALZIP_MAGIC, "NeuralZip")

    num_clusters, offset = decode_varint(blob, offset)
    clusters: list[NeuralZipCluster] = []
    for _ in range(num_clusters):
        counts, offset = deserialize_symbol_counts(blob, offset)
        clusters.append(NeuralZipCluster(symbol_counts=counts))

    num_blocks, offset = decode_varint(blob, offset)
    blocks: list[NeuralZipBlock] = []
    for _ in range(num_blocks):
        cluster_idx, offset = decode_varint(blob, offset)
        exponent_count, offset = decode_varint(blob, offset)
        if cluster_idx >= len(clusters):
            raise ValueError("NeuralZip block references a missing cluster.")
        body, offset = _extract_body(blob, offset)
        blocks.append(
            NeuralZipBlock(
                exponent_count=exponent_count,
                cluster_idx=cluster_idx,
                body=body,
            )
        )

    _check_fully_consumed(offset, len(blob), "NeuralZip")
    return blocks, clusters
