from dataclasses import dataclass


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
        strings = (self.cluster_id, self.dtype, self.codec, self.mode)
        return (
            8
            + 4
            + 1
            + sum(4 + len(value.encode("utf-8")) for value in strings)
            + 8
            + len(self.code_lengths)
            + 8
            + len(self.codebook)
            + 8
            + len(self.vocab)
        )


@dataclass(frozen=True)
class StreamChunk:
    mode: str
    symbol_count: int
    payload: bytes = b""
    code_lengths: bytes = b""
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
        base_header = 1 + 8 + 8
        if self.mode == "huffman":
            return base_header + 8 + len(self.code_lengths)
        if self.mode in {"shared_huffman", "shared_huff0"}:
            return base_header + 8
        if self.mode == "packed_huff0":
            return base_header + 8 + 8 + 1 + 8 + 8
        return base_header

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
        return (
            4
            + len(self.stream_kind.encode("utf-8"))
            + 8
            + 8
            + sum(chunk.metadata_bytes for chunk in self.chunks)
        )

    @property
    def compressed_bytes(self):
        return self.metadata_bytes + self.payload_bytes

    @property
    def huffman_chunks(self):
        modes = {
            "huffman",
            "local_huff0",
            "shared_huffman",
            "shared_huff0",
            "packed_huff0",
        }
        return sum(chunk.mode in modes for chunk in self.chunks)

    @property
    def local_huffman_chunks(self):
        return sum(chunk.mode in {"huffman", "local_huff0"} for chunk in self.chunks)

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
    def shared_codebook_defs(self):
        return sum(
            chunk.mode == "shared_huffman" and bool(chunk.code_lengths)
            for chunk in self.chunks
        )

    @property
    def shared_codebook_refs(self):
        return sum(
            (chunk.mode == "shared_huffman" and not chunk.code_lengths)
            or chunk.mode in {"shared_huff0", "packed_huff0"}
            for chunk in self.chunks
        )

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
    def metadata_bytes(self):
        strings = (
            self.name,
            self.dtype,
            self.strategy,
            self.cluster_id or "",
            self.exponent_layout,
        )
        return (
            sum(4 + len(value.encode("utf-8")) for value in strings)
            + 32
            + 8 * len(self.shape)
            + 8
            + len(self.remainder_payload)
            + 8
            + sum(group.metadata_bytes for group in self.groups)
        )

    @property
    def payload_bytes(self):
        return sum(group.payload_bytes for group in self.groups)

    @property
    def compressed_bytes(self):
        return self.metadata_bytes + self.payload_bytes

    @property
    def huffman_groups(self):
        return sum(group.huffman_chunks > 0 for group in self.groups)

    @property
    def raw_groups(self):
        return sum(group.raw_chunks > 0 for group in self.groups)

    @property
    def zero_groups(self):
        return sum(group.zero_chunks > 0 for group in self.groups)

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
    def shared_codebook_defs(self):
        return sum(getattr(group, "shared_codebook_defs", 0) for group in self.groups)

    @property
    def shared_codebook_refs(self):
        return sum(getattr(group, "shared_codebook_refs", 0) for group in self.groups)

    @property
    def local_huffman_chunks(self):
        return sum(getattr(group, "local_huffman_chunks", 0) for group in self.groups)

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
