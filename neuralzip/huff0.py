from dataclasses import dataclass

from . import _huff0_cpp

MAX_BLOCK_SIZE = int(_huff0_cpp.MAX_BLOCK_SIZE)


@dataclass(frozen=True)
class Huff0Codebook:
    handle: object
    codebook: bytes


@dataclass(frozen=True)
class Huff0CodeTable:
    """Exact canonical symbol codes reconstructed from a Huff0 header."""

    code_lengths: tuple[int, ...]
    code_values: tuple[int, ...]
    max_bits: int


def _immutable_bytes(value):
    """Pin native decoder input against concurrent caller mutation."""
    return value if isinstance(value, bytes) else bytes(value)


def _codebook(result):
    return Huff0Codebook(*result)


def _normalize_mixed_stream_modes(stream_modes):
    modes = tuple(stream_modes)
    native_modes = []
    for mode in modes:
        kind = str(mode.get("kind"))
        if kind == "raw":
            native_modes.append((kind, None, b"", 1))
            continue
        codebook = mode.get("codebook")
        if codebook is None:
            raise ValueError(f"{kind} requires a Huff0 codebook.")
        native_modes.append(
            (
                kind,
                codebook.handle,
                _immutable_bytes(mode.get("vocab", b"")),
                int(mode.get("ngram_n", 1)),
            )
        )
    return tuple(native_modes)


def compress_mixed_streams_many(
    tensor_specs,
    *,
    min_saving_bytes=0,
    threads=0,
    compression_threshold=0.95,
):
    """Encode ``(data, element_size, bit_reorder, chunk_size, modes)`` specs.

    Input buffers are borrowed for this call and must not be mutated until it
    returns.
    """
    native_specs = tuple(
        (
            data,
            int(element_size),
            int(bit_reorder),
            int(chunk_size),
            _normalize_mixed_stream_modes(stream_modes),
        )
        for data, element_size, bit_reorder, chunk_size, stream_modes in tensor_specs
    )
    return _huff0_cpp.compress_mixed_streams_many(
        native_specs,
        min_saving_bytes=int(min_saving_bytes),
        threads=int(threads),
        compression_threshold=float(compression_threshold),
    )


def decompress_mixed_streams_many(tensor_specs, *, threads=0):
    """Decode ``(element_size, bit_reorder, numel, groups)`` specs."""

    def native_chunk(chunk):
        if len(chunk) == 4:
            mode, payload, symbol_count, decodebook = chunk
            vocab, ngram_n, escape_bytes, tail = b"", 1, b"", b""
        elif len(chunk) == 8:
            (
                mode,
                payload,
                symbol_count,
                decodebook,
                vocab,
                ngram_n,
                escape_bytes,
                tail,
            ) = chunk
        else:
            raise ValueError("Decode chunks must contain 4 or 8 fields.")
        return (
            str(mode),
            _immutable_bytes(payload),
            int(symbol_count),
            decodebook,
            _immutable_bytes(vocab),
            int(ngram_n),
            _immutable_bytes(escape_bytes),
            _immutable_bytes(tail),
        )

    native_specs = tuple(
        (
            int(element_size),
            bool(bit_reorder),
            int(numel),
            tuple(tuple(native_chunk(chunk) for chunk in group) for group in groups),
        )
        for element_size, bit_reorder, numel, groups in tensor_specs
    )
    return _huff0_cpp.decompress_mixed_streams_many(native_specs, threads=int(threads))


def build_codebook_from_counts(symbol_counts, *, prior_count=0, max_bits=11):
    """Build a codebook from a complete byte-symbol histogram."""
    return _codebook(
        _huff0_cpp.build_codebook_from_counts(
            tuple(int(count) for count in symbol_counts),
            prior_count=int(prior_count),
            max_bits=int(max_bits),
        )
    )


def build_codebook_from_header(codebook):
    """Restore an encoder handle directly from a serialized Huff0 header."""
    if isinstance(codebook, Huff0Codebook):
        codebook = codebook.codebook
    return _codebook(_huff0_cpp.build_codebook_from_header(_immutable_bytes(codebook)))


def build_decodebook(codebook):
    return _huff0_cpp.build_decodebook(_immutable_bytes(codebook))


def export_code_table(codebook, *, max_bits=12):
    """Return all 256 right-aligned canonical codes from a Huff0 header."""
    if isinstance(codebook, Huff0Codebook):
        codebook = codebook.codebook
    lengths, values, actual_max_bits = _huff0_cpp.export_code_table(
        _immutable_bytes(codebook), max_bits=int(max_bits)
    )
    return Huff0CodeTable(tuple(lengths), tuple(values), int(actual_max_bits))
