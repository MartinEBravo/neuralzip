from dataclasses import dataclass
from neuralzip import _huff0_cpp

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


def _normalize_mixed_stream_modes(stream_modes):
    native_modes = []
    for mode in stream_modes:
        kind = str(mode.get("kind"))
        if kind == "raw":
            native_modes.append((kind, None, b"", 1))
            continue
        codebook = mode["codebook"]
        native_modes.append(
            (
                kind,
                codebook.handle,
                mode.get("vocab", b""),
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
    """Encode ``(data, element_size, bit_reorder, chunk_size, modes)`` specs."""
    native_specs = tuple(
        (
            data,
            int(element_size),
            bool(bit_reorder),
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
    return _huff0_cpp.decompress_mixed_streams_many(
        tuple(tensor_specs), threads=int(threads)
    )


def build_codebook_from_counts(symbol_counts, *, prior_count=0, max_bits=11):
    """Build a codebook from a complete byte-symbol histogram."""
    return Huff0Codebook(
        *_huff0_cpp.build_codebook_from_counts(
            tuple(int(count) for count in symbol_counts),
            prior_count=int(prior_count),
            max_bits=int(max_bits),
        )
    )


def build_codebook_from_header(codebook):
    """Restore an encoder handle directly from a serialized Huff0 header."""
    return Huff0Codebook(*_huff0_cpp.build_codebook_from_header(codebook))


def build_decodebook(codebook):
    return _huff0_cpp.build_decodebook(codebook)


def export_code_table(codebook, *, max_bits=12):
    """Return all 256 right-aligned canonical codes from a Huff0 header."""
    lengths, values, actual_max_bits = _huff0_cpp.export_code_table(
        codebook, max_bits=int(max_bits)
    )
    return Huff0CodeTable(tuple(lengths), tuple(values), int(actual_max_bits))
