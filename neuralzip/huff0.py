from dataclasses import dataclass

from neuralzip import _huff0_cpp

MAX_BLOCK_SIZE = int(_huff0_cpp.MAX_BLOCK_SIZE)


@dataclass(frozen=True)
class Huff0Codebook:
    handle: object
    codebook: bytes


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


def compress_direct_storage(
    tensor_specs,
    prefix,
    *,
    min_saving_bytes=0,
    threads=0,
    compression_threshold=0.95,
):
    native_specs = []
    for data, dtype, stream_id, chunk_size, mode in tensor_specs:
        kind = str(mode.get("kind"))
        if kind == "raw":
            native_mode = (kind, None, -1, b"", 1)
        elif kind in {"shared_huff0", "packed_huff0"}:
            native_mode = (
                kind,
                mode["codebook"].handle,
                int(mode["codebook_id"]),
                mode.get("vocab", b""),
                int(mode.get("ngram_n", 1)),
            )
        else:
            raise ValueError(
                "Direct storage supports raw, shared_huff0, or packed_huff0."
            )
        native_specs.append(
            (
                data,
                dtype.element_size,
                bool(dtype.bit_reorder),
                dtype.mantissa_bits,
                dtype.exponent_mask.bit_length(),
                int(stream_id),
                int(chunk_size),
                native_mode,
            )
        )
    return _huff0_cpp.compress_direct_storage(
        tuple(native_specs),
        prefix,
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


def build_codebook_from_header(codebook, *, vocab=b"", ngram_n=1):
    """Restore an encoder handle directly from a serialized Huff0 header."""
    result = Huff0Codebook(*_huff0_cpp.build_codebook_from_header(codebook))
    if ngram_n != 1:
        _huff0_cpp.prepare_ngram_codebook(result.handle, vocab, int(ngram_n))
    return result


def build_decodebook(codebook):
    return _huff0_cpp.build_decodebook(codebook)


def export_code_table(codebook, *, max_bits=12):
    return _huff0_cpp.export_code_table(codebook, max_bits=int(max_bits))
