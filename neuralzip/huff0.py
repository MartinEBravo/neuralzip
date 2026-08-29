from dataclasses import dataclass

from . import _huff0_cpp


@dataclass(frozen=True)
class Huff0Payload:
    payload: bytes
    symbol_count: int


@dataclass(frozen=True)
class Huff0Codebook:
    handle: object
    codebook: bytes


@dataclass(frozen=True)
class Huff0Decodebook:
    handle: object


@dataclass(frozen=True)
class Huff0CodeTable:
    """Exact canonical symbol codes reconstructed from a Huff0 header."""

    code_lengths: tuple[int, ...]
    code_values: tuple[int, ...]
    max_bits: int


@dataclass(frozen=True)
class Huff0NGramPayload:
    payload: bytes
    symbol_count: int
    escape_bytes: bytes
    tail: bytes
    hits: int
    misses: int


def _bytes(value):
    return value if isinstance(value, (bytes, bytearray, memoryview)) else bytes(value)


def _immutable_bytes(value):
    """Pin native decoder input against concurrent caller mutation."""
    return value if isinstance(value, bytes) else bytes(value)


def _codebook(result):
    return Huff0Codebook(*result)


def _payload(result):
    return None if result is None else Huff0Payload(*result)


def _ngram_payload(result):
    if result is None:
        return None
    encoded, symbol_count, escape_bytes, tail, hits, misses = result
    return Huff0NGramPayload(
        encoded, symbol_count, escape_bytes, tail, int(hits), int(misses)
    )


def max_block_size():
    return _huff0_cpp.max_block_size()


def _normalize_mixed_stream_modes(element_size, stream_modes):
    element_size = int(element_size)
    modes = tuple(stream_modes)
    if len(modes) != element_size:
        raise ValueError("stream_modes must contain one mode per byte stream.")

    supported = {"raw", "shared_huff0", "packed_huff0"}
    native_modes = []
    for mode in modes:
        kind = str(mode.get("kind"))
        if kind not in supported:
            raise ValueError(f"Unsupported mixed Huff0 stream mode: {kind!r}.")
        if kind == "raw":
            native_modes.append((kind, None, b"", 1, bool(mode.get("frame_4x", False))))
            continue

        codebook = mode.get("codebook")
        if codebook is None:
            raise ValueError(f"{kind} requires a Huff0 codebook.")
        ngram_n = 1
        vocab = b""
        if kind == "packed_huff0":
            ngram_n = int(mode.get("ngram_n", 1))
            if not 2 <= ngram_n <= 8:
                raise ValueError("packed_huff0 requires ngram_n in [2, 8].")
            vocab = _bytes(mode.get("vocab", b""))
            if not vocab:
                raise ValueError("packed_huff0 vocabulary is empty.")
        native_modes.append(
            (
                kind,
                codebook.handle,
                vocab,
                ngram_n,
                bool(mode.get("frame_4x", False)),
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
    native_specs = []
    for spec in tensor_specs:
        if len(spec) != 5:
            raise ValueError(
                "Each mixed tensor spec must contain data, element_size, "
                "bit_reorder, chunk_size, and stream_modes."
            )
        data, element_size, bit_reorder, chunk_size, stream_modes = spec
        native_specs.append(
            (
                data,
                int(element_size),
                int(bit_reorder),
                int(chunk_size),
                _normalize_mixed_stream_modes(element_size, stream_modes),
            )
        )

    return _huff0_cpp.compress_mixed_streams_many(
        tuple(native_specs),
        min_saving_bytes=int(min_saving_bytes),
        threads=int(threads),
        compression_threshold=float(compression_threshold),
    )


_DECODE_MODES = frozenset({"raw", "zero", "local_huff0", "shared_huff0"})


def _normalize_decode_chunk(chunk):
    if len(chunk) != 5:
        raise ValueError(
            "Each decode chunk must contain mode, payload, symbol_count, "
            "decodebook, and frame_4x."
        )
    mode, payload, symbol_count, decodebook, frame_4x = chunk
    mode = str(mode)
    if mode == "packed_huff0":
        raise ValueError("decompress_mixed_streams_many does not support packed_huff0.")
    if mode not in _DECODE_MODES:
        raise ValueError(f"Unsupported decode chunk mode: {mode!r}.")
    if isinstance(decodebook, Huff0Decodebook):
        decodebook = decodebook.handle
    return (
        mode,
        _immutable_bytes(payload),
        int(symbol_count),
        decodebook,
        bool(frame_4x),
    )


def decompress_mixed_streams_many(tensor_specs, *, threads=0):
    """Decode sharing-only ``(element_size, bit_reorder, numel, groups)`` specs."""
    native_specs = []
    for spec in tensor_specs:
        if len(spec) != 4:
            raise ValueError(
                "Each decode spec must contain element_size, bit_reorder, "
                "numel, and groups."
            )
        element_size, bit_reorder, numel, groups = spec
        element_size = int(element_size)
        groups = tuple(groups)
        if len(groups) != element_size:
            raise ValueError("groups must contain exactly one group per byte stream.")

        native_specs.append(
            (
                element_size,
                bool(bit_reorder),
                int(numel),
                tuple(
                    tuple(_normalize_decode_chunk(chunk) for chunk in group)
                    for group in groups
                ),
            )
        )

    return _huff0_cpp.decompress_mixed_streams_many(
        tuple(native_specs), threads=int(threads)
    )


def build_codebook(data, *, prior_count=0, max_bits=11):
    data = _bytes(data)
    return _codebook(
        _huff0_cpp.build_codebook(
            data, prior_count=int(prior_count), max_bits=int(max_bits)
        )
    )


def build_codebook_from_counts(symbol_counts, *, prior_count=0, max_bits=11):
    """Build a codebook from a complete byte-symbol histogram.

    Counts are normalized deterministically inside the native backend when
    their total cannot fit Huff0's unsigned counters. Every observed symbol
    and every symbol covered by ``prior_count`` remains representable.
    """

    counts = tuple(int(count) for count in symbol_counts)
    return _codebook(
        _huff0_cpp.build_codebook_from_counts(
            counts, prior_count=int(prior_count), max_bits=int(max_bits)
        )
    )


def build_codebook_from_header(codebook):
    """Restore an encoder handle directly from a serialized Huff0 header."""

    if isinstance(codebook, Huff0Codebook):
        codebook = codebook.codebook
    return _codebook(_huff0_cpp.build_codebook_from_header(_immutable_bytes(codebook)))


def build_decodebook(codebook):
    return Huff0Decodebook(_huff0_cpp.build_decodebook(_immutable_bytes(codebook)))


def export_code_table(codebook, *, max_bits=12):
    """Return all 256 right-aligned canonical codes from a Huff0 header."""
    if isinstance(codebook, Huff0Codebook):
        codebook = codebook.codebook
    lengths, values, actual_max_bits = _huff0_cpp.export_code_table(
        _immutable_bytes(codebook), max_bits=int(max_bits)
    )
    return Huff0CodeTable(tuple(lengths), tuple(values), int(actual_max_bits))


def encode_with_codebook(
    data,
    codebook,
    *,
    min_saving_bytes=0,
    max_output_bytes=None,
    _frame_4x=False,
):
    max_bytes = -1 if max_output_bytes is None else int(max_output_bytes)
    result = _huff0_cpp.encode_with_codebook(
        _bytes(data),
        codebook.handle,
        int(min_saving_bytes),
        max_bytes,
        bool(_frame_4x),
    )
    return _payload(result)


def encode_with_codebook_4x(
    data,
    codebook,
    *,
    min_saving_bytes=0,
    max_output_bytes=None,
):
    """Encode with a precomputed codebook using Huff0's four-stream frame."""
    return encode_with_codebook(
        data,
        codebook,
        min_saving_bytes=min_saving_bytes,
        max_output_bytes=max_output_bytes,
        _frame_4x=True,
    )


def encode_ngram_with_codebook(
    data,
    codebook,
    vocab,
    *,
    n,
    min_saving_bytes=0,
    max_output_bytes=None,
    _frame_4x=False,
):
    max_bytes = -1 if max_output_bytes is None else int(max_output_bytes)
    result = _huff0_cpp.encode_ngram_with_codebook(
        _bytes(data),
        codebook.handle,
        _bytes(vocab),
        int(n),
        int(min_saving_bytes),
        max_bytes,
        bool(_frame_4x),
    )
    return _ngram_payload(result)


def encode_ngram_with_codebook_4x(
    data,
    codebook,
    vocab,
    *,
    n,
    min_saving_bytes=0,
    max_output_bytes=None,
):
    """Tokenize n-grams and encode token ids as one Huff0 four-stream frame."""
    return encode_ngram_with_codebook(
        data,
        codebook,
        vocab,
        n=n,
        min_saving_bytes=min_saving_bytes,
        max_output_bytes=max_output_bytes,
        _frame_4x=True,
    )


def decompress_block(payload, symbol_count):
    return _huff0_cpp.decompress_block(_immutable_bytes(payload), int(symbol_count))


def decompress_with_decodebook(payload, decodebook, *, _frame_4x=False):
    return _huff0_cpp.decompress_with_decodebook(
        _immutable_bytes(payload.payload),
        decodebook.handle,
        payload.symbol_count,
        bool(_frame_4x),
    )


def decompress_with_decodebook_4x(payload, decodebook):
    """Decode a payload produced by ``encode_with_codebook_4x``."""
    return decompress_with_decodebook(payload, decodebook, _frame_4x=True)


def decompress_ngram_with_decodebook(
    payload,
    decodebook,
    vocab,
    *,
    n,
    _frame_4x=False,
):
    return _huff0_cpp.decompress_ngram_with_decodebook(
        _immutable_bytes(payload.payload),
        decodebook.handle,
        payload.symbol_count,
        _immutable_bytes(vocab),
        int(n),
        _immutable_bytes(payload.escape_bytes),
        _immutable_bytes(payload.tail),
        bool(_frame_4x),
    )


def decompress_ngram_with_decodebook_4x(payload, decodebook, vocab, *, n):
    """Decode four Huff0 token streams and restore their packed n-grams."""
    return decompress_ngram_with_decodebook(
        payload,
        decodebook,
        vocab,
        n=n,
        _frame_4x=True,
    )
