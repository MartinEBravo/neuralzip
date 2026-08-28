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


def _native_variant(name, *, frame_4x):
    suffix = "_4x" if frame_4x else ""
    return getattr(_huff0_cpp, f"{name}{suffix}")


def _payload(result):
    if result is None:
        return None
    encoded, symbol_count = result
    return Huff0Payload(
        payload=encoded,
        symbol_count=symbol_count,
    )


def _ngram_payload(result):
    if result is None:
        return None
    encoded, symbol_count, escape_bytes, tail, hits, misses = result
    return Huff0NGramPayload(
        payload=encoded,
        symbol_count=symbol_count,
        escape_bytes=escape_bytes,
        tail=tail,
        hits=int(hits),
        misses=int(misses),
    )


def max_block_size():
    return int(_huff0_cpp.max_block_size())


def _normalize_mixed_stream_modes(element_size, stream_modes):
    element_size = int(element_size)
    modes = tuple(stream_modes)
    if len(modes) != element_size:
        raise ValueError("stream_modes must contain one mode per byte stream.")

    native_modes = []
    supported = {"raw", "shared_huff0", "packed_huff0"}
    for mode in modes:
        kind = str(mode.get("kind"))
        if kind not in supported:
            raise ValueError(f"Unsupported mixed Huff0 stream mode: {kind!r}.")
        if kind in {"shared_huff0", "packed_huff0"}:
            codebook = mode.get("codebook")
            if codebook is None:
                raise ValueError(f"{kind} requires a Huff0 codebook.")
            handle = codebook.handle
        else:
            handle = None

        if kind == "packed_huff0":
            ngram_n = int(mode.get("ngram_n", 1))
            if not 2 <= ngram_n <= 8:
                raise ValueError("packed_huff0 requires ngram_n in [2, 8].")
            vocab = mode.get("vocab", b"")
            if not isinstance(vocab, (bytes, bytearray, memoryview)):
                vocab = bytes(vocab)
            if not vocab:
                raise ValueError("packed_huff0 vocabulary is empty.")
        else:
            ngram_n = 1
            vocab = b""
        native_modes.append(
            (
                kind,
                handle,
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
    """Encode a model's tensors through one native mixed-stream job queue.

    Each input is ``(data, element_size, bit_reorder, chunk_size,
    stream_modes)`` and produces one list per reordered byte stream. Input
    buffers and the native job queue live only for this call; no reordered or
    tokenized target representation is retained afterwards. Callers must not
    mutate input buffers until the call returns.
    """
    native_specs = []
    for spec in tensor_specs:
        if len(spec) != 5:
            raise ValueError(
                "Each mixed tensor spec must contain data, element_size, "
                "bit_reorder, chunk_size, and stream_modes."
            )
        data, element_size, bit_reorder, chunk_size, stream_modes = spec
        element_size = int(element_size)
        native_specs.append(
            (
                data,
                element_size,
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


def decompress_mixed_streams_many(tensor_specs, *, threads=0):
    """Decode sharing-only tensors through one native job queue.

    Each tensor spec is ``(element_size, bit_reorder, numel, groups)``.
    ``groups`` contains one chunk sequence per byte stream, and every chunk is
    ``(mode, payload, symbol_count, decodebook, frame_4x)``. Supported modes
    are ``raw``, ``zero``, ``local_huff0``, and ``shared_huff0``. A shared
    chunk accepts either a :class:`Huff0Decodebook` or its native handle;
    every other mode must use ``None``. Packed chunks are intentionally
    outside this sharing-only primitive.

    The returned list contains one owning ``bytearray`` per tensor in the
    original dtype byte order, after native lane recombination and inverse
    bit reorder.
    """
    native_specs = []
    supported = {"raw", "zero", "local_huff0", "shared_huff0"}
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

        native_groups = []
        for group in groups:
            native_chunks = []
            for chunk in group:
                if len(chunk) != 5:
                    raise ValueError(
                        "Each decode chunk must contain mode, payload, "
                        "symbol_count, decodebook, and frame_4x."
                    )
                mode, payload, symbol_count, decodebook, frame_4x = chunk
                mode = str(mode)
                if mode == "packed_huff0":
                    raise ValueError(
                        "decompress_mixed_streams_many does not support packed_huff0."
                    )
                if mode not in supported:
                    raise ValueError(f"Unsupported decode chunk mode: {mode!r}.")
                if isinstance(decodebook, Huff0Decodebook):
                    decodebook = decodebook.handle
                native_chunks.append(
                    (
                        mode,
                        _immutable_bytes(payload),
                        int(symbol_count),
                        decodebook,
                        bool(frame_4x),
                    )
                )
            native_groups.append(tuple(native_chunks))
        native_specs.append(
            (
                element_size,
                bool(bit_reorder),
                int(numel),
                tuple(native_groups),
            )
        )

    return _huff0_cpp.decompress_mixed_streams_many(
        tuple(native_specs), threads=int(threads)
    )


def build_codebook(data, *, prior_count=0, max_bits=11):
    if not isinstance(data, (bytes, bytearray, memoryview)):
        data = bytes(data)
    handle, codebook = _huff0_cpp.build_codebook(
        data,
        prior_count=int(prior_count),
        max_bits=int(max_bits),
    )
    return Huff0Codebook(handle=handle, codebook=codebook)


def build_codebook_from_counts(symbol_counts, *, prior_count=0, max_bits=11):
    """Build a codebook from a complete byte-symbol histogram.

    Counts are normalized deterministically inside the native backend when
    their total cannot fit Huff0's unsigned counters. Every observed symbol
    and every symbol covered by ``prior_count`` remains representable.
    """

    counts = tuple(int(count) for count in symbol_counts)
    handle, codebook = _huff0_cpp.build_codebook_from_counts(
        counts,
        prior_count=int(prior_count),
        max_bits=int(max_bits),
    )
    return Huff0Codebook(handle=handle, codebook=codebook)


def build_codebook_from_header(codebook):
    """Restore an encoder handle directly from a serialized Huff0 header."""

    if isinstance(codebook, Huff0Codebook):
        codebook = codebook.codebook
    handle, header = _huff0_cpp.build_codebook_from_header(_immutable_bytes(codebook))
    return Huff0Codebook(handle=handle, codebook=header)


def build_decodebook(codebook):
    handle = _huff0_cpp.build_decodebook(_immutable_bytes(codebook))
    return Huff0Decodebook(handle=handle)


def export_code_table(codebook, *, max_bits=12):
    """Return the exact right-aligned Huff0 code for every byte symbol.

    ``codebook`` is the serialized header stored in NeuralZip archives. The
    result always contains 256 entries; symbols absent from the table use a
    zero length and value. ``max_bits`` rejects tables that cannot fit the
    intended decoder lookup representation.
    """
    if isinstance(codebook, Huff0Codebook):
        codebook = codebook.codebook
    lengths, values, actual_max_bits = _huff0_cpp.export_code_table(
        _immutable_bytes(codebook), max_bits=int(max_bits)
    )
    return Huff0CodeTable(
        code_lengths=tuple(lengths),
        code_values=tuple(values),
        max_bits=int(actual_max_bits),
    )


def encode_with_codebook(
    data,
    codebook,
    *,
    min_saving_bytes=0,
    max_output_bytes=None,
):
    return _encode_with_codebook(
        data,
        codebook,
        min_saving_bytes=min_saving_bytes,
        max_output_bytes=max_output_bytes,
        frame_4x=False,
    )


def encode_with_codebook_4x(
    data,
    codebook,
    *,
    min_saving_bytes=0,
    max_output_bytes=None,
):
    """Encode with a precomputed codebook using Huff0's four-stream frame."""
    return _encode_with_codebook(
        data,
        codebook,
        min_saving_bytes=min_saving_bytes,
        max_output_bytes=max_output_bytes,
        frame_4x=True,
    )


def _encode_with_codebook(
    data,
    codebook,
    *,
    min_saving_bytes,
    max_output_bytes,
    frame_4x,
):
    max_bytes = -1 if max_output_bytes is None else int(max_output_bytes)
    result = _native_variant("encode_with_codebook", frame_4x=frame_4x)(
        _bytes(data),
        codebook.handle,
        min_saving_bytes=int(min_saving_bytes),
        max_output_bytes=max_bytes,
    )
    return _payload(result)


def encode_ngram_with_codebook(
    data,
    codebook,
    vocab,
    *,
    n,
    min_saving_bytes=0,
    max_output_bytes=None,
):
    return _encode_ngram_with_codebook(
        data,
        codebook,
        vocab,
        n=n,
        min_saving_bytes=min_saving_bytes,
        max_output_bytes=max_output_bytes,
        frame_4x=False,
    )


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
    return _encode_ngram_with_codebook(
        data,
        codebook,
        vocab,
        n=n,
        min_saving_bytes=min_saving_bytes,
        max_output_bytes=max_output_bytes,
        frame_4x=True,
    )


def _encode_ngram_with_codebook(
    data,
    codebook,
    vocab,
    *,
    n,
    min_saving_bytes,
    max_output_bytes,
    frame_4x,
):
    max_bytes = -1 if max_output_bytes is None else int(max_output_bytes)
    result = _native_variant("encode_ngram_with_codebook", frame_4x=frame_4x)(
        _bytes(data),
        codebook.handle,
        _bytes(vocab),
        int(n),
        min_saving_bytes=int(min_saving_bytes),
        max_output_bytes=max_bytes,
    )
    return _ngram_payload(result)


def decompress_block(payload, symbol_count):
    return _huff0_cpp.decompress_block(_immutable_bytes(payload), int(symbol_count))


def decompress_with_decodebook(payload, decodebook):
    return _decompress_with_decodebook(payload, decodebook, frame_4x=False)


def decompress_with_decodebook_4x(payload, decodebook):
    """Decode a payload produced by ``encode_with_codebook_4x``."""
    return _decompress_with_decodebook(payload, decodebook, frame_4x=True)


def _decompress_with_decodebook(payload, decodebook, *, frame_4x):
    return _native_variant("decompress_with_decodebook", frame_4x=frame_4x)(
        _immutable_bytes(payload.payload),
        decodebook.handle,
        payload.symbol_count,
    )


def decompress_ngram_with_decodebook(payload, decodebook, vocab, *, n):
    return _decompress_ngram_with_decodebook(
        payload, decodebook, vocab, n=n, frame_4x=False
    )


def decompress_ngram_with_decodebook_4x(payload, decodebook, vocab, *, n):
    """Decode four Huff0 token streams and restore their packed n-grams."""
    return _decompress_ngram_with_decodebook(
        payload, decodebook, vocab, n=n, frame_4x=True
    )


def _decompress_ngram_with_decodebook(payload, decodebook, vocab, *, n, frame_4x):
    return _native_variant("decompress_ngram_with_decodebook", frame_4x=frame_4x)(
        _immutable_bytes(payload.payload),
        decodebook.handle,
        payload.symbol_count,
        _immutable_bytes(vocab),
        int(n),
        _immutable_bytes(payload.escape_bytes),
        _immutable_bytes(payload.tail),
    )
