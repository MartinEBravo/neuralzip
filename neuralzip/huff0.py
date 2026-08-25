from dataclasses import dataclass

from . import _huff0_cpp


@dataclass(frozen=True)
class Huff0Payload:
    payload: bytes
    codebook: bytes
    symbol_count: int


@dataclass(frozen=True)
class Huff0Codebook:
    handle: object
    codebook: bytes


@dataclass(frozen=True)
class Huff0Decodebook:
    handle: object


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


def _byte_chunks(chunks):
    return [_bytes(chunk) for chunk in chunks]


def _native_variant(name, *, frame_4x):
    suffix = "_4x" if frame_4x else ""
    return getattr(_huff0_cpp, f"{name}{suffix}")


def _payload(result, codebook):
    if result is None:
        return None
    encoded, symbol_count = result
    return Huff0Payload(
        payload=encoded,
        codebook=codebook.codebook,
        symbol_count=symbol_count,
    )


def _ngram_payload(result, codebook):
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


def compress(data, *, min_saving_bytes=0, prior_count=0):
    data = _bytes(data)
    min_saving_bytes = int(min_saving_bytes)
    if min_saving_bytes < 0:
        raise ValueError("min_saving_bytes must be non-negative.")

    codebook = build_codebook(data, prior_count=prior_count)
    result = encode_with_codebook(data, codebook)
    if result is None or len(result.payload) + len(
        result.codebook
    ) + min_saving_bytes >= len(data):
        return None
    return result


def compress_many(chunks, *, min_saving_bytes=0, threads=0):
    normalized = [
        chunk if isinstance(chunk, (bytes, bytearray, memoryview)) else bytes(chunk)
        for chunk in chunks
    ]
    return _huff0_cpp.compress_many(
        normalized,
        min_saving_bytes=int(min_saving_bytes),
        threads=int(threads),
    )


def compress_zipnn(
    data,
    *,
    element_size,
    bit_reorder,
    chunk_size,
    min_saving_bytes=0,
    threads=0,
    compression_threshold=0.95,
):
    return _huff0_cpp.compress_zipnn(
        data,
        int(element_size),
        int(bit_reorder),
        int(chunk_size),
        min_saving_bytes=int(min_saving_bytes),
        threads=int(threads),
        compression_threshold=float(compression_threshold),
    )


def _normalize_zipnn_mixed_modes(element_size, stream_modes):
    element_size = int(element_size)
    modes = tuple(stream_modes)
    if len(modes) != element_size:
        raise ValueError("stream_modes must contain one mode per byte stream.")

    native_modes = []
    supported = {"raw", "local_huff0", "shared_huff0", "packed_huff0"}
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


def compress_zipnn_mixed(
    data,
    *,
    element_size,
    bit_reorder,
    chunk_size,
    stream_modes,
    min_saving_bytes=0,
    threads=0,
    compression_threshold=0.95,
):
    """Encode heterogeneous ZipNN byte streams in one native pipeline.

    ``stream_modes`` contains one mapping per reordered byte stream. Supported
    kinds are ``raw``, ``local_huff0``, ``shared_huff0``, and
    ``packed_huff0``. Shared and packed modes carry a :class:`Huff0Codebook`;
    packed modes additionally carry ``vocab`` and ``ngram_n``. The native
    result contains one list per stream. Each chunk is returned as
    ``(mode, payload, symbol_count, token_count, escapes, tail, n, hits,
    misses)`` after applying the exact local/raw fallback.

    This primitive implements the original ZipNN stream layout. Callers using
    a transposed exponent layout must retain the existing Python path.
    """
    return compress_zipnn_mixed_many(
        ((data, element_size, bit_reorder, chunk_size, stream_modes),),
        min_saving_bytes=int(min_saving_bytes),
        threads=int(threads),
        compression_threshold=float(compression_threshold),
    )[0]


def compress_zipnn_mixed_many(
    tensor_specs,
    *,
    min_saving_bytes=0,
    threads=0,
    compression_threshold=0.95,
):
    """Encode a model's tensors through one native mixed-stream job queue.

    Each input is ``(data, element_size, bit_reorder, chunk_size,
    stream_modes)`` and the output adds one outer tensor dimension to
    :func:`compress_zipnn_mixed`. Input buffers and the native job queue live
    only for this call; no reordered or tokenized target representation is
    retained afterwards.
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
                _normalize_zipnn_mixed_modes(element_size, stream_modes),
            )
        )

    return _huff0_cpp.compress_zipnn_mixed_many(
        tuple(native_specs),
        min_saving_bytes=int(min_saving_bytes),
        threads=int(threads),
        compression_threshold=float(compression_threshold),
    )


def build_codebook(data, *, prior_count=0):
    if not isinstance(data, (bytes, bytearray, memoryview)):
        data = bytes(data)
    handle, codebook = _huff0_cpp.build_codebook(data, prior_count=int(prior_count))
    return Huff0Codebook(handle=handle, codebook=codebook)


def build_decodebook(codebook):
    handle = _huff0_cpp.build_decodebook(codebook)
    return Huff0Decodebook(handle=handle)


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
    """Encode with a precomputed CTable using Huff0's four-stream frame."""
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
    return _payload(result, codebook)


def encode_many_with_codebook(
    chunks,
    codebook,
    *,
    min_saving_bytes=0,
    compression_threshold=0.95,
    metadata_bytes=0,
    threads=0,
):
    return _encode_many_with_codebook(
        chunks,
        codebook,
        min_saving_bytes=min_saving_bytes,
        compression_threshold=compression_threshold,
        metadata_bytes=metadata_bytes,
        threads=threads,
        frame_4x=False,
    )


def encode_many_with_codebook_4x(
    chunks,
    codebook,
    *,
    min_saving_bytes=0,
    compression_threshold=0.95,
    metadata_bytes=0,
    threads=0,
):
    """Encode independent blocks with one CTable as four-stream frames."""
    return _encode_many_with_codebook(
        chunks,
        codebook,
        min_saving_bytes=min_saving_bytes,
        compression_threshold=compression_threshold,
        metadata_bytes=metadata_bytes,
        threads=threads,
        frame_4x=True,
    )


def _encode_many_with_codebook(
    chunks,
    codebook,
    *,
    min_saving_bytes,
    compression_threshold,
    metadata_bytes,
    threads,
    frame_4x,
):
    return _native_variant("encode_many_with_codebook", frame_4x=frame_4x)(
        _byte_chunks(chunks),
        codebook.handle,
        min_saving_bytes=int(min_saving_bytes),
        compression_threshold=float(compression_threshold),
        metadata_bytes=int(metadata_bytes),
        threads=int(threads),
    )


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
    return _ngram_payload(result, codebook)


def encode_many_ngram_with_codebook(
    chunks,
    codebook,
    vocab,
    *,
    n,
    min_saving_bytes=0,
    compression_threshold=0.95,
    metadata_bytes=0,
    threads=0,
):
    return _encode_many_ngram_with_codebook(
        chunks,
        codebook,
        vocab,
        n=n,
        min_saving_bytes=min_saving_bytes,
        compression_threshold=compression_threshold,
        metadata_bytes=metadata_bytes,
        threads=threads,
        frame_4x=False,
    )


def encode_many_ngram_with_codebook_4x(
    chunks,
    codebook,
    vocab,
    *,
    n,
    min_saving_bytes=0,
    compression_threshold=0.95,
    metadata_bytes=0,
    threads=0,
):
    """Tokenize and encode n-gram blocks as Huff0 four-stream frames."""
    return _encode_many_ngram_with_codebook(
        chunks,
        codebook,
        vocab,
        n=n,
        min_saving_bytes=min_saving_bytes,
        compression_threshold=compression_threshold,
        metadata_bytes=metadata_bytes,
        threads=threads,
        frame_4x=True,
    )


def _encode_many_ngram_with_codebook(
    chunks,
    codebook,
    vocab,
    *,
    n,
    min_saving_bytes,
    compression_threshold,
    metadata_bytes,
    threads,
    frame_4x,
):
    results = _native_variant("encode_many_ngram_with_codebook", frame_4x=frame_4x)(
        _byte_chunks(chunks),
        codebook.handle,
        _bytes(vocab),
        int(n),
        min_saving_bytes=int(min_saving_bytes),
        compression_threshold=float(compression_threshold),
        metadata_bytes=int(metadata_bytes),
        threads=int(threads),
    )
    return [_ngram_payload(result, codebook) for result in results]


def decompress(payload):
    return decompress_with_decodebook(payload, build_decodebook(payload.codebook))


def decompress_block(payload, symbol_count):
    return _huff0_cpp.decompress_block(payload, int(symbol_count))


def decompress_with_decodebook(payload, decodebook):
    return _decompress_with_decodebook(payload, decodebook, frame_4x=False)


def decompress_with_decodebook_4x(payload, decodebook):
    """Decode a payload produced by ``encode_with_codebook_4x``."""
    return _decompress_with_decodebook(payload, decodebook, frame_4x=True)


def _decompress_with_decodebook(payload, decodebook, *, frame_4x):
    return _native_variant("decompress_with_decodebook", frame_4x=frame_4x)(
        payload.payload,
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
        payload.payload,
        decodebook.handle,
        payload.symbol_count,
        _bytes(vocab),
        int(n),
        payload.escape_bytes,
        payload.tail,
    )
