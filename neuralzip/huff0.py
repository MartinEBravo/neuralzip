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
    codebook: bytes


@dataclass(frozen=True)
class Huff0NGramPayload:
    payload: bytes
    codebook: bytes
    symbol_count: int
    escape_bytes: bytes
    tail: bytes
    hits: int
    misses: int


def max_block_size():
    return int(_huff0_cpp.max_block_size())


def compress(data, *, min_saving_bytes=0, prior_count=0):
    if not isinstance(data, (bytes, bytearray, memoryview)):
        data = bytes(data)
    result = _huff0_cpp.compress(
        data,
        min_saving_bytes=int(min_saving_bytes),
        prior_count=int(prior_count),
    )
    if result is None:
        return None
    payload, codebook, symbol_count = result
    return Huff0Payload(payload=payload, codebook=codebook, symbol_count=symbol_count)


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
    element_size = int(element_size)
    native_modes = _normalize_zipnn_mixed_modes(element_size, stream_modes)

    return _huff0_cpp.compress_zipnn_mixed(
        data,
        element_size,
        int(bit_reorder),
        int(chunk_size),
        native_modes,
        min_saving_bytes=int(min_saving_bytes),
        threads=int(threads),
        compression_threshold=float(compression_threshold),
    )


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


def build_prebuilt_block_codebook(data):
    """Build the exact CTable/header pair used by a local Huff0 block."""
    if not isinstance(data, (bytes, bytearray, memoryview)):
        data = bytes(data)
    result = _huff0_cpp.build_prebuilt_block_codebook(data)
    if result is None:
        return None
    handle, codebook = result
    return Huff0Codebook(handle=handle, codebook=codebook)


def codebook_memory_bytes(codebook):
    """Return native CTable plus serialized-header bytes, excluding Python."""
    return int(_huff0_cpp.codebook_memory_bytes(codebook.handle))


def build_decodebook(codebook):
    handle = _huff0_cpp.build_decodebook(codebook)
    return Huff0Decodebook(handle=handle, codebook=codebook)


def encode_with_codebook(
    data,
    codebook,
    *,
    min_saving_bytes=0,
    max_output_bytes=None,
):
    if not isinstance(data, (bytes, bytearray, memoryview)):
        data = bytes(data)
    max_bytes = -1 if max_output_bytes is None else int(max_output_bytes)
    result = _huff0_cpp.encode_with_codebook(
        data,
        codebook.handle,
        min_saving_bytes=int(min_saving_bytes),
        max_output_bytes=max_bytes,
    )
    if result is None:
        return None
    payload, symbol_count = result
    return Huff0Payload(
        payload=payload,
        codebook=codebook.codebook,
        symbol_count=symbol_count,
    )


def encode_with_codebook_4x(
    data,
    codebook,
    *,
    min_saving_bytes=0,
    max_output_bytes=None,
):
    """Encode with a precomputed CTable using Huff0's four-stream frame."""
    if not isinstance(data, (bytes, bytearray, memoryview)):
        data = bytes(data)
    max_bytes = -1 if max_output_bytes is None else int(max_output_bytes)
    result = _huff0_cpp.encode_with_codebook_4x(
        data,
        codebook.handle,
        min_saving_bytes=int(min_saving_bytes),
        max_output_bytes=max_bytes,
    )
    if result is None:
        return None
    payload, symbol_count = result
    return Huff0Payload(
        payload=payload,
        codebook=codebook.codebook,
        symbol_count=symbol_count,
    )


def encode_many_with_codebook(
    chunks,
    codebook,
    *,
    min_saving_bytes=0,
    compression_threshold=0.95,
    metadata_bytes=0,
    threads=0,
):
    normalized = [
        chunk if isinstance(chunk, (bytes, bytearray, memoryview)) else bytes(chunk)
        for chunk in chunks
    ]
    return _huff0_cpp.encode_many_with_codebook(
        normalized,
        codebook.handle,
        min_saving_bytes=int(min_saving_bytes),
        compression_threshold=float(compression_threshold),
        metadata_bytes=int(metadata_bytes),
        threads=int(threads),
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
    normalized = [
        chunk if isinstance(chunk, (bytes, bytearray, memoryview)) else bytes(chunk)
        for chunk in chunks
    ]
    return _huff0_cpp.encode_many_with_codebook_4x(
        normalized,
        codebook.handle,
        min_saving_bytes=int(min_saving_bytes),
        compression_threshold=float(compression_threshold),
        metadata_bytes=int(metadata_bytes),
        threads=int(threads),
    )


def encode_prebuilt_block(
    data,
    codebook,
    *,
    min_saving_bytes=0,
    compression_threshold=0.95,
):
    """Emit a self-contained local frame from a previously built CTable."""
    if not isinstance(data, (bytes, bytearray, memoryview)):
        data = bytes(data)
    return _huff0_cpp.encode_prebuilt_block(
        data,
        codebook.handle,
        min_saving_bytes=int(min_saving_bytes),
        compression_threshold=float(compression_threshold),
    )


def encode_prebuilt_blocks(
    chunks,
    codebooks,
    *,
    min_saving_bytes=0,
    compression_threshold=0.95,
    threads=0,
):
    """Encode self-contained local frames from one prebuilt CTable per block."""
    normalized = [
        chunk if isinstance(chunk, (bytes, bytearray, memoryview)) else bytes(chunk)
        for chunk in chunks
    ]
    handles = [codebook.handle for codebook in codebooks]
    return _huff0_cpp.encode_prebuilt_blocks(
        normalized,
        handles,
        min_saving_bytes=int(min_saving_bytes),
        compression_threshold=float(compression_threshold),
        threads=int(threads),
    )


def encode_many_pretokenized_with_codebook_4x(
    token_chunks,
    codebook,
    max_output_bytes,
    *,
    threads=0,
):
    """Encode materialized token blocks with exact per-block payload budgets.

    This function neither tokenizes nor applies a raw/local fallback.  Each
    result is the encoded payload bytes, or ``None`` when the four-stream frame
    cannot fit its corresponding limit.  Callers remain responsible for
    accounting for metadata, escapes, and tails when computing those limits.
    A limit of ``-1`` means unlimited.
    """
    normalized = [
        chunk if isinstance(chunk, (bytes, bytearray, memoryview)) else bytes(chunk)
        for chunk in token_chunks
    ]
    limits = [int(value) for value in max_output_bytes]
    return _huff0_cpp.encode_many_pretokenized_with_codebook_4x(
        normalized,
        codebook.handle,
        limits,
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
    if not isinstance(data, (bytes, bytearray, memoryview)):
        data = bytes(data)
    if not isinstance(vocab, (bytes, bytearray, memoryview)):
        vocab = bytes(vocab)
    max_bytes = -1 if max_output_bytes is None else int(max_output_bytes)
    result = _huff0_cpp.encode_ngram_with_codebook(
        data,
        codebook.handle,
        vocab,
        int(n),
        min_saving_bytes=int(min_saving_bytes),
        max_output_bytes=max_bytes,
    )
    if result is None:
        return None
    payload, symbol_count, escape_bytes, tail, hits, misses = result
    return Huff0NGramPayload(
        payload=payload,
        codebook=codebook.codebook,
        symbol_count=symbol_count,
        escape_bytes=escape_bytes,
        tail=tail,
        hits=int(hits),
        misses=int(misses),
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
    if not isinstance(data, (bytes, bytearray, memoryview)):
        data = bytes(data)
    if not isinstance(vocab, (bytes, bytearray, memoryview)):
        vocab = bytes(vocab)
    max_bytes = -1 if max_output_bytes is None else int(max_output_bytes)
    result = _huff0_cpp.encode_ngram_with_codebook_4x(
        data,
        codebook.handle,
        vocab,
        int(n),
        min_saving_bytes=int(min_saving_bytes),
        max_output_bytes=max_bytes,
    )
    if result is None:
        return None
    payload, symbol_count, escape_bytes, tail, hits, misses = result
    return Huff0NGramPayload(
        payload=payload,
        codebook=codebook.codebook,
        symbol_count=symbol_count,
        escape_bytes=escape_bytes,
        tail=tail,
        hits=int(hits),
        misses=int(misses),
    )


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
    normalized = [
        chunk if isinstance(chunk, (bytes, bytearray, memoryview)) else bytes(chunk)
        for chunk in chunks
    ]
    if not isinstance(vocab, (bytes, bytearray, memoryview)):
        vocab = bytes(vocab)
    results = _huff0_cpp.encode_many_ngram_with_codebook(
        normalized,
        codebook.handle,
        vocab,
        int(n),
        min_saving_bytes=int(min_saving_bytes),
        compression_threshold=float(compression_threshold),
        metadata_bytes=int(metadata_bytes),
        threads=int(threads),
    )
    output = []
    for result in results:
        if result is None:
            output.append(None)
            continue
        payload, symbol_count, escape_bytes, tail, hits, misses = result
        output.append(
            Huff0NGramPayload(
                payload=payload,
                codebook=codebook.codebook,
                symbol_count=symbol_count,
                escape_bytes=escape_bytes,
                tail=tail,
                hits=int(hits),
                misses=int(misses),
            )
        )
    return output


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
    normalized = [
        chunk if isinstance(chunk, (bytes, bytearray, memoryview)) else bytes(chunk)
        for chunk in chunks
    ]
    if not isinstance(vocab, (bytes, bytearray, memoryview)):
        vocab = bytes(vocab)
    results = _huff0_cpp.encode_many_ngram_with_codebook_4x(
        normalized,
        codebook.handle,
        vocab,
        int(n),
        min_saving_bytes=int(min_saving_bytes),
        compression_threshold=float(compression_threshold),
        metadata_bytes=int(metadata_bytes),
        threads=int(threads),
    )
    output = []
    for result in results:
        if result is None:
            output.append(None)
            continue
        payload, symbol_count, escape_bytes, tail, hits, misses = result
        output.append(
            Huff0NGramPayload(
                payload=payload,
                codebook=codebook.codebook,
                symbol_count=symbol_count,
                escape_bytes=escape_bytes,
                tail=tail,
                hits=int(hits),
                misses=int(misses),
            )
        )
    return output


def decompress(payload):
    return _huff0_cpp.decompress(
        payload.payload,
        payload.codebook,
        payload.symbol_count,
    )


def decompress_block(payload, symbol_count):
    return _huff0_cpp.decompress_block(payload, int(symbol_count))


def decompress_with_decodebook(payload, decodebook):
    return _huff0_cpp.decompress_with_decodebook(
        payload.payload,
        decodebook.handle,
        payload.symbol_count,
    )


def decompress_with_decodebook_4x(payload, decodebook):
    """Decode a payload produced by ``encode_with_codebook_4x``."""
    return _huff0_cpp.decompress_with_decodebook_4x(
        payload.payload,
        decodebook.handle,
        payload.symbol_count,
    )


def decompress_ngram_with_decodebook(payload, decodebook, vocab, *, n):
    if not isinstance(vocab, (bytes, bytearray, memoryview)):
        vocab = bytes(vocab)
    return _huff0_cpp.decompress_ngram_with_decodebook(
        payload.payload,
        decodebook.handle,
        payload.symbol_count,
        vocab,
        int(n),
        payload.escape_bytes,
        payload.tail,
    )


def decompress_ngram_with_decodebook_4x(payload, decodebook, vocab, *, n):
    """Decode four Huff0 token streams and restore their packed n-grams."""
    if not isinstance(vocab, (bytes, bytearray, memoryview)):
        vocab = bytes(vocab)
    return _huff0_cpp.decompress_ngram_with_decodebook_4x(
        payload.payload,
        decodebook.handle,
        payload.symbol_count,
        vocab,
        int(n),
        payload.escape_bytes,
        payload.tail,
    )
