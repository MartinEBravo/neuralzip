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
class Huff0CodeTable:
    """Exact canonical symbol codes reconstructed from a Huff0 header."""

    code_lengths: tuple[int, ...]
    code_values: tuple[int, ...]
    max_bits: int


@dataclass(frozen=True)
class GPUEncodedBFloat16:
    """Native CPU export of BF16 weights for the CUDA decoder."""

    encoded_exponents: bytes
    sign_mantissa: bytes
    packed_gaps: bytes
    output_positions: tuple[int, ...]
    bit_length: int
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


def backend_info():
    """Return the native Huff0 CPU-dispatch configuration."""
    return dict(_huff0_cpp.backend_info())


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
    misses)`` after applying direct raw fallback to rejected shared or packed
    chunks. Explicit ``local_huff0`` modes still use their own local/raw path.

    This primitive implements the original ZipNN stream layout used by every
    production NeuralZip encoding entrypoint.
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


def decompress_zipnn_mixed_many(tensor_specs, *, threads=0):
    """Decode sharing-only ZipNN tensors through one native job queue.

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
                        "decompress_zipnn_mixed_many does not support packed_huff0."
                    )
                if mode not in supported:
                    raise ValueError(f"Unsupported ZipNN decode chunk mode: {mode!r}.")
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

    return _huff0_cpp.decompress_zipnn_mixed_many(
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


def encode_gpu_bfloat16(
    raw_bf16,
    codebook_or_header,
    bytes_per_thread=8,
    threads_per_block=512,
):
    """Encode little-endian BF16 bytes into the CUDA inference layout.

    The exponent lane is emitted as one continuous MSB-first canonical Huffman
    stream. The sign/mantissa lane remains verbatim, while ``packed_gaps`` and
    ``output_positions`` let independent CUDA threads and blocks restart inside
    that stream without scanning from its beginning.
    """

    if isinstance(codebook_or_header, Huff0Codebook):
        codebook_header = codebook_or_header.codebook
    else:
        codebook_header = codebook_or_header
    result = _huff0_cpp.encode_gpu_bfloat16(
        _immutable_bytes(raw_bf16),
        _immutable_bytes(codebook_header),
        bytes_per_thread=int(bytes_per_thread),
        threads_per_block=int(threads_per_block),
    )
    (
        encoded_exponents,
        sign_mantissa,
        packed_gaps,
        output_positions,
        bit_length,
        max_bits,
    ) = result
    return GPUEncodedBFloat16(
        encoded_exponents=encoded_exponents,
        sign_mantissa=sign_mantissa,
        packed_gaps=packed_gaps,
        output_positions=tuple(output_positions),
        bit_length=int(bit_length),
        max_bits=int(max_bits),
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
