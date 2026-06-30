from dataclasses import dataclass

from . import _arithmetic_cpp, _huffman_cpp


@dataclass(frozen=True)
class HuffmanPayload:
    payload: bytes
    code_lengths: bytes
    symbol_count: int


@dataclass(frozen=True)
class HuffmanCodebook:
    handle: object
    code_lengths: bytes


@dataclass(frozen=True)
class HuffmanDecodebook:
    handle: object
    code_lengths: bytes


@dataclass(frozen=True)
class SparseNGramArithmeticCodebook:
    payload: bytes


@dataclass(frozen=True)
class SparseNGramArithmeticPayload:
    payload: bytes
    codebook: bytes
    symbol_count: int
    tail: bytes = b""


def compress(data, *, min_saving_bytes=16):
    if not isinstance(data, (bytes, bytearray, memoryview)):
        data = bytes(data)

    result = _huffman_cpp.compress(data, min_saving_bytes=min_saving_bytes)
    if result is None:
        return None

    payload, code_lengths, symbol_count = result
    return HuffmanPayload(
        payload=payload,
        code_lengths=code_lengths,
        symbol_count=symbol_count,
    )


def build_code_lengths(data, *, prior_count=0):
    if not isinstance(data, (bytes, bytearray, memoryview)):
        data = bytes(data)
    return _huffman_cpp.build_code_lengths(data, prior_count=int(prior_count))


def build_codebook(data, *, prior_count=0):
    if not isinstance(data, (bytes, bytearray, memoryview)):
        data = bytes(data)

    handle, code_lengths = _huffman_cpp.build_codebook(
        data,
        prior_count=int(prior_count),
    )
    return HuffmanCodebook(handle=handle, code_lengths=code_lengths)


def build_decodebook(code_lengths):
    handle = _huffman_cpp.build_decodebook(code_lengths)
    return HuffmanDecodebook(handle=handle, code_lengths=code_lengths)


def encode_with_code_lengths(
    data,
    code_lengths,
    *,
    min_saving_bytes=16,
    max_output_bytes=None,
):
    if not isinstance(data, (bytes, bytearray, memoryview)):
        data = bytes(data)

    max_bytes = -1 if max_output_bytes is None else int(max_output_bytes)
    result = _huffman_cpp.encode_with_code_lengths(
        data,
        code_lengths,
        min_saving_bytes=min_saving_bytes,
        max_output_bytes=max_bytes,
    )
    if result is None:
        return None

    payload, symbol_count = result
    return HuffmanPayload(
        payload=payload,
        code_lengths=code_lengths,
        symbol_count=symbol_count,
    )


def encode_with_codebook(
    data,
    codebook,
    *,
    min_saving_bytes=16,
    max_output_bytes=None,
):
    if not isinstance(data, (bytes, bytearray, memoryview)):
        data = bytes(data)

    max_bytes = -1 if max_output_bytes is None else int(max_output_bytes)
    result = _huffman_cpp.encode_with_codebook(
        data,
        codebook.handle,
        min_saving_bytes=min_saving_bytes,
        max_output_bytes=max_bytes,
    )
    if result is None:
        return None

    payload, symbol_count = result
    return HuffmanPayload(
        payload=payload,
        code_lengths=codebook.code_lengths,
        symbol_count=symbol_count,
    )


def decompress(payload):
    return _huffman_cpp.decompress(
        payload.payload,
        payload.code_lengths,
        payload.symbol_count,
    )


def decompress_with_decodebook(payload, decodebook):
    return _huffman_cpp.decompress_with_decodebook(
        payload.payload,
        decodebook.handle,
        payload.symbol_count,
    )


def build_sparse_ngram_arithmetic_codebook(data, *, n, max_total=1 << 20):
    if not isinstance(data, (bytes, bytearray, memoryview)):
        data = bytes(data)
    payload = _arithmetic_cpp.sparse_ngram_build_codebook(
        data,
        int(n),
        max_total=int(max_total),
    )
    return SparseNGramArithmeticCodebook(payload=payload)


def sparse_ngram_arithmetic_encode_with_codebook(
    data,
    codebook,
    *,
    min_saving_bytes=0,
    max_output_bytes=None,
):
    if not isinstance(data, (bytes, bytearray, memoryview)):
        data = bytes(data)

    max_bytes = -1 if max_output_bytes is None else int(max_output_bytes)
    try:
        result = _arithmetic_cpp.sparse_ngram_encode_with_codebook(
            data,
            codebook.payload,
            min_saving_bytes=min_saving_bytes,
            max_output_bytes=max_bytes,
        )
    except ValueError as error:
        if "missing from sparse codebook" not in str(error):
            raise
        return None
    if result is None:
        return None

    payload, symbol_count, tail = result
    return SparseNGramArithmeticPayload(
        payload=payload,
        codebook=codebook.payload,
        symbol_count=symbol_count,
        tail=tail,
    )


def sparse_ngram_arithmetic_decompress(payload):
    return _arithmetic_cpp.sparse_ngram_decompress(
        payload.payload,
        payload.codebook,
        payload.symbol_count,
        payload.tail,
    )
