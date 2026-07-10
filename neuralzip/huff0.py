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


def build_codebook(data, *, prior_count=0):
    if not isinstance(data, (bytes, bytearray, memoryview)):
        data = bytes(data)
    handle, codebook = _huff0_cpp.build_codebook(data, prior_count=int(prior_count))
    return Huff0Codebook(handle=handle, codebook=codebook)


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


def decompress(payload):
    return _huff0_cpp.decompress(
        payload.payload,
        payload.codebook,
        payload.symbol_count,
    )


def decompress_with_decodebook(payload, decodebook):
    return _huff0_cpp.decompress_with_decodebook(
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
