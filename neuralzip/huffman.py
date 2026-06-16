from dataclasses import dataclass

from . import _huffman_cpp


@dataclass(frozen=True)
class HuffmanPayload:
    payload: bytes
    code_lengths: bytes
    symbol_count: int


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


def decompress(payload):
    return _huffman_cpp.decompress(
        payload.payload,
        payload.code_lengths,
        payload.symbol_count,
    )
