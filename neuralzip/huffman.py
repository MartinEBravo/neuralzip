import heapq
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class HuffmanPayload:
    payload: bytes
    code_lengths: bytes
    symbol_count: int


def _code_lengths_from_counts(counts):
    lengths = [0] * 256
    heap = [
        (count, symbol, [symbol])
        for symbol, count in enumerate(counts)
        if count > 0
    ]

    if not heap:
        return lengths
    if len(heap) == 1:
        lengths[heap[0][1]] = 1
        return lengths

    heapq.heapify(heap)
    next_index = 256
    while len(heap) > 1:
        left_count, _, left_symbols = heapq.heappop(heap)
        right_count, _, right_symbols = heapq.heappop(heap)
        for symbol in left_symbols:
            lengths[symbol] += 1
        for symbol in right_symbols:
            lengths[symbol] += 1
        heapq.heappush(
            heap,
            (
                left_count + right_count,
                next_index,
                left_symbols + right_symbols,
            ),
        )
        next_index += 1

    return lengths


def _canonical_codes(code_lengths):
    codes = {}
    code = 0
    previous_length = 0
    for symbol, length in sorted(
        enumerate(code_lengths),
        key=lambda item: (item[1], item[0]),
    ):
        if length == 0:
            continue
        code <<= length - previous_length
        codes[symbol] = (code, length)
        code += 1
        previous_length = length
    return codes


def _encode_bits(data, codes):
    output = bytearray()
    bit_buffer = 0
    bit_count = 0

    for symbol in data:
        code, length = codes[symbol]
        bit_buffer = (bit_buffer << length) | code
        bit_count += length

        while bit_count >= 8:
            shift = bit_count - 8
            output.append((bit_buffer >> shift) & 0xFF)
            bit_count -= 8
            bit_buffer &= (1 << bit_count) - 1 if bit_count else 0

    if bit_count:
        output.append((bit_buffer << (8 - bit_count)) & 0xFF)

    return bytes(output)


def _decode_tree(code_lengths):
    tree = {}
    for symbol, (code, length) in _canonical_codes(code_lengths).items():
        node = tree
        for bit_index in range(length - 1, -1, -1):
            bit = (code >> bit_index) & 1
            node = node.setdefault(bit, {})
        node["symbol"] = symbol
    return tree


def _decode_bits(payload, code_lengths, symbol_count):
    tree = _decode_tree(code_lengths)
    output = bytearray()
    node = tree

    for byte in payload:
        for bit_index in range(7, -1, -1):
            bit = (byte >> bit_index) & 1
            node = node[bit]
            symbol = node.get("symbol")
            if symbol is not None:
                output.append(symbol)
                if len(output) == symbol_count:
                    return bytes(output)
                node = tree

    if len(output) != symbol_count:
        raise ValueError("Truncated Huffman payload.")
    return bytes(output)


def compress(data, *, min_saving_bytes=16):
    if not data:
        return None

    counter = Counter(data)
    counts = [counter.get(symbol, 0) for symbol in range(256)]
    code_lengths = _code_lengths_from_counts(counts)
    encoded_bits = sum(counts[symbol] * code_lengths[symbol] for symbol in range(256))
    encoded_bytes = (encoded_bits + 7) // 8

    # The serialized codebook is one byte per symbol. If Huffman does not pay
    # for its own metadata, the caller should store the stream raw.
    if encoded_bytes + 256 + min_saving_bytes >= len(data):
        return None

    payload = _encode_bits(data, _canonical_codes(code_lengths))
    if len(payload) + 256 + min_saving_bytes >= len(data):
        return None

    return HuffmanPayload(
        payload=payload,
        code_lengths=bytes(code_lengths),
        symbol_count=len(data),
    )


def decompress(payload):
    code_lengths = list(payload.code_lengths)
    return _decode_bits(payload.payload, code_lengths, payload.symbol_count)
