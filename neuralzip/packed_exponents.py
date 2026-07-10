import numpy as np


ESCAPE_SYMBOL = 255


def _validate_ngram(n):
    n = int(n)
    if not 2 <= n <= 4:
        raise ValueError("Packed exponent n-grams require n in [2, 4].")
    return n


def pack_ngram_symbols(data, n):
    """Pack consecutive bytes into little-endian uint32 symbols."""
    n = _validate_ngram(n)
    data = bytes(data)
    complete_length = (len(data) // n) * n
    if complete_length == 0:
        return np.empty(0, dtype=np.dtype("<u4")), data

    values = np.frombuffer(data[:complete_length], dtype=np.uint8).reshape(-1, n)
    packed = np.zeros(values.shape[0], dtype=np.dtype("<u4"))
    for offset in range(n):
        packed |= values[:, offset].astype(np.uint32) << np.uint32(8 * offset)
    return packed, data[complete_length:]


def topk_vocab(data, n, top_k=254):
    """Return a sorted little-endian uint32 vocabulary for frequent n-grams."""
    n = _validate_ngram(n)
    top_k = int(top_k)
    if not 1 <= top_k <= 254:
        raise ValueError("top_k must be in [1, 254].")

    packed, _ = pack_ngram_symbols(data, n)
    if packed.size == 0:
        return b""
    symbols, counts = np.unique(packed, return_counts=True)
    order = np.argsort(counts, kind="stable")[::-1]
    selected = np.sort(symbols[order[:top_k]]).astype(np.dtype("<u4"), copy=False)
    return selected.tobytes()


def tokenize_ngrams(data, n, vocab):
    """Tokenize n-grams with byte IDs and an explicit lossless escape stream."""
    n = _validate_ngram(n)
    packed, tail = pack_ngram_symbols(data, n)
    vocab_values = np.frombuffer(bytes(vocab), dtype=np.dtype("<u4"))
    if packed.size == 0:
        return b"", b"", tail, 0, 0
    if vocab_values.size == 0:
        complete = bytes(data)[: packed.size * n]
        return bytes([ESCAPE_SYMBOL]) * packed.size, complete, tail, 0, int(packed.size)

    positions = np.searchsorted(vocab_values, packed)
    safe_positions = np.minimum(positions, vocab_values.size - 1)
    hits = (positions < vocab_values.size) & (
        vocab_values[safe_positions] == packed
    )
    tokens = np.full(packed.size, ESCAPE_SYMBOL, dtype=np.uint8)
    tokens[hits] = positions[hits].astype(np.uint8)

    source = np.frombuffer(bytes(data)[: packed.size * n], dtype=np.uint8).reshape(-1, n)
    escapes = source[~hits].reshape(-1).tobytes() if np.any(~hits) else b""
    hit_count = int(np.count_nonzero(hits))
    return tokens.tobytes(), escapes, tail, hit_count, int(packed.size - hit_count)


def restore_ngrams(tokens, n, vocab, escapes, tail):
    """Inverse of :func:`tokenize_ngrams`, used as a portable fallback."""
    n = _validate_ngram(n)
    vocab_values = np.frombuffer(bytes(vocab), dtype=np.dtype("<u4"))
    escape_view = memoryview(bytes(escapes))
    escape_offset = 0
    output = bytearray()

    for token in bytes(tokens):
        if token == ESCAPE_SYMBOL:
            end = escape_offset + n
            if end > len(escape_view):
                raise ValueError("Packed exponent escape stream is truncated.")
            output.extend(escape_view[escape_offset:end])
            escape_offset = end
            continue
        if token >= vocab_values.size:
            raise ValueError(f"Packed exponent token {token} is outside the vocabulary.")
        value = int(vocab_values[token])
        output.extend(value.to_bytes(4, "little")[:n])

    if escape_offset != len(escape_view):
        raise ValueError("Packed exponent escape stream contains unused bytes.")
    output.extend(bytes(tail))
    return bytes(output)


def transpose_2d_stream(stream, shape):
    if len(shape) != 2 or int(np.prod(shape)) != len(stream):
        return bytes(stream), "original"
    values = np.frombuffer(bytes(stream), dtype=np.uint8).reshape(tuple(shape))
    return values.T.copy().reshape(-1).tobytes(), "transpose2d"


def restore_2d_stream(stream, shape, layout):
    if layout == "original":
        return bytes(stream)
    if layout != "transpose2d" or len(shape) != 2:
        raise ValueError(f"Unsupported exponent layout {layout!r} for shape {shape!r}.")
    rows, columns = (int(value) for value in shape)
    values = np.frombuffer(bytes(stream), dtype=np.uint8).reshape(columns, rows)
    return values.T.copy().reshape(-1).tobytes()
