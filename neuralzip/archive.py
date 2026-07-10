import io
import struct
from pathlib import Path

from .payloads import ByteGroupPayload, CodebookPayload, StreamChunk, TensorPayload


MAGIC = b"NZP1"
VERSION = 1

CHUNK_MODE_IDS = {
    "raw": 0,
    "zero": 1,
    "huffman": 2,
    "shared_huffman": 3,
    "shared_huff0": 4,
    "packed_huff0": 5,
}
CHUNK_MODES = {value: key for key, value in CHUNK_MODE_IDS.items()}


class _CountingWriter:
    def __init__(self):
        self.count = 0

    def write(self, data):
        self.count += len(data)
        return len(data)


def _write_struct(handle, fmt, *values):
    handle.write(struct.pack("<" + fmt, *values))


def _write_bytes(handle, value):
    value = bytes(value)
    _write_struct(handle, "Q", len(value))
    handle.write(value)


def _write_text(handle, value):
    encoded = str(value).encode("utf-8")
    _write_struct(handle, "I", len(encoded))
    handle.write(encoded)


def _write_codebook(handle, codebook):
    _write_struct(handle, "q", int(codebook.codebook_id))
    _write_text(handle, codebook.cluster_id)
    _write_struct(handle, "i", int(codebook.stream_id))
    _write_text(handle, codebook.dtype)
    _write_text(handle, codebook.codec)
    _write_text(handle, codebook.mode)
    _write_struct(handle, "B", int(codebook.ngram_n))
    _write_bytes(handle, codebook.code_lengths)
    _write_bytes(handle, codebook.codebook)
    _write_bytes(handle, codebook.vocab)


def _write_chunk(handle, chunk):
    if chunk.mode not in CHUNK_MODE_IDS:
        raise ValueError(f"Unsupported archive chunk mode {chunk.mode!r}.")
    _write_struct(handle, "BQ", CHUNK_MODE_IDS[chunk.mode], int(chunk.symbol_count))
    _write_bytes(handle, chunk.payload)
    if chunk.mode == "huffman":
        _write_bytes(handle, chunk.code_lengths)
    elif chunk.mode in {"shared_huffman", "shared_huff0"}:
        if chunk.codebook_id is None:
            raise ValueError(f"{chunk.mode} chunk is missing its codebook id.")
        _write_struct(handle, "q", int(chunk.codebook_id))
    elif chunk.mode == "packed_huff0":
        if chunk.codebook_id is None or chunk.token_count is None:
            raise ValueError("packed_huff0 chunk is missing coding metadata.")
        _write_struct(
            handle,
            "qQB",
            int(chunk.codebook_id),
            int(chunk.token_count),
            int(chunk.ngram_n),
        )
        _write_bytes(handle, chunk.escape_payload)
        _write_bytes(handle, chunk.tail_payload)


def _write_group(handle, group):
    _write_text(handle, group.stream_kind)
    _write_struct(handle, "QQ", int(group.symbol_count), len(group.chunks))
    for chunk in group.chunks:
        _write_chunk(handle, chunk)


def _write_tensor(handle, tensor):
    _write_text(handle, tensor.name)
    _write_text(handle, tensor.dtype)
    _write_text(handle, tensor.strategy)
    _write_text(handle, tensor.cluster_id or "")
    _write_text(handle, tensor.exponent_layout)
    _write_struct(
        handle,
        "QQiiiI",
        int(tensor.element_size),
        int(tensor.numel),
        int(tensor.byte_reorder),
        int(tensor.bit_reorder),
        int(tensor.num_buffers),
        len(tensor.shape),
    )
    for dimension in tensor.shape:
        _write_struct(handle, "q", int(dimension))
    _write_bytes(handle, tensor.remainder_payload)
    _write_struct(handle, "Q", len(tensor.groups))
    for group in tensor.groups:
        _write_group(handle, group)


def _write_archive(handle, method, codebooks, tensors):
    handle.write(MAGIC)
    _write_struct(handle, "H", VERSION)
    _write_text(handle, method)
    _write_struct(handle, "Q", len(codebooks))
    for codebook in codebooks:
        _write_codebook(handle, codebook)
    _write_struct(handle, "Q", len(tensors))
    for tensor in tensors:
        _write_tensor(handle, tensor)


def serialized_size(method, codebooks, tensors):
    counter = _CountingWriter()
    _write_archive(counter, method, tuple(codebooks), tuple(tensors))
    return counter.count


def serialized_codebooks_size(codebooks):
    counter = _CountingWriter()
    for codebook in codebooks:
        _write_codebook(counter, codebook)
    return counter.count


def serialized_tensor_size(tensor):
    counter = _CountingWriter()
    _write_tensor(counter, tensor)
    return counter.count


def serialized_group_size(group):
    counter = _CountingWriter()
    _write_group(counter, group)
    return counter.count


def dump(compressed_model, target):
    codebooks = tuple(
        compressed_model.get("codebooks")
        or (compressed_model.get("header") or {}).get("codebooks")
        or ()
    )
    tensors = []
    for tensor_result in compressed_model.get("tensor_results", ()):
        payload = tensor_result.get("payload")
        if payload is None:
            raise ValueError(
                "The compressed result has no tensor payloads. Use keep_payloads=True."
            )
        tensors.append(payload)

    close = False
    if isinstance(target, (str, Path)):
        target = Path(target).open("wb")
        close = True
    try:
        _write_archive(target, compressed_model.get("method", "neuralzip"), codebooks, tensors)
    finally:
        if close:
            target.close()


def dumps(compressed_model):
    output = io.BytesIO()
    dump(compressed_model, output)
    return output.getvalue()


class _Reader:
    def __init__(self, handle):
        self.handle = handle

    def read_exact(self, size):
        data = self.handle.read(size)
        if len(data) != size:
            raise ValueError("NeuralZip archive is truncated.")
        return data

    def unpack(self, fmt):
        fmt = "<" + fmt
        return struct.unpack(fmt, self.read_exact(struct.calcsize(fmt)))

    def bytes(self):
        (size,) = self.unpack("Q")
        return self.read_exact(size)

    def text(self):
        (size,) = self.unpack("I")
        try:
            return self.read_exact(size).decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("NeuralZip archive contains invalid UTF-8 metadata.") from error


def _read_codebook(reader):
    (codebook_id,) = reader.unpack("q")
    cluster_id = reader.text()
    (stream_id,) = reader.unpack("i")
    dtype = reader.text()
    codec = reader.text()
    mode = reader.text()
    (ngram_n,) = reader.unpack("B")
    return CodebookPayload(
        codebook_id=codebook_id,
        cluster_id=cluster_id,
        stream_id=stream_id,
        dtype=dtype,
        codec=codec,
        mode=mode,
        ngram_n=ngram_n,
        code_lengths=reader.bytes(),
        codebook=reader.bytes(),
        vocab=reader.bytes(),
    )


def _read_chunk(reader):
    mode_id, symbol_count = reader.unpack("BQ")
    if mode_id not in CHUNK_MODES:
        raise ValueError(f"Unsupported NeuralZip chunk mode id {mode_id}.")
    mode = CHUNK_MODES[mode_id]
    values = {
        "mode": mode,
        "symbol_count": symbol_count,
        "payload": reader.bytes(),
    }
    if mode == "huffman":
        values["code_lengths"] = reader.bytes()
    elif mode in {"shared_huffman", "shared_huff0"}:
        values["codebook_id"] = reader.unpack("q")[0]
    elif mode == "packed_huff0":
        codebook_id, token_count, ngram_n = reader.unpack("qQB")
        values.update(
            codebook_id=codebook_id,
            token_count=token_count,
            ngram_n=ngram_n,
            escape_payload=reader.bytes(),
            tail_payload=reader.bytes(),
        )
    return StreamChunk(**values)


def _read_group(reader):
    stream_kind = reader.text()
    symbol_count, chunk_count = reader.unpack("QQ")
    chunks = tuple(_read_chunk(reader) for _ in range(chunk_count))
    return ByteGroupPayload(
        stream_kind=stream_kind,
        symbol_count=symbol_count,
        chunks=chunks,
    )


def _read_tensor(reader):
    name = reader.text()
    dtype = reader.text()
    strategy = reader.text()
    cluster_id = reader.text() or None
    exponent_layout = reader.text()
    element_size, numel, byte_reorder, bit_reorder, num_buffers, rank = reader.unpack(
        "QQiiiI"
    )
    shape = tuple(reader.unpack("q")[0] for _ in range(rank))
    remainder_payload = reader.bytes()
    (group_count,) = reader.unpack("Q")
    groups = tuple(_read_group(reader) for _ in range(group_count))
    return TensorPayload(
        name=name,
        shape=shape,
        dtype=dtype,
        element_size=element_size,
        numel=numel,
        strategy=strategy,
        groups=groups,
        byte_reorder=byte_reorder,
        bit_reorder=bit_reorder,
        num_buffers=num_buffers,
        remainder_payload=remainder_payload,
        cluster_id=cluster_id,
        exponent_layout=exponent_layout,
    )


def load(source):
    close = False
    if isinstance(source, (str, Path)):
        source = Path(source).open("rb")
        close = True
    reader = _Reader(source)
    try:
        if reader.read_exact(len(MAGIC)) != MAGIC:
            raise ValueError("Not a NeuralZip archive.")
        (version,) = reader.unpack("H")
        if version != VERSION:
            raise ValueError(f"Unsupported NeuralZip archive version {version}.")
        method = reader.text()
        (codebook_count,) = reader.unpack("Q")
        codebooks = tuple(_read_codebook(reader) for _ in range(codebook_count))
        (tensor_count,) = reader.unpack("Q")
        tensors = tuple(_read_tensor(reader) for _ in range(tensor_count))
        if source.read(1):
            raise ValueError("NeuralZip archive contains trailing bytes.")
    finally:
        if close:
            source.close()
    return {
        "method": method,
        "header": {"codebooks": codebooks},
        "codebooks": codebooks,
        "tensor_results": [{"payload": tensor} for tensor in tensors],
    }


def loads(data):
    return load(io.BytesIO(bytes(data)))
