import heapq
from dataclasses import dataclass

from neuralzip import _huff0_cpp

MAX_BLOCK_SIZE = int(_huff0_cpp.MAX_BLOCK_SIZE)


@dataclass(frozen=True)
class Huff0Codebook:
    handle: object
    codebook: bytes


def count_mixed_tokens(chunks, vocab):
    return _huff0_cpp.count_mixed_tokens(tuple(chunks), vocab)


def build_mixed_codebook(counts, vocab, *, prior_count=1):
    heap = [(int(count) + prior_count, i, i) for i, count in enumerate(counts)]
    heapq.heapify(heap)
    serial = len(heap)
    while len(heap) > 1:
        left, _, a = heapq.heappop(heap)
        right, _, b = heapq.heappop(heap)
        heapq.heappush(heap, (left + right, serial, (a, b)))
        serial += 1
    lengths = bytearray(len(counts))
    stack = [(heap[0][2], 0)]
    while stack:
        node, depth = stack.pop()
        if isinstance(node, int):
            lengths[node] = max(1, depth)
        else:
            stack.extend((child, depth + 1) for child in node)
    return build_codebook_from_header(b"NZM1" + lengths, vocab=vocab)


def compress_direct_storage(tensor_specs, prefix, *, min_saving_bytes=0, threads=0, compression_threshold=0.95):
    native_specs = tuple(
        (
            data, dtype.element_size, bool(dtype.bit_reorder), dtype.mantissa_bits,
            dtype.exponent_mask.bit_length(), int(stream_id), int(chunk_size),
            tuple((mode["kind"], mode["codebook"].handle if "codebook" in mode else None,
                   mode.get("codebook_id", -1)) for mode in modes),
        )
        for data, dtype, stream_id, chunk_size, modes in tensor_specs
    )
    return _huff0_cpp.compress_direct_storage(
        native_specs, prefix, min_saving_bytes=int(min_saving_bytes),
        threads=int(threads), compression_threshold=float(compression_threshold),
    )


def decompress_mixed_streams_many(tensor_specs, *, threads=0):
    return _huff0_cpp.decompress_mixed_streams_many(tuple(tensor_specs), threads=int(threads))


def build_codebook_from_counts(symbol_counts, *, prior_count=0, max_bits=11):
    return Huff0Codebook(*_huff0_cpp.build_codebook_from_counts(
        tuple(int(count) for count in symbol_counts), prior_count=int(prior_count), max_bits=int(max_bits),
    ))


def build_codebook_from_header(codebook, *, vocab=b""):
    return Huff0Codebook(*_huff0_cpp.build_codebook_from_header(codebook, vocab))


def build_decodebook(codebook, *, vocab=b""):
    return _huff0_cpp.build_decodebook(codebook, vocab)


def export_code_table(codebook, *, max_bits=12):
    return _huff0_cpp.export_code_table(codebook, max_bits=int(max_bits))
