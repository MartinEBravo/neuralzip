"""Canonical byte accounting for the NeuralZip archive wire format.

The archive writer and payload diagnostics both depend on these functions.
Keeping the field layout here prevents compression ratios from drifting away
from the number of bytes that :mod:`neuralzip.archive` actually emits.
"""

import struct

CHUNK_MODE_IDS = {
    "raw": 0,
    "zero": 1,
    "shared_huff0": 4,
    "packed_huff0": 5,
    "local_huff0": 6,
}
CHUNK_MODES = {value: key for key, value in CHUNK_MODE_IDS.items()}


def _struct_size(fmt):
    return struct.calcsize("<" + fmt)


def serialized_bytes_size(value):
    """Size of a length-prefixed byte buffer."""

    return _struct_size("Q") + len(value)


def serialized_text_size(value):
    """Size of a UTF-8, length-prefixed text field."""

    return _struct_size("I") + len(str(value).encode("utf-8"))


RAW_CHUNK_METADATA_BYTES = _struct_size("BQ") + _struct_size("Q")
SHARED_CHUNK_METADATA_BYTES = RAW_CHUNK_METADATA_BYTES + _struct_size("q")
PACKED_CHUNK_METADATA_BYTES = (
    RAW_CHUNK_METADATA_BYTES + _struct_size("qQB") + 2 * _struct_size("Q")
)


def stream_chunk_metadata_bytes(mode):
    """Return archive metadata overhead for one encoded stream chunk."""

    if mode not in CHUNK_MODE_IDS:
        raise ValueError(f"Unsupported archive chunk mode {mode!r}.")
    if mode == "shared_huff0":
        return SHARED_CHUNK_METADATA_BYTES
    if mode == "packed_huff0":
        return PACKED_CHUNK_METADATA_BYTES
    return RAW_CHUNK_METADATA_BYTES


def serialized_chunk_size(chunk):
    return (
        stream_chunk_metadata_bytes(chunk.mode)
        + len(chunk.payload)
        + len(chunk.escape_payload)
        + len(chunk.tail_payload)
    )


def serialized_group_size(group):
    return (
        serialized_text_size(group.stream_kind)
        + _struct_size("QQ")
        + sum(serialized_chunk_size(chunk) for chunk in group.chunks)
    )


def serialized_codebook_size(codebook):
    return (
        _struct_size("q")
        + serialized_text_size(codebook.cluster_id)
        + _struct_size("i")
        + serialized_text_size(codebook.dtype)
        + serialized_text_size(codebook.codec)
        + serialized_text_size(codebook.mode)
        + _struct_size("B")
        + serialized_bytes_size(codebook.code_lengths)
        + serialized_bytes_size(codebook.codebook)
        + serialized_bytes_size(codebook.vocab)
    )


def serialized_codebooks_size(codebooks):
    return sum(serialized_codebook_size(codebook) for codebook in codebooks)


def serialized_tensor_size(tensor):
    return (
        serialized_text_size(tensor.name)
        + serialized_text_size(tensor.dtype)
        + serialized_text_size(tensor.strategy)
        + serialized_text_size(tensor.cluster_id or "")
        + serialized_text_size(tensor.exponent_layout)
        + _struct_size("QQiiiI")
        + len(tensor.shape) * _struct_size("q")
        + serialized_bytes_size(tensor.remainder_payload)
        + _struct_size("Q")
        + sum(serialized_group_size(group) for group in tensor.groups)
    )


def serialized_size(method, codebooks, tensors):
    """Return the exact byte size of a complete NeuralZip archive."""

    return (
        len(b"NZP1")
        + _struct_size("H")
        + serialized_text_size(method)
        + _struct_size("Q")
        + serialized_codebooks_size(codebooks)
        + _struct_size("Q")
        + sum(serialized_tensor_size(tensor) for tensor in tensors)
    )
