"""Parser for the stream-size metadata emitted by official ZipNN archives."""

from dataclasses import dataclass
import math
import struct
import sys


@dataclass(frozen=True)
class ZipNNStreamSize:
    stream_id: int
    raw_bytes: int
    payload_bytes: int
    metadata_bytes: int

    @property
    def compressed_bytes(self):
        return self.payload_bytes + self.metadata_bytes


@dataclass(frozen=True)
class ZipNNArchiveLayout:
    original_bytes: int
    archive_bytes: int
    header_bytes: int
    chunk_bytes: int
    chunk_count: int
    element_size: int
    byte_reorder: int
    bit_reorder: int
    streams: tuple[ZipNNStreamSize, ...]

    def stream(self, stream_id):
        return self.streams[int(stream_id)]

    @property
    def exponent_stream(self):
        return self.streams[self.element_size - 1]


def _shape_header_end(payload):
    if len(payload) < 33:
        raise ValueError("ZipNN payload is shorter than its tensor header.")
    dimensions = int(payload[32])
    offset = 33
    for _ in range(dimensions):
        if offset >= len(payload):
            raise ValueError("Truncated ZipNN tensor-shape header.")
        width = int(payload[offset])
        if width not in {1, 2, 4, 8}:
            raise ValueError(f"Invalid ZipNN shape width {width}.")
        offset += 1 + width
        if offset > len(payload):
            raise ValueError("Truncated ZipNN tensor-shape value.")
    return offset


def parse_zipnn_streams(payload, *, element_size, expected_raw_bytes=None):
    """Return exact stream sizes from a tensor-formatted ZipNN frame.

    Official ZipNN currently writes native ``size_t`` cumulative lengths.  The
    parser therefore uses the host pointer width and byte order, matching the
    process that produced and consumes the archive.
    """

    view = memoryview(payload).cast("B")
    if len(view) < 33 or bytes(view[:2]) != b"ZN":
        raise ValueError("Invalid ZipNN payload header.")

    element_size = int(element_size)
    if element_size not in {2, 4}:
        raise ValueError("ZipNN stream parsing supports 2- or 4-byte floats.")
    original_bytes = int.from_bytes(view[16:24], byteorder="little")
    archive_bytes = int.from_bytes(view[24:32], byteorder="little")
    if archive_bytes != len(view):
        raise ValueError(
            "ZipNN compressed length mismatch: "
            f"{archive_bytes} != {len(view)}."
        )
    if expected_raw_bytes is not None and original_bytes != int(expected_raw_bytes):
        raise ValueError(
            "ZipNN original length mismatch: "
            f"{original_bytes} != {int(expected_raw_bytes)}."
        )
    if original_bytes % element_size:
        raise ValueError(
            f"ZipNN original length is not divisible by {element_size}."
        )

    input_format = int(view[8])
    if input_format not in {2, 3}:
        raise ValueError(
            "ZipNN stream parsing requires a torch or numpy tensor frame."
        )
    header_bytes = _shape_header_end(view)
    chunk_exponent = int(view[14])
    if chunk_exponent >= 63:
        raise ValueError(f"Invalid ZipNN chunk-size exponent {chunk_exponent}.")
    chunk_bytes = 1 << chunk_exponent
    chunk_count = math.ceil(original_bytes / chunk_bytes)
    entry_count = element_size * chunk_count
    type_bytes = entry_count
    size_width = struct.calcsize("P")
    size_table_bytes = entry_count * size_width
    sizes_start = header_bytes + type_bytes
    payload_start = sizes_start + size_table_bytes
    if payload_start > len(view):
        raise ValueError("ZipNN payload metadata is truncated.")

    cumulative = []
    for index in range(entry_count):
        start = sizes_start + index * size_width
        cumulative.append(
            int.from_bytes(view[start : start + size_width], sys.byteorder)
        )
    totals = (
        tuple(
            cumulative[stream_id * chunk_count + chunk_count - 1]
            for stream_id in range(element_size)
        )
        if chunk_count
        else (0,) * element_size
    )
    if payload_start + sum(totals) != len(view):
        raise ValueError(
            "ZipNN group lengths do not reconstruct the payload size: "
            f"{payload_start + sum(totals)} != {len(view)}."
        )

    raw_stream_bytes = original_bytes // element_size
    per_stream_metadata = chunk_count * (1 + size_width)
    streams = tuple(
        ZipNNStreamSize(
            stream_id=stream_id,
            raw_bytes=raw_stream_bytes,
            payload_bytes=int(total),
            metadata_bytes=per_stream_metadata,
        )
        for stream_id, total in enumerate(totals)
    )
    return ZipNNArchiveLayout(
        original_bytes=original_bytes,
        archive_bytes=archive_bytes,
        header_bytes=header_bytes,
        chunk_bytes=chunk_bytes,
        chunk_count=chunk_count,
        element_size=element_size,
        byte_reorder=int(view[5]),
        bit_reorder=int(view[6]),
        streams=streams,
    )
