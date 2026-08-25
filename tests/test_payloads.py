import unittest

from neuralzip.payloads import (
    ByteGroupPayload,
    StreamChunk,
)
from neuralzip.serialization import (
    PACKED_CHUNK_METADATA_BYTES,
    RAW_CHUNK_METADATA_BYTES,
    SHARED_CHUNK_METADATA_BYTES,
    serialized_group_size,
    stream_chunk_metadata_bytes,
)


class StreamChunkMetadataTests(unittest.TestCase):
    def test_metadata_and_serialized_size_for_every_archive_mode(self):
        cases = (
            (StreamChunk("raw", 3, b"raw"), RAW_CHUNK_METADATA_BYTES),
            (StreamChunk("zero", 3), RAW_CHUNK_METADATA_BYTES),
            (
                StreamChunk("local_huff0", 3, b"local"),
                RAW_CHUNK_METADATA_BYTES,
            ),
            (
                StreamChunk(
                    "shared_huff0",
                    3,
                    b"bits",
                    codebook_id=7,
                ),
                SHARED_CHUNK_METADATA_BYTES,
            ),
            (
                StreamChunk(
                    "packed_huff0",
                    7,
                    b"bits",
                    codebook_id=7,
                    token_count=2,
                    escape_payload=b"escape",
                    tail_payload=b"tail",
                    ngram_n=3,
                ),
                PACKED_CHUNK_METADATA_BYTES,
            ),
        )

        for chunk, expected_metadata_bytes in cases:
            with self.subTest(mode=chunk.mode):
                self.assertEqual(chunk.metadata_bytes, expected_metadata_bytes)
                self.assertEqual(
                    stream_chunk_metadata_bytes(chunk.mode),
                    expected_metadata_bytes,
                )
                group = ByteGroupPayload("stream", chunk.symbol_count, (chunk,))
                self.assertEqual(serialized_group_size(group), group.compressed_bytes)

    def test_metadata_constants_preserve_archive_wire_sizes(self):
        self.assertEqual(RAW_CHUNK_METADATA_BYTES, 17)
        self.assertEqual(SHARED_CHUNK_METADATA_BYTES, 25)
        self.assertEqual(PACKED_CHUNK_METADATA_BYTES, 50)


if __name__ == "__main__":
    unittest.main()
