import unittest

from neuralzip.archive import dumps, loads
from neuralzip.payloads import (
    ByteGroupPayload,
    CodebookPayload,
    StreamChunk,
    TensorPayload,
)
from neuralzip.serialization import (
    serialized_codebooks_size,
    serialized_size,
    serialized_tensor_size,
)


class ArchiveTests(unittest.TestCase):
    def test_roundtrip(self):
        codebook = CodebookPayload(
            codebook_id=0,
            cluster_id="js_000",
            stream_id=1,
            dtype="bfloat16",
            codec="huff0",
            mode="packed_huff0",
            codebook=b"header",
            ngram_n=3,
            vocab=b"\x01\x02\x03\x00",
        )
        chunk = StreamChunk(
            mode="packed_huff0",
            symbol_count=7,
            payload=b"bits",
            codebook_id=0,
            token_count=2,
            escape_payload=b"abc",
            tail_payload=b"z",
            ngram_n=3,
        )
        tensor = TensorPayload(
            name="layer.weight",
            shape=(7,),
            dtype="bfloat16",
            element_size=2,
            numel=7,
            strategy="neuralzip",
            groups=(
                ByteGroupPayload("byte_group", 7, (StreamChunk("raw", 7, b"1234567"),)),
                ByteGroupPayload("exponent", 7, (chunk,)),
            ),
            byte_reorder=10,
            bit_reorder=1,
            num_buffers=2,
            cluster_id="js_000",
        )
        result = {
            "method": "neuralzip",
            "codebooks": (codebook,),
            "tensor_results": [{"payload": tensor}],
        }

        encoded = dumps(result)
        self.assertEqual(
            codebook.metadata_bytes, serialized_codebooks_size((codebook,))
        )
        self.assertEqual(tensor.compressed_bytes, serialized_tensor_size(tensor))
        self.assertEqual(
            len(encoded), serialized_size("neuralzip", (codebook,), (tensor,))
        )
        decoded = loads(encoded)
        self.assertEqual(decoded["codebooks"], (codebook,))
        self.assertEqual(decoded["tensor_results"][0]["payload"], tensor)

    def test_local_huff0_chunk_roundtrip(self):
        chunk = StreamChunk(
            mode="local_huff0",
            symbol_count=12,
            payload=b"huff0-frame",
        )
        tensor = TensorPayload(
            name="weight",
            shape=(6,),
            dtype="bfloat16",
            element_size=2,
            numel=6,
            strategy="zipnn",
            groups=(ByteGroupPayload("byte_group", 12, (chunk,)),),
        )
        result = {
            "method": "zipnn-local",
            "tensor_results": [{"payload": tensor}],
        }
        restored = loads(dumps(result))["tensor_results"][0]["payload"]
        self.assertEqual(restored, tensor)

    def test_contiguous_raw_memoryview_serializes_without_normalization(self):
        raw = memoryview(b"cached-raw-stream")
        tensor = TensorPayload(
            name="raw.weight",
            shape=(len(raw),),
            dtype="uint8",
            element_size=1,
            numel=len(raw),
            strategy="neuralzip",
            groups=(
                ByteGroupPayload(
                    "byte_group",
                    len(raw),
                    (StreamChunk("raw", len(raw), raw),),
                ),
            ),
        )
        result = {
            "method": "neuralzip",
            "tensor_results": [{"payload": tensor}],
        }

        encoded = dumps(result)
        self.assertEqual(len(encoded), serialized_size("neuralzip", (), (tensor,)))
        restored = loads(encoded)["tensor_results"][0]["payload"]
        self.assertEqual(restored.groups[0].chunks[0].payload, raw.tobytes())


if __name__ == "__main__":
    unittest.main()
