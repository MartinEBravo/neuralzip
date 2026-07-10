import unittest

from neuralzip.archive import (
    dumps,
    loads,
    serialized_codebooks_size,
    serialized_size,
    serialized_tensor_size,
)
from neuralzip.payloads import (
    ByteGroupPayload,
    CodebookPayload,
    StreamChunk,
    TensorPayload,
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
        self.assertEqual(codebook.metadata_bytes, serialized_codebooks_size((codebook,)))
        self.assertEqual(tensor.compressed_bytes, serialized_tensor_size(tensor))
        self.assertEqual(len(encoded), serialized_size("neuralzip", (codebook,), (tensor,)))
        decoded = loads(encoded)
        self.assertEqual(decoded["codebooks"], (codebook,))
        self.assertEqual(decoded["tensor_results"][0]["payload"], tensor)


if __name__ == "__main__":
    unittest.main()
