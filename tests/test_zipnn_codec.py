import unittest

import torch

from neuralzip.neuralzip_codec import NeuralZipCodec
from neuralzip.zipnn_codec import ZipNNCodec


class ZipNNCodecTests(unittest.TestCase):
    def test_native_huff0_roundtrip_for_supported_dtypes(self):
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            with self.subTest(dtype=dtype):
                source = torch.randn(257, 263, dtype=dtype)
                codec = ZipNNCodec(threads=4)
                payload = codec.compress_tensor("weight", source)
                restored = codec.decompress_tensor(payload)

                self.assertTrue(
                    torch.equal(source.view(torch.uint8), restored.view(torch.uint8))
                )
                self.assertGreater(payload.huffman_chunks, 0)

    def test_neuralzip_inherits_shared_batch_executor(self):
        self.assertNotIn("_encode_huff0_chunks", NeuralZipCodec.__dict__)
        self.assertIs(
            NeuralZipCodec._encode_huff0_chunks,
            ZipNNCodec._encode_huff0_chunks,
        )


if __name__ == "__main__":
    unittest.main()
