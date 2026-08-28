import unittest

import torch

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

    def test_scalar_and_empty_tensors_round_trip(self):
        codec = ZipNNCodec(threads=2)
        for name, source in (
            ("scalar", torch.tensor(0.5, dtype=torch.bfloat16)),
            ("empty", torch.empty(0, 3, dtype=torch.float16)),
        ):
            with self.subTest(name=name):
                payload = codec.compress_tensor(name, source)
                restored = codec.decompress_tensor(payload)

                self.assertEqual(restored.shape, source.shape)
                self.assertEqual(restored.dtype, source.dtype)
                self.assertTrue(
                    torch.equal(
                        source.reshape(-1).view(torch.uint8),
                        restored.reshape(-1).view(torch.uint8),
                    )
                )


if __name__ == "__main__":
    unittest.main()
