import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError as error:
    raise unittest.SkipTest("NeuralZip codec tests require torch") from error

from neuralzip import Compressor


class NeuralZipCodecTests(unittest.TestCase):
    def test_full_checkpoint_roundtrip_and_archive(self):
        repeated = torch.tensor(
            [[1.0, 1.0, 0.5, 1.0], [1.0, 0.5, 1.0, 1.0]],
            dtype=torch.bfloat16,
        )
        model = {
            "encoder.0.weight": repeated.repeat(128, 128),
            "encoder.1.weight": repeated.repeat(128, 128),
        }
        compressor = Compressor(
            "neuralzip",
            verify=True,
            keep_payloads=True,
            cluster_js_threshold=0.05,
            max_codebook_sample_bytes=64 << 10,
        )
        result = compressor.compress(model)
        self.assertTrue(result["roundtrip_ok"])
        self.assertEqual(result["method"], "neuralzip")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.nzip"
            compressor.save(result, path)
            self.assertEqual(path.stat().st_size, result["compressed_bytes"])
            restored = compressor.decompress_file(path)

        self.assertEqual(set(restored), set(model))
        for name, tensor in model.items():
            self.assertTrue(
                torch.equal(
                    restored[name].view(torch.uint8),
                    tensor.contiguous().view(torch.uint8),
                )
            )

    def test_zero_packed_budget_uses_canonical_shared_huffman(self):
        repeated = torch.tensor(
            [[1.0, 1.0, 0.5, 1.0], [1.0, 0.5, 1.0, 1.0]],
            dtype=torch.bfloat16,
        )
        model = {
            "encoder.0.weight": repeated.repeat(128, 128),
            "encoder.1.weight": repeated.repeat(128, 128),
        }
        result = Compressor(
            "neuralzip",
            verify=True,
            packed_time_budget_ratio=0.0,
            max_codebook_sample_bytes=64 << 10,
        ).compress(model)

        self.assertTrue(result["roundtrip_ok"])
        self.assertEqual(result["packed_ngram_chunks"], 0)
        self.assertGreater(result["selected_modes"].get("shared_huffman", 0), 0)
        self.assertEqual(result["selection_stats"]["packed_selected"], 0)

    def test_empty_ngram_values_disable_packed_candidates(self):
        repeated = torch.tensor(
            [[1.0, 1.0, 0.5, 1.0], [1.0, 0.5, 1.0, 1.0]],
            dtype=torch.bfloat16,
        )
        model = {
            "encoder.0.weight": repeated.repeat(128, 128),
            "encoder.1.weight": repeated.repeat(128, 128),
        }
        result = Compressor(
            "neuralzip",
            verify=True,
            ngram_values=(),
            max_codebook_sample_bytes=64 << 10,
        ).compress(model)

        self.assertTrue(result["roundtrip_ok"])
        self.assertEqual(result["packed_ngram_chunks"], 0)
        self.assertEqual(result["selection_stats"]["packed_candidates"], 0)

    def test_large_packed_budget_can_select_ngram_mode(self):
        pattern = torch.tensor(
            [1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 0.25],
            dtype=torch.bfloat16,
        )
        model = {
            "encoder.0.weight": pattern.repeat(1 << 15),
            "encoder.1.weight": pattern.repeat(1 << 15),
        }
        result = Compressor(
            "neuralzip",
            verify=True,
            packed_time_budget_ratio=1_000.0,
            max_codebook_sample_bytes=1 << 20,
        ).compress(model)

        self.assertTrue(result["roundtrip_ok"])
        self.assertGreater(result["packed_ngram_chunks"], 0)
        self.assertGreater(result["selection_stats"]["packed_selected"], 0)


if __name__ == "__main__":
    unittest.main()
