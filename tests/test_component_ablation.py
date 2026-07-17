import unittest

import torch

from neuralzip.ablations import neuralzip_component_ablation


class ComponentAblationTests(unittest.TestCase):
    def test_isolates_sharing_and_packing_with_the_same_setup(self):
        pattern = torch.tensor(
            [1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 0.25],
            dtype=torch.bfloat16,
        )
        model = {
            "encoder.0.weight": pattern.repeat(1 << 12),
            "encoder.1.weight": pattern.repeat(1 << 12),
        }

        runs, summary = neuralzip_component_ablation(
            model,
            repeats=1,
            max_sample_bytes=64 << 10,
        )

        self.assertEqual(
            list(summary["variant"]),
            ["ZipNN", "Sharing only", "Packing only", "NeuralZip"],
        )
        self.assertEqual(len(runs), 4)
        zipnn = summary[summary["variant"] == "ZipNN"].iloc[0]
        sharing = summary[summary["variant"] == "Sharing only"].iloc[0]
        packing = summary[summary["variant"] == "Packing only"].iloc[0]
        self.assertEqual(sharing["packed_ngram_chunks"], 0)
        self.assertEqual(sharing["compressed_bytes"], zipnn["compressed_bytes"])
        self.assertGreater(sharing["precompute_seconds"], 0.0)
        self.assertGreater(packing["precompute_seconds"], 0.0)
        self.assertNotIn("shared_huff0", packing["selected_modes"])


if __name__ == "__main__":
    unittest.main()
