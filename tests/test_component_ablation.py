import unittest

import torch

from neuralzip.ablations import neuralzip_component_ablation


class ComponentAblationTests(unittest.TestCase):
    def test_isolates_fast_and_ratio_modes(self):
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
            ["ZipNN", "Fast only", "Ratio only", "NeuralZip"],
        )
        self.assertEqual(len(runs), 4)
        fast = summary[summary["variant"] == "Fast only"].iloc[0]
        ratio = summary[summary["variant"] == "Ratio only"].iloc[0]
        self.assertEqual(fast["packed_ngram_chunks"], 0)
        self.assertEqual(ratio["precompute_seconds"], 0.0)


if __name__ == "__main__":
    unittest.main()
