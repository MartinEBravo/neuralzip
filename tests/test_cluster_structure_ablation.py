import unittest

import torch
from torch import nn

from neuralzip.ablations import (
    architecture_js_alignment,
    cluster_structure_ablation,
    layer_signature,
)


class ClusterStructureAblationTests(unittest.TestCase):
    @staticmethod
    def repeated_model():
        model = nn.Module()
        model.layers = nn.ModuleList(
            [nn.Linear(8, 8, bias=True), nn.Linear(8, 8, bias=True)]
        )
        with torch.no_grad():
            for layer in model.layers:
                layer.weight.fill_(1.0)
                layer.bias.fill_(0.125)
        return model.to(torch.bfloat16)

    def test_layer_signature_removes_block_indices(self):
        self.assertEqual(
            layer_signature("encoder.layer.11.attention.query.weight"),
            "encoder.layer.#.attention.query.weight",
        )
        self.assertEqual(
            layer_signature("layer2.1.conv1.weight"),
            "layer2.#.conv1.weight",
        )

    def test_js_clusters_recover_repeated_weight_and_bias_signatures(self):
        result = architecture_js_alignment(
            self.repeated_model(),
            cluster_js_threshold=0.01,
        )
        summary = result["summary"].iloc[0]

        self.assertEqual(summary["tensors"], 4)
        self.assertEqual(summary["architecture_signatures"], 2)
        self.assertEqual(summary["js_clusters"], 2)
        self.assertAlmostEqual(summary["adjusted_rand_index"], 1.0)
        self.assertAlmostEqual(summary["normalized_mutual_information"], 1.0)
        self.assertEqual(len(result["repeated_block_summary"]), 1)
        self.assertEqual(
            result["repeated_block_summary"].iloc[0]["module_path"], "layers"
        )

    def test_compression_ablation_reports_all_partition_controls(self):
        result = cluster_structure_ablation(
            self.repeated_model(),
            cluster_js_threshold=0.01,
            repeats=1,
            max_codebook_sample_bytes=64 << 10,
        )
        strategies = set(result["compression_summary"]["strategy"])

        self.assertEqual(
            strategies,
            {
                "ZipNN local",
                "Global shared",
                "Module type",
                "Architecture signature",
                "Random matched to JS",
                "JS distance",
            },
        )
        self.assertTrue(result["compression_summary"]["roundtrip_ok"].all())


if __name__ == "__main__":
    unittest.main()
