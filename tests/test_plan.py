import unittest

import torch

from neuralzip import NeuralZipPlan


class NeuralZipPlanTests(unittest.TestCase):
    @staticmethod
    def tensors(offset=0.0):
        pattern = torch.tensor(
            [1.0, 2.0, 1.0, 2.0],
            dtype=torch.bfloat16,
        ).repeat(8192)
        return (
            ("layers.0.weight", (pattern + offset).contiguous()),
            ("layers.1.weight", (pattern + offset).contiguous()),
        )

    def test_fit_encode_transfer_and_roundtrip(self):
        source = dict(self.tensors())
        plan = NeuralZipPlan.fit(
            source,
            threads=2,
            max_codebook_sample_bytes=1 << 20,
        )

        encoded = plan.encode(source)
        restored = plan.decode(encoded.payloads)
        self.assertTrue(encoded.used_batch)
        self.assertEqual(tuple(restored), tuple(source))
        for name, tensor in source.items():
            self.assertTrue(
                torch.equal(
                    restored[name].view(torch.uint8),
                    tensor.view(torch.uint8),
                )
            )

        target = self.tensors(offset=1.0)
        transfer = plan.map_target(target, strategy="structural")
        self.assertEqual(transfer["matched_tensors"], len(target))
        transferred = plan.encode(target)
        transferred_restored = plan.decode(transferred.payloads)
        for name, tensor in target:
            self.assertTrue(
                torch.equal(
                    transferred_restored[name].view(torch.uint8),
                    tensor.view(torch.uint8),
                )
            )

    def test_saved_plan_rebuilds_native_codebooks(self):
        source = dict(self.tensors())
        plan = NeuralZipPlan.fit(
            source,
            threads=2,
            max_codebook_sample_bytes=1 << 20,
        )
        expected_digest = plan.digest

        with self.subTest("save-load"):
            import tempfile
            from pathlib import Path

            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "plan.pkl"
                plan.save(path, source)
                loaded = NeuralZipPlan.load(path)

                self.assertEqual(loaded.digest, expected_digest)
                self.assertEqual(
                    loaded.encode(source).payloads,
                    plan.encode(source).payloads,
                )

    def test_failed_save_leaves_live_plan_usable(self):
        import tempfile
        from pathlib import Path

        source = {
            "layers.0.weight": torch.ones(32768, dtype=torch.bfloat16),
            "layers.1.weight": torch.arange(32768, dtype=torch.bfloat16),
        }
        plan = NeuralZipPlan.fit(
            source,
            threads=2,
            max_codebook_sample_bytes=1 << 20,
            cluster_js_threshold=0.0,
        )
        self.assertGreaterEqual(plan.codec.cluster_count, 2)
        expected_digest = plan.digest
        expected_payloads = plan.encode(source).payloads

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "incomplete.pkl"
            with self.assertRaises(KeyError):
                plan.save(path, {"layers.0.weight": source["layers.0.weight"]})

        self.assertEqual(plan.digest, expected_digest)
        self.assertEqual(plan.encode(source).payloads, expected_payloads)

if __name__ == "__main__":
    unittest.main()
