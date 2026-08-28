import pickle
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import torch
from torch import nn

from neuralzip import (
    NeuralZipModel,
    NeuralZipPlan,
    NeuralZipRuntime,
    export_model_for_gpu,
    huff0,
)
from neuralzip.gpu_inference import (
    NeuralZipRuntime as GPUInferenceRuntime,
)
from neuralzip.gpu_inference import (
    export_model_for_gpu as gpu_export_model,
)
from neuralzip.neuralzip_codec import NeuralZipCodec
from neuralzip.plan import LEGACY_PLAN_MARKER


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
        self.assertNotIn("local_huff0", plan.codec.selected_modes)
        self.assertTrue(
            all(
                chunk.mode != "local_huff0"
                for payload in encoded.payloads
                for group in payload.groups
                for chunk in group.chunks
            )
        )

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
        transfer = plan.map_target(target)
        self.assertEqual(transfer["matched_tensors"], len(target))
        transferred = plan.encode(target)
        self.assertTrue(
            all(
                chunk.mode != "local_huff0"
                for payload in transferred.payloads
                for group in payload.groups
                for chunk in group.chunks
            )
        )
        transferred_restored = plan.decode(transferred.payloads)
        for name, tensor in target:
            self.assertTrue(
                torch.equal(
                    transferred_restored[name].view(torch.uint8),
                    tensor.view(torch.uint8),
                )
            )

    def test_prepare_inference_uses_public_model_facade(self):
        plan = NeuralZipPlan(codec=NeuralZipCodec())
        model = nn.Linear(2, 2)

        with mock.patch.object(
            NeuralZipModel,
            "from_model",
            return_value=model,
        ) as from_model:
            result = plan.prepare_inference(
                model,
                device="cuda:1",
                cpu_offload=True,
                cpu_offload_blocks=3,
                pin_memory=False,
            )

        self.assertIs(result, model)
        from_model.assert_called_once_with(
            model,
            plan,
            adapter=None,
            include_embeddings=False,
            device="cuda:1",
            block_devices=None,
            cpu_offload=True,
            cpu_offload_blocks=3,
            pin_memory=False,
        )

    def test_gpu_export_stages_are_available_from_public_api(self):
        self.assertIs(NeuralZipRuntime, GPUInferenceRuntime)
        self.assertIs(export_model_for_gpu, gpu_export_model)

    def test_saved_plan_rebuilds_native_codebooks(self):
        source = dict(self.tensors())
        plan = NeuralZipPlan.fit(
            source,
            threads=2,
            max_codebook_sample_bytes=1 << 20,
        )
        expected_digest = plan.digest

        with (
            self.subTest("save-load"),
            tempfile.TemporaryDirectory() as directory,
        ):
            path = Path(directory) / "plan.pkl"
            plan.save(path, source)
            loaded = NeuralZipPlan.load(path)

            self.assertEqual(loaded.digest, expected_digest)
            self.assertEqual(
                loaded.encode(source).payloads,
                plan.encode(source).payloads,
            )

    def test_save_rebuilds_from_headers_without_recollecting_samples(self):
        source = dict(self.tensors())
        plan = NeuralZipPlan.fit(
            source,
            threads=2,
            max_codebook_sample_bytes=1 << 20,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "header-only-plan.pkl"
            with mock.patch.object(
                NeuralZipCodec,
                "_collect_codebook_samples",
                side_effect=AssertionError("setup samples were recollected"),
            ):
                plan.save(path, source)
            loaded = NeuralZipPlan.load(path)

        self.assertEqual(loaded.digest, plan.digest)
        self.assertEqual(loaded.encode(source).payloads, plan.encode(source).payloads)

    def test_load_rebuilds_real_v1_codebook_markers(self):
        source = dict(self.tensors())
        plan = NeuralZipPlan.fit(
            source,
            threads=2,
            ngram_values=(),
            max_codebook_sample_bytes=1 << 20,
        )
        expected_digest = plan.digest
        expected_payloads = plan.encode(source).payloads
        sample_parts, _, _, _, _ = plan.codec._collect_codebook_samples(
            tuple(source.items())
        )
        for key, mode in plan.codec._mode_by_key.items():
            codebook = mode.get("codebook")
            if codebook is None:
                continue
            tokens = b"".join(sample_parts[key])
            prior_count = (
                plan.codec.exponent_prior_count
                if key[2] == 1
                else plan.codec.prior_count
            )
            legacy = huff0.build_codebook(tokens, prior_count=prior_count)
            self.assertEqual(legacy.codebook, codebook.codebook)
            mode["codebook"] = (
                LEGACY_PLAN_MARKER,
                tokens,
                prior_count,
                legacy.codebook,
            )
        plan.codec._decode_codebooks = {}

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v1-plan.pkl"
            with path.open("wb") as handle:
                pickle.dump(plan.codec, handle, protocol=5)
            loaded = NeuralZipPlan.load(path)

        self.assertEqual(loaded.digest, expected_digest)
        self.assertEqual(loaded.encode(source).payloads, expected_payloads)

    def test_load_rejects_marker_payload_codebook_mismatch(self):
        source = dict(self.tensors())
        plan = NeuralZipPlan.fit(
            source,
            threads=2,
            ngram_values=(),
            max_codebook_sample_bytes=1 << 20,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.pkl"
            plan.save(path, source)
            with path.open("rb") as handle:
                codec = pickle.load(handle)

            payloads = list(codec._codebook_payloads)
            wrong = huff0.build_codebook(
                (b"x" * 8192) + bytes(range(256)), prior_count=1
            )
            self.assertNotEqual(payloads[0].codebook, wrong.codebook)
            payloads[0] = replace(payloads[0], codebook=wrong.codebook)
            codec._codebook_payloads = payloads
            with path.open("wb") as handle:
                pickle.dump(codec, handle, protocol=5)

            with self.assertRaisesRegex(ValueError, "disagrees with its payload"):
                NeuralZipPlan.load(path)

    def test_failed_save_leaves_live_plan_usable(self):
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

    def test_load_rejects_legacy_transposed_plan(self):
        codec = NeuralZipCodec()
        codec.exponent_layout = "transpose2d"

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy-transposed.pkl"
            with path.open("wb") as handle:
                pickle.dump(codec, handle, protocol=5)

            with self.assertRaisesRegex(ValueError, "transposed exponent streams"):
                NeuralZipPlan.load(path)

    def test_load_rejects_plan_with_target_local_huffman(self):
        codec = NeuralZipCodec()
        codec._mode_by_key[("cluster", "bfloat16", 1)] = {"kind": "local_huff0"}

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy-local.pkl"
            with path.open("wb") as handle:
                pickle.dump(codec, handle, protocol=5)

            with self.assertRaisesRegex(ValueError, "Refit the setup"):
                NeuralZipPlan.load(path)

    def test_load_rejects_zero_prior_transfer_plan(self):
        codec = NeuralZipCodec()
        codec.exponent_prior_count = 0

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "zero-prior.pkl"
            with path.open("wb") as handle:
                pickle.dump(codec, handle, protocol=5)

            with self.assertRaisesRegex(ValueError, "positive Huffman priors"):
                NeuralZipPlan.load(path)

    def test_load_migrates_legacy_original_plan(self):
        source = dict(self.tensors())
        plan = NeuralZipPlan.fit(
            source,
            threads=2,
            max_codebook_sample_bytes=1 << 20,
        )
        expected_digest = plan.digest
        expected_payloads = plan.encode(source).payloads
        plan.codec.exponent_layout = "original"

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy-original.pkl"
            plan.save(path, source)
            loaded = NeuralZipPlan.load(path)

        self.assertFalse(hasattr(loaded.codec, "exponent_layout"))
        self.assertEqual(loaded.digest, expected_digest)
        self.assertEqual(loaded.encode(source).payloads, expected_payloads)


if __name__ == "__main__":
    unittest.main()
