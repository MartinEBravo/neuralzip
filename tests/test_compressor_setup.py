import pickle
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import torch
from torch import nn

import neuralzip.compressor as compressor_module
from neuralzip import (
    Compressor,
    NeuralZipModel,
    NeuralZipRuntime,
    export_model_for_gpu,
    huff0,
)
from neuralzip._setup import (
    LEGACY_SETUP_MARKER,
    load_setup,
    materialize_setup,
    setup_from_codec,
)
from neuralzip.gpu_inference import (
    NeuralZipRuntime as GPUInferenceRuntime,
)
from neuralzip.gpu_inference import (
    export_model_for_gpu as gpu_export_model,
)
from neuralzip.neuralzip_codec import NeuralZipCodec


class CompressorSetupTests(unittest.TestCase):
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
        compressor = Compressor(
            "neuralzip",
            threads=2,
            max_codebook_sample_bytes=1 << 20,
        )
        compressor.fit(source)

        encoded = compressor.encode(source)
        self.assertNotIn("local_huff0", compressor.codec.selected_modes)
        self.assertTrue(
            all(
                chunk.mode != "local_huff0"
                for payload in encoded.payloads
                for group in payload.groups
                for chunk in group.chunks
            )
        )

        restored = compressor.decode(encoded.payloads)
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
        transfer = compressor.map_target(target)
        self.assertEqual(transfer["matched_tensors"], len(target))
        transferred = compressor.encode(target)
        self.assertTrue(
            all(
                chunk.mode != "local_huff0"
                for payload in transferred.payloads
                for group in payload.groups
                for chunk in group.chunks
            )
        )
        transferred_restored = compressor.decode(transferred.payloads)
        for name, tensor in target:
            self.assertTrue(
                torch.equal(
                    transferred_restored[name].view(torch.uint8),
                    tensor.view(torch.uint8),
                )
            )

    def test_prepare_inference_uses_public_model_facade(self):
        compressor = Compressor("neuralzip")
        compressor.fit(dict(self.tensors()))
        model = nn.Linear(2, 2)

        with mock.patch.object(
            NeuralZipModel,
            "from_model",
            return_value=model,
        ) as from_model:
            result = compressor.prepare_inference(
                model,
                device="cuda:1",
                cpu_offload=True,
                cpu_offload_blocks=3,
                pin_memory=False,
            )

        self.assertIs(result, model)
        from_model.assert_called_once_with(
            model,
            compressor,
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

    def test_fit_and_setup_are_neuralzip_only(self):
        source = dict(self.tensors())
        with self.assertRaisesRegex(ValueError, "only for method='neuralzip'"):
            Compressor("zipnn").fit(source)
        with self.assertRaisesRegex(ValueError, "only for method='neuralzip'"):
            Compressor("zipnn-local").save_setup("unused.setup")
        with self.assertRaisesRegex(ValueError, "only for method='neuralzip'"):
            Compressor("zipnn", setup="missing.setup")

    def test_save_setup_requires_explicit_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unused.setup"
            with self.assertRaisesRegex(RuntimeError, "Call fit"):
                Compressor("neuralzip").save_setup(path)
            self.assertFalse(path.exists())

    def test_one_shot_refits_while_explicit_setup_is_reused(self):
        source = dict(self.tensors())
        one_shot = Compressor("neuralzip", verify=False)
        with mock.patch(
            "neuralzip.compressor.prepare_neuralzip_codec",
            wraps=compressor_module.prepare_neuralzip_codec,
        ) as prepare:
            first = one_shot.compress(source)
            second = one_shot.compress(source)
        self.assertEqual(prepare.call_count, 2)
        self.assertFalse(first["setup_reused"])
        self.assertFalse(second["setup_reused"])

        reusable = Compressor("neuralzip", verify=False)
        reusable.fit(source)
        with mock.patch(
            "neuralzip.compressor.prepare_neuralzip_codec",
            side_effect=AssertionError("setup was repeated"),
        ):
            result = reusable.compress(source)
        self.assertTrue(result["setup_reused"])
        self.assertEqual(result["setup_seconds"], 0.0)
        self.assertEqual(result["mapped_tensors"], len(source))

    def test_saved_setup_rebuilds_native_codebooks(self):
        source = dict(self.tensors())
        compressor = Compressor(
            "neuralzip",
            threads=2,
            max_codebook_sample_bytes=1 << 20,
        )
        compressor.fit(source)
        expected_digest = compressor.setup_digest

        with (
            self.subTest("save-load"),
            tempfile.TemporaryDirectory() as directory,
        ):
            path = Path(directory) / "compressor.pkl"
            compressor.save_setup(path)
            loaded = Compressor("neuralzip", setup=path)

            self.assertEqual(loaded.setup_digest, expected_digest)
            self.assertEqual(
                loaded.encode(source).payloads,
                compressor.encode(source).payloads,
            )

    def test_save_rebuilds_from_headers_without_recollecting_samples(self):
        source = dict(self.tensors())
        compressor = Compressor(
            "neuralzip",
            threads=2,
            max_codebook_sample_bytes=1 << 20,
        )
        compressor.fit(source)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "header-only-compressor.pkl"
            with mock.patch.object(
                NeuralZipCodec,
                "_collect_codebook_samples",
                side_effect=AssertionError("setup samples were recollected"),
            ):
                compressor.save_setup(path)
            loaded = Compressor("neuralzip", setup=path)

        self.assertEqual(loaded.setup_digest, compressor.setup_digest)
        self.assertEqual(
            loaded.encode(source).payloads, compressor.encode(source).payloads
        )

    def test_load_rebuilds_real_v1_codebook_markers(self):
        source = dict(self.tensors())
        compressor = Compressor(
            "neuralzip",
            threads=2,
            ngram_values=(),
            max_codebook_sample_bytes=1 << 20,
        )
        compressor.fit(source)
        expected_digest = compressor.setup_digest
        expected_payloads = compressor.encode(source).payloads
        sample_parts, _, _, _, _ = compressor.codec._collect_codebook_samples(
            tuple(source.items())
        )
        for key, mode in compressor.codec._mode_by_key.items():
            codebook = mode.get("codebook")
            if codebook is None:
                continue
            tokens = b"".join(sample_parts[key])
            prior_count = (
                compressor.codec.exponent_prior_count
                if key[2] == 1
                else compressor.codec.prior_count
            )
            legacy = huff0.build_codebook(tokens, prior_count=prior_count)
            self.assertEqual(legacy.codebook, codebook.codebook)
            mode["codebook"] = (
                LEGACY_SETUP_MARKER,
                tokens,
                prior_count,
                legacy.codebook,
            )
        compressor.codec._decode_codebooks = {}

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v1-compressor.pkl"
            with path.open("wb") as handle:
                pickle.dump(compressor.codec, handle, protocol=5)
            loaded = Compressor("neuralzip", setup=path)

        self.assertEqual(loaded.setup_digest, expected_digest)
        self.assertEqual(loaded.encode(source).payloads, expected_payloads)

    def test_load_rejects_marker_payload_codebook_mismatch(self):
        source = dict(self.tensors())
        compressor = Compressor(
            "neuralzip",
            threads=2,
            ngram_values=(),
            max_codebook_sample_bytes=1 << 20,
        )
        compressor.fit(source)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "compressor.pkl"
            compressor.save_setup(path)
            stored_setup = load_setup(path)
            codec = pickle.loads(stored_setup._codec_state)

            payloads = list(codec._codebook_payloads)
            wrong = huff0.build_codebook(
                (b"x" * 8192) + bytes(range(256)), prior_count=1
            )
            self.assertNotEqual(payloads[0].codebook, wrong.codebook)
            payloads[0] = replace(payloads[0], codebook=wrong.codebook)
            codec._codebook_payloads = payloads
            corrupted_setup = replace(
                stored_setup,
                _codec_state=pickle.dumps(codec, protocol=5),
            )
            with path.open("wb") as handle:
                pickle.dump(corrupted_setup, handle, protocol=5)

            with self.assertRaisesRegex(ValueError, "disagrees with its payload"):
                Compressor("neuralzip", setup=path)

    def test_failed_save_leaves_live_setup_usable(self):
        source = {
            "layers.0.weight": torch.ones(32768, dtype=torch.bfloat16),
            "layers.1.weight": torch.arange(32768, dtype=torch.bfloat16),
        }
        compressor = Compressor(
            "neuralzip",
            threads=2,
            max_codebook_sample_bytes=1 << 20,
            cluster_js_threshold=0.0,
        )
        compressor.fit(source)
        self.assertGreaterEqual(compressor.codec.cluster_count, 2)
        expected_digest = compressor.setup_digest
        expected_payloads = compressor.encode(source).payloads

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "incomplete.pkl"
            with (
                mock.patch(
                    "neuralzip._setup.pickle.dump",
                    side_effect=RuntimeError("simulated write failure"),
                ),
                self.assertRaisesRegex(RuntimeError, "simulated write failure"),
            ):
                compressor.save_setup(path)
            self.assertFalse(path.exists())

        self.assertEqual(compressor.setup_digest, expected_digest)
        self.assertEqual(compressor.encode(source).payloads, expected_payloads)

    def test_load_rejects_legacy_transposed_setup(self):
        codec = NeuralZipCodec()
        codec.exponent_layout = "transpose2d"

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy-transposed.pkl"
            with path.open("wb") as handle:
                pickle.dump(codec, handle, protocol=5)

            with self.assertRaisesRegex(ValueError, "transposed exponent streams"):
                Compressor("neuralzip", setup=path)

    def test_load_rejects_setup_with_target_local_huffman(self):
        codec = NeuralZipCodec()
        codec._mode_by_key[("cluster", "bfloat16", 1)] = {"kind": "local_huff0"}

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy-local.pkl"
            with path.open("wb") as handle:
                pickle.dump(codec, handle, protocol=5)

            with self.assertRaisesRegex(ValueError, "Refit the setup"):
                Compressor("neuralzip", setup=path)

    def test_load_rejects_zero_prior_transfer_setup(self):
        codec = NeuralZipCodec()
        codec.exponent_prior_count = 0

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "zero-prior.pkl"
            with path.open("wb") as handle:
                pickle.dump(codec, handle, protocol=5)

            with self.assertRaisesRegex(ValueError, "positive Huffman priors"):
                Compressor("neuralzip", setup=path)

    def test_load_migrates_legacy_original_setup(self):
        source = dict(self.tensors())
        compressor = Compressor(
            "neuralzip",
            threads=2,
            max_codebook_sample_bytes=1 << 20,
        )
        setup = compressor.fit(source)
        expected_digest = compressor.setup_digest
        expected_payloads = compressor.encode(source).payloads
        legacy_codec = materialize_setup(setup)
        legacy_codec.exponent_layout = "original"
        legacy_setup = setup_from_codec(legacy_codec)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy-original.pkl"
            compressor.save_setup(path, setup=legacy_setup)
            loaded = Compressor("neuralzip", setup=path)

        self.assertFalse(hasattr(loaded.codec, "exponent_layout"))
        self.assertEqual(loaded.setup_digest, expected_digest)
        self.assertEqual(loaded.encode(source).payloads, expected_payloads)


if __name__ == "__main__":
    unittest.main()
