import os
import pickle
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

from neuralzip._setup import (
    NeuralZipSetup,
    load_setup,
    materialize_setup,
    save_setup,
    setup_from_codec,
)
from neuralzip.codec_protocol import (
    encode_tensor_batch,
    prepare_neuralzip_codec,
)
from neuralzip.neuralzip_codec import NeuralZipCodec


class SetupValueTests(unittest.TestCase):
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

    def prepared_codec(self):
        codec = NeuralZipCodec(
            threads=2,
            ngram_values=(),
            max_codebook_sample_bytes=1 << 20,
        )
        report = prepare_neuralzip_codec(codec, self.tensors())
        return codec, report

    def test_setup_is_data_only_and_materializes_independent_codecs(self):
        codec, report = self.prepared_codec()
        setup = setup_from_codec(codec, report)

        self.assertIsInstance(setup, NeuralZipSetup)
        self.assertEqual(setup.setup_report, report)
        for operation in ("encode", "decode", "map_target"):
            self.assertFalse(hasattr(setup, operation))

        first = materialize_setup(setup)
        second = materialize_setup(setup)
        self.assertIsNot(first, second)
        self.assertIsNot(first._mode_by_key, second._mode_by_key)

        first_codebooks = [
            mode["codebook"]
            for mode in first._mode_by_key.values()
            if mode.get("codebook") is not None
        ]
        second_codebooks = [
            mode["codebook"]
            for mode in second._mode_by_key.values()
            if mode.get("codebook") is not None
        ]
        self.assertTrue(first_codebooks)
        self.assertEqual(len(first_codebooks), len(second_codebooks))
        for left, right in zip(first_codebooks, second_codebooks, strict=True):
            self.assertIsNot(left, right)
            self.assertEqual(left.codebook, right.codebook)

        expected = encode_tensor_batch(codec, self.tensors()).payloads
        self.assertEqual(encode_tensor_batch(first, self.tensors()).payloads, expected)
        self.assertEqual(encode_tensor_batch(second, self.tensors()).payloads, expected)

        first.prepare_direct_transfer_target(self.tensors(offset=1.0))
        self.assertTrue(first._transfer_active)
        self.assertFalse(second._transfer_active)
        self.assertFalse(second._transfer_mode_by_tensor_stream)

    def test_capture_drops_transfer_state_without_mutating_live_codec(self):
        codec, _ = self.prepared_codec()
        codec.prepare_direct_transfer_target(self.tensors(offset=1.0))
        source_mode_ids = {
            key: id(mode) for key, mode in codec._mode_by_tensor_stream.items()
        }
        source_codebook_ids = {
            key: id(mode.get("codebook"))
            for key, mode in codec._mode_by_key.items()
            if mode.get("codebook") is not None
        }
        transfer_modes = dict(codec._transfer_mode_by_tensor_stream)
        transfer_clusters = dict(codec._transfer_cluster_by_name)
        decode_codebooks = codec._decode_codebooks

        setup = setup_from_codec(codec)

        self.assertTrue(codec._transfer_active)
        self.assertEqual(codec._transfer_mode_by_tensor_stream, transfer_modes)
        self.assertEqual(codec._transfer_cluster_by_name, transfer_clusters)
        self.assertIs(codec._decode_codebooks, decode_codebooks)
        self.assertEqual(
            {key: id(mode) for key, mode in codec._mode_by_tensor_stream.items()},
            source_mode_ids,
        )
        self.assertEqual(
            {
                key: id(mode.get("codebook"))
                for key, mode in codec._mode_by_key.items()
                if mode.get("codebook") is not None
            },
            source_codebook_ids,
        )

        materialized = materialize_setup(setup)
        self.assertFalse(materialized._transfer_active)
        self.assertFalse(materialized._transfer_mode_by_tensor_stream)
        self.assertFalse(materialized._transfer_cluster_by_name)

    def test_save_is_durable_and_failed_write_preserves_destination_and_codec(self):
        codec, report = self.prepared_codec()
        expected = encode_tensor_batch(codec, self.tensors()).payloads
        mode_ids = {key: id(mode) for key, mode in codec._mode_by_key.items()}

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.setup"
            with mock.patch("neuralzip._setup.os.fsync", wraps=os.fsync) as fsync:
                self.assertEqual(save_setup(codec, path, report), path)
            self.assertGreaterEqual(fsync.call_count, 2)
            self.assertFalse(tuple(path.parent.glob(f".{path.name}.*.tmp")))

            loaded = load_setup(path)
            self.assertIsInstance(loaded, NeuralZipSetup)
            self.assertEqual(loaded.setup_report, report)
            restored = materialize_setup(loaded)
            self.assertEqual(
                encode_tensor_batch(restored, self.tensors()).payloads,
                expected,
            )

            original_file = path.read_bytes()
            with (
                mock.patch(
                    "neuralzip._setup.pickle.dump",
                    side_effect=RuntimeError("simulated write failure"),
                ),
                self.assertRaisesRegex(RuntimeError, "simulated write failure"),
            ):
                save_setup(codec, path, report)
            self.assertEqual(path.read_bytes(), original_file)
            self.assertFalse(tuple(path.parent.glob(f".{path.name}.*.tmp")))

        self.assertEqual(
            {key: id(mode) for key, mode in codec._mode_by_key.items()},
            mode_ids,
        )
        self.assertEqual(encode_tensor_batch(codec, self.tensors()).payloads, expected)

    def test_loads_historical_raw_codec_and_plan_wrapper(self):
        codec, report = self.prepared_codec()
        setup = setup_from_codec(codec, report)
        frozen_codec = pickle.loads(setup._codec_state)

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            raw_path = directory / "raw-codec.pkl"
            with raw_path.open("wb") as handle:
                pickle.dump(frozen_codec, handle, protocol=5)
            raw_setup = load_setup(raw_path)
            self.assertIsNone(raw_setup.setup_report)
            self.assertEqual(
                encode_tensor_batch(
                    materialize_setup(raw_setup), self.tensors()
                ).payloads,
                encode_tensor_batch(codec, self.tensors()).payloads,
            )

            legacy_module = types.ModuleType("neuralzip.plan")
            legacy_type = type("NeuralZipPlan", (), {})
            legacy_type.__module__ = legacy_module.__name__
            legacy_module.NeuralZipPlan = legacy_type
            sys.modules[legacy_module.__name__] = legacy_module
            try:
                wrapper = legacy_type()
                wrapper.codec = frozen_codec
                wrapper.setup_report = report
                plan_path = directory / "plan.pkl"
                with plan_path.open("wb") as handle:
                    pickle.dump(wrapper, handle, protocol=5)
            finally:
                del sys.modules[legacy_module.__name__]

            migrated = load_setup(plan_path)
            self.assertEqual(migrated.setup_report, report)
            self.assertEqual(
                encode_tensor_batch(
                    materialize_setup(migrated), self.tensors()
                ).payloads,
                encode_tensor_batch(codec, self.tensors()).payloads,
            )


if __name__ == "__main__":
    unittest.main()
