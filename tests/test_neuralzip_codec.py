import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError as error:
    raise unittest.SkipTest("NeuralZip codec tests require torch") from error

from neuralzip import Compressor
from neuralzip.neuralzip_codec import NeuralZipCodec


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
        self.assertAlmostEqual(
            result["compression_seconds"],
            result["codebook_setup_seconds"] + result["encode_seconds"],
        )
        self.assertAlmostEqual(
            result["total_compression_seconds"],
            result["precompute_seconds"] + result["compression_seconds"],
        )

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

    def test_codebooks_are_built_during_setup(self):
        tensor = torch.tensor([1.0, 0.5], dtype=torch.bfloat16).repeat(4096)
        named_tensors = [
            ("encoder.0.weight", tensor),
            ("encoder.1.weight", tensor.clone()),
        ]
        codec = NeuralZipCodec(
            ngram_values=(),
            max_codebook_sample_bytes=64 << 10,
        )

        with self.assertRaises(RuntimeError):
            codec.compress_tensor(named_tensors[0][0], tensor)

        codec.prepare_clusters(named_tensors)
        self.assertGreater(codec.cluster_count, 0)
        with self.assertRaises(RuntimeError):
            codec.compress_tensor(named_tensors[0][0], tensor)

        codec.build_codebooks_and_select_modes(named_tensors)
        self.assertGreater(len(codec.export_codebooks()), 0)
        codec.compress_tensor(named_tensors[0][0], tensor)

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
            packed_source_budget_ratio=0.0,
            max_codebook_sample_bytes=64 << 10,
        ).compress(model)

        self.assertTrue(result["roundtrip_ok"])
        self.assertEqual(result["compressed_bytes"], 66_915)
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
        compressor = Compressor(
            "neuralzip",
            verify=True,
            keep_payloads=True,
            ngram_values=(),
            max_codebook_sample_bytes=64 << 10,
        )
        result = compressor.compress(model)

        self.assertTrue(result["roundtrip_ok"])
        self.assertEqual(result["packed_ngram_chunks"], 0)
        self.assertEqual(result["selection_stats"]["packed_candidates"], 0)
        self.assertEqual(
            result["selection_stats"]["selection_strategy"],
            "source-byte-budget",
        )
        self.assertGreater(result["codebook_setup_seconds"], 0.0)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lazy-model.nzip"
            compressor.save(result, path)
            restored = compressor.decompress_file(path)
        for name, tensor in model.items():
            self.assertTrue(
                torch.equal(
                    restored[name].view(torch.uint8),
                    tensor.contiguous().view(torch.uint8),
                )
            )

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
            packed_source_budget_ratio=1.0,
            max_codebook_sample_bytes=1 << 20,
        ).compress(model)

        self.assertTrue(result["roundtrip_ok"])
        self.assertGreater(result["packed_ngram_chunks"], 0)
        self.assertGreater(result["selection_stats"]["packed_selected"], 0)

    def test_packed_selection_is_deterministic(self):
        pattern = torch.tensor(
            [1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 0.25],
            dtype=torch.bfloat16,
        )
        model = {
            "encoder.0.weight": pattern.repeat(1 << 12),
            "encoder.1.weight": pattern.repeat(1 << 12),
        }

        results = [
            Compressor(
                "neuralzip",
                verify=True,
                max_codebook_sample_bytes=64 << 10,
                packed_source_budget_ratio=1.0,
            ).compress(model)
            for _ in range(3)
        ]

        self.assertEqual(
            len({result["compressed_bytes"] for result in results}),
            1,
        )
        self.assertEqual(results[0]["compressed_bytes"], 12_272)
        self.assertEqual(
            len({repr(result["selected_modes"]) for result in results}),
            1,
        )
        self.assertTrue(all(result["roundtrip_ok"] for result in results))

    def test_shared_codebook_failure_falls_back_to_local_huffman(self):
        codec = NeuralZipCodec(ngram_values=())
        mode = codec._build_candidate(
            bytes([1]) * 4096,
            kind="shared_huffman",
            prior_count=0,
        )
        group = codec._compress_stream_with_mode(
            bytes([2]) * 4096,
            "exponent",
            element_size=2,
            mode=mode,
        )

        self.assertEqual(group.chunks[0].mode, "huffman")
        self.assertEqual(codec._decompress_stream(group), bytes([2]) * 4096)

    def test_tensor_probe_rejects_cluster_codebook_for_outlier(self):
        common = torch.tensor(
            [1.0, 1.0, 1.0, 1.0, 0.5],
            dtype=torch.bfloat16,
        ).repeat(1000)
        outlier = torch.tensor(
            [0.5, 0.5, 0.5, 0.25],
            dtype=torch.bfloat16,
        ).repeat(1000)
        named_tensors = [
            (f"common.{index}", common.clone()) for index in range(9)
        ] + [("outlier", outlier)]
        codec = NeuralZipCodec(
            cluster_map={name: "cluster" for name, _ in named_tensors},
            ngram_values=(),
            max_codebook_sample_bytes=1 << 20,
            max_tensor_probe_bytes=64 << 10,
        )

        codec.prepare_clusters(named_tensors)
        codec.build_codebooks_and_select_modes(named_tensors)

        self.assertEqual(
            codec._mode_by_tensor_stream[("common.0", "bfloat16", 1)]["kind"],
            "shared_huffman",
        )
        self.assertEqual(
            codec._mode_by_tensor_stream[("outlier", "bfloat16", 1)]["kind"],
            "local_huffman",
        )

    def test_transfer_assigns_matching_target_by_js_distance(self):
        pattern = torch.tensor(
            [1.0, 1.0, 0.5, 1.0, 0.5, 1.0],
            dtype=torch.bfloat16,
        ).repeat(4096)
        source = [
            ("source.0.weight", pattern),
            ("source.1.weight", pattern.clone()),
        ]
        target = [("target.weight", pattern.clone())]
        codec = NeuralZipCodec(
            ngram_values=(),
            max_codebook_sample_bytes=64 << 10,
        )
        codec.prepare_clusters(source)
        codec.build_codebooks_and_select_modes(source)
        codec.compress_tensor(*source[0])

        stats = codec.prepare_transfer_target(target)
        payload = codec.compress_tensor(*target[0])
        restored = codec.decompress_tensor(payload)

        self.assertEqual(stats["matched_tensors"], 1)
        self.assertEqual(stats["matched_exponent_pct"], 100.0)
        self.assertTrue(
            any(
                chunk.mode == "shared_huffman"
                for group in payload.groups
                for chunk in group.chunks
            )
        )
        self.assertTrue(
            torch.equal(restored.view(torch.uint8), pattern.view(torch.uint8))
        )

    def test_transfer_falls_back_locally_when_js_distance_is_too_high(self):
        source_tensor = torch.ones(1 << 14, dtype=torch.bfloat16)
        target_tensor = torch.full(
            (1 << 14,),
            2.0**40,
            dtype=torch.bfloat16,
        )
        source = [("source.weight", source_tensor)]
        target = [("target.weight", target_tensor)]
        codec = NeuralZipCodec(
            ngram_values=(),
            cluster_js_threshold=0.01,
            max_codebook_sample_bytes=64 << 10,
        )
        codec.prepare_clusters(source)
        codec.build_codebooks_and_select_modes(source)
        codec.compress_tensor(*source[0])

        stats = codec.prepare_transfer_target(target)
        payload = codec.compress_tensor(*target[0])
        restored = codec.decompress_tensor(payload)

        self.assertEqual(stats["matched_tensors"], 0)
        self.assertFalse(
            any(
                chunk.mode in {"shared_huffman", "shared_huff0", "packed_huff0"}
                for group in payload.groups
                for chunk in group.chunks
            )
        )
        self.assertTrue(
            torch.equal(restored.view(torch.uint8), target_tensor.view(torch.uint8))
        )

    def test_transfer_probe_rejects_bad_packed_vocabulary(self):
        codec = NeuralZipCodec(max_codebook_sample_bytes=64 << 10)
        source = bytes([127, 127, 127, 126]) * 4096
        target = bytes(range(256)) * 64
        packed = codec._build_candidate(
            source,
            kind="packed_huff0",
            prior_count=1,
            ngram_n=4,
        )

        selected, probe = codec._select_transfer_mode(
            target,
            packed,
            element_size=2,
            stream_kind="exponent",
        )

        self.assertIsNot(selected, packed)
        self.assertFalse(probe["uses_source_codebook"])
        self.assertGreaterEqual(probe["transfer_cost"], probe["selected_cost"])


if __name__ == "__main__":
    unittest.main()
