import tempfile
import unittest
from unittest import mock
from pathlib import Path

try:
    import torch
except ImportError as error:
    raise unittest.SkipTest("NeuralZip codec tests require torch") from error

from neuralzip import Compressor
from neuralzip import huff0
from neuralzip.archive import dumps
from neuralzip.neuralzip_codec import NeuralZipCodec
from neuralzip.packed_exponents import tokenize_ngrams, topk_vocab
from neuralzip.payloads import CodebookPayload
from neuralzip.zipnn_codec import (
    _zipnn_dtype_config,
    _zipnn_reordered_bytes,
)


class NeuralZipCodecTests(unittest.TestCase):
    @staticmethod
    def _prepared_direct_transfer_codec():
        source_tensor = torch.tensor(
            [1.0, 2.0, 1.0, 2.0], dtype=torch.bfloat16
        ).repeat(16384)
        target_tensor = torch.tensor(
            [1.0, 2.0, 1.0, 1.0], dtype=torch.bfloat16
        ).repeat(8193)
        source = [("family.weight", source_tensor)]
        target = [("family.weight", target_tensor)]
        codec = NeuralZipCodec(
            threads=2,
            ngram_values=(2,),
            packed_source_budget_ratio=1.0,
            max_codebook_sample_bytes=1 << 20,
        )
        codec.prepare_clusters(source)
        codec.build_codebooks_and_select_modes(source)
        return codec, source, target

    def test_direct_transfer_skips_histogram_js_and_probe(self):
        codec, _, target = self._prepared_direct_transfer_codec()
        source_codebooks = codec.export_codebooks()
        source_mode_ids = {
            key: id(mode) for key, mode in codec._mode_by_tensor_stream.items()
        }

        with (
            mock.patch(
                "neuralzip.neuralzip_codec._exponent_distribution",
                side_effect=AssertionError("target histogram was computed"),
            ),
            mock.patch(
                "neuralzip.neuralzip_codec.jensen_shannon_distance",
                side_effect=AssertionError("target JS matching was computed"),
            ),
            mock.patch.object(
                codec,
                "_select_transfer_mode",
                side_effect=AssertionError("target mode probing was run"),
            ),
            mock.patch.object(
                codec,
                "_evaluate_modes",
                side_effect=AssertionError("target modes were evaluated"),
            ),
        ):
            stats = codec.prepare_direct_transfer_target(target)
            payload = codec.compress_tensors(target)[0]

        restored = codec.decompress_tensor(payload)

        self.assertFalse(stats["target_histograms_computed"])
        self.assertFalse(stats["target_js_matching"])
        self.assertFalse(stats["target_mode_probing"])
        self.assertEqual(stats["matched_tensor_pct"], 100.0)
        self.assertEqual(stats["missing_streams"], 0)
        self.assertEqual(codec.export_codebooks(), source_codebooks)
        self.assertEqual(
            {key: id(mode) for key, mode in codec._mode_by_tensor_stream.items()},
            source_mode_ids,
        )
        self.assertTrue(
            torch.equal(
                restored.view(torch.uint8),
                target[0][1].view(torch.uint8),
            )
        )

    def test_direct_native_path_matches_python_stream_path(self):
        codec, _, target = self._prepared_direct_transfer_codec()
        codec.prepare_direct_transfer_target(target)

        # A one-dimensional tensor is unchanged by transpose2d, but selecting
        # that layout deliberately exercises the legacy Python stream path.
        codec.exponent_layout = "transpose2d"
        legacy = codec.compress_tensors(target)[0]
        codec.exponent_layout = "original"

        with (
            mock.patch.object(
                huff0,
                "compress_zipnn_mixed_many",
                wraps=huff0.compress_zipnn_mixed_many,
            ) as mixed_many,
            mock.patch.object(
                huff0,
                "compress_zipnn_mixed",
                wraps=huff0.compress_zipnn_mixed,
            ) as mixed_single,
            mock.patch(
                "neuralzip.neuralzip_codec._zipnn_reordered_bytes",
                side_effect=AssertionError("direct encode constructed reordered bytes"),
            ),
            mock.patch(
                "neuralzip.neuralzip_codec.tokenize_ngrams",
                side_effect=AssertionError("direct encode used Python tokenization"),
            ),
            mock.patch.object(
                codec,
                "_compress_stream_with_mode",
                side_effect=AssertionError("direct encode used the legacy stream path"),
            ),
        ):
            direct = codec.compress_tensors(target)[0]

        mixed_many.assert_called_once()
        mixed_single.assert_not_called()
        self.assertEqual(direct, legacy)
        self.assertTrue(torch.equal(codec.decompress_tensor(direct), target[0][1]))

    def test_direct_transfer_requires_complete_exact_name_mapping(self):
        codec, _, target = self._prepared_direct_transfer_codec()
        unknown = [("other.weight", target[0][1])]

        with self.assertRaisesRegex(ValueError, "missing other.weight"):
            codec.prepare_direct_transfer_target(unknown)

        self.assertFalse(codec._transfer_active)

    def test_mode_cost_uses_local_fallback_after_shared_rejection(self):
        codec = NeuralZipCodec(
            threads=2,
            compression_threshold=1.0,
            ngram_values=(),
        )
        source = bytes([3, 3, 3, 3, 4, 3]) * 4096
        mode = {
            "kind": "shared_huff0",
            "codebook": huff0.build_codebook(source, prior_count=1),
            "frame_4x": True,
        }
        local = codec._compress_local_huff0_chunks((source,))[0]

        with mock.patch.object(codec, "_encode_chunk", return_value=None):
            estimated = codec._mode_cost(source, mode, element_size=2)

        self.assertEqual(estimated, local.compressed_bytes)
        self.assertLess(estimated, codec._raw_chunk(source).compressed_bytes)

    def test_model_batch_uses_one_native_queue_and_matches_sequential_payloads(self):
        tensors = [
            (
                "encoder.packed.weight",
                torch.tensor(
                    [1.0, 1.0, 0.5, 1.0, 0.5, 0.5, 1.0, 1.0],
                    dtype=torch.bfloat16,
                ).repeat(16 << 10),
            ),
            (
                "encoder.shared.weight",
                torch.tensor(
                    [2.0, 2.0, 4.0, 2.0, 4.0, 4.0, 2.0, 2.0],
                    dtype=torch.bfloat16,
                ).repeat(16 << 10),
            ),
            (
                "encoder.local.weight",
                torch.tensor(
                    [0.25, 0.5, 0.25, 1.0, 0.25, 0.5, 0.25, 1.0],
                    dtype=torch.bfloat16,
                ).repeat(16 << 10),
            ),
            (
                "encoder.raw.weight",
                torch.randint(
                    0,
                    256,
                    (2 * ((64 << 10) + 1),),
                    generator=torch.Generator().manual_seed(19),
                    dtype=torch.uint8,
                ).view(torch.bfloat16),
            ),
        ]
        exponent_streams = {
            name: _zipnn_reordered_bytes(
                tensor,
                _zipnn_dtype_config(tensor.dtype),
            )[1::2]
            for name, tensor in tensors
        }
        n = 2
        packed_name = "encoder.packed.weight"
        shared_name = "encoder.shared.weight"
        vocab = topk_vocab(exponent_streams[packed_name], n, top_k=16)
        token_stream = tokenize_ngrams(
            exponent_streams[packed_name], n, vocab
        )[0]
        packed_codebook = huff0.build_codebook(token_stream, prior_count=1)
        shared_codebook = huff0.build_codebook(
            exponent_streams[shared_name], prior_count=1
        )

        codec = NeuralZipCodec(
            threads=4,
            compression_threshold=1.0,
            exponent_layout="original",
        )
        packed_mode = {
            "kind": "packed_huff0",
            "codebook": packed_codebook,
            "vocab": vocab,
            "ngram_n": n,
            "codebook_id": 7,
        }
        shared_mode = {
            "kind": "shared_huff0",
            "codebook": shared_codebook,
            "codebook_id": 8,
        }
        codec._mode_by_tensor_stream = {
            (packed_name, "bfloat16", 0): {"kind": "local_huff0"},
            (packed_name, "bfloat16", 1): packed_mode,
            (shared_name, "bfloat16", 0): {"kind": "raw"},
            (shared_name, "bfloat16", 1): shared_mode,
            ("encoder.local.weight", "bfloat16", 0): {"kind": "local_huff0"},
            ("encoder.local.weight", "bfloat16", 1): {"kind": "local_huff0"},
            ("encoder.raw.weight", "bfloat16", 0): {"kind": "raw"},
            ("encoder.raw.weight", "bfloat16", 1): {"kind": "raw"},
        }
        codebook_payloads = (
            CodebookPayload(
                codebook_id=7,
                cluster_id="packed",
                stream_id=1,
                dtype="bfloat16",
                codec="huff0",
                mode="packed_huff0",
                codebook=packed_codebook.codebook,
                ngram_n=n,
                vocab=vocab,
            ),
            CodebookPayload(
                codebook_id=8,
                cluster_id="shared",
                stream_id=1,
                dtype="bfloat16",
                codec="huff0",
                mode="shared_huff0",
                codebook=shared_codebook.codebook,
            ),
        )
        codec.load_codebooks(codebook_payloads)

        sequential = tuple(
            codec.compress_tensor(name, tensor) for name, tensor in tensors
        )

        with (
            mock.patch.object(
                huff0,
                "compress_zipnn_mixed_many",
                wraps=huff0.compress_zipnn_mixed_many,
            ) as model_encoder,
            mock.patch.object(
                huff0,
                "compress_zipnn_mixed",
                wraps=huff0.compress_zipnn_mixed,
            ) as tensor_encoder,
        ):
            batched = codec.compress_tensors(tensors)

        model_encoder.assert_called_once()
        tensor_encoder.assert_not_called()
        self.assertEqual(batched, sequential)
        self.assertEqual(
            dumps(
                {
                    "method": "neuralzip",
                    "codebooks": codebook_payloads,
                    "tensor_results": [
                        {"payload": payload} for payload in batched
                    ],
                }
            ),
            dumps(
                {
                    "method": "neuralzip",
                    "codebooks": codebook_payloads,
                    "tensor_results": [
                        {"payload": payload} for payload in sequential
                    ],
                }
            ),
        )
        for (name, tensor), payload in zip(tensors, batched, strict=True):
            with self.subTest(name=name):
                restored = codec.decompress_tensor(payload)
                self.assertTrue(
                    torch.equal(
                        restored.view(torch.uint8),
                        tensor.view(torch.uint8),
                    )
                )
    def test_direct_native_batches_respect_input_byte_bound(self):
        tensors = [
            (f"encoder.block.{index}.weight", torch.arange(64, dtype=torch.bfloat16))
            for index in range(3)
        ]
        codec = NeuralZipCodec(
            threads=2,
            exponent_layout="original",
            direct_batch_bytes=200,
        )
        codec._clusters_prepared = True
        codec._mode_by_tensor_stream = {
            (name, "bfloat16", stream_id): {"kind": "raw"}
            for name, _ in tensors
            for stream_id in range(2)
        }

        with mock.patch.object(
            huff0,
            "compress_zipnn_mixed_many",
            wraps=huff0.compress_zipnn_mixed_many,
        ) as model_encoder:
            payloads = codec.compress_tensors(tensors)

        self.assertEqual(model_encoder.call_count, 3)
        for (_, tensor), payload in zip(tensors, payloads, strict=True):
            restored = codec.decompress_tensor(payload)
            self.assertTrue(
                torch.equal(restored.view(torch.uint8), tensor.view(torch.uint8))
            )
    def test_direct_mode_timing_uses_mixed_path_warmup_and_best_of_three(self):
        codec = NeuralZipCodec(
            threads=3,
            compression_threshold=0.75,
            min_saving_bytes=7,
        )
        self.assertEqual(codec.packed_time_budget_ratio, 1.0)
        clock = iter((100.0, 110.0, 200.0, 209.0, 300.0, 307.0, 400.0, 408.0))

        with (
            mock.patch.object(
                huff0,
                "compress_zipnn_mixed_many",
                return_value=[],
            ) as mixed,
            mock.patch(
                "neuralzip.neuralzip_codec.time.perf_counter",
                side_effect=lambda: next(clock),
            ),
        ):
            elapsed = codec._measure_direct_mode_seconds(
                {"kind": "local_huff0"},
                2,
                (b"abcd", b"ef"),
            )

        self.assertEqual(elapsed, 7.0)
        self.assertEqual(mixed.call_count, 4)
        for call in mixed.call_args_list:
            specs = call.args[0]
            self.assertEqual(len(specs), 2)
            self.assertTrue(
                all(
                    spec[1:4] == (2, 0, codec.chunk_size // 2)
                    for spec in specs
                )
            )
            self.assertTrue(
                all(
                    tuple(mode["kind"] for mode in spec[4])
                    == ("local_huff0", "raw")
                    for spec in specs
                )
            )
            self.assertEqual(call.kwargs["min_saving_bytes"], 7)
            self.assertEqual(call.kwargs["threads"], 3)
            self.assertEqual(call.kwargs["compression_threshold"], 0.75)

    def test_packed_ratio_best_is_not_hidden_by_faster_non_saving_ngram(self):
        codec = NeuralZipCodec(
            ngram_values=(2, 3),
            enable_shared_codebooks=False,
            speed_ratio_tolerance=0.03,
        )
        costs = {
            ("raw", 1): (120.0, 0.0),
            ("local_huff0", 1): (100.0, 0.5),
            ("packed_huff0", 2): (101.0, 0.1),
            ("packed_huff0", 3): (99.0, 1.0),
        }

        def build_candidate(_sample, *, kind, prior_count, ngram_n=1):
            del prior_count
            return {
                "kind": kind,
                "ngram_n": int(ngram_n),
                "vocab": b"v",
                "codebook": object(),
            }

        def score_mode(
            _sample,
            mode,
            _element_size,
            _codebook_payload=None,
            sample_parts=None,
        ):
            del sample_parts
            cost, seconds = costs[(mode["kind"], int(mode.get("ngram_n", 1)))]
            mode["sample_cost"] = cost
            mode["sample_seconds"] = seconds
            mode.setdefault("fixed_cost", 0)
            return mode

        with (
            mock.patch.object(codec, "_build_candidate", side_effect=build_candidate),
            mock.patch.object(codec, "_score_mode", side_effect=score_mode),
            mock.patch.object(codec, "_candidate_payload", return_value=None),
        ):
            evaluation = codec._evaluate_modes(
                ("cluster", "bfloat16", 1),
                b"x" * 100,
                total_size=100,
                element_size=2,
                is_exponent=True,
            )

        self.assertEqual(evaluation["base"]["kind"], "local_huff0")
        self.assertEqual(evaluation["opportunity"]["mode"]["ngram_n"], 3)
        self.assertEqual(evaluation["opportunity"]["saved_bytes"], 1.0)
        self.assertEqual(
            evaluation["packed_selection"]["ratio_tolerance"],
            0.03,
        )
        diagnostics = {
            row["ngram_n"]: row
            for row in evaluation["packed_candidate_diagnostics"]
        }
        self.assertFalse(diagnostics[2]["improves_selected_base"])
        self.assertTrue(diagnostics[3]["improves_selected_base"])
        self.assertTrue(diagnostics[3]["ratio_best"])
        self.assertTrue(diagnostics[3]["selected_for_time_budget"])

    def test_packed_selection_keeps_fastest_near_best_saving_ngram(self):
        codec = NeuralZipCodec(
            ngram_values=(2, 3),
            enable_shared_codebooks=False,
            speed_ratio_tolerance=0.03,
        )
        costs = {
            ("raw", 1): (120.0, 0.0),
            ("local_huff0", 1): (100.0, 0.5),
            ("packed_huff0", 2): (99.0, 0.6),
            ("packed_huff0", 3): (98.0, 10.0),
        }

        def build_candidate(_sample, *, kind, prior_count, ngram_n=1):
            del prior_count
            return {
                "kind": kind,
                "ngram_n": int(ngram_n),
                "vocab": b"v",
                "codebook": object(),
            }

        def score_mode(
            _sample,
            mode,
            _element_size,
            _codebook_payload=None,
            sample_parts=None,
        ):
            del sample_parts
            cost, seconds = costs[(mode["kind"], int(mode.get("ngram_n", 1)))]
            mode["sample_cost"] = cost
            mode["sample_seconds"] = seconds
            mode.setdefault("fixed_cost", 0)
            return mode

        with (
            mock.patch.object(codec, "_build_candidate", side_effect=build_candidate),
            mock.patch.object(codec, "_score_mode", side_effect=score_mode),
            mock.patch.object(codec, "_candidate_payload", return_value=None),
        ):
            evaluation = codec._evaluate_modes(
                ("cluster", "bfloat16", 1),
                b"x" * 100,
                total_size=100,
                element_size=2,
                is_exponent=True,
            )

        self.assertEqual(evaluation["opportunity"]["mode"]["ngram_n"], 2)
        self.assertEqual(evaluation["opportunity"]["saved_bytes"], 1.0)
        diagnostics = {
            row["ngram_n"]: row
            for row in evaluation["packed_candidate_diagnostics"]
        }
        self.assertTrue(diagnostics[3]["ratio_best"])
        self.assertFalse(diagnostics[3]["selected_for_time_budget"])
        self.assertTrue(diagnostics[2]["selected_for_time_budget"])

    def test_fast_path_is_selected_per_tensor(self):
        codec = NeuralZipCodec(threads=2, exponent_layout="original")
        tensor = torch.ones((8, 8), dtype=torch.float32)
        # This tensor is baseline-only even though another tensor in the model
        # uses an advanced mode.  The latter must not disable this fast path.
        codec._mode_by_tensor_stream = {
            **{
                ("baseline.weight", "float32", stream_id): {
                    "kind": "local_huff0"
                }
                for stream_id in range(4)
            },
            ("advanced.weight", "float32", 3): {"kind": "packed_huff0"},
        }
        baseline_payload = codec.__class__.__bases__[0](threads=2)._compress_zipnn(
            "baseline.weight",
            tensor,
            _zipnn_dtype_config(tensor.dtype),
        )
        with mock.patch.object(
            codec.__class__.__bases__[0],
            "_compress_zipnn",
            return_value=baseline_payload,
        ) as baseline:
            result = codec._compress_neuralzip(
                "baseline.weight", tensor, _zipnn_dtype_config(tensor.dtype)
            )
        self.assertEqual(result.strategy, "neuralzip")
        self.assertEqual(result.groups, baseline_payload.groups)
        baseline.assert_called_once()

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
            result["setup_seconds"],
            result["clustering_seconds"] + result["codebook_setup_seconds"],
        )
        self.assertAlmostEqual(
            result["total_seconds"],
            result["setup_seconds"] + result["encode_seconds"],
        )
        self.assertEqual(
            result["setup_memory_bytes"],
            compressor.codec.setup_memory_bytes,
        )
        self.assertEqual(
            result["selection_stats"]["selection_strategy"],
            "fastest-within-ratio-and-time-budget",
        )
        self.assertGreater(
            result["selection_stats"]["packed_modes_evaluated"],
            0,
        )
        self.assertEqual(
            result["selection_diagnostics"],
            compressor.codec.selection_diagnostics,
        )
        self.assertTrue(
            all(
                diagnostic["decision"]
                in {
                    "no-packed-mode",
                    "no-ratio-saving",
                    "time-budget",
                    "source-budget",
                    "selected",
                }
                for diagnostic in result["selection_diagnostics"]
            )
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

    def test_zero_packed_budget_uses_shared_huff0(self):
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
        self.assertEqual(result["compressed_bytes"], 66_183)
        self.assertEqual(result["packed_ngram_chunks"], 0)
        self.assertGreater(result["selected_modes"].get("shared_huff0", 0), 0)
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
            "fastest-within-ratio-and-time-budget",
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
            packed_time_budget_ratio=1.0,
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
                packed_time_budget_ratio=1.0,
            ).compress(model)
            for _ in range(3)
        ]

        self.assertEqual(
            len({result["compressed_bytes"] for result in results}),
            1,
        )
        self.assertEqual(results[0]["compressed_bytes"], 12_057)
        self.assertEqual(
            len({repr(result["selected_modes"]) for result in results}),
            1,
        )
        self.assertTrue(all(result["roundtrip_ok"] for result in results))

    def test_shared_codebook_failure_falls_back_to_local_huff0(self):
        codec = NeuralZipCodec(ngram_values=())
        mode = codec._build_candidate(
            bytes([1, 3]) * 2048,
            kind="shared_huff0",
            prior_count=1,
        )
        mode["codebook_id"] = 0
        group = codec._compress_stream_with_mode(
            bytes([2, 4]) * 2048,
            "exponent",
            element_size=2,
            mode=mode,
        )

        self.assertEqual(group.chunks[0].mode, "local_huff0")
        self.assertEqual(codec._decompress_stream(group), bytes([2, 4]) * 2048)

    def test_packed_candidate_records_sample_hit_diagnostics(self):
        codec = NeuralZipCodec(top_k=16)
        source = bytes([1, 2, 1, 2]) * 1024
        mode = codec._build_candidate(
            source,
            kind="packed_huff0",
            prior_count=1,
            ngram_n=2,
        )

        self.assertEqual(mode["sample_token_count"], len(source) // 2)
        self.assertEqual(
            mode["sample_ngram_hits"] + mode["sample_ngram_misses"],
            len(source) // 2,
        )
        self.assertEqual(mode["sample_ngram_misses"], 0)
        self.assertEqual(mode["sample_escape_bytes"], 0)
        self.assertEqual(mode["sample_tail_bytes"], 0)

    def test_tensor_probe_rejects_cluster_codebook_for_outlier(self):
        codec = NeuralZipCodec(ngram_values=())
        # With the new four-stream shared frame, this distribution keeps the
        # source table one byte ahead of a local frame, while a skewed outlier
        # still benefits materially from rebuilding locally.
        common = bytes(range(16)) * 512
        outlier = bytes([0, 0, 0, 0, 1, 2, 3]) * 1170
        source_mode = codec._build_candidate(
            common,
            kind="shared_huff0",
            prior_count=0,
        )
        source_mode["codebook_id"] = 0

        common_mode, common_probe = codec._select_transfer_mode(
            common,
            source_mode,
            element_size=2,
            stream_kind="exponent",
        )
        outlier_mode, outlier_probe = codec._select_transfer_mode(
            outlier,
            source_mode,
            element_size=2,
            stream_kind="exponent",
        )

        self.assertEqual(common_mode["kind"], "shared_huff0")
        self.assertTrue(common_probe["uses_source_codebook"])
        self.assertEqual(outlier_mode["kind"], "local_huff0")
        self.assertFalse(outlier_probe["uses_source_codebook"])

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
                chunk.mode == "shared_huff0"
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
                chunk.mode in {"shared_huff0", "packed_huff0"}
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
