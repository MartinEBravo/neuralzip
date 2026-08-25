import unittest
from dataclasses import replace
from unittest import mock

import torch

from neuralzip.archive import (
    serialized_codebooks_size,
    serialized_group_size,
    serialized_tensor_size,
)
from neuralzip.codec_protocol import (
    CODEC_PROTOCOL_VERSION,
    assert_tensor_roundtrip,
    encode_tensor_batch,
    measure_compressor_exponents,
    measure_exponent_streams,
    payload_exponent_sizes,
    prepare_neuralzip_codec,
)
from neuralzip.experiment_scenarios import (
    _compress_mapping,
    _model_methods_completed,
    _upsert_model_compressor_row,
)
from neuralzip.payloads import (
    ByteGroupPayload,
    CodebookPayload,
    StreamChunk,
    TensorPayload,
)


def _payload(name, tensor):
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    group = ByteGroupPayload(
        stream_kind="exponent",
        symbol_count=len(raw),
        chunks=(StreamChunk(mode="raw", symbol_count=len(raw), payload=raw),),
    )
    return TensorPayload(
        name=name,
        shape=tuple(tensor.shape),
        dtype=str(tensor.dtype).replace("torch.", ""),
        element_size=tensor.element_size(),
        numel=tensor.numel(),
        strategy="test",
        groups=(group,) * tensor.element_size(),
        num_buffers=tensor.element_size(),
    )


class _BatchCodec:
    def __init__(self):
        self.batch_calls = 0
        self.tensor_calls = 0
        self.prepared = []

    def compress_tensors(self, named_tensors):
        self.batch_calls += 1
        return tuple(_payload(name, tensor) for name, tensor in named_tensors)

    def compress_tensor(self, name, tensor):
        self.tensor_calls += 1
        return _payload(name, tensor)

    def prepare_clusters(self, named_tensors):
        self.prepared.append(("clusters", tuple(name for name, _ in named_tensors)))

    def build_codebooks_and_select_modes(self, named_tensors):
        self.prepared.append(("codebooks", tuple(name for name, _ in named_tensors)))


class _SequentialCodec:
    def __init__(self):
        self.tensor_calls = 0

    def compress_tensor(self, name, tensor):
        self.tensor_calls += 1
        return _payload(name, tensor)


class CodecProtocolTests(unittest.TestCase):
    def setUp(self):
        self.tensors = {
            "first": torch.arange(8, dtype=torch.float32),
            "second": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        }

    def test_batch_is_preferred_and_accounting_uses_serialized_bytes(self):
        codec = _BatchCodec()
        result = encode_tensor_batch(codec, self.tensors)

        self.assertTrue(result.used_batch)
        self.assertEqual(result.encoder_path, "compress_tensors")
        self.assertFalse(result.materialization_active)
        self.assertEqual(result.materialization_bytes, 0)
        self.assertEqual(result.materialization_hits, 0)
        self.assertEqual(codec.batch_calls, 1)
        self.assertEqual(codec.tensor_calls, 0)
        self.assertEqual(
            result.compressed_bytes,
            sum(serialized_tensor_size(payload) for payload in result.payloads),
        )
        self.assertEqual(
            tuple(payload.name for payload in result.payloads),
            tuple(self.tensors),
        )

    def test_sequential_fallback_preserves_payloads(self):
        batch = encode_tensor_batch(_BatchCodec(), self.tensors)
        sequential_codec = _SequentialCodec()
        sequential = encode_tensor_batch(sequential_codec, self.tensors)

        self.assertFalse(sequential.used_batch)
        self.assertEqual(sequential.encoder_path, "compress_tensor")
        self.assertEqual(sequential_codec.tensor_calls, len(self.tensors))
        self.assertEqual(sequential.payloads, batch.payloads)
        self.assertEqual(sequential.compressed_bytes, batch.compressed_bytes)

    def test_exponent_metrics_use_the_archive_group_size(self):
        payload = _payload("weight", self.tensors["first"])
        raw_bytes, compressed_bytes = payload_exponent_sizes(payload)
        exponent_group = payload.groups[payload.element_size - 1]

        self.assertEqual(raw_bytes, exponent_group.symbol_count)
        self.assertEqual(compressed_bytes, serialized_group_size(exponent_group))

    def test_exponent_accounting_includes_only_referenced_codebooks(self):
        payload = _payload("weight", self.tensors["first"])
        exponent = payload.groups[payload.element_size - 1]
        chunks = list(exponent.chunks)
        chunks[0] = replace(chunks[0], mode="shared_huff0", codebook_id=7)
        exponent = replace(exponent, chunks=tuple(chunks))
        payload = replace(payload, groups=(*payload.groups[:-1], exponent))
        referenced = CodebookPayload(7, "cluster", 3, "float32")
        unused = CodebookPayload(9, "cluster", 0, "float32")

        report = measure_exponent_streams((payload,), (referenced, unused))

        self.assertEqual(report.raw_bytes, exponent.symbol_count)
        self.assertEqual(report.group_bytes, serialized_group_size(exponent))
        self.assertEqual(
            report.codebook_bytes,
            serialized_codebooks_size((referenced,)),
        )
        self.assertEqual(report.referenced_codebooks, 1)
        self.assertEqual(
            report.compressed_bytes,
            report.group_bytes + report.codebook_bytes,
        )
        self.assertEqual(
            report.ratio_pct,
            100 * report.compressed_bytes / report.raw_bytes,
        )
        with self.assertRaisesRegex(ValueError, "missing codebooks: 7"):
            measure_exponent_streams((payload,), ())
        stream_report = measure_exponent_streams(
            (payload,),
            (referenced, unused),
            codebook_scope="stream",
        )
        self.assertEqual(
            stream_report.codebook_bytes,
            serialized_codebooks_size((referenced,)),
        )
        same_stream_unused = replace(unused, stream_id=3)
        stream_report = measure_exponent_streams(
            (payload,),
            (referenced, same_stream_unused),
            codebook_scope="stream",
        )
        self.assertEqual(
            stream_report.codebook_bytes,
            serialized_codebooks_size((referenced, same_stream_unused)),
        )
        compressor_report = measure_compressor_exponents(
            {
                "tensor_results": ({"payload": payload}, {"payload": None}),
                "codebooks": (referenced, same_stream_unused),
            },
            dtypes={"float32"},
        )
        self.assertEqual(compressor_report, stream_report)

    def test_setup_report_preserves_both_timing_phases(self):
        codec = _BatchCodec()
        report = prepare_neuralzip_codec(codec, self.tensors)

        self.assertEqual(
            codec.prepared,
            [
                ("clusters", tuple(self.tensors)),
                ("codebooks", tuple(self.tensors)),
            ],
        )
        self.assertGreaterEqual(report.precompute_seconds, 0.0)
        self.assertGreaterEqual(report.codebook_setup_seconds, 0.0)
        self.assertEqual(
            report.total_setup_seconds,
            report.precompute_seconds + report.codebook_setup_seconds,
        )

    def test_roundtrip_comparison_is_bitwise(self):
        restored = {name: tensor.clone() for name, tensor in self.tensors.items()}
        assert_tensor_roundtrip(
            self.tensors,
            restored,
            method="test",
            checkpoint="checkpoint",
        )
        restored["first"][0] += 1
        with self.assertRaises(AssertionError):
            assert_tensor_roundtrip(
                self.tensors,
                restored,
                method="test",
                checkpoint="checkpoint",
            )

    def test_mapping_preserves_codec_timing_fields(self):
        result = {
            "method": "neuralzip",
            "raw_bytes": 100,
            "compressed_bytes": 50,
            "compression_ratio": 0.5,
            "encode_seconds": 1.0,
            "compression_seconds": 2.0,
            "total_compression_seconds": 3.0,
            "cold_encode_seconds": 4.0,
            "precompute_seconds": 0.5,
            "codebook_setup_seconds": 0.5,
            "tensor_results": (),
        }
        compressor = mock.Mock()
        compressor.compress.return_value = result
        with mock.patch(
            "neuralzip.experiment_scenarios.Compressor",
            return_value=compressor,
        ):
            rows = _compress_mapping(
                "model",
                {"weight": torch.ones(1)},
                methods=(("NeuralZip", "neuralzip"),),
            )

        self.assertEqual(rows[0]["compression_seconds"], 2.0)
        self.assertEqual(rows[0]["total_compression_seconds"], 3.0)
        self.assertEqual(rows[0]["cold_encode_seconds"], 4.0)

    def test_mapping_cascades_missing_timing_fields_from_encode(self):
        result = {
            "method": "zipnn-src",
            "raw_bytes": 100,
            "compressed_bytes": 50,
            "compression_ratio": 0.5,
            "encode_seconds": 1.25,
            "tensor_results": (),
        }
        compressor = mock.Mock()
        compressor.compress.return_value = result
        with mock.patch(
            "neuralzip.experiment_scenarios.Compressor",
            return_value=compressor,
        ):
            row = _compress_mapping(
                "model",
                {"weight": torch.ones(1)},
                methods=(("ZipNN", "zipnn-src"),),
            )[0]

        self.assertEqual(row["compression_seconds"], 1.25)
        self.assertEqual(row["total_compression_seconds"], 1.25)
        self.assertEqual(row["cold_encode_seconds"], 1.25)

    def test_model_resume_rejects_rows_without_current_encoder_protocol(self):
        methods = (("NeuralZip", "neuralzip"),)
        legacy = {
            "model_name": "model",
            "compressor": "NeuralZip",
            "method": "neuralzip",
            "status": "ok",
        }
        self.assertFalse(_model_methods_completed([legacy], "model", methods))

        current = {
            **legacy,
            "codec_protocol_version": float(CODEC_PROTOCOL_VERSION),
            "encoder_path": "compress_tensors",
        }
        self.assertTrue(_model_methods_completed([current], "model", methods))
        current["codec_protocol_version"] = CODEC_PROTOCOL_VERSION + 0.5
        self.assertFalse(_model_methods_completed([current], "model", methods))

    def test_model_result_upsert_replaces_legacy_compressor_row(self):
        rows = [
            {
                "model_name": "model",
                "compressor": "NeuralZip",
                "codec_protocol_version": None,
            }
        ]
        current = {
            "model_name": "model",
            "compressor": "NeuralZip",
            "codec_protocol_version": CODEC_PROTOCOL_VERSION,
        }

        _upsert_model_compressor_row(rows, current)

        self.assertEqual(rows, [current])


if __name__ == "__main__":
    unittest.main()
