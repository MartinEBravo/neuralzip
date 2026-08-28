import unittest
from dataclasses import replace

import torch

from neuralzip.codec_protocol import (
    decode_tensor_batch,
    encode_tensor_batch,
    measure_compressor_exponents,
    measure_exponent_streams,
    payload_exponent_sizes,
    prepare_neuralzip_codec,
)
from neuralzip.payloads import (
    ByteGroupPayload,
    CodebookPayload,
    StreamChunk,
    TensorPayload,
)
from neuralzip.serialization import (
    serialized_codebooks_size,
    serialized_group_size,
    serialized_tensor_size,
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


class _BatchDecoder:
    def __init__(self):
        self.batch_calls = 0
        self.tensor_calls = 0

    def decompress_tensors(self, payloads):
        self.batch_calls += 1
        return tuple(f"batch:{payload.name}" for payload in payloads)

    def decompress_tensor(self, payload):
        self.tensor_calls += 1
        return f"single:{payload.name}"


class _SequentialDecoder:
    def __init__(self):
        self.tensor_calls = 0

    def decompress_tensor(self, payload):
        self.tensor_calls += 1
        return f"single:{payload.name}"


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
        self.assertGreaterEqual(result.encode_seconds, 0.0)

    def test_sequential_fallback_preserves_payloads(self):
        batch = encode_tensor_batch(_BatchCodec(), self.tensors)
        sequential_codec = _SequentialCodec()
        sequential = encode_tensor_batch(sequential_codec, self.tensors)

        self.assertFalse(sequential.used_batch)
        self.assertEqual(sequential.encoder_path, "compress_tensor")
        self.assertEqual(sequential_codec.tensor_calls, len(self.tensors))
        self.assertEqual(sequential.payloads, batch.payloads)
        self.assertEqual(sequential.compressed_bytes, batch.compressed_bytes)
        self.assertGreaterEqual(sequential.encode_seconds, 0.0)

    def test_decode_prefers_batch_and_preserves_payload_order(self):
        payloads = tuple(
            _payload(name, tensor) for name, tensor in self.tensors.items()
        )
        codec = _BatchDecoder()

        decoded, seconds = decode_tensor_batch(codec, payloads)

        self.assertEqual(codec.batch_calls, 1)
        self.assertEqual(codec.tensor_calls, 0)
        self.assertEqual(
            decoded,
            {name: f"batch:{name}" for name in self.tensors},
        )
        self.assertGreaterEqual(seconds, 0.0)

    def test_decode_falls_back_to_per_tensor_frontend(self):
        payloads = tuple(
            _payload(name, tensor) for name, tensor in self.tensors.items()
        )
        codec = _SequentialDecoder()

        decoded, seconds = decode_tensor_batch(codec, payloads)

        self.assertEqual(codec.tensor_calls, len(payloads))
        self.assertEqual(
            decoded,
            {name: f"single:{name}" for name in self.tensors},
        )
        self.assertGreaterEqual(seconds, 0.0)

    def test_decode_rejects_wrong_batch_cardinality(self):
        payloads = tuple(
            _payload(name, tensor) for name, tensor in self.tensors.items()
        )

        class MissingDecoder(_BatchDecoder):
            def decompress_tensors(self, payloads):
                return super().decompress_tensors(payloads)[:-1]

        with self.assertRaisesRegex(RuntimeError, "different number"):
            decode_tensor_batch(MissingDecoder(), payloads)

    def test_every_frontend_gets_the_same_payload_validation(self):
        class ReorderedCodec(_BatchCodec):
            def compress_tensors(self, named_tensors):
                payloads = super().compress_tensors(named_tensors)
                return tuple(reversed(payloads))

        with self.assertRaisesRegex(RuntimeError, "changed tensor names or order"):
            encode_tensor_batch(ReorderedCodec(), self.tensors)

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
        self.assertGreaterEqual(report.clustering_seconds, 0.0)
        self.assertGreaterEqual(report.codebook_setup_seconds, 0.0)
        self.assertEqual(
            report.setup_seconds,
            report.clustering_seconds + report.codebook_setup_seconds,
        )


if __name__ == "__main__":
    unittest.main()
