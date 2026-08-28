import unittest

from examples.experiments.direct_compression import exponent_metrics as direct_metrics
from examples.experiments.run_setup_transfer_suite import (
    exponent_metrics as family_metrics,
)
from neuralzip.codec_protocol import measure_exponent_streams
from neuralzip.payloads import (
    ByteGroupPayload,
    CodebookPayload,
    StreamChunk,
    TensorPayload,
)
from neuralzip.serialization import serialized_codebooks_size


def _fixture():
    mantissa = ByteGroupPayload(
        stream_kind="mantissa",
        symbol_count=6,
        chunks=(StreamChunk(mode="raw", symbol_count=6, payload=b"abcdef"),),
    )
    exponent = ByteGroupPayload(
        stream_kind="exponent",
        symbol_count=6,
        chunks=(
            StreamChunk(
                mode="shared_huff0",
                symbol_count=4,
                payload=b"a",
                codebook_id=7,
                ngram_hits=3,
                ngram_escapes=1,
            ),
            StreamChunk(mode="raw", symbol_count=2, payload=b"bc"),
        ),
    )
    payload = TensorPayload(
        name="weight",
        shape=(6,),
        dtype="bfloat16",
        element_size=2,
        numel=6,
        strategy="test",
        groups=(mantissa, exponent),
        num_buffers=2,
    )
    referenced = CodebookPayload(
        codebook_id=7,
        cluster_id="cluster",
        stream_id=1,
        dtype="bfloat16",
        codebook=b"referenced",
    )
    unused = CodebookPayload(
        codebook_id=9,
        cluster_id="cluster",
        stream_id=1,
        dtype="bfloat16",
        codebook=b"unused",
    )
    return payload, referenced, unused


class _Codec:
    def __init__(self, codebooks):
        self._codebooks = tuple(codebooks)

    def export_codebooks(self):
        return self._codebooks


class ExperimentMetricWrapperTests(unittest.TestCase):
    def test_family_wrapper_preserves_schema_and_mode_breakdown(self):
        payload, referenced, unused = _fixture()
        codebooks = (referenced, unused)
        report = measure_exponent_streams((payload,), codebooks)

        metrics = family_metrics((payload,), codebooks)

        self.assertEqual(
            set(metrics),
            {
                "raw_exponents",
                "compressed_exponent_bytes",
                "compressed_exponent_group_bytes",
                "exponent_codebook_metadata_bytes",
                "exponent_ratio_pct",
                "shared_exponent_share_pct",
                "packed_exponent_share_pct",
                "local_exponent_share_pct",
                "raw_exponent_share_pct",
                "mode_symbols",
                "mode_chunks",
                "referenced_codebooks",
            },
        )
        self.assertEqual(metrics["raw_exponents"], report.raw_bytes)
        self.assertEqual(metrics["compressed_exponent_group_bytes"], report.group_bytes)
        self.assertEqual(
            metrics["exponent_codebook_metadata_bytes"], report.codebook_bytes
        )
        self.assertEqual(metrics["compressed_exponent_bytes"], report.compressed_bytes)
        self.assertEqual(metrics["exponent_ratio_pct"], report.ratio_pct)
        self.assertEqual(metrics["referenced_codebooks"], 1)
        self.assertEqual(metrics["mode_symbols"], {"shared_huff0": 4, "raw": 2})
        self.assertEqual(metrics["mode_chunks"], {"shared_huff0": 1, "raw": 1})

    def test_direct_wrapper_preserves_schema_and_diagnostic_counters(self):
        payload, referenced, unused = _fixture()
        codebooks = (referenced, unused)
        report = measure_exponent_streams(
            (payload,),
            codebooks,
            codebook_scope="stream",
        )

        metrics = direct_metrics(_Codec(codebooks), (payload,))

        self.assertEqual(
            set(metrics),
            {
                "exponent_raw_bytes",
                "exponent_data_bytes",
                "exponent_group_metadata_bytes",
                "exponent_codebook_metadata_bytes",
                "all_codebook_metadata_bytes",
                "exponent_compressed_bytes",
                "exponent_ratio_pct",
                "local_huff0_chunks",
                "shared_huff0_chunks",
                "packed_ngram_chunks",
                "raw_chunks",
                "zero_chunks",
                "ngram_hits",
                "ngram_escapes",
            },
        )
        self.assertEqual(metrics["exponent_raw_bytes"], report.raw_bytes)
        self.assertEqual(metrics["exponent_data_bytes"], 3)
        self.assertEqual(
            metrics["exponent_group_metadata_bytes"], report.group_bytes - 3
        )
        self.assertEqual(
            metrics["exponent_codebook_metadata_bytes"], report.codebook_bytes
        )
        self.assertEqual(
            metrics["all_codebook_metadata_bytes"],
            serialized_codebooks_size(codebooks),
        )
        self.assertEqual(metrics["exponent_compressed_bytes"], report.compressed_bytes)
        self.assertEqual(metrics["exponent_ratio_pct"], report.ratio_pct)
        self.assertEqual(metrics["shared_huff0_chunks"], 1)
        self.assertEqual(metrics["raw_chunks"], 1)
        self.assertEqual(metrics["ngram_hits"], 3)
        self.assertEqual(metrics["ngram_escapes"], 1)


if __name__ == "__main__":
    unittest.main()
