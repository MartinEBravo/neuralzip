import random

from neuralzip.setup import NeuralZipSetup, NeuralZipSetupReport


def test_setup_report():
    left = random.random()
    right = random.random()
    report = NeuralZipSetupReport(clustering_seconds=left, codebook_setup_seconds=right)
    assert report.clustering_seconds == left
    assert report.codebook_setup_seconds == right
    assert report.setup_seconds == left + right


def test_setup():
    setup = NeuralZipSetup(
        min_saving_bytes=0,
        chunk_size=256 << 10,
        compression_threshold=0.95,
        threads=0,
        prior_count=1,
        exponent_prior_count=1,
        cluster_map=None,
        cluster_js_threshold=0.05,
        max_codebook_sample_bytes=8 << 20,
        ngram_values=(2, 3, 4),
        top_k=254,
        packed_source_budget_ratio=0.30,
        packed_time_budget_ratio=1.0,
        speed_ratio_tolerance=0.03,
        enable_shared_codebooks=True,
        direct_batch_bytes=128 << 20,
    )
    assert setup.min_saving_bytes == 0
    assert setup.chunk_size == 256 << 10
    assert setup.compression_threshold == 0.95
    assert setup.threads == 0
    assert setup.prior_count == 1
    assert setup.exponent_prior_count == 1
    assert setup.cluster_map == {}
    assert setup.cluster_js_threshold == 0.05
    assert setup.max_codebook_sample_bytes == 8 << 20
    assert setup.ngram_values == (2, 3, 4)
    assert setup.top_k == 254
    assert setup.packed_source_budget_ratio == 0.30
    assert setup.packed_time_budget_ratio == 1.0
    assert setup.speed_ratio_tolerance == 0.03
    assert setup.enable_shared_codebooks is True
    assert setup.direct_batch_bytes == 128 << 20

    assert setup.codebooks == ()
    assert setup.tensor_codebooks == {}
    assert setup.global_codebooks == {}
    assert setup.setup_report is None
