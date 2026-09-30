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
    setup = NeuralZipSetup()
    assert setup.min_saving_bytes == 0
    assert setup.chunk_size == 256 << 10
    assert setup.compression_threshold == 0.95
    assert setup.prior_count == 1
    assert setup.cluster_map == {}
    assert setup.cluster_js_threshold == 0.05
    assert setup.max_codebook_sample_bytes == 8 << 20
    assert setup.packed_min_gain == 0.02
    assert setup.enable_shared_codebooks is True
    assert setup.enable_packed_codebooks is True

    assert setup.codebooks == ()
    assert setup.tensor_codebooks == {}
    assert setup.global_codebooks == {}
    assert setup.setup_report.setup_seconds == 0.0
    
    # Test build
    
