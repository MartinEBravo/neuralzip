import random

import pytest

import neuralzip.setup as setup_api
from neuralzip.setup import SetupReport


def test_setup_report():
    left = random.random()
    right = random.random()
    report = SetupReport(clustering_seconds=left, codebook_setup_seconds=right)
    assert report.clustering_seconds == left
    assert report.codebook_setup_seconds == right
    assert report.setup_seconds == left + right
