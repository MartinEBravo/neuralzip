"""Stable result records stored or returned by the public API."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SetupReport:
    clustering_seconds: float
    codebook_setup_seconds: float

    @property
    def setup_seconds(self):
        return self.clustering_seconds + self.codebook_setup_seconds


@dataclass(frozen=True)
class DecodeStats:
    decode_seconds: float


@dataclass(frozen=True)
class EncodeStats:
    """Stable size and timing summary returned by ``NeuralZip.encode``."""

    raw_bytes: int
    compressed_bytes: int
    compression_ratio: float
    mapping_seconds: float
    encode_seconds: float
    total_seconds: float
