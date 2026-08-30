import copy
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from neuralzip import huff0


@dataclass(frozen=True)
class SetupReport:
    clustering_seconds: float
    codebook_setup_seconds: float

    @property
    def setup_seconds(self):
        return self.clustering_seconds + self.codebook_setup_seconds


@dataclass(frozen=True, slots=True)
class NeuralZipSetup:
    """Opaque, immutable setup returned by :meth:`NeuralZip.fit`."""

    _codec_state: bytes = field(repr=False)
    setup_report: SetupReport | None = None


def _detached_codec_snapshot(codec):
    snapshot = copy.copy(codec)
    modes = {}
    snapshot._mode_by_tensor_stream = {
        key: modes.setdefault(id(mode), dict(mode))
        for key, mode in codec._mode_by_tensor_stream.items()
    }
    snapshot._transfer_mode_by_tensor_stream = {}
    snapshot._decode_codebooks = {}
    snapshot._codebook_payload_by_id = {}
    for mode in modes.values():
        if mode.get("codebook") is not None:
            mode["codebook"] = None
    return snapshot


def _restore_live_codebooks(codec):
    payloads = codec.export_codebooks()
    handles = {
        int(payload.codebook_id): huff0.build_codebook_from_header(payload.codebook)
        for payload in payloads
    }
    for mode in codec._mode_by_tensor_stream.values():
        codebook_id = mode.get("codebook_id")
        if codebook_id is not None:
            mode["codebook"] = handles[int(codebook_id)]
    codec.load_codebooks(payloads)


def setup_from_codec(codec, setup_report=None):
    snapshot = _detached_codec_snapshot(codec)
    return NeuralZipSetup(pickle.dumps(snapshot, protocol=5), setup_report)


def load_setup(value_or_path):
    if isinstance(value_or_path, NeuralZipSetup):
        return value_or_path
    with Path(value_or_path).open("rb") as handle:
        return pickle.load(handle)


def materialize_setup(value_or_path, *, threads=None):
    setup = load_setup(value_or_path)
    codec = pickle.loads(setup._codec_state)
    _restore_live_codebooks(codec)
    if threads is not None:
        threads = int(threads)
        if threads < 0:
            raise ValueError("threads must be non-negative.")
        codec.threads = threads
    return codec


def save_setup(value, path):
    setup = load_setup(value)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(setup, handle, protocol=5)
    return path
