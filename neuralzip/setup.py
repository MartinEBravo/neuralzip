import copy
import pickle
from dataclasses import dataclass, field
from pathlib import Path

from neuralzip import huff0
from neuralzip.codec.codec import BF16_NAME, EXPONENT_STREAM, NeuralZipCodec
from neuralzip.payloads import CodebookPayload
from neuralzip.utils import _model_tensors, time_call


@dataclass(frozen=True)
class SetupReport:
    clustering_seconds: float
    codebook_setup_seconds: float

    @property
    def setup_seconds(self):
        return self.clustering_seconds + self.codebook_setup_seconds


@dataclass(frozen=True, slots=True)
class NeuralZipSetup:
    _codec_state: bytes = field(repr=False)
    setup_report: SetupReport | None = None

    @property
    def setup_seconds(self):
        return None if self.setup_report is None else self.setup_report.setup_seconds

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle, protocol=5)
        return path


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


def load_setup(value):
    if isinstance(value, NeuralZipSetup):
        return value
    with Path(value).open("rb") as handle:
        setup = pickle.load(handle)
    if not isinstance(setup, NeuralZipSetup):
        raise TypeError("The file does not contain a NeuralZip setup.")
    return setup


def _add_global_codebook(codec):
    if not hasattr(codec, "_global_modes"):
        codec._global_modes = {}
    key = (BF16_NAME, EXPONENT_STREAM)
    if key in codec._global_modes:
        return
    header = Path(__file__).with_name("codebooks").joinpath("bf16.huff0").read_bytes()
    codebook_id = (
        max(
            (int(payload.codebook_id) for payload in codec._codebook_payloads),
            default=-1,
        )
        + 1
    )
    codec._codebook_payloads.append(
        CodebookPayload(codebook_id, EXPONENT_STREAM, BF16_NAME, codebook=header)
    )
    codec._global_modes[key] = {
        "kind": "shared_huff0",
        "codebook_id": codebook_id,
        "codebook": huff0.build_codebook_from_header(header),
    }


def materialize_setup(value=None, *, threads=None, gpu=False):
    if value is None:
        codec = NeuralZipCodec(ngram_values=())
    else:
        codec = pickle.loads(load_setup(value)._codec_state)
        _restore_live_codebooks(codec)
    if threads is not None:
        codec.threads = int(threads)
    if gpu:
        _add_global_codebook(codec)
    return codec


def build_setup(model, **setup_options):
    named_tensors = _model_tensors(model)
    codec = NeuralZipCodec(**setup_options)
    _, clustering_seconds = time_call(codec.prepare_clusters, named_tensors)
    _, codebook_setup_seconds = time_call(
        codec.build_codebooks_and_select_modes, named_tensors
    )
    report = SetupReport(clustering_seconds, codebook_setup_seconds)
    return setup_from_codec(codec, report)
