"""Immutable values and persistence helpers for reusable NeuralZip setups.

Setup files are pickle based and must only be loaded from trusted sources.
Native Huff0 handles are process-local, so the persisted representation keeps
only canonical codebook headers and rebuilds independent handles on demand.
"""

import os
import pickle
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from . import huff0
from .neuralzip_codec import NeuralZipCodec
from .operations import SetupReport

SETUP_MARKER = "frozen-huff0-codebook-v2"
POST_SETUP_MODES = frozenset({"raw", "shared_huff0", "packed_huff0"})
_SETUP_FORMAT_VERSION = 2


@dataclass(frozen=True, slots=True)
class NeuralZipSetup:
    """Opaque, immutable data produced by fitting NeuralZip.

    A setup intentionally has no mapping, encoding, or decoding methods. It
    can be shared safely between operations because every consumer
    materializes its own :class:`NeuralZipCodec` and native Huff0 handles.
    """

    _codec_state: bytes = field(repr=False)
    setup_report: SetupReport | None = None
    _format_version: int = field(default=_SETUP_FORMAT_VERSION, repr=False)

    def __post_init__(self):
        if not isinstance(self._codec_state, bytes):
            raise TypeError("NeuralZip setup state must be immutable bytes.")
        if not self._codec_state:
            raise ValueError("NeuralZip setup state cannot be empty.")
        if self.setup_report is not None and not isinstance(
            self.setup_report, SetupReport
        ):
            raise TypeError("setup_report must be a SetupReport or None.")
        if int(self._format_version) != _SETUP_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported NeuralZip setup version {self._format_version!r}."
            )


def validate_setup(codec):
    """Reject codec state that is unsafe to reuse on a later target."""

    if not isinstance(codec, NeuralZipCodec):
        raise TypeError(f"Expected NeuralZipCodec, got {type(codec)!r}.")
    if (
        int(getattr(codec, "prior_count", 1)) <= 0
        or int(getattr(codec, "exponent_prior_count", 1)) <= 0
    ):
        raise ValueError(
            "Transferable NeuralZip setups require positive Huffman priors."
        )
    invalid = sorted(
        {
            str(mode.get("kind"))
            for mode in codec._mode_by_tensor_stream.values()
            if str(mode.get("kind")) not in POST_SETUP_MODES
        }
    )
    if invalid:
        joined = ", ".join(invalid)
        raise ValueError(
            "NeuralZip setups may only use raw, shared, or packed modes; "
            f"found {joined}. Refit the setup."
        )
    if not bool(getattr(codec, "_clusters_prepared", False)) or not getattr(
        codec, "_mode_by_tensor_stream", None
    ):
        raise ValueError("NeuralZip setup is empty. Call NeuralZip.fit() first.")


def _unique_modes(codec):
    seen = set()
    for mode in codec._mode_by_tensor_stream.values():
        if id(mode) in seen:
            continue
        seen.add(id(mode))
        yield mode


def _detached_codec_snapshot(codec):
    """Copy the serializable setup graph without touching the live codec."""

    snapshot = object.__new__(NeuralZipCodec)
    snapshot.__dict__ = dict(codec.__dict__)

    # Mode dictionaries are shared across the source indexes. Recreate that
    # alias graph so replacing native handles below cannot mutate the live
    # codec, while preserving the relationships expected by encoding.
    copied_modes = {}

    snapshot._mode_by_tensor_stream = {}
    for key, mode in codec._mode_by_tensor_stream.items():
        identity = id(mode)
        copied = copied_modes.get(identity)
        if copied is None:
            copied = copied_modes[identity] = dict(mode)
        snapshot._mode_by_tensor_stream[key] = copied

    # Transfer mapping depends on one particular target. It is runtime state,
    # not part of the reusable source setup, and must never cross persistence
    # or NeuralZip operation boundaries.
    snapshot._transfer_mode_by_tensor_stream = {}

    # Runtime lookup tables are reconstructed from immutable payload headers.
    snapshot._decode_codebooks = {}
    snapshot._codebook_payload_by_id = {}
    return snapshot


def _freeze_native_codebooks(codec):
    payload_by_id = {
        int(payload.codebook_id): payload for payload in codec.export_codebooks()
    }
    for mode in _unique_modes(codec):
        codebook = mode.get("codebook")
        if codebook is None or isinstance(codebook, tuple):
            continue
        expected_header = bytes(codebook.codebook)
        codebook_id = mode.get("codebook_id")
        payload = payload_by_id.get(codebook_id)
        if payload is None:
            raise ValueError(f"Setup is missing codebook {codebook_id!r}.")
        if bytes(payload.codebook) != expected_header:
            raise ValueError(f"Codebook {codebook_id} disagrees with its payload.")
        mode["codebook"] = (SETUP_MARKER, expected_header)


def _restore_live_codebooks(codec):
    payload_by_id = {
        int(payload.codebook_id): payload for payload in codec.export_codebooks()
    }
    for mode in _unique_modes(codec):
        marker = mode.get("codebook")
        if not isinstance(marker, tuple):
            continue
        if len(marker) != 2 or marker[0] != SETUP_MARKER:
            raise ValueError("Unsupported frozen Huff0 codebook marker.")
        _, expected_header = marker
        codebook = huff0.build_codebook_from_header(expected_header)
        if codebook.codebook != expected_header:
            raise RuntimeError("Rebuilt Huff0 codebook differs from the frozen setup.")

        codebook_id = mode.get("codebook_id")
        payload = payload_by_id.get(codebook_id)
        if payload is None:
            raise ValueError(f"Frozen setup is missing codebook {codebook_id!r}.")
        if bytes(payload.codebook) != bytes(expected_header):
            raise ValueError(
                f"Frozen codebook {codebook_id} disagrees with its payload."
            )
        mode["codebook"] = codebook
    codec.load_codebooks(codec.export_codebooks())


def setup_from_codec(codec, setup_report=None):
    """Capture a prepared codec as an immutable, shareable setup value."""

    validate_setup(codec)
    if setup_report is not None and not isinstance(setup_report, SetupReport):
        raise TypeError("setup_report must be a SetupReport or None.")
    snapshot = _detached_codec_snapshot(codec)
    _freeze_native_codebooks(snapshot)
    state = pickle.dumps(snapshot, protocol=5)
    return NeuralZipSetup(state, setup_report=setup_report)


def load_setup(value_or_path):
    """Normalize a setup value or trusted pickle path to ``NeuralZipSetup``.

    Pickle can execute arbitrary code. ``value_or_path`` must therefore refer
    only to a current-format file produced by a fully trusted NeuralZip process.
    """

    if isinstance(value_or_path, NeuralZipSetup):
        NeuralZipSetup.__post_init__(value_or_path)
        return value_or_path
    path = Path(value_or_path)
    with path.open("rb") as handle:
        setup = pickle.load(handle)
    if not isinstance(setup, NeuralZipSetup):
        raise TypeError(f"Unsupported NeuralZip setup payload: {type(setup)!r}.")
    NeuralZipSetup.__post_init__(setup)
    return setup


def materialize_setup(value_or_path, *, threads=None):
    """Build one independent live codec from a setup value or trusted path."""

    setup = load_setup(value_or_path)
    value = pickle.loads(setup._codec_state)
    if not isinstance(value, NeuralZipCodec):
        raise TypeError(f"Unsupported NeuralZip setup state: {type(value)!r}.")
    validate_setup(value)
    _restore_live_codebooks(value)
    if threads is not None:
        threads = int(threads)
        if threads < 0:
            raise ValueError("threads must be non-negative.")
        value.threads = threads
    return value


def _fsync_directory(path):
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save_setup(value, path):
    """Atomically and durably persist a current ``NeuralZipSetup`` value."""

    setup = load_setup(value)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            pickle.dump(setup, handle, protocol=5)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        _fsync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path
