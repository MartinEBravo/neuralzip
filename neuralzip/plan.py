"""Public facade for fitting, transferring, encoding, and persisting setup."""

import pickle
from dataclasses import dataclass
from pathlib import Path

import torch

from . import huff0
from .codec_protocol import (
    SetupReport,
    decode_tensor_batch,
    encode_tensor_batch,
    named_tensor_items,
    prepare_neuralzip_codec,
)
from .hashing import sha256_bytes
from .neuralzip_codec import NeuralZipCodec
from .zipnn_codec import _zipnn_dtype_config

PLAN_MARKER = "frozen-huff0-codebook-v2"
LEGACY_PLAN_MARKER = "frozen-huff0-codebook-v1"
POST_SETUP_MODES = frozenset({"raw", "shared_huff0", "packed_huff0"})


def codebook_digest(codebooks):
    """Return the stable digest historically used by experiment runners."""

    return sha256_bytes(pickle.dumps(tuple(codebooks), protocol=5))


def _replace_native_codebooks(codec, named_tensors):
    available_streams = set()
    for name, tensor in named_tensors:
        if (
            not torch.is_tensor(tensor)
            or not tensor.is_floating_point()
            or tensor.numel() == 0
        ):
            continue
        config = _zipnn_dtype_config(tensor.dtype)
        available_streams.update(
            (str(name), str(tensor.dtype).removeprefix("torch."), stream_id)
            for stream_id in range(int(config["num_buffers"]))
        )
    missing = sorted(set(codec._mode_by_tensor_stream) - available_streams)
    if missing:
        raise KeyError(missing[0])

    replaced = []
    try:
        for mode in codec._mode_by_key.values():
            codebook = mode.get("codebook")
            if codebook is None:
                continue
            marker = (
                PLAN_MARKER,
                bytes(codebook.codebook),
            )
            replaced.append((mode, codebook))
            mode["codebook"] = marker
    except Exception:
        for mode, codebook in reversed(replaced):
            mode["codebook"] = codebook
        raise
    return replaced


def _restore_native_codebooks(codec):
    seen = set()
    payload_by_id = {
        int(payload.codebook_id): payload for payload in codec.export_codebooks()
    }
    mode_maps = (
        codec._mode_by_key,
        codec._mode_by_tensor_stream,
        codec._transfer_mode_by_tensor_stream,
    )
    for mode_map in mode_maps:
        for mode in mode_map.values():
            if id(mode) in seen:
                continue
            seen.add(id(mode))
            marker = mode.get("codebook")
            if not isinstance(marker, tuple):
                continue
            if len(marker) == 2 and marker[0] == PLAN_MARKER:
                _, expected_header = marker
                codebook = huff0.build_codebook_from_header(expected_header)
            elif len(marker) == 4 and marker[0] == LEGACY_PLAN_MARKER:
                _, tokens, prior_count, expected_header = marker
                codebook = huff0.build_codebook(tokens, prior_count=int(prior_count))
            else:
                raise ValueError("Unsupported frozen Huff0 codebook marker.")
            if codebook.codebook != expected_header:
                raise RuntimeError(
                    "Rebuilt Huff0 codebook differs from the frozen plan."
                )
            codebook_id = mode.get("codebook_id")
            payload = payload_by_id.get(codebook_id)
            if payload is None:
                raise ValueError(
                    f"Frozen plan is missing codebook payload {codebook_id!r}."
                )
            if bytes(payload.codebook) != bytes(expected_header):
                raise ValueError(
                    f"Frozen codebook {codebook_id} disagrees with its payload."
                )
            mode["codebook"] = codebook
    codec.load_codebooks(codec.export_codebooks())


def _validate_post_setup_modes(codec):
    """Reject plans that can construct a target-local Huffman tree."""

    if (
        int(getattr(codec, "prior_count", 1)) <= 0
        or int(getattr(codec, "exponent_prior_count", 1)) <= 0
    ):
        raise ValueError(
            "Transferable NeuralZip plans require positive Huffman priors."
        )

    mode_maps = (
        codec._mode_by_key,
        codec._mode_by_tensor_stream,
        codec._transfer_mode_by_tensor_stream,
    )
    invalid = sorted(
        {
            str(mode.get("kind"))
            for mode_map in mode_maps
            for mode in mode_map.values()
            if str(mode.get("kind")) not in POST_SETUP_MODES
        }
    )
    if invalid:
        joined = ", ".join(invalid)
        raise ValueError(
            "NeuralZip post-setup plans may only use raw, shared, or packed "
            f"modes; found {joined}. Refit the setup."
        )


@dataclass
class NeuralZipPlan:
    """A fitted NeuralZip setup with explicit transfer and encoding methods."""

    codec: NeuralZipCodec
    setup_report: SetupReport | None = None

    @classmethod
    def fit(cls, named_tensors, *, codec=None, **codec_options):
        named_tensors = named_tensor_items(named_tensors)
        codec = NeuralZipCodec(**codec_options) if codec is None else codec
        setup_report = prepare_neuralzip_codec(codec, named_tensors)
        _validate_post_setup_modes(codec)
        return cls(codec=codec, setup_report=setup_report)

    @property
    def codebooks(self):
        return self.codec.export_codebooks()

    @property
    def digest(self):
        return codebook_digest(self.codebooks)

    def map_target(self, named_tensors):
        """Apply the frozen setup by exact tensor name, dtype, and stream."""

        _validate_post_setup_modes(self.codec)
        named_tensors = named_tensor_items(named_tensors)
        return self.codec.prepare_direct_transfer_target(
            named_tensors,
        )

    def encode(self, named_tensors):
        _validate_post_setup_modes(self.codec)
        return encode_tensor_batch(self.codec, named_tensor_items(named_tensors))

    def decode(self, payloads):
        tensors, _ = decode_tensor_batch(self.codec, payloads)
        return tensors

    def prepare_inference(
        self,
        model,
        *,
        adapter=None,
        include_embeddings=False,
        device=None,
        block_devices=None,
        cpu_offload=False,
        cpu_offload_blocks=None,
        pin_memory=True,
    ):
        """Attach post-setup BF16 GPU inference and return ``model``.

        The setup must use sharing-only exponent modes for weights selected by
        the runtime. Packed exponent groups currently require a storage decode
        and are rejected by the GPU path.
        """

        from .gpu_inference import NeuralZipModel

        return NeuralZipModel.from_model(
            model,
            self,
            adapter=adapter,
            include_embeddings=include_embeddings,
            device=device,
            block_devices=block_devices,
            cpu_offload=cpu_offload,
            cpu_offload_blocks=cpu_offload_blocks,
            pin_memory=pin_memory,
        )

    def save(self, path, source_tensors):
        """Persist the plan without serializing process-local native handles."""

        _validate_post_setup_modes(self.codec)
        path = Path(path)
        source_tensors = named_tensor_items(source_tensors)
        replaced = []
        decodebooks = self.codec._decode_codebooks
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            replaced = _replace_native_codebooks(self.codec, source_tensors)
            self.codec._decode_codebooks = {}
            with temporary.open("wb") as handle:
                pickle.dump(self.codec, handle, protocol=5)
        finally:
            self.codec._decode_codebooks = decodebooks
            for mode, codebook in replaced:
                mode["codebook"] = codebook
        temporary.replace(path)

    @classmethod
    def load(cls, path):
        with Path(path).open("rb") as handle:
            value = pickle.load(handle)
        codec = value.codec if isinstance(value, cls) else value
        if not isinstance(codec, NeuralZipCodec):
            raise TypeError(f"Unsupported NeuralZip plan payload: {type(codec)!r}.")
        legacy_layout = getattr(codec, "exponent_layout", "original")
        if legacy_layout != "original":
            raise ValueError(
                "Legacy NeuralZip plans with transposed exponent streams cannot "
                "be used by the native-only encoder. Refit the setup with the "
                "original tensor layout."
            )
        codec.__dict__.pop("exponent_layout", None)
        _validate_post_setup_modes(codec)
        _restore_native_codebooks(codec)
        return cls(codec=codec)
