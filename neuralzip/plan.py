"""Public facade for fitting, transferring, encoding, and persisting setup."""

from dataclasses import dataclass
import hashlib
from pathlib import Path
import pickle

import torch

from . import huff0
from .codec_protocol import (
    SetupReport,
    decode_tensor_batch,
    encode_tensor_batch,
    named_tensor_items,
    prepare_neuralzip_codec,
)
from .neuralzip_codec import NeuralZipCodec
from .packed_exponents import tokenize_ngrams
from .zipnn_codec import _zipnn_dtype_config, _zipnn_reordered_bytes


PLAN_MARKER = "frozen-huff0-codebook-v1"


def codebook_digest(codebooks):
    """Return the stable digest historically used by experiment runners."""

    return hashlib.sha256(
        pickle.dumps(tuple(codebooks), protocol=5)
    ).hexdigest()


def _source_samples(codec, named_tensors):
    sample_parts = {}
    sample_sizes = {}
    for name, tensor in sorted(named_tensors, key=lambda item: item[0]):
        original = tensor.detach().cpu().contiguous()
        config = _zipnn_dtype_config(original.dtype)
        reordered = _zipnn_reordered_bytes(original, config)
        element_size = int(config["element_size"])
        for stream_id in range(int(config["num_buffers"])):
            is_exponent = codec._is_exponent_stream(config, stream_id)
            stream = reordered[stream_id::element_size]
            stream, _ = codec._apply_exponent_layout(
                original,
                stream,
                is_exponent,
            )
            key = codec._codebook_key(name, original, stream_id)
            remaining = codec.max_codebook_sample_bytes - sample_sizes.get(key, 0)
            if remaining <= 0:
                continue
            part = codec._bounded_stream_sample(stream, remaining)
            if not part:
                continue
            sample_parts.setdefault(key, []).append(bytes(part))
            sample_sizes[key] = sample_sizes.get(key, 0) + len(part)
    return {key: b"".join(parts) for key, parts in sample_parts.items()}


def _replace_native_codebooks(codec, named_tensors):
    samples = _source_samples(codec, named_tensors)
    replaced = []
    try:
        for key, mode in codec._mode_by_key.items():
            codebook = mode.get("codebook")
            if codebook is None:
                continue
            sample = samples[key]
            if mode["kind"] == "packed_huff0":
                tokens, _, _, _, _ = tokenize_ngrams(
                    sample,
                    int(mode["ngram_n"]),
                    mode["vocab"],
                )
            else:
                tokens = sample
            dtype = str(key[1])
            stream_id = int(key[2])
            element_size = int(
                _zipnn_dtype_config(getattr(torch, dtype))["element_size"]
            )
            prior_count = (
                codec.exponent_prior_count
                if stream_id == element_size - 1
                else codec.prior_count
            )
            marker = (
                PLAN_MARKER,
                bytes(tokens),
                int(prior_count),
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
            if not (
                isinstance(marker, tuple)
                and len(marker) == 4
                and marker[0] == PLAN_MARKER
            ):
                continue
            _, tokens, prior_count, expected_header = marker
            codebook = huff0.build_codebook(tokens, prior_count=int(prior_count))
            if codebook.codebook != expected_header:
                raise RuntimeError(
                    "Rebuilt Huff0 codebook differs from the frozen plan."
                )
            mode["codebook"] = codebook
    codec.load_codebooks(codec.export_codebooks())


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
        codec.clear_prepacked_cache()
        return cls(codec=codec, setup_report=setup_report)

    @property
    def codebooks(self):
        return self.codec.export_codebooks()

    @property
    def digest(self):
        return codebook_digest(self.codebooks)

    def map_target(
        self,
        named_tensors,
        *,
        strategy="structural",
        require_full_match=True,
        js_threshold=None,
    ):
        named_tensors = named_tensor_items(named_tensors)
        if strategy in {"structural", "direct"}:
            return self.codec.prepare_direct_transfer_target(
                named_tensors,
                require_full_match=require_full_match,
            )
        if strategy in {"js", "jensen-shannon"}:
            return self.codec.prepare_transfer_target(
                named_tensors,
                js_threshold=js_threshold,
            )
        raise ValueError("strategy must be 'structural' or 'js'.")

    def materialize(self, named_tensors):
        return self.codec.materialize_precompute(named_tensor_items(named_tensors))

    def encode(self, named_tensors, *, clear_materialization=False):
        if clear_materialization:
            self.codec.clear_prepacked_cache()
        return encode_tensor_batch(self.codec, named_tensor_items(named_tensors))

    def decode(self, payloads):
        tensors, _ = decode_tensor_batch(self.codec, payloads)
        return tensors

    def save(self, path, source_tensors):
        """Persist the plan without serializing process-local native handles."""

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
        _restore_native_codebooks(codec)
        return cls(codec=codec)
