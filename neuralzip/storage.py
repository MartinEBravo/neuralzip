import time
from dataclasses import dataclass

import torch

from neuralzip import huff0
from neuralzip.codec.codec import NeuralZipCodec
from neuralzip.payloads import (
    CHUNK_METADATA_BYTES,
    ByteGroupPayload,
    CodebookPayload,
    StreamChunk,
    TensorPayload,
    serialized_size,
)
from neuralzip.setup import load_setup, materialize_setup
from neuralzip.utils import DTYPES, _model_tensors

FORMAT_VERSION = 1


@dataclass(frozen=True)
class NeuralZipStats:
    setup_seconds: float
    encoding_seconds: float
    raw_bytes: int
    compressed_bytes: int
    exponent_raw_bytes: int
    exponent_compressed_bytes: int
    representation: str = "storage"
    decoding_seconds: float = 0.0
    runtime_encoding_seconds: float = 0.0

    @property
    def compression_ratio(self):
        return self.compressed_bytes / self.raw_bytes if self.raw_bytes else 1.0

    @property
    def exponent_ratio(self):
        return (
            self.exponent_compressed_bytes / self.exponent_raw_bytes
            if self.exponent_raw_bytes
            else 1.0
        )

    @property
    def total_seconds(self):
        return self.setup_seconds + self.encoding_seconds


def setup_seconds(setup):
    return 0.0 if setup is None else float(load_setup(setup).setup_seconds or 0.0)


def storage_stats(setup, sizes, encoding_seconds):
    return NeuralZipStats(
        setup_seconds=setup_seconds(setup),
        encoding_seconds=float(encoding_seconds),
        raw_bytes=int(sizes["raw_bytes"]),
        compressed_bytes=int(sizes["compressed_bytes"]),
        exponent_raw_bytes=int(sizes["exponent_raw_bytes"]),
        exponent_compressed_bytes=int(sizes["exponent_compressed_bytes"]),
    )


def _bytes_tensor(data):
    if not data:
        return torch.empty(0, dtype=torch.uint8)
    buffer = data if isinstance(data, bytearray) else bytearray(data)
    return torch.frombuffer(buffer, dtype=torch.uint8)


def pack_storage(codebooks, tensors):
    """Pack the full storage codec into one byte tensor and JSON-safe metadata."""
    data = bytearray()

    def append(value):
        value = bytes(value)
        span = [len(data), len(value)]
        data.extend(value)
        return span

    codebook_metadata = []
    for value in codebooks:
        codebook_metadata.append(
            {
                "id": int(value.codebook_id),
                "stream": int(value.stream_id),
                "dtype": str(value.dtype),
                "mode": str(value.mode),
                "ngram": int(value.ngram_n),
                "codebook": append(value.codebook),
                "vocab": append(value.vocab),
            }
        )

    tensor_metadata = []
    for value in tensors:
        groups = []
        for group in value.groups:
            chunks = []
            for chunk in group.chunks:
                chunks.append(
                    {
                        "mode": str(chunk.mode),
                        "symbols": int(chunk.symbol_count),
                        "codebook": (
                            None
                            if chunk.codebook_id is None
                            else int(chunk.codebook_id)
                        ),
                        "payload": append(chunk.payload),
                        "escapes": append(chunk.escape_payload),
                        "tail": append(chunk.tail_payload),
                    }
                )
            groups.append(chunks)
        tensor_metadata.append(
            {
                "name": str(value.name),
                "shape": [int(dimension) for dimension in value.shape],
                "dtype": str(value.dtype),
                "element_size": int(value.element_size),
                "numel": int(value.numel),
                "bit_reorder": int(value.bit_reorder),
                "groups": groups,
            }
        )
    metadata = {
        "version": FORMAT_VERSION,
        "codebooks": codebook_metadata,
        "tensors": tensor_metadata,
    }
    return _bytes_tensor(data), metadata


def unpack_storage(blob, metadata):
    """Recreate full storage payloads from ``pack_storage`` output."""
    if int(metadata["version"]) != FORMAT_VERSION:
        raise ValueError("Unsupported NeuralZip storage checkpoint.")
    if torch.is_tensor(blob):
        blob = blob.detach().cpu().contiguous().numpy().tobytes()
    else:
        blob = bytes(blob)

    def take(span):
        offset, size = map(int, span)
        if offset < 0 or size < 0 or offset + size > len(blob):
            raise ValueError("A NeuralZip storage span is outside its data tensor.")
        return blob[offset : offset + size]

    codebooks = tuple(
        CodebookPayload(
            codebook_id=value["id"],
            stream_id=value["stream"],
            dtype=value["dtype"],
            mode=value["mode"],
            codebook=take(value["codebook"]),
            ngram_n=value["ngram"],
            vocab=take(value["vocab"]),
        )
        for value in metadata["codebooks"]
    )
    tensors = []
    for value in metadata["tensors"]:
        groups = tuple(
            ByteGroupPayload(
                chunks=tuple(
                    StreamChunk(
                        mode=chunk["mode"],
                        symbol_count=chunk["symbols"],
                        codebook_id=chunk["codebook"],
                        payload=take(chunk["payload"]),
                        escape_payload=take(chunk["escapes"]),
                        tail_payload=take(chunk["tail"]),
                    )
                    for chunk in group
                )
            )
            for group in value["groups"]
        )
        tensors.append(
            TensorPayload(
                name=value["name"],
                shape=tuple(value["shape"]),
                dtype=value["dtype"],
                element_size=value["element_size"],
                numel=value["numel"],
                groups=groups,
                bit_reorder=value["bit_reorder"],
            )
        )
    return codebooks, tuple(tensors)


def storage_bytes(codebooks, tensors):
    """Return canonical full-model and exponent sizes for experiment reports."""
    codebooks = tuple(codebooks)
    tensors = tuple(tensors)
    raw = sum(tensor.raw_bytes for tensor in tensors)
    exponent_raw = sum(tensor.numel for tensor in tensors)
    exponent_streams = {(tensor.dtype, len(tensor.groups) - 1) for tensor in tensors}
    exponent_compressed = sum(
        chunk.compressed_bytes
        for tensor in tensors
        for chunk in tensor.groups[-1].chunks
    ) + sum(
        value.metadata_bytes
        for value in codebooks
        if (value.dtype, value.stream_id) in exponent_streams
    )
    return {
        "raw_bytes": int(raw),
        "compressed_bytes": int(serialized_size(codebooks, tensors)),
        "exponent_raw_bytes": int(exponent_raw),
        "exponent_compressed_bytes": int(exponent_compressed),
    }


def _text_size(value):
    return 4 + len(str(value).encode("utf-8"))


def _direct_storage_bytes(codebooks, tensors, blob_bytes):
    raw_bytes = sum(tensor["numel"] * tensor["element_size"] for tensor in tensors)
    exponent_raw_bytes = sum(tensor["numel"] for tensor in tensors)
    structural_bytes = 4 + 2 + _text_size("neuralzip") + 8
    for codebook in codebooks:
        structural_bytes += (
            8 + 4 + _text_size(codebook.dtype) + _text_size(codebook.mode) + 1 + 8 + 8
        )
    structural_bytes += 8
    for tensor in tensors:
        structural_bytes += (
            _text_size(tensor["name"])
            + _text_size(tensor["dtype"])
            + 24
            + 8 * len(tensor["shape"])
            + 8
        )
        for group in tensor["groups"]:
            structural_bytes += 8 + sum(
                CHUNK_METADATA_BYTES[chunk["mode"]] for chunk in group
            )
    exponent_compressed_bytes = sum(
        CHUNK_METADATA_BYTES[chunk["mode"]]
        + chunk["payload"][1]
        + chunk["escapes"][1]
        + chunk["tail"][1]
        for tensor in tensors
        for chunk in tensor["groups"][-1]
    )
    exponent_streams = {(tensor["dtype"], 1) for tensor in tensors}
    exponent_compressed_bytes += sum(
        codebook.metadata_bytes
        for codebook in codebooks
        if (codebook.dtype, codebook.stream_id) in exponent_streams
    )
    return {
        "raw_bytes": int(raw_bytes),
        "compressed_bytes": int(blob_bytes + structural_bytes),
        "exponent_raw_bytes": int(exponent_raw_bytes),
        "exponent_compressed_bytes": int(exponent_compressed_bytes),
    }


def _direct_bf16_storage(codec, named_tensors):
    specs = []
    tensor_info = []
    codebook_ids = set()
    for name, tensor in named_tensors:
        if tensor.dtype != torch.bfloat16:
            return None
        original = tensor.detach().cpu().contiguous()
        config = DTYPES[original.dtype]
        if original.numel():
            modes = tuple(
                codec._mode_for_tensor_stream(name, original, stream)
                for stream in range(2)
            )
        else:
            modes = ({"kind": "raw"}, {"kind": "raw"})
        if modes[0].get("kind") != "raw" or modes[1].get("kind") not in {
            "raw",
            "shared_huff0",
            "packed_huff0",
        }:
            return None
        if modes[1].get("kind") != "raw":
            codebook_ids.add(int(modes[1]["codebook_id"]))
        specs.append(
            (
                original.reshape(-1).view(torch.uint8).numpy(),
                codec._stream_chunk_size(2),
                modes[1],
            )
        )
        tensor_info.append(
            {
                "name": str(name),
                "shape": [int(dimension) for dimension in original.shape],
                "dtype": config["name"],
                "element_size": 2,
                "numel": int(original.numel()),
                "bit_reorder": int(config["bit_reorder"]),
            }
        )

    codebooks = tuple(
        value
        for value in codec.export_codebooks()
        if int(value.codebook_id) in codebook_ids
    )
    prefix = bytearray()
    codebook_metadata = []
    for value in codebooks:
        codebook_offset = len(prefix)
        prefix.extend(value.codebook)
        vocab_offset = len(prefix)
        prefix.extend(value.vocab)
        codebook_metadata.append(
            {
                "id": int(value.codebook_id),
                "stream": int(value.stream_id),
                "dtype": str(value.dtype),
                "mode": str(value.mode),
                "ngram": int(value.ngram_n),
                "codebook": [codebook_offset, len(value.codebook)],
                "vocab": [vocab_offset, len(value.vocab)],
            }
        )

    packed, native_groups = huff0.compress_bf16_storage(
        specs,
        prefix,
        min_saving_bytes=codec.min_saving_bytes,
        threads=codec.threads,
        compression_threshold=codec.compression_threshold,
    )
    tensors = []
    for info, groups in zip(tensor_info, native_groups, strict=True):
        encoded_groups = []
        for group in groups:
            encoded_groups.append(
                [
                    {
                        "mode": str(mode),
                        "symbols": int(symbols),
                        "codebook": None if int(codebook_id) < 0 else int(codebook_id),
                        "payload": [int(offset), int(size)],
                        "escapes": [int(escape_offset), int(escape_size)],
                        "tail": [int(tail_offset), int(tail_size)],
                    }
                    for (
                        mode,
                        symbols,
                        codebook_id,
                        offset,
                        size,
                        escape_offset,
                        escape_size,
                        tail_offset,
                        tail_size,
                    ) in group
                ]
            )
        tensors.append({**info, "groups": encoded_groups})
    blob = _bytes_tensor(packed)
    metadata = {
        "version": FORMAT_VERSION,
        "codebooks": codebook_metadata,
        "tensors": tensors,
    }
    return blob, metadata, _direct_storage_bytes(codebooks, tensors, blob.numel())


def encode_storage(named_tensors, setup):
    """Run the full precomputed codec and return its persistent representation."""
    named_tensors = tuple(named_tensors)
    codec = materialize_setup(setup, gpu=True)
    codec.prepare_storage_transfer_target(named_tensors)
    started = time.perf_counter()
    direct = _direct_bf16_storage(codec, named_tensors)
    if direct is not None:
        blob, metadata, sizes = direct
        return blob, metadata, sizes, time.perf_counter() - started
    payloads = codec.compress_tensors(named_tensors)
    encoding_seconds = time.perf_counter() - started
    used_codebooks = {
        chunk.codebook_id
        for tensor in payloads
        for group in tensor.groups
        for chunk in group.chunks
        if chunk.codebook_id is not None
    }
    codebooks = tuple(
        value
        for value in codec.export_codebooks()
        if value.codebook_id in used_codebooks
    )
    blob, metadata = pack_storage(codebooks, payloads)
    return blob, metadata, storage_bytes(codebooks, payloads), encoding_seconds


def decode_storage(blob, metadata):
    """Decode a persistent full-storage representation to named CPU tensors."""
    codebooks, payloads = unpack_storage(blob, metadata)
    codec = NeuralZipCodec()
    codec.load_codebooks(codebooks)
    tensors = codec.decompress_tensors(payloads)
    return {
        payload.name: tensor for payload, tensor in zip(payloads, tensors, strict=True)
    }


def _aliases(model):
    state_names = set(model.state_dict())
    aliases = {}
    for values in (
        model.named_parameters(remove_duplicate=False),
        model.named_buffers(remove_duplicate=False),
    ):
        canonical = {}
        for name, tensor in values:
            if name not in state_names:
                continue
            source = canonical.setdefault(id(tensor), str(name))
            if source != name:
                aliases[str(name)] = source
    return aliases


def _tensor_bytes(tensors):
    seen = set()
    total = 0
    for tensor in tensors:
        key = (
            tensor.device,
            tensor.data_ptr(),
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.dtype,
        )
        if key not in seen:
            seen.add(key)
            total += tensor.numel() * tensor.element_size()
    return total


def encode_model_storage(model, setup, runtime=None):
    """Encode all floating model state and retain only auxiliary tensors."""
    named_tensors = dict(_model_tensors(model))
    if runtime is not None:
        named_tensors.update(runtime.materialized_weights())
    named_tensors = tuple(sorted(named_tensors.items()))
    blob, metadata, sizes, seconds = encode_storage(named_tensors, setup)
    encoded_names = {name for name, _ in named_tensors}
    aliases = _aliases(model)
    resident = {
        name: tensor
        for name, tensor in model.state_dict().items()
        if name not in aliases
        and name not in encoded_names
        and aliases.get(name, name) not in encoded_names
    }
    auxiliary_bytes = _tensor_bytes(resident.values())
    sizes["raw_bytes"] += auxiliary_bytes
    sizes["compressed_bytes"] += auxiliary_bytes
    return blob, metadata, resident, aliases, sizes, seconds


def decode_model_storage(blob, metadata, resident, aliases):
    state = dict(resident)
    state.update(decode_storage(blob, metadata))
    for alias, canonical in aliases.items():
        state[alias] = state[canonical]
    return state
