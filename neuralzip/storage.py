from dataclasses import dataclass

import torch

from neuralzip import huff0
from neuralzip.compressor import NeuralZipCompressor
from neuralzip.dtypes import DTYPES
from neuralzip.payloads import (
    ByteGroupPayload,
    CodebookPayload,
    StreamChunk,
    TensorPayload,
    storage_bytes,
)
from neuralzip.setup import NeuralZipSetup
from neuralzip.utils import _model_tensors, _stream_chunk_size, time_call

FORMAT_VERSION = 1


@dataclass(frozen=True)
class NeuralZipStats:
    setup_seconds: float
    encoding_seconds: float
    raw_bytes: int
    compressed_bytes: int
    exponent_raw_bytes: int | None
    exponent_compressed_bytes: int | None
    representation: str = "storage"
    decoding_seconds: float = 0.0
    runtime_encoding_seconds: float = 0.0
    stream_raw_bytes: int | None = None
    stream_compressed_bytes: int | None = None

    @property
    def compression_ratio(self):
        return self.compressed_bytes / self.raw_bytes if self.raw_bytes else 1.0

    @property
    def exponent_ratio(self):
        if self.exponent_raw_bytes is None or self.exponent_compressed_bytes is None:
            return None
        return (
            self.exponent_compressed_bytes / self.exponent_raw_bytes
            if self.exponent_raw_bytes
            else 1.0
        )

    @property
    def stream_ratio(self):
        """Selected GPU lanes plus their codebooks; excludes resident lanes."""
        if not self.stream_raw_bytes or self.stream_compressed_bytes is None:
            return None
        return self.stream_compressed_bytes / self.stream_raw_bytes

    @property
    def total_seconds(self):
        return self.setup_seconds + self.encoding_seconds


def setup_seconds(setup):
    return (
        0.0 if setup is None else float(NeuralZipSetup.load(setup).setup_seconds or 0.0)
    )


def storage_stats(setup, sizes, encoding_seconds):
    return NeuralZipStats(
        setup_seconds=setup_seconds(setup),
        encoding_seconds=float(encoding_seconds),
        raw_bytes=int(sizes["raw_bytes"]),
        compressed_bytes=int(sizes["compressed_bytes"]),
        exponent_raw_bytes=sizes["exponent_raw_bytes"],
        exponent_compressed_bytes=sizes["exponent_compressed_bytes"],
    )


def _bytes_tensor(data):
    if not data:
        return torch.empty(0, dtype=torch.uint8)
    buffer = data if isinstance(data, bytearray) else bytearray(data)
    return torch.frombuffer(buffer, dtype=torch.uint8)


def _pack_codebooks(codebooks):
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
    return prefix, codebook_metadata


def pack_storage(codebooks, tensors):
    """Pack the full storage codec into one byte tensor and JSON-safe metadata."""
    data, codebook_metadata = _pack_codebooks(codebooks)

    def append(value):
        value = bytes(value)
        span = [len(data), len(value)]
        data.extend(value)
        return span

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


def _direct_storage(compressor, named_tensors):
    specs = []
    tensor_info = []
    codebook_ids = set()
    for name, tensor in named_tensors:
        config = DTYPES[tensor.dtype]
        chunk_size = _stream_chunk_size(compressor.chunk_size, config.element_size)
        if chunk_size > huff0.MAX_BLOCK_SIZE:
            return None
        modes = (
            tuple(
                compressor._mode_for_tensor_stream(name, tensor, stream)
                for stream in config.stream_ids
            )
            if tensor.numel()
            else ({"kind": "raw"},) * config.element_size
        )
        selected = [
            stream for stream in config.stream_ids if modes[stream]["kind"] != "raw"
        ]
        if len(selected) > 1:
            return None
        stream = selected[0] if selected else config.exponent_stream
        if stream is None:
            stream = config.element_size - 1
        mode = modes[stream]
        if mode["kind"] not in {"raw", "shared_huff0", "packed_huff0"}:
            return None
        if selected:
            codebook_ids.add(int(mode["codebook_id"]))
        original = tensor.detach().cpu().contiguous()
        specs.append(
            (
                original.reshape(-1).view(torch.uint8).numpy(),
                config,
                stream,
                chunk_size,
                mode,
            )
        )
        tensor_info.append(
            {
                "name": str(name),
                "shape": [int(dimension) for dimension in original.shape],
                "dtype": config.name,
                "element_size": config.element_size,
                "numel": int(original.numel()),
                "bit_reorder": int(config.bit_reorder),
            }
        )

    codebooks = tuple(
        value
        for value in compressor.export_codebooks()
        if int(value.codebook_id) in codebook_ids
    )
    prefix, codebook_metadata = _pack_codebooks(codebooks)
    packed, native_groups = huff0.compress_direct_storage(
        specs,
        prefix,
        min_saving_bytes=compressor.min_saving_bytes,
        threads=compressor.threads,
        compression_threshold=compressor.compression_threshold,
    )
    removed_prefix_bytes = 0
    used = {chunk[2] for groups in native_groups for group in groups for chunk in group}
    retained = tuple(value for value in codebooks if value.codebook_id in used)
    if len(retained) != len(codebooks):
        filtered_prefix, codebook_metadata = _pack_codebooks(retained)
        removed_prefix_bytes = len(prefix) - len(filtered_prefix)
        packed[: len(prefix)] = filtered_prefix
        codebooks = retained

    tensors = []
    for info, spec, groups in zip(tensor_info, specs, native_groups, strict=True):
        encoded_groups = []
        for stream, group in enumerate(groups):
            if stream != spec[2] and group:
                offset = group[0][3]
                # Raw sibling lanes are contiguous, so rechunk only their spans.
                group = [
                    (
                        "raw",
                        min(spec[3], info["numel"] - start),
                        -1,
                        offset + start,
                        min(spec[3], info["numel"] - start),
                        0,
                        0,
                        0,
                        0,
                    )
                    for start in range(0, info["numel"], spec[3])
                ]
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
    if removed_prefix_bytes:
        for info in tensors:
            for group in info["groups"]:
                for chunk in group:
                    for field in ("payload", "escapes", "tail"):
                        span = chunk[field]
                        if span[0] >= len(prefix):
                            span[0] -= removed_prefix_bytes
    blob = _bytes_tensor(packed)
    metadata = {
        "version": FORMAT_VERSION,
        "codebooks": codebook_metadata,
        "tensors": tensors,
    }
    return blob, metadata, storage_bytes(codebooks, tensors, blob.numel())


def _encode_storage(compressor, named_tensors):
    direct = _direct_storage(compressor, named_tensors)
    if direct is not None:
        return direct
    payloads = compressor.compress_tensors(named_tensors)
    used_codebooks = {
        chunk.codebook_id
        for tensor in payloads
        for group in tensor.groups
        for chunk in group.chunks
        if chunk.codebook_id is not None
    }
    codebooks = tuple(
        value
        for value in compressor.export_codebooks()
        if value.codebook_id in used_codebooks
    )
    blob, metadata = pack_storage(codebooks, payloads)
    return blob, metadata, storage_bytes(codebooks, metadata["tensors"], blob.numel())


def encode_storage(named_tensors, setup):
    """Time encoding and assembly; prepare setup and target mapping separately."""
    named_tensors = tuple(named_tensors)
    compressor = NeuralZipCompressor(setup, gpu=True)
    compressor.prepare_storage_transfer_target(named_tensors)
    encoded, seconds = time_call(_encode_storage, compressor, named_tensors)
    return *encoded, seconds


def decode_storage(blob, metadata):
    """Decode a persistent full-storage representation to named CPU tensors."""
    codebooks, payloads = unpack_storage(blob, metadata)
    compressor = NeuralZipCompressor()
    compressor.load_codebooks(codebooks)
    tensors = compressor.decompress_tensors(payloads)
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
