import torch

from neuralzip.payloads import (
    ByteGroupPayload,
    CodebookPayload,
    StreamChunk,
    TensorPayload,
    storage_bytes,
)

FORMAT_VERSION = 2


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
                "codebook": [codebook_offset, len(value.codebook)],
                "vocab": [vocab_offset, len(value.vocab)],
            }
        )
    return prefix, codebook_metadata




def unpack_storage(blob, metadata):
    """Borrow read-only payload views that keep the storage buffer alive.

    The caller must not mutate the underlying storage while using these views.
    """
    if int(metadata["version"]) != FORMAT_VERSION:
        raise ValueError("Unsupported NeuralZip storage checkpoint.")
    if torch.is_tensor(blob):
        blob = blob.detach().cpu().contiguous().numpy().reshape(-1)
    blob = memoryview(blob)
    if not blob.c_contiguous or not blob.nbytes:
        blob = memoryview(blob.tobytes())
    blob = blob.cast("B").toreadonly()

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


def pack_direct_storage(
    named_tensors, specs, codebooks, prefix, codebook_metadata, packed, native_groups
):
    """Assemble native buffer spans without materializing per-chunk payloads."""
    removed_prefix_bytes = 0
    used = {chunk[2] for groups in native_groups for group in groups for chunk in group}
    retained = tuple(value for value in codebooks if value.codebook_id in used)
    if len(retained) != len(codebooks):
        filtered_prefix, codebook_metadata = _pack_codebooks(retained)
        removed_prefix_bytes = len(prefix) - len(filtered_prefix)
        packed[: len(prefix)] = filtered_prefix
        codebooks = retained

    tensors = []
    for (name, tensor), spec, groups in zip(
        named_tensors, specs, native_groups, strict=True
    ):
        config = spec[1]
        info = {
            "name": str(name),
            "shape": [int(dimension) for dimension in tensor.shape],
            "dtype": config.name,
            "element_size": config.element_size,
            "numel": int(tensor.numel()),
            "bit_reorder": int(config.bit_reorder),
        }
        encoded_groups = []
        for group in groups:
            encoded_groups.append(
                [
                    {
                        "mode": str(mode),
                        "symbols": int(symbols),
                        "codebook": None if int(codebook_id) < 0 else int(codebook_id),
                        "payload": [int(offset), int(size)],
                    }
                    for (
                        mode,
                        symbols,
                        codebook_id,
                        offset,
                        size,
                    ) in group
                ]
            )
        tensors.append({**info, "groups": encoded_groups})
    if removed_prefix_bytes:
        for info in tensors:
            for group in info["groups"]:
                for chunk in group:
                    span = chunk["payload"]
                    if span[0] >= len(prefix):
                        span[0] -= removed_prefix_bytes
    blob = _bytes_tensor(packed)
    metadata = {
        "version": FORMAT_VERSION,
        "codebooks": codebook_metadata,
        "tensors": tensors,
    }
    return blob, metadata, storage_bytes(codebooks, tensors, blob.numel())
