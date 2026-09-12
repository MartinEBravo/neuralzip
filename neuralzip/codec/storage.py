import torch

from neuralzip import huff0
from neuralzip.payloads import ByteGroupPayload, StreamChunk, TensorPayload
from neuralzip.utils import DTYPES, _tensor_from_raw_bytes


class _StorageMixin:
    def _compress_direct_native_batch(self, tensor_specs):
        """Encode one input-size-windowed tensor batch through one job queue."""
        native_specs = [
            (
                tensor.reshape(-1).view(torch.uint8).numpy(),
                int(config["element_size"]),
                int(config["bit_reorder"]),
                self._stream_chunk_size(int(config["element_size"])),
                tensor_modes,
            )
            for _, tensor, config, tensor_modes in tensor_specs
        ]
        native_batch = huff0.compress_mixed_streams_many(
            native_specs,
            min_saving_bytes=self.min_saving_bytes,
            threads=self.threads,
            compression_threshold=self.compression_threshold,
        )
        return [
            self._tensor_payload_from_native_mixed(
                name, tensor, config, tensor_modes, native_groups
            )
            for (name, tensor, config, tensor_modes), native_groups in zip(
                tensor_specs, native_batch, strict=True
            )
        ]

    def compress_tensors(self, named_tensors):
        """Compress a model without retaining target-derived data."""
        named_tensors = list(named_tensors)
        outputs = []
        batch = []
        batch_bytes = 0

        def flush_batch():
            nonlocal batch, batch_bytes
            if batch:
                outputs.extend(self._compress_direct_native_batch(batch))
                batch = []
                batch_bytes = 0

        for name, tensor in named_tensors:
            config = DTYPES[tensor.dtype]
            raw_bytes = int(tensor.numel()) * int(config["element_size"])
            if batch and batch_bytes + raw_bytes > self.direct_batch_bytes:
                flush_batch()
            original = tensor.detach().cpu().contiguous()
            stream_count = int(config["element_size"])
            if original.numel() == 0:
                tensor_modes = ({"kind": "raw"},) * stream_count
            else:
                tensor_modes = tuple(
                    self._mode_for_tensor_stream(name, original, stream_id)
                    for stream_id in range(stream_count)
                )
            batch.append((name, original, config, tensor_modes))
            batch_bytes += raw_bytes
            if batch_bytes >= self.direct_batch_bytes:
                flush_batch()
        flush_batch()
        return tuple(outputs)

    def _tensor_payload_from_native_mixed(
        self, name, tensor, config, tensor_modes, native_groups
    ):
        """Assemble the stable archive objects returned by the native engine."""
        groups = []
        for planned_mode, native_chunks in zip(
            tensor_modes, native_groups, strict=True
        ):
            chunks = []
            for native_chunk in native_chunks:
                final_kind, payload, symbol_count, escapes, tail = native_chunk
                final_kind = str(final_kind)
                chunks.append(
                    StreamChunk(
                        mode=final_kind,
                        symbol_count=int(symbol_count),
                        payload=payload,
                        codebook_id=(
                            planned_mode.get("codebook_id")
                            if final_kind != "raw"
                            else None
                        ),
                        escape_payload=escapes,
                        tail_payload=tail,
                    )
                )
            groups.append(ByteGroupPayload(chunks=tuple(chunks)))

        return TensorPayload(
            name=name,
            shape=tuple(int(dim) for dim in tensor.shape),
            dtype=config["name"],
            element_size=int(config["element_size"]),
            numel=int(tensor.numel()),
            groups=tuple(groups),
            bit_reorder=int(config["bit_reorder"]),
        )

    def _decode_assets(self, chunk):
        codebook = self._codebook_payload_by_id[chunk.codebook_id]
        decodebook = self._decode_codebooks.get(chunk.codebook_id)
        if decodebook is None:
            decodebook = huff0.build_decodebook(codebook.codebook)
            self._decode_codebooks[chunk.codebook_id] = decodebook
        return codebook, decodebook

    def _native_decode_spec(self, compressed):
        """Translate one stable payload into the native mixed-stream ABI."""
        dtype = getattr(torch, compressed.dtype)
        config = DTYPES[dtype]
        native_groups = []
        for group in compressed.groups:
            native_chunks = []
            for chunk in group.chunks:
                mode = str(chunk.mode)
                decodebook_handle = None
                vocab = b""
                ngram_n = 1
                escape_payload = b""
                tail_payload = b""
                if mode in {"shared_huff0", "packed_huff0"}:
                    codebook, decodebook_handle = self._decode_assets(chunk)
                    if mode == "packed_huff0":
                        vocab = codebook.vocab
                        ngram_n = int(codebook.ngram_n)
                        escape_payload = chunk.escape_payload
                        tail_payload = chunk.tail_payload
                native_chunks.append(
                    (
                        mode,
                        chunk.payload,
                        int(chunk.symbol_count),
                        decodebook_handle,
                        vocab,
                        ngram_n,
                        escape_payload,
                        tail_payload,
                    )
                )
            native_groups.append(tuple(native_chunks))
        return (
            int(config["element_size"]),
            bool(config["bit_reorder"]),
            int(compressed.numel),
            tuple(native_groups),
        ), dtype

    def _decompress_direct_native_batch(self, payloads):
        prepared = [self._native_decode_spec(compressed) for compressed in payloads]
        raw_buffers = huff0.decompress_mixed_streams_many(
            tuple(spec for spec, _ in prepared),
            threads=self.threads,
        )
        return [
            _tensor_from_raw_bytes(raw, dtype, compressed.shape)
            for compressed, (_, dtype), raw in zip(
                payloads, prepared, raw_buffers, strict=True
            )
        ]

    def decompress_tensors(self, payloads):
        """Decode tensors through input-size-windowed native queues."""
        outputs = []
        batch = []
        batch_bytes = 0

        def flush_batch():
            nonlocal batch, batch_bytes
            if batch:
                outputs.extend(self._decompress_direct_native_batch(batch))
                batch = []
                batch_bytes = 0

        for compressed in payloads:
            raw_bytes = int(compressed.raw_bytes)
            if batch and batch_bytes + raw_bytes > self.direct_batch_bytes:
                flush_batch()
            batch.append(compressed)
            batch_bytes += raw_bytes
            if batch_bytes >= self.direct_batch_bytes:
                flush_batch()
        flush_batch()
        return tuple(outputs)
