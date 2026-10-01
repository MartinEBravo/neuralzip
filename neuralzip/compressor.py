from pathlib import Path

import torch

from neuralzip import huff0
from neuralzip.dtypes import DTYPES
from neuralzip.payloads import CodebookPayload
from neuralzip.setup import NeuralZipSetup
from neuralzip.storage import _pack_codebooks, pack_direct_storage, unpack_storage
from neuralzip.utils import _stream_chunk_size, _tensor_from_raw_bytes, time_call

_TRANSFER_NAME_FALLBACKS = {"lm_head.weight": ("model.embed_tokens.weight",)}


class NeuralZipCompressor:
    def __init__(self, setup=None, *, threads=0, direct_batch_bytes=128 << 20, gpu=False):
        setup = NeuralZipSetup() if setup is None else NeuralZipSetup.load(setup)
        self.min_saving_bytes = setup.min_saving_bytes
        self.chunk_size = setup.chunk_size
        self.compression_threshold = setup.compression_threshold
        self.threads = int(threads)
        self.direct_batch_bytes = int(direct_batch_bytes)
        self._codebook_payloads = list(setup.codebooks)
        modes = {
            payload.codebook_id: {
                "kind": payload.mode, "codebook_id": payload.codebook_id,
                "codebook": huff0.build_codebook_from_header(payload.codebook, vocab=payload.vocab),
            }
            for payload in setup.codebooks
        }
        def mode(codebook_id):
            return {"kind": "raw"} if codebook_id is None else modes[codebook_id]
        self._chunk_modes = {
            key: tuple(mode(i) for i in ids) for key, ids in setup.chunk_codebooks.items()
        }
        self._mode_by_tensor_stream = {key: mode(i) for key, i in setup.tensor_codebooks.items()}
        self._global_modes = {key: mode(i) for key, i in setup.global_codebooks.items()}
        self._transfer_mode_by_tensor_stream = {}
        self._transfer_chunk_modes = {}
        if gpu:
            self._add_global_codebooks()
        self.load_codebooks(self._codebook_payloads)

    def encode(self, named_tensors):
        named_tensors = tuple(named_tensors)
        self._add_global_codebooks()
        self.prepare_storage_transfer_target(named_tensors)
        encoded, seconds = time_call(self._encode, named_tensors)
        return *encoded, seconds

    def decode(self, blob, metadata):
        codebooks, payloads = unpack_storage(blob, metadata)
        previous = self._codebook_payload_by_id, self._decode_codebooks
        try:
            self.load_codebooks(codebooks)
            tensors = self._decompress_tensors(payloads)
        finally:
            self._codebook_payload_by_id, self._decode_codebooks = previous
        return {payload.name: tensor for payload, tensor in zip(payloads, tensors, strict=True)}

    def _encode(self, named_tensors):
        specs, used = [], set()
        for name, tensor in named_tensors:
            config = DTYPES[tensor.dtype]
            stream = config.exponent_stream
            if stream is None:
                stream = config.element_size - 1
            key = self._transfer_plan_key(name, tensor, stream)
            modes = self._transfer_chunk_modes[key]
            used.update(mode["codebook_id"] for mode in modes if mode["kind"] != "raw")
            original = tensor.detach().cpu().contiguous()
            specs.append((
                original.reshape(-1).view(torch.uint8).numpy(), config, stream,
                _stream_chunk_size(self.chunk_size, config.element_size), modes,
            ))
        codebooks = tuple(value for value in self.export_codebooks() if value.codebook_id in used)
        prefix, codebook_metadata = _pack_codebooks(codebooks)
        packed, groups = huff0.compress_direct_storage(
            specs, prefix, min_saving_bytes=self.min_saving_bytes,
            threads=self.threads, compression_threshold=self.compression_threshold,
        )
        return pack_direct_storage(
            named_tensors, specs, codebooks, prefix, codebook_metadata, packed, groups,
        )

    def _add_global_codebooks(self):
        codebook_id = max((payload.codebook_id for payload in self._codebook_payloads), default=-1) + 1
        for config in DTYPES.values():
            for stream_id, filename in config.global_codebooks:
                key = config.name, stream_id
                if key in self._global_modes:
                    continue
                path = Path(__file__).with_name("codebooks") / filename
                if not path.is_file():
                    continue
                header = path.read_bytes()
                self._codebook_payloads.append(CodebookPayload(codebook_id, stream_id, config.name, codebook=header))
                self._global_modes[key] = {
                    "kind": "shared_huff0", "codebook_id": codebook_id,
                    "codebook": huff0.build_codebook_from_header(header),
                }
                codebook_id += 1

    def export_codebooks(self):
        return tuple(self._codebook_payloads)

    def load_codebooks(self, codebooks):
        self._codebook_payload_by_id = {book.codebook_id: book for book in codebooks}
        self._decode_codebooks = {}

    @staticmethod
    def _transfer_plan_key(name, tensor, stream_id):
        return str(name), DTYPES[tensor.dtype].name, int(stream_id)

    def prepare_storage_transfer_target(self, named_tensors):
        parents, chunks = {}, {}
        for name, tensor in named_tensors:
            config = DTYPES[tensor.dtype]
            candidates = (str(name), *_TRANSFER_NAME_FALLBACKS.get(str(name), ()))
            step = _stream_chunk_size(self.chunk_size, config.element_size)
            count = (tensor.numel() + step - 1) // step
            for stream in config.stream_ids:
                target_key = self._transfer_plan_key(name, tensor, stream)
                source_key = next(
                    ((candidate, config.name, stream) for candidate in candidates
                     if (candidate, config.name, stream) in self._mode_by_tensor_stream),
                    None,
                )
                parent = self._mode_by_tensor_stream.get(
                    source_key, self._global_modes.get((config.name, stream), {"kind": "raw"})
                )
                parents[target_key] = parent
                exact = (*source_key, tensor.numel()) if source_key is not None else None
                chunks[target_key] = self._chunk_modes.get(exact, (parent,) * count)
        self._transfer_mode_by_tensor_stream = parents
        self._transfer_chunk_modes = chunks

    def prepare_gpu_transfer_target(self, named_tensors):
        self.prepare_storage_transfer_target(named_tensors)

    def _mode_for_tensor_stream(self, name, tensor, stream_id):
        return self._transfer_mode_by_tensor_stream[self._transfer_plan_key(name, tensor, stream_id)]

    def _native_decode_spec(self, tensor):
        config = DTYPES[getattr(torch, tensor.dtype)]
        groups = []
        for group in tensor.groups:
            chunks = []
            for chunk in group.chunks:
                handle = None
                if chunk.mode != "raw":
                    book = self._codebook_payload_by_id[chunk.codebook_id]
                    if (book.dtype, book.stream_id, book.mode) != (tensor.dtype, len(groups), chunk.mode):
                        raise ValueError("Chunk codebook does not match its tensor stream.")
                    if chunk.codebook_id not in self._decode_codebooks:
                        self._decode_codebooks[chunk.codebook_id] = huff0.build_decodebook(book.codebook, vocab=book.vocab)
                    handle = self._decode_codebooks[chunk.codebook_id]
                chunks.append((chunk.mode, chunk.payload, chunk.symbol_count, handle))
            groups.append(tuple(chunks))
        return config.element_size, bool(config.bit_reorder), tensor.numel, tuple(groups)

    def _decompress_tensors(self, payloads):
        outputs, batch, batch_bytes = [], [], 0
        def flush():
            nonlocal batch_bytes
            raw = huff0.decompress_mixed_streams_many(
                tuple(self._native_decode_spec(tensor) for tensor in batch), threads=self.threads,
            )
            outputs.extend(
                _tensor_from_raw_bytes(data, getattr(torch, tensor.dtype), tensor.shape)
                for tensor, data in zip(batch, raw, strict=True)
            )
            batch.clear()
            batch_bytes = 0
        for tensor in payloads:
            if batch and batch_bytes + tensor.raw_bytes > self.direct_batch_bytes:
                flush()
            batch.append(tensor)
            batch_bytes += tensor.raw_bytes
        if batch:
            flush()
        return tuple(outputs)
