from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from itertools import pairwise

import torch
from torch import nn
from transformers.pytorch_utils import Conv1D

from neuralzip import _huff0_cpp
from neuralzip.compressor import NeuralZipCompressor
from neuralzip.dtypes import DTYPES, DTypeSpec
from neuralzip.gpu import decoder as gpu_decoder
from neuralzip.utils import _stream_chunk_size

RUNTIME_ATTRIBUTE = "neuralzip_runtime"
_GPU_WEIGHT_TYPES = (nn.Linear, nn.Conv2d, Conv1D)


@dataclass(frozen=True)
class GPUCodebook:
    code_lengths: tuple[int, ...]
    code_values: tuple[int, ...]
    vocab: bytes = b""
    max_bits: int = field(init=False)
    decode_lut: tuple[int, ...] = field(init=False, repr=False)

    def __post_init__(self):
        longest = max(self.code_lengths, default=0)
        if longest > (56 if self.vocab else 12):
            raise ValueError("The GPU codebook exceeds its decoder's code-length limit.")
        max_bits = min(12, longest)
        object.__setattr__(self, "max_bits", max_bits)
        if not longest:
            object.__setattr__(self, "decode_lut", ())
            return
        if len(self.code_lengths) != 256 + len(self.vocab) // 2:
            raise ValueError("The GPU alphabet does not match its pair vocabulary.")
        lut = [0] * (1 << max_bits)
        nodes = [[0, 0]]
        for symbol, (length, value) in enumerate(
            zip(self.code_lengths, self.code_values, strict=True)
        ):
            if not length:
                continue
            pair = self.vocab[2 * (symbol - 256):2 * (symbol - 255)] if symbol >= 256 else bytes((symbol,))
            entry = ((length << 24) | (len(pair) << 16) | int.from_bytes(pair, "little")
                     if self.vocab else (length << 8) | symbol)
            if longest <= max_bits:
                begin = value << (max_bits - length)
                lut[begin:begin + (1 << (max_bits - length))] = [entry] * (1 << (max_bits - length))
            else:
                node = 0
                for bit in range(length - 1, -1, -1):
                    branch = (value >> bit) & 1
                    if bit == 0:
                        nodes[node][branch] = entry
                    else:
                        if not nodes[node][branch]:
                            nodes[node][branch] = len(nodes)
                            nodes.append([0, 0])
                        node = nodes[node][branch]
        if longest > max_bits:
            for prefix in range(len(lut)):
                node = 0
                for bit in range(max_bits - 1, -1, -1):
                    node = nodes[node][(prefix >> bit) & 1]
                    if node >> 24:
                        break
                lut[prefix] = node if node >> 24 else len(lut) + 2 * node
            base = len(lut)
            lut.extend(child if child >> 24 else base + 2 * child for node in nodes for child in node)
        if not all(lut):
            raise ValueError("The Huffman codebook does not cover every prefix.")
        object.__setattr__(self, "decode_lut", tuple(lut))


@dataclass(frozen=True)
class GPUEncodedSymbols:
    bitstream: bytes
    bit_length: int
    symbol_count: int
    packed_gaps: bytes
    output_positions: tuple[int, ...]

    @property
    def compressed_bytes(self):
        return (
            len(self.bitstream) + len(self.packed_gaps) + 4 * len(self.output_positions)
        )


@dataclass(frozen=True)
class GPUEncoded:
    dtype: DTypeSpec
    numel: int
    streams: tuple[bytes | GPUEncodedSymbols, ...]  # Indexed by byte-lane ID.

    @property
    def raw_bytes(self):
        return self.numel * self.dtype.element_size

    @property
    def compressed_bytes(self):
        return sum(
            len(stream) if isinstance(stream, bytes) else stream.compressed_bytes
            for stream in self.streams
        )


def encode_gpu_native(raw, codebook_header, *, dtype, stream_id, vocab=b"", codebook_handle=None):
    raw = bytes(raw)
    encoded, raw_streams, gaps, positions, bit_length, _max_bits = (
        _huff0_cpp.encode_gpu_tensor(
            raw,
            bytes(codebook_header),
            dtype.element_size,
            bool(dtype.bit_reorder),
            dtype.mantissa_bits,
            dtype.exponent_mask.bit_length(),
            stream_id,
            vocab=vocab,
            codebook_handle=codebook_handle,
        )
    )
    numel = len(raw) // dtype.element_size
    streams = list(raw_streams)
    streams[stream_id] = GPUEncodedSymbols(
        encoded, int(bit_length), numel, gaps, tuple(positions)
    )
    return GPUEncoded(dtype, numel, tuple(streams))


@dataclass(frozen=True)
class LinearWeightBinding:
    module_name: str
    tensor_name: str
    shape: tuple[int, ...]
    numel: int
    workspace_offset: int


@dataclass(frozen=True)
class GPUWeightGroup:
    block_name: str
    codebook_id: int
    codebook: GPUCodebook
    encoded: GPUEncoded
    bindings: tuple[LinearWeightBinding, ...]
    workspace_offset: int

    @property
    def layout(self):
        (stream_id,) = (
            index
            for index, stream in enumerate(self.encoded.streams)
            if isinstance(stream, GPUEncodedSymbols)
        )
        dtype = self.encoded.dtype
        return (
            dtype.element_size,
            bool(dtype.bit_reorder),
            dtype.mantissa_bits,
            dtype.exponent_mask.bit_length(),
            stream_id,
            bool(self.codebook.vocab),
            max(self.codebook.code_lengths, default=0) > 32,
        )


@dataclass(frozen=True)
class NeuralZipModelExport:
    groups: tuple[GPUWeightGroup, ...]

    @property
    def block_order(self):
        return tuple(dict.fromkeys(group.block_name for group in self.groups))

    @property
    def bindings(self):
        return tuple(dict.fromkeys(binding for group in self.groups for binding in group.bindings))

    @property
    def workspace_sizes(self):
        sizes = {}
        for group in self.groups:
            dtype = group.encoded.dtype.torch_dtype
            sizes[dtype] = max(
                sizes.get(dtype, 0), group.workspace_offset + group.encoded.numel
            )
        return sizes


@dataclass(frozen=True)
class _Candidate:
    block_name: str
    module_name: str
    tensor_name: str
    parameter: nn.Parameter = field(repr=False, compare=False)


@dataclass(frozen=True)
class _UploadedGroup:
    group: GPUWeightGroup
    lut: torch.Tensor
    encoded_symbols: torch.Tensor
    raw_streams: tuple[torch.Tensor | None, ...]
    packed_gaps: torch.Tensor
    output_positions: torch.Tensor
    output: torch.Tensor
    max_bits: int
    max_block_elements: int

    @property
    def layout(self):
        return self.group.layout


def _join(parent, child):
    return f"{parent}.{child}" if parent and child else parent or child


def _decode_hook_module(model, block_name):
    parent_name, _, child_name = block_name.rpartition(".")
    parent = model.get_submodule(parent_name)
    # MultiheadAttention reads out_proj.weight without calling the child Linear.
    if child_name == "out_proj" and isinstance(parent, nn.MultiheadAttention):
        return parent
    return model.get_submodule(block_name)


def _gpu_candidates(model):
    """Keep linear block batching; hook ungrouped weight modules individually."""

    possible_blocks = [
        (_join(parent_name, child_name), child)
        for parent_name, parent in model.named_modules(remove_duplicate=False)
        if isinstance(parent, nn.ModuleList)
        for child_name, child in parent._modules.items()
        if child is not None
    ]
    names = {name for name, _ in possible_blocks}
    blocks = []
    for name, module in possible_blocks:
        if any(name.startswith(parent + ".") for parent, _ in blocks):
            continue
        nested = tuple(child + "." for child in names if child.startswith(name + "."))
        if nested and all(
            (_join(name, child) + ".").startswith(nested)
            for child, layer in module.named_modules(remove_duplicate=False)
            if isinstance(layer, nn.Linear)
        ):
            continue
        blocks.append((name, module))
    candidates = []
    for module_name, module in model.named_modules(remove_duplicate=False):
        if not isinstance(module, _GPU_WEIGHT_TYPES):
            continue
        parameter = module._parameters.get("weight")
        if isinstance(parameter, nn.Parameter):
            block_name = next(
                (name for name, _ in blocks
                 if module_name == name or module_name.startswith(name + ".")),
                module_name,
            )
            candidates.append(
                _Candidate(
                    block_name,
                    module_name,
                    _join(module_name, "weight"),
                    parameter,
                )
            )
    return tuple(dict.fromkeys(c.block_name for c in candidates)), tuple(candidates)


def _unique_candidates(model, candidates):
    occurrences = defaultdict(int)
    for _name, parameter in model.named_parameters(remove_duplicate=False):
        occurrences[id(parameter)] += 1
    return tuple(
        candidate
        for candidate in candidates
        if candidate.parameter.numel() and occurrences[id(candidate.parameter)] == 1
    )


def _assemble_export(block_order, candidates, compressor):
    payloads = {
        int(payload.codebook_id): payload for payload in compressor.export_codebooks()
    }
    codebooks = {}
    block_index = {name: index for index, name in enumerate(block_order)}
    workspace_offsets = defaultdict(int)
    groups = []
    for member in sorted(candidates, key=lambda c: (block_index[c.block_name], c.module_name)):
        parameter = member.parameter
        config = DTYPES[parameter.dtype]
        stream_id = config.exponent_stream if config.exponent_stream is not None else config.element_size - 1
        key = compressor._transfer_plan_key(member.tensor_name, parameter, stream_id)
        modes = compressor._transfer_chunk_modes[key]
        step = _stream_chunk_size(compressor.chunk_size, config.element_size)
        raw = parameter.detach().cpu().reshape(-1).view(torch.uint8).numpy().tobytes()
        workspace_offset = workspace_offsets[(member.block_name, parameter.dtype)]
        binding = LinearWeightBinding(member.module_name, member.tensor_name, tuple(parameter.shape),
                                      parameter.numel(), workspace_offset)
        fragments = []
        for chunk, mode in enumerate(modes):
            start = chunk * step
            part = raw[start * config.element_size:(start + step) * config.element_size]
            codebook_id = mode.get("codebook_id", -1)
            if codebook_id not in codebooks:
                if codebook_id == -1:
                    codebooks[-1] = GPUCodebook((), ())
                else:
                    payload = payloads[codebook_id]
                    lengths, values, _ = _huff0_cpp.export_code_table(payload.codebook, max_bits=56, vocab=payload.vocab)
                    codebooks[codebook_id] = GPUCodebook(tuple(lengths), tuple(values), bytes(payload.vocab))
            if codebook_id == -1:
                reordered = config.reorder_bytes(part)
                streams = [reordered[lane::config.element_size] for lane in config.stream_ids]
                streams[stream_id] = GPUEncodedSymbols(streams[stream_id], len(streams[stream_id]) * 8,
                                                     len(streams[stream_id]), b"", ())
                encoded = GPUEncoded(config, len(part) // config.element_size, tuple(streams))
            else:
                payload = payloads[codebook_id]
                encoded = encode_gpu_native(part, payload.codebook, dtype=config, stream_id=stream_id,
                                            vocab=payload.vocab, codebook_handle=mode["codebook"].handle)
            fragments.append(GPUWeightGroup(member.block_name, codebook_id, codebooks[codebook_id],
                                            encoded, (binding,), workspace_offset + start))
        if sum(group.encoded.compressed_bytes for group in fragments) >= len(raw):
            continue
        groups.extend(fragments)
        workspace_offsets[(member.block_name, parameter.dtype)] += parameter.numel()

    return NeuralZipModelExport(tuple(groups))


def export_model_for_gpu(model, setup, *, allowed_dtypes=DTYPES):
    """Export the frozen chunk tables, including adaptive singleton/pair codes."""

    blocks, candidates = _gpu_candidates(model)
    candidates = _unique_candidates(model, candidates)
    candidates = tuple(
        candidate
        for candidate in candidates
        if candidate.parameter.dtype in DTYPES
        and candidate.parameter.dtype in allowed_dtypes
        and candidate.parameter.is_contiguous()
    )
    if not candidates:
        return NeuralZipModelExport(())
    compressor = NeuralZipCompressor(setup, gpu=True)
    named_tensors = tuple(
        (candidate.tensor_name, candidate.parameter) for candidate in candidates
    )
    compressor.prepare_gpu_transfer_target(named_tensors)
    return _assemble_export(blocks, candidates, compressor)


def _bytes_tensor(data, device):
    if not data:
        return torch.empty(0, dtype=torch.uint8, device=device)
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).to(device=device)


def _binding_module(model, binding):
    module = model.get_submodule(binding.module_name)
    if not isinstance(module, _GPU_WEIGHT_TYPES):
        raise TypeError(f"{binding.module_name!r} is not a supported weight module.")
    return module


def _binding_view(workspaces, dtype, binding):
    offset = binding.workspace_offset
    return workspaces[dtype][offset:offset + binding.numel].view(binding.shape)


class NeuralZipRuntime:
    """Sequential runtime attached to one model on one CUDA device."""

    def __init__(self, model, exported, device):
        self.model = model
        self.export = exported
        self.device = torch.device(device)
        self._decoder = gpu_decoder.default_decoder()
        self._hook_handles = []
        self.workspaces = {
            dtype: torch.empty(size, dtype=dtype, device=self.device)
            for dtype, size in exported.workspace_sizes.items()
        }

        luts = {
            group.codebook_id: torch.tensor(
                group.codebook.decode_lut,
                dtype=torch.uint32 if group.codebook.vocab else torch.uint16,
                device=self.device,
            )
            for group in exported.groups
        }
        uploaded = []
        for group in exported.groups:
            (symbols,) = (
                stream
                for stream in group.encoded.streams
                if isinstance(stream, GPUEncodedSymbols)
            )
            positions = symbols.output_positions
            uploaded.append(
                _UploadedGroup(
                    group,
                    luts[group.codebook_id],
                    _bytes_tensor(symbols.bitstream, self.device),
                    tuple(
                        _bytes_tensor(stream, self.device)
                        if isinstance(stream, bytes)
                        else None
                        for stream in group.encoded.streams
                    ),
                    _bytes_tensor(symbols.packed_gaps, self.device),
                    torch.tensor(positions, dtype=torch.int32, device=self.device),
                    self.workspaces[group.encoded.dtype.torch_dtype][
                        group.workspace_offset : group.workspace_offset
                        + group.encoded.numel
                    ],
                    group.codebook.max_bits,
                    0 if not group.codebook.max_bits else max(
                        (right - left for left, right in pairwise(positions)),
                        default=0,
                    ),
                )
            )
        by_block = defaultdict(list)
        for group in uploaded:
            by_block[group.group.block_name].append(group)
        self._groups_by_block = {
            name: tuple(by_block[name]) for name in exported.block_order
        }
        self._prepared_batches = {}
        for name, groups in self._groups_by_block.items():
            layouts = defaultdict(list)
            for group in groups:
                layouts[group.layout].append(group)
            self._prepared_batches[name] = tuple(
                self._decoder.prepare_batch(batch) for batch in layouts.values()
            )
        torch.cuda.synchronize(self.device)

    @classmethod
    def attach(cls, model, exported, *, device=None):
        if hasattr(model, RUNTIME_ATTRIBUTE):
            raise RuntimeError("This model already has a NeuralZip runtime.")
        if not exported.groups:
            raise ValueError("The setup produced no GPU-compressible weights.")
        if device is None:
            current = next(model.parameters()).device
            device = current if current.type == "cuda" else "cuda"
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("NeuralZip GPU inference requires a CUDA device.")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        decoder = gpu_decoder.default_decoder()
        for group in exported.groups:
            decoder.prepare_group(device.index, group)

        for binding in exported.bindings:
            module = _binding_module(model, binding)
            module._parameters.pop("weight", None)
            module.__dict__.pop("weight", None)
        model.to(device).eval()
        runtime = cls(model, exported, device)
        runtime._install()
        return runtime

    def _install(self):
        for groups in self._groups_by_block.values():
            for uploaded in groups:
                for binding in uploaded.group.bindings:
                    _binding_module(self.model, binding).__dict__["weight"] = (
                        _binding_view(self.workspaces, uploaded.group.encoded.dtype.torch_dtype, binding)
                    )
        for block_name in self.export.block_order:
            handle = _decode_hook_module(self.model, block_name).register_forward_pre_hook(
                partial(self._decode_block, block_name),
                prepend=True,
            )
            self._hook_handles.append(handle)
        setattr(self.model, RUNTIME_ATTRIBUTE, self)

    def _decode_block(self, block_name, _module=None, _args=None):
        for prepared in self._prepared_batches[block_name]:
            self._decoder.decode_batch(prepared)

    @torch.no_grad()
    def materialized_weights(self):
        """Yield compressed weights as bit-exact CPU tensors, one block at a time."""

        yielded = set()
        for block_name in self.export.block_order:
            self._decode_block(block_name)
            torch.cuda.synchronize(self.device)
            for uploaded in self._groups_by_block[block_name]:
                for binding in uploaded.group.bindings:
                    if binding.tensor_name in yielded:
                        continue
                    yielded.add(binding.tensor_name)
                    weight = _binding_view(self.workspaces, uploaded.group.encoded.dtype.torch_dtype, binding)
                    yield binding.tensor_name, weight.detach().to("cpu", copy=True)
