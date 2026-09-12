from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from itertools import pairwise

import torch
from torch import nn

from neuralzip import _huff0_cpp
from neuralzip.compressor import NeuralZipCompressor
from neuralzip.gpu import decoder as gpu_decoder

RUNTIME_ATTRIBUTE = "neuralzip_runtime"


@dataclass(frozen=True)
class GPUCodebook:
    code_lengths: tuple[int, ...]
    code_values: tuple[int, ...]
    max_bits: int = field(init=False)
    decode_lut: tuple[int, ...] = field(init=False, repr=False)

    def __post_init__(self):
        max_bits = max(self.code_lengths)
        lut = [0] * (1 << max_bits)
        for symbol, (length, value) in enumerate(
            zip(self.code_lengths, self.code_values, strict=True)
        ):
            if not length:
                continue
            begin = value << (max_bits - length)
            end = begin + (1 << (max_bits - length))
            entry = (length << 8) | symbol
            for index in range(begin, end):
                lut[index] = entry
        if not all(lut):
            raise ValueError("The Huffman codebook does not cover every prefix.")
        object.__setattr__(self, "max_bits", max_bits)
        object.__setattr__(self, "decode_lut", tuple(lut))


@dataclass(frozen=True)
class GPUEncodedExponents:
    bitstream: bytes
    bit_length: int
    symbol_count: int
    packed_gaps: bytes
    output_positions: tuple[int, ...]


@dataclass(frozen=True)
class GPUEncodedBF16:
    sign_mantissa: bytes
    exponents: GPUEncodedExponents

    @property
    def numel(self):
        return len(self.sign_mantissa)


def encode_bfloat16_native(raw_bfloat16, codebook_header):
    encoded, sign_mantissa, gaps, positions, bit_length, _max_bits = (
        _huff0_cpp.encode_gpu_bfloat16(
            bytes(raw_bfloat16),
            bytes(codebook_header),
        )
    )
    return GPUEncodedBF16(
        sign_mantissa,
        GPUEncodedExponents(
            encoded,
            int(bit_length),
            len(sign_mantissa),
            gaps,
            tuple(positions),
        ),
    )


@dataclass(frozen=True)
class LinearWeightBinding:
    module_name: str
    tensor_name: str
    shape: tuple[int, ...]
    numel: int
    workspace_offset: int


@dataclass(frozen=True)
class BF16WeightGroup:
    block_name: str
    codebook_id: int
    codebook: GPUCodebook
    encoded: GPUEncodedBF16
    bindings: tuple[LinearWeightBinding, ...]
    workspace_offset: int


@dataclass(frozen=True)
class NeuralZipModelExport:
    groups: tuple[BF16WeightGroup, ...]

    @property
    def block_order(self):
        return tuple(dict.fromkeys(group.block_name for group in self.groups))

    @property
    def bindings(self):
        return tuple(binding for group in self.groups for binding in group.bindings)

    @property
    def workspace_numel(self):
        return max(
            (group.workspace_offset + group.encoded.numel for group in self.groups),
            default=0,
        )


@dataclass(frozen=True)
class _Candidate:
    block_name: str
    module_name: str
    tensor_name: str
    parameter: nn.Parameter = field(repr=False, compare=False)


@dataclass(frozen=True)
class _UploadedGroup:
    group: BF16WeightGroup
    lut: torch.Tensor
    encoded_exponents: torch.Tensor
    sign_mantissa: torch.Tensor
    packed_gaps: torch.Tensor
    output_positions: torch.Tensor
    output: torch.Tensor
    max_bits: int
    max_block_elements: int


def _join(parent, child):
    return f"{parent}.{child}" if parent and child else parent or child


def _gpu_candidates(model):
    """Find direct ModuleList children and their linear-like weights."""

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
        ancestor = name
        while "." in ancestor:
            ancestor = ancestor.rsplit(".", 1)[0]
            if ancestor in names:
                break
        else:
            blocks.append((name, module))
    candidates = []
    for block_name, block in blocks:
        for relative_name, module in block.named_modules(remove_duplicate=False):
            if not isinstance(module, nn.Linear):
                continue
            module_name = _join(block_name, relative_name)
            parameter = module._parameters.get("weight")
            if isinstance(parameter, nn.Parameter):
                candidates.append(
                    _Candidate(
                        block_name,
                        module_name,
                        f"{module_name}.weight",
                        parameter,
                    )
                )
    return tuple(name for name, _ in blocks), tuple(candidates)


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
    grouped = defaultdict(list)
    for candidate in candidates:
        parameter = candidate.parameter
        mode = compressor._mode_for_tensor_stream(candidate.tensor_name, parameter, 1)
        kind = mode.get("kind")
        if kind == "raw":
            continue
        if kind != "shared_huff0":
            raise ValueError(
                f"GPU inference needs a sharing-only setup; {candidate.tensor_name} uses {kind!r}."
            )
        codebook_id = int(mode["codebook_id"])
        payload = payloads[codebook_id]
        if codebook_id not in codebooks:
            lengths, values, _max_bits = _huff0_cpp.export_code_table(payload.codebook)
            codebooks[codebook_id] = GPUCodebook(
                tuple(lengths),
                tuple(values),
            )
        grouped[(candidate.block_name, codebook_id)].append(candidate)

    block_index = {name: index for index, name in enumerate(block_order)}
    workspace_offsets = defaultdict(int)
    groups = []
    for (block_name, codebook_id), members in sorted(
        grouped.items(),
        key=lambda item: (block_index[item[0][0]], item[0][1]),
    ):
        members.sort(key=lambda member: member.module_name)
        workspace_offset = workspace_offsets[block_name]
        group_offset = 0
        bindings = []
        raw_parts = []
        for member in members:
            parameter = member.parameter
            bindings.append(
                LinearWeightBinding(
                    member.module_name,
                    member.tensor_name,
                    tuple(parameter.shape),
                    parameter.numel(),
                    workspace_offset + group_offset,
                )
            )
            raw = parameter.detach().to(device="cpu").contiguous()
            raw_parts.append(raw.reshape(-1).view(torch.uint8).numpy().tobytes())
            group_offset += parameter.numel()

        encoded = encode_bfloat16_native(
            b"".join(raw_parts),
            payloads[codebook_id].codebook,
        )
        encoded_bytes = (
            len(encoded.sign_mantissa)
            + len(encoded.exponents.bitstream)
            + len(encoded.exponents.packed_gaps)
            + 4 * len(encoded.exponents.output_positions)
        )
        if encoded_bytes >= 2 * encoded.numel:
            continue
        groups.append(
            BF16WeightGroup(
                block_name,
                codebook_id,
                codebooks[codebook_id],
                encoded,
                tuple(bindings),
                workspace_offset,
            )
        )
        workspace_offsets[block_name] += group_offset

    return NeuralZipModelExport(tuple(groups))


def export_model_for_gpu(model, setup):
    """Create the immutable sharing-only BF16 export used by the runtime."""

    blocks, candidates = _gpu_candidates(model)
    candidates = _unique_candidates(model, candidates)
    candidates = tuple(
        candidate
        for candidate in candidates
        if candidate.parameter.dtype == torch.bfloat16
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
    if not isinstance(module, nn.Linear):
        raise TypeError(f"{binding.module_name!r} is not a Linear module.")
    return module


class NeuralZipRuntime:
    """Sequential runtime attached to one model on one CUDA device."""

    def __init__(self, model, exported, device):
        self.model = model
        self.export = exported
        self.device = torch.device(device)
        self._decoder = gpu_decoder.default_decoder()
        self._hook_handles = []
        self.workspace = torch.empty(
            exported.workspace_numel,
            dtype=torch.bfloat16,
            device=self.device,
        )

        luts = {
            group.codebook_id: torch.tensor(
                group.codebook.decode_lut,
                dtype=torch.uint16,
                device=self.device,
            )
            for group in exported.groups
        }
        uploaded = []
        for group in exported.groups:
            positions = group.encoded.exponents.output_positions
            uploaded.append(
                _UploadedGroup(
                    group,
                    luts[group.codebook_id],
                    _bytes_tensor(group.encoded.exponents.bitstream, self.device),
                    _bytes_tensor(group.encoded.sign_mantissa, self.device),
                    _bytes_tensor(group.encoded.exponents.packed_gaps, self.device),
                    torch.tensor(positions, dtype=torch.int32, device=self.device),
                    self.workspace[
                        group.workspace_offset : group.workspace_offset
                        + group.encoded.numel
                    ],
                    group.codebook.max_bits,
                    max(
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
        self._prepared_batches = {
            name: self._decoder.prepare_batch(groups)
            for name, groups in self._groups_by_block.items()
        }
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

        for binding in exported.bindings:
            module = _binding_module(model, binding)
            module._parameters.pop("weight", None)
            module.__dict__.pop("weight", None)
        model.to(device).eval()
        runtime = cls(model, exported, device)
        runtime._install()
        return runtime

    def _install(self):
        for binding in self.export.bindings:
            view = self.workspace[
                binding.workspace_offset : binding.workspace_offset + binding.numel
            ].view(binding.shape)
            _binding_module(self.model, binding).__dict__["weight"] = view
        for block_name in self.export.block_order:
            handle = self.model.get_submodule(block_name).register_forward_pre_hook(
                partial(self._decode_block, block_name),
                prepend=True,
            )
            self._hook_handles.append(handle)
        setattr(self.model, RUNTIME_ATTRIBUTE, self)

    def _decode_block(self, block_name, _module=None, _args=None):
        self._decoder.decode_batch(self._prepared_batches[block_name])

    @torch.no_grad()
    def materialized_weights(self):
        """Yield compressed weights as bit-exact CPU tensors, one block at a time."""

        for block_name in self.export.block_order:
            self._decode_block(block_name)
            torch.cuda.synchronize(self.device)
            for uploaded in self._groups_by_block[block_name]:
                for binding in uploaded.group.bindings:
                    weight = self.workspace[
                        binding.workspace_offset : binding.workspace_offset
                        + binding.numel
                    ].view(binding.shape)
                    yield binding.tensor_name, weight.detach().to("cpu", copy=True)
