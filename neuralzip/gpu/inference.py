"""Post-setup BF16 inference from NeuralZip-compressed model weights.

Adapters discover block-local ``nn.Linear`` and optional ``nn.Embedding``
weights whose exponent stream uses a prepared ``shared_huff0`` codebook. A
block pre-hook decodes them into one reusable workspace per execution device
immediately before the block runs. Persistent checkpoints may distribute
blocks across GPUs or keep compressed payloads in host memory.

The runtime is inference-only and sequential within each device. Packed
exponents remain an archive feature until a corresponding GPU decoder exists.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import partial

import torch
from torch import nn

from .. import huff0
from .._setup import materialize_setup
from ..neuralzip_codec import NeuralZipCodec
from . import decoder as gpu_decoder
from .adapters import AutoModuleListAdapter, GPUModelAdapter, GPUWeightKind
from .export import (
    DEFAULT_BYTES_PER_THREAD,
    DEFAULT_THREADS_PER_BLOCK,
    GPUCodebook,
    GPUEncodedBF16,
    encode_bfloat16_native,
)

_RUNTIME_ATTRIBUTE = "neuralzip_runtime"


def _require_non_negative_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")


def _has_external_dispatch(model: nn.Module) -> bool:
    if hasattr(model, "hf_device_map"):
        return True
    return any(
        getattr(module, "_hf_hook", None) is not None for module in model.modules()
    )


@dataclass(frozen=True)
class LinearWeightBinding:
    """One compressed two-dimensional weight view inside a block workspace."""

    block_name: str
    module_name: str
    tensor_name: str
    shape: tuple[int, ...]
    numel: int
    workspace_offset: int
    group_offset: int
    requires_grad: bool
    module_kind: str = "linear"

    def __post_init__(self) -> None:
        if not self.block_name or not self.module_name or not self.tensor_name:
            raise ValueError("Binding names must be non-empty.")
        shape = tuple(self.shape)
        if len(shape) != 2 or any(
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension <= 0
            for dimension in shape
        ):
            raise ValueError("Linear weight shape must contain two positive integers.")
        for name, value in (
            ("numel", self.numel),
            ("workspace_offset", self.workspace_offset),
            ("group_offset", self.group_offset),
        ):
            _require_non_negative_integer(name, value)
        if self.numel != shape[0] * shape[1]:
            raise ValueError("Binding numel does not match its shape.")
        if not isinstance(self.requires_grad, bool):
            raise TypeError("requires_grad must be a boolean.")
        if self.module_kind not in {"linear", "embedding"}:
            raise ValueError("module_kind must be 'linear' or 'embedding'.")
        object.__setattr__(self, "shape", shape)

    @property
    def raw_bytes(self) -> int:
        return self.numel * 2

    @property
    def workspace_bytes(self) -> int:
        return self.raw_bytes


@dataclass(frozen=True)
class BF16WeightGroup:
    """Weights in one transformer block that share one exponent codebook."""

    block_name: str
    codebook_id: int
    codebook: GPUCodebook
    encoded: GPUEncodedBF16
    bindings: tuple[LinearWeightBinding, ...]
    workspace_offset: int

    def __post_init__(self) -> None:
        if not self.block_name:
            raise ValueError("block_name must be non-empty.")
        _require_non_negative_integer("codebook_id", self.codebook_id)
        _require_non_negative_integer("workspace_offset", self.workspace_offset)
        if not isinstance(self.codebook, GPUCodebook):
            raise TypeError("codebook must be a GPUCodebook.")
        if not isinstance(self.encoded, GPUEncodedBF16):
            raise TypeError("encoded must be GPUEncodedBF16.")
        bindings = tuple(self.bindings)
        if not bindings:
            raise ValueError("A compressed weight group must contain bindings.")

        next_group_offset = 0
        next_workspace_offset = self.workspace_offset
        for binding in bindings:
            if not isinstance(binding, LinearWeightBinding):
                raise TypeError("bindings must contain LinearWeightBinding values.")
            if binding.block_name != self.block_name:
                raise ValueError("Every binding must belong to the group's block.")
            if binding.group_offset != next_group_offset:
                raise ValueError("Group binding offsets must be contiguous.")
            if binding.workspace_offset != next_workspace_offset:
                raise ValueError("Workspace binding offsets must be contiguous.")
            next_group_offset += binding.numel
            next_workspace_offset += binding.numel
        if next_group_offset != self.encoded.numel:
            raise ValueError("Encoded symbol count does not match group bindings.")
        object.__setattr__(self, "bindings", bindings)

    @property
    def numel(self) -> int:
        return self.encoded.numel

    @property
    def raw_bytes(self) -> int:
        return self.numel * 2

    @property
    def exponent_bytes(self) -> int:
        return len(self.encoded.exponents.bitstream)

    @property
    def sign_mantissa_bytes(self) -> int:
        return len(self.encoded.sign_mantissa)

    @property
    def payload_bytes(self) -> int:
        return self.exponent_bytes + self.sign_mantissa_bytes

    @property
    def metadata_bytes(self) -> int:
        return len(self.encoded.exponents.packed_gaps) + (
            4 * len(self.encoded.exponents.output_positions)
        )

    @property
    def compressed_bytes(self) -> int:
        """Encoded group buffers, excluding its shared LUT and descriptors."""

        return self.payload_bytes + self.metadata_bytes

    @property
    def codebook_bytes(self) -> int:
        return self.codebook.lut_nbytes

    @property
    def workspace_bytes(self) -> int:
        return self.raw_bytes


@dataclass(frozen=True)
class NeuralZipModelExport:
    """Immutable compressed buffers and exact candidate-weight statistics."""

    groups: tuple[BF16WeightGroup, ...]
    block_order: tuple[str, ...]
    resident_weight_names: tuple[str, ...] = ()
    resident_raw_bytes: int = 0

    def __post_init__(self) -> None:
        groups = tuple(self.groups)
        block_order = tuple(self.block_order)
        resident_names = tuple(self.resident_weight_names)
        _require_non_negative_integer("resident_raw_bytes", self.resident_raw_bytes)
        if len(block_order) != len(set(block_order)):
            raise ValueError("block_order must not contain duplicates.")
        if any(not name for name in block_order):
            raise ValueError("block_order entries must be non-empty.")
        if len(resident_names) != len(set(resident_names)):
            raise ValueError("resident_weight_names must not contain duplicates.")

        known_blocks = set(block_order)
        tensor_names = set()
        codebooks = {}
        expected_offsets = defaultdict(int)
        last_block_index = -1
        for group in groups:
            if not isinstance(group, BF16WeightGroup):
                raise TypeError("groups must contain BF16WeightGroup values.")
            if group.block_name not in known_blocks:
                raise ValueError("A group references a block outside block_order.")
            block_index = block_order.index(group.block_name)
            if block_index < last_block_index:
                raise ValueError("Groups must follow block_order.")
            last_block_index = block_index
            if group.workspace_offset != expected_offsets[group.block_name]:
                raise ValueError("Groups in a block must occupy contiguous workspace.")
            expected_offsets[group.block_name] += group.numel
            previous = codebooks.setdefault(group.codebook_id, group.codebook)
            if previous != group.codebook:
                raise ValueError("One codebook_id maps to inconsistent GPU codebooks.")
            for binding in group.bindings:
                if binding.tensor_name in tensor_names:
                    raise ValueError("A tensor cannot appear in multiple groups.")
                tensor_names.add(binding.tensor_name)
        if tensor_names.intersection(resident_names):
            raise ValueError("A compressed tensor cannot also be marked resident.")

        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "block_order", block_order)
        object.__setattr__(self, "resident_weight_names", resident_names)

    @property
    def bindings(self) -> tuple[LinearWeightBinding, ...]:
        return tuple(binding for group in self.groups for binding in group.bindings)

    @property
    def compressed_weight_count(self) -> int:
        return len(self.bindings)

    @property
    def resident_weight_count(self) -> int:
        return len(self.resident_weight_names)

    @property
    def raw_bytes(self) -> int:
        """Original BF16 bytes represented by compressed groups."""

        return sum(group.raw_bytes for group in self.groups)

    @property
    def original_candidate_bytes(self) -> int:
        return self.raw_bytes + self.resident_raw_bytes

    @property
    def payload_bytes(self) -> int:
        return sum(group.payload_bytes for group in self.groups)

    @property
    def metadata_bytes(self) -> int:
        return sum(group.metadata_bytes for group in self.groups)

    @property
    def codebook_bytes(self) -> int:
        unique = {}
        for group in self.groups:
            unique.setdefault(group.codebook_id, group.codebook)
        return sum(codebook.lut_nbytes for codebook in unique.values())

    @property
    def compressed_bytes(self) -> int:
        """Compressed streams and LUTs, excluding launch-time allocations."""

        return self.payload_bytes + self.metadata_bytes + self.codebook_bytes

    @property
    def launch_descriptor_bytes(self) -> int:
        """Exact batched-launch pointer and integer descriptor bytes."""

        blocks = sum(group.encoded.exponents.block_count for group in self.groups)
        return 56 * len(self.groups) + 8 * blocks

    @property
    def workspace_numel(self) -> int:
        totals = defaultdict(int)
        for group in self.groups:
            totals[group.block_name] += group.numel
        return max(totals.values(), default=0)

    @property
    def workspace_bytes(self) -> int:
        return self.workspace_numel * 2

    @property
    def runtime_candidate_bytes(self) -> int:
        """Compressed buffers + shared workspace + untouched candidate weights."""

        return (
            self.compressed_bytes
            + self.launch_descriptor_bytes
            + self.workspace_bytes
            + self.resident_raw_bytes
        )


@dataclass(frozen=True)
class _LinearCandidate:
    block_name: str
    module_name: str
    tensor_name: str
    module: nn.Module = field(repr=False, compare=False)
    parameter: nn.Parameter = field(repr=False, compare=False)
    module_kind: str = "linear"


@dataclass(frozen=True)
class _UploadedGroup:
    group: BF16WeightGroup
    lut: torch.Tensor
    encoded_exponents: torch.Tensor
    sign_mantissa: torch.Tensor
    packed_gaps: torch.Tensor
    output_positions: torch.Tensor
    output: torch.Tensor
    payload_offloaded: bool = False


def _join_name(parent: str, child: str) -> str:
    return f"{parent}.{child}" if parent else child


def _gpu_candidates(
    model: nn.Module,
    *,
    adapter: GPUModelAdapter | None,
    include_embeddings: bool,
) -> tuple[tuple[str, ...], tuple[_LinearCandidate, ...]]:
    selected = (
        AutoModuleListAdapter(include_embeddings=include_embeddings)
        if adapter is None
        else adapter
    )
    discovery = selected.discover(model)
    candidates = tuple(
        _LinearCandidate(
            block_name=binding.block_name,
            module_name=binding.module_name,
            tensor_name=binding.tensor_name,
            module=binding.module,
            parameter=binding.parameter,
            module_kind=(
                "embedding" if binding.kind is GPUWeightKind.EMBEDDING else "linear"
            ),
        )
        for binding in discovery.bindings
    )
    return discovery.block_order, candidates


def _tensor_raw_bytes(tensor: torch.Tensor) -> bytes:
    cpu = tensor.detach().to(device="cpu").contiguous()
    return cpu.reshape(-1).view(torch.uint8).numpy().tobytes()


def _native_gpu_payload(raw_bf16: bytes, codebook_header: bytes) -> GPUEncodedBF16:
    return encode_bfloat16_native(
        raw_bf16,
        codebook_header,
        bytes_per_thread=DEFAULT_BYTES_PER_THREAD,
        threads_per_block=DEFAULT_THREADS_PER_BLOCK,
    )


def _parameter_occurrences(model: nn.Module) -> dict[int, int]:
    occurrences = defaultdict(int)
    try:
        named_parameters = model.named_parameters(remove_duplicate=False)
    except TypeError:  # pragma: no cover - compatibility with older PyTorch.
        named_parameters = (
            (name, parameter)
            for name, module in model.named_modules()
            for parameter_name, parameter in module._parameters.items()
            if parameter is not None
            for name in (_join_name(name, parameter_name),)
        )
    for _name, parameter in named_parameters:
        occurrences[id(parameter)] += 1
    return dict(occurrences)


def _assemble_gpu_export(
    model: nn.Module,
    block_order: tuple[str, ...],
    candidates: tuple[_LinearCandidate, ...],
    codebook_payloads,
    resolve,
) -> NeuralZipModelExport:
    """Build one GPU export from a source-specific candidate resolver."""

    occurrences = _parameter_occurrences(model)
    payload_by_id = {
        int(payload.codebook_id): payload for payload in tuple(codebook_payloads)
    }
    codebooks = {}
    grouped_candidates = defaultdict(list)
    resident_names = []
    resident_parameter_bytes = {}

    def keep_resident(candidate: _LinearCandidate) -> None:
        resident_names.append(candidate.tensor_name)
        resident_parameter_bytes.setdefault(
            id(candidate.parameter),
            candidate.parameter.numel() * candidate.parameter.element_size(),
        )

    for candidate in candidates:
        parameter = candidate.parameter
        if (
            occurrences[id(parameter)] != 1
            or parameter.dtype != torch.bfloat16
            or parameter.numel() == 0
            or not parameter.is_contiguous()
        ):
            keep_resident(candidate)
            continue

        selected = resolve(candidate)
        if selected is None:
            keep_resident(candidate)
            continue
        codebook_id, raw_loader = selected
        if isinstance(codebook_id, bool) or not isinstance(codebook_id, int):
            raise TypeError(
                f"Shared exponent mode for {candidate.tensor_name} has no "
                "integer codebook_id."
            )
        if not callable(raw_loader):
            raise TypeError("A GPU candidate raw-data loader must be callable.")

        payload = payload_by_id.get(codebook_id)
        if payload is None:
            raise ValueError(f"Missing codebook payload {codebook_id}.")
        if (
            payload.codec != "huff0"
            or payload.mode not in {"shared_huff0", "shared_huff0_4x"}
            or payload.dtype != "bfloat16"
            or int(payload.stream_id) != 1
        ):
            raise ValueError(
                f"Codebook {codebook_id} is not a shared BF16 exponent codebook."
            )
        if codebook_id not in codebooks:
            table = huff0.export_code_table(payload.codebook)
            codebook = GPUCodebook.from_codes(
                table.code_lengths,
                table.code_values,
            )
            if codebook.max_bits != table.max_bits:
                raise RuntimeError("Exported Huff0 maximum code length changed.")
            codebooks[codebook_id] = codebook
        grouped_candidates[(candidate.block_name, codebook_id)].append(
            (candidate, raw_loader)
        )

    block_index = {name: index for index, name in enumerate(block_order)}
    groups = []
    workspace_offsets = defaultdict(int)
    for (block_name, codebook_id), members in sorted(
        grouped_candidates.items(),
        key=lambda item: (block_index[item[0][0]], item[0][1]),
    ):
        members.sort(key=lambda item: item[0].module_name)
        workspace_offset = workspace_offsets[block_name]
        group_offset = 0
        bindings = []
        raw_parts = []
        for candidate, raw_loader in members:
            parameter = candidate.parameter
            bindings.append(
                LinearWeightBinding(
                    block_name=block_name,
                    module_name=candidate.module_name,
                    tensor_name=candidate.tensor_name,
                    shape=tuple(parameter.shape),
                    numel=parameter.numel(),
                    workspace_offset=workspace_offset + group_offset,
                    group_offset=group_offset,
                    requires_grad=bool(parameter.requires_grad),
                    module_kind=candidate.module_kind,
                )
            )
            raw = raw_loader()
            if not isinstance(raw, bytes):
                raw = bytes(raw)
            expected_bytes = parameter.numel() * parameter.element_size()
            if len(raw) != expected_bytes:
                raise RuntimeError(
                    f"Encoded weight {candidate.tensor_name} has {len(raw)} raw "
                    f"bytes; expected {expected_bytes}."
                )
            raw_parts.append(raw)
            group_offset += parameter.numel()

        payload = payload_by_id[codebook_id]
        encoded = _native_gpu_payload(b"".join(raw_parts), payload.codebook)
        if encoded.exponents.symbol_count != group_offset:
            raise RuntimeError("GPU encoder returned the wrong BF16 element count.")
        groups.append(
            BF16WeightGroup(
                block_name=block_name,
                codebook_id=codebook_id,
                codebook=codebooks[codebook_id],
                encoded=encoded,
                bindings=tuple(bindings),
                workspace_offset=workspace_offset,
            )
        )
        workspace_offsets[block_name] += group_offset

    compressed_blocks = {group.block_name for group in groups}
    return NeuralZipModelExport(
        groups=tuple(groups),
        block_order=tuple(name for name in block_order if name in compressed_blocks),
        resident_weight_names=tuple(sorted(resident_names)),
        resident_raw_bytes=sum(resident_parameter_bytes.values()),
    )


def export_model_for_gpu(
    model: nn.Module,
    setup,
    *,
    threads: int = 0,
    adapter: GPUModelAdapter | None = None,
    include_embeddings: bool = False,
) -> NeuralZipModelExport:
    """Export compatible block-local BF16 weights without mutating ``model``."""

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module.")
    if setup is None:
        raise TypeError("setup is required.")

    codec = materialize_setup(setup, threads=threads)
    named_tensors = tuple(
        (str(name), parameter)
        for name, parameter in model.named_parameters()
        if parameter.is_floating_point()
    )
    if not named_tensors:
        raise ValueError("No floating parameters were found.")
    codec.prepare_direct_transfer_target(named_tensors)
    return _export_model_for_gpu_current_setup(
        model,
        codec,
        adapter=adapter,
        include_embeddings=include_embeddings,
    )


def _export_model_for_gpu_current_setup(
    model: nn.Module,
    codec: NeuralZipCodec,
    *,
    adapter: GPUModelAdapter | None,
    include_embeddings: bool,
) -> NeuralZipModelExport:
    """Export with an independently materialized and target-mapped codec."""

    block_order, candidates = _gpu_candidates(
        model,
        adapter=adapter,
        include_embeddings=include_embeddings,
    )

    def resolve(candidate: _LinearCandidate):
        parameter = candidate.parameter
        mode = codec._mode_for_tensor_stream(
            candidate.tensor_name,
            parameter,
            1,
        )
        kind = str(mode.get("kind"))
        if kind == "raw":
            return None
        if kind == "packed_huff0":
            raise ValueError(
                "GPU inference does not yet decode packed exponents; "
                f"{candidate.tensor_name} uses packed_huff0. Fit or transfer "
                "a sharing-only setup."
            )
        if kind != "shared_huff0":
            raise ValueError(
                f"GPU inference requires raw or shared_huff0 exponent modes; "
                f"{candidate.tensor_name} uses {kind!r}."
            )

        codebook_id = mode.get("codebook_id")
        if isinstance(codebook_id, bool) or not isinstance(codebook_id, int):
            raise TypeError(
                f"Shared exponent mode for {candidate.tensor_name} has no codebook_id."
            )
        return codebook_id, partial(_tensor_raw_bytes, parameter)

    return _assemble_gpu_export(
        model,
        block_order,
        candidates,
        codec.export_codebooks(),
        resolve,
    )


def _encoded_tensor_raw_bytes(payload, decoder: NeuralZipCodec) -> bytes:
    tensor = decoder.decompress_tensor(payload)
    if tuple(tensor.shape) != tuple(payload.shape) or tensor.dtype != torch.bfloat16:
        raise RuntimeError(f"Decoded GPU weight {payload.name!r} changed schema.")
    return _tensor_raw_bytes(tensor)


def export_encoded_model_for_gpu(
    model: nn.Module,
    encoded,
    *,
    adapter: GPUModelAdapter | None = None,
    include_embeddings: bool = False,
) -> NeuralZipModelExport:
    """Convert a NeuralZip storage result into the CUDA inference layout."""

    from ..encoded_model import EncodedModel
    from ..payloads import TensorPayload

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module.")
    if not isinstance(encoded, EncodedModel):
        raise TypeError("encoded must be an EncodedModel.")
    if encoded.source_kind == "mapping":
        raise ValueError("GPU inference requires an encoded torch.nn.Module.")

    block_order, candidates = _gpu_candidates(
        model,
        adapter=adapter,
        include_embeddings=include_embeddings,
    )
    payload_by_name = dict(zip(encoded.names, encoded.payloads, strict=True))
    decoder = NeuralZipCodec(ngram_values=())
    decoder.load_codebooks(encoded.codebooks)

    def resolve(candidate: _LinearCandidate):
        payload = payload_by_name.get(candidate.tensor_name)
        if payload is None:
            raise ValueError(
                f"Encoded model has no tensor payload for {candidate.tensor_name!r}."
            )
        if not isinstance(payload, TensorPayload):
            raise TypeError(
                f"Encoded payload for {candidate.tensor_name!r} is not a TensorPayload."
            )
        parameter = candidate.parameter
        if (
            payload.strategy != "neuralzip"
            or payload.dtype != "bfloat16"
            or tuple(payload.shape) != tuple(parameter.shape)
            or int(payload.numel) != parameter.numel()
            or int(payload.element_size) != 2
            or int(payload.bit_reorder) != 1
            or len(payload.groups) != 2
            or payload.exponent_layout != "original"
        ):
            raise ValueError(
                f"Encoded tensor {candidate.tensor_name!r} is incompatible with "
                "BF16 GPU inference."
            )

        exponent_chunks = tuple(payload.groups[1].chunks)
        modes = {str(chunk.mode) for chunk in exponent_chunks}
        if "packed_huff0" in modes:
            raise ValueError(
                "GPU inference does not yet decode packed exponents; "
                f"{candidate.tensor_name} uses packed_huff0. Encode with a "
                "sharing-only setup."
            )
        unsupported = modes - {"raw", "shared_huff0"}
        if unsupported:
            raise ValueError(
                f"GPU inference cannot convert exponent modes {sorted(unsupported)!r} "
                f"for {candidate.tensor_name}."
            )
        codebook_ids = {
            int(chunk.codebook_id)
            for chunk in exponent_chunks
            if chunk.mode == "shared_huff0"
        }
        if not codebook_ids:
            return None
        if len(codebook_ids) != 1:
            raise ValueError(
                f"Encoded tensor {candidate.tensor_name!r} references multiple "
                "exponent codebooks."
            )
        codebook_id = codebook_ids.pop()
        return codebook_id, partial(_encoded_tensor_raw_bytes, payload, decoder)

    return _assemble_gpu_export(
        model,
        block_order,
        candidates,
        encoded.codebooks,
        resolve,
    )


def _upload_bytes(data: bytes, *, device: torch.device) -> torch.Tensor:
    if not data:
        return torch.empty(0, dtype=torch.uint8, device=device)
    owner = bytearray(data)
    return torch.frombuffer(owner, dtype=torch.uint8).clone().to(device=device)


def _host_bytes(data: bytes, *, pin_memory: bool) -> torch.Tensor:
    if not data:
        tensor = torch.empty(0, dtype=torch.uint8)
    else:
        owner = bytearray(data)
        tensor = torch.frombuffer(owner, dtype=torch.uint8).clone()
    return tensor.pin_memory() if pin_memory else tensor


def _upload_group(
    group: BF16WeightGroup,
    lut: torch.Tensor,
    workspace: torch.Tensor,
    *,
    device: torch.device,
    offload_payload: bool = False,
    pin_memory: bool = True,
) -> _UploadedGroup:
    positions = torch.tensor(
        group.encoded.exponents.output_positions,
        dtype=torch.int32,
        device=device,
    )
    return _UploadedGroup(
        group=group,
        lut=lut,
        encoded_exponents=(
            _host_bytes(
                group.encoded.exponents.bitstream,
                pin_memory=pin_memory,
            )
            if offload_payload
            else _upload_bytes(
                group.encoded.exponents.bitstream,
                device=device,
            )
        ),
        sign_mantissa=(
            _host_bytes(group.encoded.sign_mantissa, pin_memory=pin_memory)
            if offload_payload
            else _upload_bytes(group.encoded.sign_mantissa, device=device)
        ),
        packed_gaps=_upload_bytes(
            group.encoded.exponents.packed_gaps,
            device=device,
        ),
        output_positions=positions,
        output=workspace[group.workspace_offset : group.workspace_offset + group.numel],
        payload_offloaded=offload_payload,
    )


def _resolve_device(model: nn.Module, device) -> torch.device:
    states = tuple(model.parameters()) + tuple(model.buffers())
    state_devices = {value.device for value in states}
    if device is None:
        if len(state_devices) > 1:
            raise ValueError("Model parameters and buffers must use one device.")
        return next(iter(state_devices), torch.device("cpu"))
    resolved = torch.device(device)
    if resolved.type == "cuda" and resolved.index is None:
        cuda_devices = {
            state_device
            for state_device in state_devices
            if state_device.type == "cuda"
        }
        if len(cuda_devices) == 1 and len(cuda_devices) == len(state_devices):
            resolved = next(iter(cuda_devices))
        else:
            resolved = torch.device("cuda", torch.cuda.current_device())
    if any(state_device != resolved for state_device in state_devices):
        raise ValueError(
            "Move the complete model to the inference device before attaching "
            "NeuralZip."
        )
    return resolved


def _resolve_block_devices(
    model: nn.Module,
    exported: NeuralZipModelExport,
    *,
    device,
    block_devices: Mapping[str, object] | None,
) -> dict[str, torch.device]:
    if device is not None and block_devices is not None:
        raise ValueError("Pass either device or block_devices, not both.")
    if block_devices is not None:
        if not isinstance(block_devices, Mapping):
            raise TypeError(
                "block_devices must be a mapping of block names to devices."
            )
        missing = set(exported.block_order) - set(block_devices)
        extra = set(block_devices) - set(exported.block_order)
        if missing or extra:
            raise ValueError(
                "block_devices must cover exactly the compressed blocks; "
                f"missing={sorted(missing)!r}, extra={sorted(extra)!r}."
            )
        resolved = {
            name: torch.device(block_devices[name]) for name in exported.block_order
        }
        if any(value.type == "meta" for value in resolved.values()):
            raise ValueError("Compressed blocks cannot execute on the meta device.")
        return resolved

    if device is not None:
        resolved = _resolve_device(model, device)
        return {name: resolved for name in exported.block_order}

    model_states = tuple(model.parameters()) + tuple(model.buffers())
    model_devices = {
        value.device for value in model_states if value.device.type != "meta"
    }
    discovered = {}
    for block_name in exported.block_order:
        block = model.get_submodule(block_name)
        states = tuple(block.parameters()) + tuple(block.buffers())
        devices = {value.device for value in states if value.device.type != "meta"}
        if not devices and len(model_devices) == 1:
            devices = set(model_devices)
        if len(devices) != 1:
            raise ValueError(
                f"Cannot infer one execution device for block {block_name!r}; "
                "pass block_devices explicitly."
            )
        discovered[block_name] = devices.pop()
    return discovered


def _offloaded_blocks(
    exported: NeuralZipModelExport,
    *,
    cpu_offload: bool,
    cpu_offload_blocks: int | None,
    cpu_offload_block_names: tuple[str, ...] | None = None,
) -> frozenset[str]:
    if not isinstance(cpu_offload, bool):
        raise TypeError("cpu_offload must be a boolean.")
    if cpu_offload_blocks is not None:
        _require_non_negative_integer("cpu_offload_blocks", cpu_offload_blocks)
        if not cpu_offload:
            raise ValueError("cpu_offload_blocks requires cpu_offload=True.")
    if not cpu_offload:
        selected = set()
    else:
        count = (
            len(exported.block_order)
            if cpu_offload_blocks is None
            else min(cpu_offload_blocks, len(exported.block_order))
        )
        selected = set(exported.block_order[:count])
    if cpu_offload_block_names is not None:
        if not isinstance(cpu_offload_block_names, tuple) or any(
            not isinstance(name, str) for name in cpu_offload_block_names
        ):
            raise TypeError("cpu_offload_block_names must be a tuple of block names.")
        unknown = set(cpu_offload_block_names) - set(exported.block_order)
        if unknown:
            raise ValueError(
                f"Unknown CPU-offloaded NeuralZip blocks: {sorted(unknown)!r}."
            )
        selected.update(cpu_offload_block_names)
    return frozenset(selected)


class NeuralZipRuntime:
    """Attached sequential runtime for one compressed model instance."""

    def __init__(
        self,
        model: nn.Module,
        exported: NeuralZipModelExport,
        *,
        block_devices: Mapping[str, torch.device],
        decoder: Callable | object,
        precompressed: bool = False,
        cpu_offload: bool = False,
        cpu_offload_blocks: int | None = None,
        cpu_offload_block_names: tuple[str, ...] | None = None,
        pin_memory: bool = True,
    ):
        self.model = model
        self.export = exported
        self.block_devices = {
            name: torch.device(block_devices[name]) for name in exported.block_order
        }
        self.devices = tuple(dict.fromkeys(self.block_devices.values()))
        self.device = self.devices[0] if len(self.devices) == 1 else None
        self._decoder_backend = decoder
        self._decoder = decoder.decode if hasattr(decoder, "decode") else decoder
        if not callable(self._decoder):
            raise TypeError("decoder must be callable or provide a decode method.")
        prepare_batch = getattr(decoder, "prepare_batch", None)
        decode_batch = getattr(decoder, "decode_batch", None)
        self._prepare_batch = prepare_batch if callable(prepare_batch) else None
        self._decode_batch = decode_batch if callable(decode_batch) else None
        if (self._prepare_batch is None) != (self._decode_batch is None):
            raise TypeError(
                "A batched decoder must provide both prepare_batch and decode_batch."
            )
        self._state_lock = threading.Lock()
        self._forward_claims = threading.local()
        self._active_block = None
        self._cuda_stream = None
        self._cuda_streams = {}
        self._hook_handles = []
        self._state_dict_hook_handle = None
        self._restored = False
        self._precompressed = bool(precompressed)
        if not isinstance(pin_memory, bool):
            raise TypeError("pin_memory must be a boolean.")
        self._offloaded_blocks = _offloaded_blocks(
            exported,
            cpu_offload=cpu_offload,
            cpu_offload_blocks=cpu_offload_blocks,
            cpu_offload_block_names=cpu_offload_block_names,
        )
        self._pin_memory = pin_memory

        self.workspaces = {
            device: torch.empty(
                exported.workspace_numel,
                dtype=torch.bfloat16,
                device=device,
            )
            for device in self.devices
        }
        self.workspace = (
            self.workspaces[self.device] if self.device is not None else None
        )
        lut_by_id = {}
        for group in exported.groups:
            group_device = self.block_devices[group.block_name]
            key = (group_device, group.codebook_id)
            if key not in lut_by_id:
                lut_by_id[key] = torch.tensor(
                    group.codebook.decode_lut,
                    dtype=torch.uint16,
                    device=group_device,
                )
        self._uploaded_groups = tuple(
            _upload_group(
                group,
                lut_by_id[(self.block_devices[group.block_name], group.codebook_id)],
                self.workspaces[self.block_devices[group.block_name]],
                device=self.block_devices[group.block_name],
                offload_payload=group.block_name in self._offloaded_blocks,
                pin_memory=(
                    pin_memory and self.block_devices[group.block_name].type == "cuda"
                ),
            )
            for group in exported.groups
        )
        groups_by_block = defaultdict(list)
        for uploaded in self._uploaded_groups:
            groups_by_block[uploaded.group.block_name].append(uploaded)
        self._groups_by_block = {
            name: tuple(groups_by_block[name]) for name in exported.block_order
        }
        self._prepared_batches = {}
        if self._prepare_batch is not None:
            for block_name, uploaded_groups in self._groups_by_block.items():
                if block_name in self._offloaded_blocks:
                    continue
                thread_counts = {
                    uploaded.group.encoded.exponents.threads_per_block
                    for uploaded in uploaded_groups
                }
                if len(thread_counts) != 1:
                    raise ValueError(
                        "Every compressed group in one block must use the same "
                        "CUDA thread layout."
                    )
                descriptors = tuple(
                    gpu_decoder.BF16DecodeGroup(
                        lut=uploaded.lut,
                        encoded_exponents=uploaded.encoded_exponents,
                        sign_mantissa=uploaded.sign_mantissa,
                        packed_gaps=uploaded.packed_gaps,
                        output_positions=uploaded.output_positions,
                        output=uploaded.output,
                        max_bits=uploaded.group.codebook.max_bits,
                    )
                    for uploaded in uploaded_groups
                )
                self._prepared_batches[block_name] = self._prepare_batch(
                    descriptors,
                    threads_per_block=thread_counts.pop(),
                )

        tensors_by_device = defaultdict(list)
        for workspace in self.workspaces.values():
            if workspace.is_cuda:
                tensors_by_device[workspace.device].append(workspace)
        for uploaded in self._uploaded_groups:
            for tensor in (
                uploaded.lut,
                uploaded.encoded_exponents,
                uploaded.sign_mantissa,
                uploaded.packed_gaps,
                uploaded.output_positions,
                uploaded.output,
            ):
                if tensor.is_cuda:
                    tensors_by_device[tensor.device].append(tensor)
        for prepared in self._prepared_batches.values():
            values = vars(prepared).values() if hasattr(prepared, "__dict__") else ()
            for value in values:
                if torch.is_tensor(value) and value.is_cuda:
                    tensors_by_device[value.device].append(value)
        self._cuda_tensors_by_device = {}
        self._attach_events = {}
        for state_device, tensors in tensors_by_device.items():
            unique = {}
            for tensor in tensors:
                unique.setdefault(id(tensor), tensor)
            self._cuda_tensors_by_device[state_device] = tuple(unique.values())
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(device=state_device))
            self._attach_events[state_device] = event
        self._cuda_tensors = tuple(
            tensor
            for tensors in self._cuda_tensors_by_device.values()
            for tensor in tensors
        )
        self._attach_event = (
            self._attach_events.get(self.device) if self.device is not None else None
        )

    @classmethod
    def _prepare_runtime(
        cls,
        model: nn.Module,
        exported: NeuralZipModelExport,
        *,
        device=None,
        block_devices: Mapping[str, object] | None = None,
        decoder=None,
        precompressed: bool = False,
        cpu_offload: bool = False,
        cpu_offload_blocks: int | None = None,
        cpu_offload_block_names: tuple[str, ...] | None = None,
        pin_memory: bool = True,
    ) -> NeuralZipRuntime:
        """Allocate and preflight runtime state without mutating ``model``."""

        if hasattr(model, _RUNTIME_ATTRIBUTE):
            raise RuntimeError("This model already has a NeuralZip runtime.")
        if not exported.groups:
            raise ValueError(
                "The GPU export contains no eligible shared BF16 linear weights."
            )
        if model.training:
            raise ValueError("Call model.eval() before attaching NeuralZip inference.")
        resolved_block_devices = _resolve_block_devices(
            model,
            exported,
            device=device,
            block_devices=block_devices,
        )
        if decoder is None:
            if any(value.type != "cuda" for value in resolved_block_devices.values()):
                raise ValueError(
                    "The default NeuralZip decoder requires CUDA block devices; "
                    "a decoder may only be injected for testing."
                )
            decoder = gpu_decoder.default_decoder()

        # All compressed buffers and the reusable workspace are allocated
        # before the first model parameter is removed.
        runtime = cls(
            model,
            exported,
            block_devices=resolved_block_devices,
            decoder=decoder,
            precompressed=precompressed,
            cpu_offload=cpu_offload,
            cpu_offload_blocks=cpu_offload_blocks,
            cpu_offload_block_names=cpu_offload_block_names,
            pin_memory=pin_memory,
        )
        runtime._preflight_decoder()
        return runtime

    @classmethod
    def attach(
        cls,
        model: nn.Module,
        exported: NeuralZipModelExport,
        *,
        device=None,
        block_devices: Mapping[str, object] | None = None,
        decoder=None,
        precompressed: bool = False,
        cpu_offload: bool = False,
        cpu_offload_blocks: int | None = None,
        cpu_offload_block_names: tuple[str, ...] | None = None,
        pin_memory: bool = True,
    ) -> NeuralZipRuntime:
        runtime = cls._prepare_runtime(
            model,
            exported,
            device=device,
            block_devices=block_devices,
            decoder=decoder,
            precompressed=precompressed,
            cpu_offload=cpu_offload,
            cpu_offload_blocks=cpu_offload_blocks,
            cpu_offload_block_names=cpu_offload_block_names,
            pin_memory=pin_memory,
        )
        runtime._install()
        return runtime

    def _preflight_decoder(self) -> None:
        preflight = getattr(self._decoder_backend, "preflight", None)
        if not callable(preflight):
            return
        if self._prepared_batches:
            preflight(tuple(self._prepared_batches.values()))
        for block_name in self.export.block_order:
            if block_name not in self._offloaded_blocks:
                continue
            if self._prepare_batch is None:
                raise TypeError(
                    "Decoder preflight for CPU-offloaded payloads requires "
                    "batched prepare/decode methods."
                )
            transient = self._prepare_offloaded_batch(block_name)
            preflight((transient,))

    def _binding_module(self, binding: LinearWeightBinding) -> nn.Module:
        module = self.model.get_submodule(binding.module_name)
        expected = nn.Embedding if binding.module_kind == "embedding" else nn.Linear
        if not isinstance(module, expected):
            raise TypeError(
                f"{binding.module_name} is no longer an {expected.__name__}."
            )
        return module

    def _install_weight_views(self) -> None:
        for binding in self.export.bindings:
            begin = binding.workspace_offset
            workspace = self.workspaces[self.block_devices[binding.block_name]]
            view = workspace[begin : begin + binding.numel].view(binding.shape)
            self._binding_module(binding).__dict__["weight"] = view

    def _remove_weight_views(self) -> None:
        for binding in self.export.bindings:
            self._binding_module(binding).__dict__.pop("weight", None)

    def _release_device_state(self) -> None:
        """Drop every runtime-owned tensor after a successful restore."""

        self._uploaded_groups = ()
        self._groups_by_block = {}
        self._prepared_batches = {}
        self._cuda_tensors = ()
        self._cuda_tensors_by_device = {}
        self._attach_events = {}
        self._cuda_streams = {}
        self.workspaces = {}
        self.workspace = None
        self._attach_event = None
        self._decoder_backend = None
        self._decoder = None
        self._prepare_batch = None
        self._decode_batch = None

    def _install(self) -> None:
        original_parameters = []
        handles = []
        try:
            for binding in self.export.bindings:
                module = self._binding_module(binding)
                parameter = module._parameters.get("weight")
                if self._precompressed:
                    if parameter is not None or "weight" in module.__dict__:
                        raise RuntimeError(
                            f"{binding.tensor_name} must be absent before attaching "
                            "a precompressed NeuralZip checkpoint."
                        )
                    continue
                if not isinstance(parameter, nn.Parameter):
                    raise TypeError(f"{binding.tensor_name} changed after GPU export.")
                if (
                    parameter.dtype != torch.bfloat16
                    or tuple(parameter.shape) != binding.shape
                    or parameter.device != self.block_devices[binding.block_name]
                ):
                    raise RuntimeError(
                        f"{binding.tensor_name} is incompatible with its GPU export."
                    )
                original_parameters.append((module, parameter))

            for module, _parameter in original_parameters:
                module._parameters.pop("weight")
            self._install_weight_views()

            try:
                model_pre = self.model.register_forward_pre_hook(
                    self._model_pre_hook,
                    prepend=True,
                )
            except TypeError:  # pragma: no cover - older PyTorch.
                model_pre = self.model.register_forward_pre_hook(self._model_pre_hook)
            try:
                model_post = self.model.register_forward_hook(
                    self._model_post_hook,
                    always_call=True,
                )
            except TypeError as error:  # pragma: no cover - older PyTorch.
                raise RuntimeError(
                    "NeuralZip GPU inference requires a PyTorch version whose "
                    "forward hooks support always_call=True."
                ) from error
            handles.extend((model_pre, model_post))
            state_dict_handle = self.model.register_state_dict_pre_hook(
                self._state_dict_pre_hook
            )
            handles.append(state_dict_handle)
            self._state_dict_hook_handle = state_dict_handle

            for block_name in self.export.block_order:
                block = self.model.get_submodule(block_name)
                try:
                    pre = block.register_forward_pre_hook(
                        partial(self._block_pre_hook, block_name),
                        prepend=True,
                    )
                except TypeError:  # pragma: no cover - older PyTorch.
                    pre = block.register_forward_pre_hook(
                        partial(self._block_pre_hook, block_name)
                    )
                handles.append(pre)
            self._hook_handles = handles
            setattr(self.model, _RUNTIME_ATTRIBUTE, self)
        except Exception:
            for handle in handles:
                handle.remove()
            self._remove_weight_views()
            for module, parameter in original_parameters:
                if "weight" not in module._parameters:
                    module.register_parameter("weight", parameter)
            self._state_dict_hook_handle = None
            raise

    def _check_stream(self, device: torch.device) -> None:
        if device.type != "cuda":
            return
        stream = torch.cuda.current_stream(device=device)
        current = int(stream.cuda_stream)
        previous = self._cuda_streams.get(device)
        if previous is None:
            event = self._attach_events.get(device)
            if event is not None:
                stream.wait_event(event)
            for tensor in self._cuda_tensors_by_device.get(device, ()):
                tensor.record_stream(stream)
            self._cuda_streams[device] = current
            if self.device == device:
                self._cuda_stream = current
        elif current != previous:
            raise RuntimeError(
                "NeuralZip supports one CUDA stream per runtime device."
            )

    def _decode_descriptor(
        self,
        uploaded: _UploadedGroup,
    ) -> gpu_decoder.BF16DecodeGroup:
        encoded_exponents = uploaded.encoded_exponents
        sign_mantissa = uploaded.sign_mantissa
        if uploaded.payload_offloaded:
            target_device = uploaded.output.device
            encoded_exponents = encoded_exponents.to(
                device=target_device,
                non_blocking=self._pin_memory,
            )
            sign_mantissa = sign_mantissa.to(
                device=target_device,
                non_blocking=self._pin_memory,
            )
        return gpu_decoder.BF16DecodeGroup(
            lut=uploaded.lut,
            encoded_exponents=encoded_exponents,
            sign_mantissa=sign_mantissa,
            packed_gaps=uploaded.packed_gaps,
            output_positions=uploaded.output_positions,
            output=uploaded.output,
            max_bits=uploaded.group.codebook.max_bits,
        )

    def _decode_uploaded_group(self, uploaded: _UploadedGroup) -> torch.Tensor:
        group = uploaded.group
        descriptor = self._decode_descriptor(uploaded)
        self._decoder(
            descriptor.lut,
            descriptor.encoded_exponents,
            descriptor.sign_mantissa,
            descriptor.packed_gaps,
            descriptor.output_positions,
            descriptor.output,
            max_bits=group.codebook.max_bits,
            threads_per_block=group.encoded.exponents.threads_per_block,
        )
        return descriptor.output

    def _prepare_offloaded_batch(self, block_name: str):
        uploaded_groups = self._groups_by_block[block_name]
        thread_counts = {
            uploaded.group.encoded.exponents.threads_per_block
            for uploaded in uploaded_groups
        }
        if len(thread_counts) != 1:
            raise RuntimeError(
                "Every offloaded group in one block must use the same "
                "CUDA thread layout."
            )
        descriptors = tuple(
            self._decode_descriptor(uploaded) for uploaded in uploaded_groups
        )
        return self._prepare_batch(
            descriptors,
            threads_per_block=thread_counts.pop(),
        )

    def _decode_block(self, block_name: str) -> None:
        prepared = self._prepared_batches.get(block_name)
        if prepared is not None:
            self._decode_batch(prepared)
            return
        uploaded_groups = self._groups_by_block[block_name]
        if self._prepare_batch is not None and block_name in self._offloaded_blocks:
            transient = self._prepare_offloaded_batch(block_name)
            self._decode_batch(transient)
            return
        for uploaded in uploaded_groups:
            self._decode_uploaded_group(uploaded)

    def _inject_block(self, block_name: str) -> None:
        self._decode_block(block_name)

    def _model_pre_hook(self, _module, _args) -> None:
        claims = getattr(self._forward_claims, "stack", None)
        if claims is None:
            claims = []
            self._forward_claims.stack = claims
        claims.append(False)
        if self.model.training:
            raise RuntimeError(
                "NeuralZip is inference-only; keep the attached model in "
                "evaluation mode."
            )
        if torch.is_grad_enabled():
            raise RuntimeError(
                "NeuralZip is inference-only; run it under torch.no_grad() "
                "or torch.inference_mode()."
            )
        with self._state_lock:
            if self._restored:
                raise RuntimeError("NeuralZip runtime has already been restored.")
            if self._active_block is not None:
                raise RuntimeError(
                    "Concurrent or re-entrant NeuralZip inference is unsupported."
                )
            for device in self.devices:
                self._check_stream(device)
            self._active_block = "<forward>"
            claims[-1] = True

    def _model_post_hook(self, _module, _args, _output) -> None:
        claims = getattr(self._forward_claims, "stack", None)
        claimed = bool(claims and claims.pop())
        if not claimed:
            return
        with self._state_lock:
            if self._active_block == "<forward>":
                self._active_block = None

    def _state_dict_pre_hook(self, _module, _prefix, _keep_vars) -> None:
        raise RuntimeError(
            "Call model.neuralzip_runtime.restore() before state_dict() or "
            "save_pretrained(); attached compressed weights are not registered "
            "Parameters."
        )

    def resident_state_dict(self) -> dict[str, torch.Tensor]:
        """Return registered state while intentionally omitting compressed views."""

        with self._state_lock:
            if self._restored:
                raise RuntimeError("NeuralZip runtime has already been restored.")
            if self._active_block is not None:
                raise RuntimeError("Cannot serialize during an active model forward.")
            if _has_external_dispatch(self.model):
                raise RuntimeError(
                    "Cannot re-save an attached model while Accelerate placement or "
                    "offload hooks are active; copy the original NeuralZip checkpoint."
                )
            handle = self._state_dict_hook_handle
            if handle is None:
                raise RuntimeError("The NeuralZip serialization guard is missing.")
            try:
                index = self._hook_handles.index(handle)
            except ValueError as error:  # pragma: no cover - internal invariant.
                raise RuntimeError(
                    "The NeuralZip serialization guard is invalid."
                ) from error
            handle.remove()
            try:
                state = self.model.state_dict()
            finally:
                replacement = self.model.register_state_dict_pre_hook(
                    self._state_dict_pre_hook
                )
                self._hook_handles[index] = replacement
                self._state_dict_hook_handle = replacement
            return state

    def _block_pre_hook(self, block_name, _module, _args) -> None:
        if self._active_block != "<forward>":
            raise RuntimeError(
                "Invoke the attached root model rather than a compressed block "
                "directly."
            )
        self._inject_block(block_name)

    def _materialize_tensors(self) -> dict[str, torch.Tensor]:
        tensors = {}
        with torch.no_grad():
            for block_name in self.export.block_order:
                self._decode_block(block_name)
                for uploaded in self._groups_by_block[block_name]:
                    for binding in uploaded.group.bindings:
                        begin = binding.workspace_offset
                        workspace = self.workspaces[
                            self.block_devices[binding.block_name]
                        ]
                        tensors[binding.tensor_name] = (
                            workspace[begin : begin + binding.numel]
                            .view(binding.shape)
                            .clone()
                        )
        return tensors

    def materialize(self) -> dict[str, torch.Tensor]:
        """Decode independent tensors for every compressed linear weight."""

        marker = "<materialize>"
        with self._state_lock:
            if self._restored:
                raise RuntimeError("NeuralZip runtime has already been restored.")
            if self._active_block is not None:
                raise RuntimeError("Cannot materialize during an active block forward.")
            for device in self.devices:
                self._check_stream(device)
            self._active_block = marker
        try:
            return self._materialize_tensors()
        finally:
            with self._state_lock:
                if self._active_block == marker:
                    self._active_block = None

    def restore(self) -> nn.Module:
        """Materialize parameters, remove hooks, and return the ordinary model."""

        marker = "<restore>"
        with self._state_lock:
            if self._restored:
                raise RuntimeError("NeuralZip runtime has already been restored.")
            if self._active_block is not None:
                raise RuntimeError("Cannot restore during an active block forward.")
            if _has_external_dispatch(self.model):
                raise RuntimeError(
                    "Cannot restore registered Parameters while Accelerate placement "
                    "or offload hooks are active; use materialize() or load on one "
                    "direct device first."
                )
            for device in self.devices:
                self._check_stream(device)
            self._active_block = marker
        try:
            tensors = self._materialize_tensors()
            parameters = []
            for binding in self.export.bindings:
                tensor = tensors[binding.tensor_name]
                parameters.append(
                    (
                        self._binding_module(binding),
                        nn.Parameter(tensor, requires_grad=binding.requires_grad),
                    )
                )

            registered_modules = []
            try:
                self._remove_weight_views()
                for module, parameter in parameters:
                    module.register_parameter("weight", parameter)
                    registered_modules.append(module)
            except Exception:
                for module in registered_modules:
                    module._parameters.pop("weight", None)
                self._install_weight_views()
                raise

            for handle in self._hook_handles:
                handle.remove()
            self._hook_handles = []
            self._state_dict_hook_handle = None
            with self._state_lock:
                self._restored = True
                self._active_block = None
            if getattr(self.model, _RUNTIME_ATTRIBUTE, None) is self:
                delattr(self.model, _RUNTIME_ATTRIBUTE)
            self._release_device_state()
            return self.model
        except Exception:
            with self._state_lock:
                if self._active_block == marker:
                    self._active_block = None
            raise


__all__ = [
    "BF16WeightGroup",
    "LinearWeightBinding",
    "NeuralZipModelExport",
    "NeuralZipRuntime",
    "export_encoded_model_for_gpu",
    "export_model_for_gpu",
]
