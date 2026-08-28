"""In-memory DFloat11 encoding for controlled compression experiments.

The public :func:`dfloat11.compress_model` entry point also mutates the model,
checks CUDA reconstruction, and serializes a checkpoint.  Those operations are
useful for producing distributable artifacts, but they obscure the CPU encoding
time in a codec comparison.  This module calls DFloat11 0.5.0's official
codebook, LUT, stream encoder, and runtime-hook primitives while keeping model
loading and disk I/O outside the measured interval.

The intended call order is::

    model = load_bfloat16_model_on_cpu()
    encoding = encode_dfloat11_in_memory(model, pattern_dict)
    install_dfloat11_runtime(model, encoding)
    model = model.to("cuda:0").eval()

Encoding does not mutate ``model``.  Runtime installation removes the covered
BF16 parameters and installs DFloat11's compressed buffers and official decode
pre-hook, matching its checkpoint loader without an intermediate file.
"""

from __future__ import annotations

import contextlib
import importlib.metadata
import os
import re
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import torch
from torch import nn

_QUIET_LOCK = threading.RLock()


def _tensor_nbytes(value: torch.Tensor) -> int:
    return value.numel() * value.element_size()


@dataclass(frozen=True)
class DFloat11BlockEncoding:
    """Official DFloat11 buffers and timing for one matched compression block."""

    block_name: str
    injection_paths: tuple[str, ...]
    weight_names: tuple[str, ...]
    weight_shapes: tuple[tuple[int, ...], ...]
    weight_numels: tuple[int, ...]
    luts: torch.Tensor
    encoded_exponent: torch.Tensor
    sign_mantissa: torch.Tensor
    output_positions: torch.Tensor
    gaps: torch.Tensor
    split_positions: torch.Tensor
    extraction_seconds: float
    codebook_seconds: float
    stream_encode_seconds: float

    @property
    def encode_seconds(self) -> float:
        return (
            self.extraction_seconds + self.codebook_seconds + self.stream_encode_seconds
        )

    @property
    def numel(self) -> int:
        return sum(self.weight_numels)

    @property
    def raw_bfloat16_bytes(self) -> int:
        return 2 * self.numel

    @property
    def raw_exponent_bytes(self) -> int:
        return self.numel

    @property
    def exponent_stream_bytes(self) -> int:
        return _tensor_nbytes(self.encoded_exponent)

    @property
    def exponent_metadata_bytes(self) -> int:
        return _tensor_nbytes(self.output_positions) + _tensor_nbytes(self.gaps)

    @property
    def exponent_codebook_bytes(self) -> int:
        return _tensor_nbytes(self.luts)

    @property
    def exponent_compressed_bytes(self) -> int:
        return (
            self.exponent_stream_bytes
            + self.exponent_metadata_bytes
            + self.exponent_codebook_bytes
        )

    @property
    def sign_mantissa_bytes(self) -> int:
        return _tensor_nbytes(self.sign_mantissa)

    @property
    def gpu_payload_bytes(self) -> int:
        """Persistent CUDA buffers, excluding the reusable BF16 workspace."""

        return self.exponent_compressed_bytes + self.sign_mantissa_bytes

    @property
    def split_positions_host_bytes(self) -> int:
        return _tensor_nbytes(self.split_positions)


@dataclass(frozen=True)
class DFloat11InMemoryEncoding:
    """Model-level in-memory result produced with official DFloat11 primitives."""

    blocks: tuple[DFloat11BlockEncoding, ...]
    encode_seconds: float
    dfloat11_version: str | None
    bytes_per_thread: int
    threads_per_block: tuple[int, ...]

    @property
    def block_names(self) -> tuple[str, ...]:
        return tuple(block.block_name for block in self.blocks)

    @property
    def weight_names(self) -> tuple[str, ...]:
        return tuple(name for block in self.blocks for name in block.weight_names)

    @property
    def raw_bfloat16_bytes(self) -> int:
        return sum(block.raw_bfloat16_bytes for block in self.blocks)

    @property
    def raw_exponent_bytes(self) -> int:
        return sum(block.raw_exponent_bytes for block in self.blocks)

    @property
    def exponent_stream_bytes(self) -> int:
        return sum(block.exponent_stream_bytes for block in self.blocks)

    @property
    def exponent_metadata_bytes(self) -> int:
        return sum(block.exponent_metadata_bytes for block in self.blocks)

    @property
    def exponent_codebook_bytes(self) -> int:
        return sum(block.exponent_codebook_bytes for block in self.blocks)

    @property
    def exponent_compressed_bytes(self) -> int:
        return sum(block.exponent_compressed_bytes for block in self.blocks)

    @property
    def exponent_ratio(self) -> float:
        if not self.raw_exponent_bytes:
            return 0.0
        return self.exponent_compressed_bytes / self.raw_exponent_bytes

    @property
    def sign_mantissa_bytes(self) -> int:
        return sum(block.sign_mantissa_bytes for block in self.blocks)

    @property
    def gpu_payload_bytes(self) -> int:
        return sum(block.gpu_payload_bytes for block in self.blocks)

    @property
    def split_positions_host_bytes(self) -> int:
        return sum(block.split_positions_host_bytes for block in self.blocks)


@dataclass(frozen=True)
class _OfficialDFloat11API:
    get_codec: Callable[..., Any]
    get_32bit_codec: Callable[..., Any]
    get_luts: Callable[..., torch.Tensor]
    encode_weights: Callable[..., tuple[torch.Tensor, ...]]
    get_hook: Callable[..., Callable[..., Any]]
    utils_module: ModuleType
    bytes_per_thread: int
    threads_per_block: tuple[int, ...]
    version: str | None


def _load_official_api() -> _OfficialDFloat11API:
    try:
        import dfloat11.dfloat11 as runtime
        import dfloat11.dfloat11_utils as utils
    except ImportError as exc:
        raise RuntimeError(
            "DFloat11 is required for this benchmark helper. Install dfloat11==0.5.0."
        ) from exc

    try:
        version = importlib.metadata.version("dfloat11")
    except importlib.metadata.PackageNotFoundError:
        version = getattr(runtime, "version", None)
    return _OfficialDFloat11API(
        get_codec=utils.get_codec,
        get_32bit_codec=utils.get_32bit_codec,
        get_luts=utils.get_luts,
        encode_weights=utils.encode_weights,
        get_hook=runtime.get_hook,
        utils_module=utils,
        bytes_per_thread=int(runtime.bytes_per_thread),
        threads_per_block=tuple(int(value) for value in runtime.threads_per_block),
        version=version,
    )


@contextlib.contextmanager
def _quiet_official_output(api: _OfficialDFloat11API, *, quiet: bool) -> Iterator[None]:
    """Disable DFloat11 progress/diagnostic output without changing its codec."""

    if not quiet:
        yield
        return

    with _QUIET_LOCK:
        original_tqdm = api.utils_module.tqdm
        api.utils_module.tqdm = lambda iterable, *args, **kwargs: iterable
        try:
            with (
                open(os.devnull, "w", encoding="utf-8") as sink,
                contextlib.redirect_stdout(sink),
                contextlib.redirect_stderr(sink),
            ):
                yield
        finally:
            api.utils_module.tqdm = original_tqdm


def _join_name(prefix: str, suffix: str) -> str:
    if not prefix:
        return suffix
    if not suffix:
        return prefix
    return f"{prefix}.{suffix}"


def _injection_paths(module: nn.Module, attr_names: Sequence[str]) -> tuple[str, ...]:
    if isinstance(module, (nn.Linear, nn.Embedding)):
        return ("",)
    paths = tuple(str(path) for path in attr_names)
    if not paths:
        raise ValueError("Grouped DFloat11 blocks require at least one weight path.")
    return paths


def _weight_module(block: nn.Module, path: str) -> nn.Module:
    return block if not path else block.get_submodule(path)


def _collect_weights(
    block_name: str,
    block: nn.Module,
    paths: tuple[str, ...],
) -> tuple[
    list[torch.Tensor],
    tuple[str, ...],
    tuple[tuple[int, ...], ...],
    tuple[int, ...],
]:
    flattened = []
    names = []
    shapes = []
    numels = []
    for path in paths:
        module = _weight_module(block, path)
        parameter = module._parameters.get("weight")
        if not isinstance(module, (nn.Linear, nn.Embedding)):
            raise TypeError(
                f"DFloat11 target {_join_name(block_name, path)!r} must be a "
                "Linear or Embedding module."
            )
        if not isinstance(parameter, nn.Parameter):
            raise TypeError(
                f"DFloat11 target {_join_name(block_name, path)!r} has no weight."
            )
        if parameter.dtype != torch.bfloat16:
            raise TypeError(
                f"DFloat11 target {_join_name(block_name, path)!r}.weight must be "
                f"bfloat16, not {parameter.dtype}."
            )
        if parameter.device.type != "cpu":
            raise ValueError(
                "In-memory DFloat11 timing requires a CPU-resident source model so "
                "host-device transfer is not hidden inside encoding."
            )
        name = _join_name(_join_name(block_name, path), "weight")
        flattened.append(parameter.detach().flatten())
        names.append(name)
        shapes.append(tuple(parameter.shape))
        numels.append(parameter.numel())
    return flattened, tuple(names), tuple(shapes), tuple(numels)


def encode_dfloat11_in_memory(
    model: nn.Module,
    pattern_dict: Mapping[str, Sequence[str]],
    *,
    quiet: bool = True,
) -> DFloat11InMemoryEncoding:
    """Encode matched BF16 weights without loading, mutating, or serializing a model.

    ``encode_seconds`` includes target discovery, CPU views/concatenation, Huffman
    construction, LUT construction, exponent/sign-mantissa extraction, Huffman
    stream encoding, and restart metadata construction.  It excludes model
    loading, disk I/O, CUDA correctness checks, runtime installation, and H2D
    transfer.
    """

    api = _load_official_api()
    blocks = []
    seen_blocks = set()
    seen_weights = set()
    with _quiet_official_output(api, quiet=quiet):
        # Progress-stream suppression and lock acquisition are harness overhead,
        # so begin the measured interval only after the quiet context is ready.
        total_started = time.perf_counter()
        for pattern, attr_names in pattern_dict.items():
            for block_name, block in model.named_modules():
                if not re.fullmatch(pattern, block_name):
                    continue
                if block_name in seen_blocks:
                    raise ValueError(
                        f"DFloat11 block {block_name!r} matched more than one pattern."
                    )
                seen_blocks.add(block_name)

                extraction_started = time.perf_counter()
                paths = _injection_paths(block, attr_names)
                weights, names, shapes, numels = _collect_weights(
                    block_name, block, paths
                )
                duplicates = seen_weights.intersection(names)
                if duplicates:
                    raise ValueError(
                        f"DFloat11 weights matched more than once: {sorted(duplicates)}"
                    )
                seen_weights.update(names)
                extraction_seconds = time.perf_counter() - extraction_started

                codebook_started = time.perf_counter()
                _codec, counter = api.get_codec(torch.cat(weights))
                codec, _, table = api.get_32bit_codec(counter)
                luts = api.get_luts(table)
                codebook_seconds = time.perf_counter() - codebook_started

                stream_started = time.perf_counter()
                (
                    encoded_exponent,
                    sign_mantissa,
                    output_positions,
                    gaps,
                    split_positions,
                ) = api.encode_weights(
                    weights,
                    codec,
                    api.bytes_per_thread,
                    api.threads_per_block[0],
                )
                stream_encode_seconds = time.perf_counter() - stream_started

                result = DFloat11BlockEncoding(
                    block_name=block_name,
                    injection_paths=paths,
                    weight_names=names,
                    weight_shapes=shapes,
                    weight_numels=numels,
                    luts=luts.contiguous(),
                    encoded_exponent=encoded_exponent.contiguous(),
                    sign_mantissa=sign_mantissa.contiguous(),
                    output_positions=output_positions.contiguous(),
                    gaps=gaps.contiguous(),
                    split_positions=split_positions.contiguous(),
                    extraction_seconds=extraction_seconds,
                    codebook_seconds=codebook_seconds,
                    stream_encode_seconds=stream_encode_seconds,
                )
                if result.sign_mantissa.numel() != result.numel:
                    raise RuntimeError(
                        f"DFloat11 block {block_name!r} encoded "
                        f"{result.sign_mantissa.numel()} elements; expected {result.numel}."
                    )
                blocks.append(result)

        encode_seconds = time.perf_counter() - total_started
    if not blocks:
        raise ValueError("No model modules matched the DFloat11 pattern dictionary.")
    return DFloat11InMemoryEncoding(
        blocks=tuple(blocks),
        encode_seconds=encode_seconds,
        dfloat11_version=api.version,
        bytes_per_thread=api.bytes_per_thread,
        threads_per_block=api.threads_per_block,
    )


def _validate_runtime_target(
    model: nn.Module,
    encoding: DFloat11InMemoryEncoding,
) -> list[tuple[DFloat11BlockEncoding, nn.Module, tuple[nn.Module, ...]]]:
    validated = []
    seen_blocks = set()
    seen_modules = set()
    for block_result in encoding.blocks:
        if block_result.block_name in seen_blocks:
            raise ValueError(
                f"Duplicate DFloat11 block {block_result.block_name!r} in encoding."
            )
        seen_blocks.add(block_result.block_name)
        block = model.get_submodule(block_result.block_name)
        targets = tuple(
            _weight_module(block, path) for path in block_result.injection_paths
        )
        if len(targets) != len(block_result.weight_shapes):
            raise RuntimeError(
                f"DFloat11 block {block_result.block_name!r} target count changed."
            )
        for target, expected_shape, expected_numel in zip(
            targets,
            block_result.weight_shapes,
            block_result.weight_numels,
            strict=True,
        ):
            if id(target) in seen_modules:
                raise ValueError(
                    "A DFloat11 runtime target appears in multiple blocks."
                )
            seen_modules.add(id(target))
            parameter = target._parameters.get("weight")
            if not isinstance(parameter, nn.Parameter):
                raise TypeError("A DFloat11 runtime target has no weight parameter.")
            if parameter.dtype != torch.bfloat16:
                raise TypeError("DFloat11 runtime target weights must remain bfloat16.")
            if (
                tuple(parameter.shape) != expected_shape
                or parameter.numel() != expected_numel
            ):
                raise RuntimeError(
                    "A DFloat11 runtime target changed shape after encoding."
                )
        for name in (
            "luts",
            "encoded_exponent",
            "sign_mantissa",
            "output_positions",
            "gaps",
            "split_positions",
            "weight_injection_modules",
        ):
            if hasattr(block, name):
                raise RuntimeError(
                    f"DFloat11 runtime target {block_result.block_name!r} already has {name!r}."
                )
        validated.append((block_result, block, targets))
    return validated


def install_dfloat11_runtime(
    model: nn.Module,
    encoding: DFloat11InMemoryEncoding,
) -> nn.Module:
    """Install official DFloat11 decode hooks and buffers without a disk round trip.

    Install on CPU before calling ``model.to(cuda_device)``.  Validation of every
    target happens before the first parameter is removed, so common structural
    mismatches leave the model unchanged.
    """

    api = _load_official_api()
    if (
        encoding.bytes_per_thread != api.bytes_per_thread
        or encoding.threads_per_block != api.threads_per_block
    ):
        raise RuntimeError(
            "The DFloat11 runtime configuration differs from the encoder configuration."
        )
    validated = _validate_runtime_target(model, encoding)
    hook = api.get_hook(api.threads_per_block, api.bytes_per_thread)

    for block_result, block, targets in validated:
        block.register_buffer("luts", block_result.luts)
        block.register_buffer("encoded_exponent", block_result.encoded_exponent)
        block.register_buffer("sign_mantissa", block_result.sign_mantissa)
        block.register_buffer(
            "output_positions", block_result.output_positions.view(torch.uint8)
        )
        block.register_buffer("gaps", block_result.gaps)
        # The official checkpoint loader restores this int64 tensor as a host list.
        block.split_positions = [
            int(value) for value in block_result.split_positions.tolist()
        ]
        positions = (
            block_result.output_positions.view(torch.uint32).cpu().to(torch.int64)
        )
        if positions.numel() < 2:
            raise RuntimeError("DFloat11 output positions must contain two boundaries.")
        maximum_output_span = int((positions[1:] - positions[:-1]).max().item())
        block.shared_mem_size = (
            api.threads_per_block[0] * 4 + 4 + maximum_output_span * 2
        )
        block.weight_injection_modules = list(targets)
        for target in targets:
            delattr(target, "weight")
        block.register_forward_pre_hook(hook)

    return model


def stable_cuda_allocated_bytes(device: torch.device | str) -> int:
    """Return active PyTorch CUDA bytes after callers release transient outputs.

    For comparable stable-memory measurements, run at least one warmup forward,
    delete its output, synchronize, and call ``gc.collect()`` before this helper.
    The value includes model buffers and reusable decode workspace but excludes
    CUDA context memory and unused allocator reservations.
    """

    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("Stable DFloat11 runtime memory requires a CUDA device.")
    torch.cuda.synchronize(device)
    return int(torch.cuda.memory_allocated(device))
