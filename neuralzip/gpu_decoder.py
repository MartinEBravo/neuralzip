"""Optional CuPy launcher for NeuralZip's BF16 CUDA decoder.

CuPy is imported only when the first decode is launched. This keeps the core
CPU library usable without a CUDA toolchain or an additional binary wheel.
"""

from __future__ import annotations

import importlib
import threading
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch

from .gpu_export import GAP_BITS, MAX_CODE_BITS

_BYTES_PER_THREAD = 8
_DEFAULT_THREADS_PER_BLOCK = 512
_INT32_MAX = (1 << 31) - 1
_KERNEL_NAME = "decode_bf16"
_BATCH_KERNEL_NAME = "decode_bf16_batched"
_DEFAULT_KERNEL_PATH = Path(__file__).with_name("cuda") / "decode_bf16.cu"
_MAX_CACHED_LAUNCHES = 256
_MAX_CACHED_STREAMS = 64


@dataclass(frozen=True)
class _DecodeLaunch:
    device_index: int
    grid: int
    block: int
    shared_memory: int
    n_bytes: int
    n_elements: int
    kernel_grid: tuple[int, ...] = field(repr=False, compare=False)
    kernel_block: tuple[int, ...] = field(repr=False, compare=False)
    kernel_args: tuple[object, ...] = field(repr=False, compare=False)


@dataclass
class _CompiledKernel:
    module: object
    function: object
    batch_function: object
    default_shared_memory: int | None = None
    optin_shared_memory: int | None = None
    configured_shared_memory: dict[str, int] = field(default_factory=dict)
    configuration_lock: threading.Lock = field(
        default_factory=threading.Lock,
        repr=False,
    )


@dataclass(frozen=True)
class BF16DecodeGroup:
    """CUDA buffers for one independently coded BF16 exponent stream."""

    lut: torch.Tensor
    encoded_exponents: torch.Tensor
    sign_mantissa: torch.Tensor
    packed_gaps: torch.Tensor
    output_positions: torch.Tensor
    output: torch.Tensor
    max_bits: int


@dataclass(frozen=True)
class PreparedBF16Batch:
    """Immutable descriptors for one allocation-free batched CUDA launch."""

    groups: tuple[BF16DecodeGroup, ...]
    lut_ptrs: torch.Tensor
    code_ptrs: torch.Tensor
    sign_mantissa_ptrs: torch.Tensor
    gap_ptrs: torch.Tensor
    output_position_ptrs: torch.Tensor
    output_ptrs: torch.Tensor
    max_bits: torch.Tensor
    n_bytes: torch.Tensor
    block_group_ids: torch.Tensor
    block_local_ids: torch.Tensor
    device_index: int
    grid: int
    block: int
    shared_memory: int
    kernel_grid: tuple[int, ...] = field(repr=False, compare=False)
    kernel_block: tuple[int, ...] = field(repr=False, compare=False)
    kernel_args: tuple[object, ...] = field(repr=False, compare=False)
    _outputs: tuple[torch.Tensor, ...] = field(repr=False, compare=False)

    @property
    def outputs(self) -> tuple[torch.Tensor, ...]:
        return self._outputs


class _TorchStreamAdapter:
    """Expose a PyTorch CUDA stream through CUDA Stream Protocol v0."""

    def __init__(self, stream):
        self.stream = stream

    def __cuda_stream__(self):
        return 0, int(self.stream.cuda_stream)


def _cupy_external_stream(cupy, torch_stream):
    from_external = getattr(cupy.cuda.Stream, "from_external", None)
    if callable(from_external):
        return from_external(_TorchStreamAdapter(torch_stream))
    return cupy.cuda.ExternalStream(int(torch_stream.cuda_stream))


def _load_cupy():
    try:
        return importlib.import_module("cupy")
    except ImportError as error:
        raise ImportError(
            "NeuralZip GPU decoding requires an optional CuPy wheel that "
            "matches the installed CUDA runtime (for example, install "
            "`cupy-cuda12x` for CUDA 12)."
        ) from error


def _position_values(output_positions):
    return output_positions.detach().to(device="cpu", dtype=torch.int64).tolist()


def _tensor_launch_key(tensor, *, track_contents=False):
    if not torch.is_tensor(tensor):
        return None
    try:
        key = (
            id(tensor),
            int(tensor.data_ptr()),
            int(tensor.numel()),
            tensor.dtype,
            tensor.device,
            int(tensor.dim()),
            bool(tensor.is_contiguous()),
        )
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None
    if track_contents:
        try:
            version = int(tensor._version)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            # PyTorch inference tensors intentionally omit version counters.
            # Their metadata buffers are immutable by the runtime contract.
            version = None
        key += (version,)
    return key


def _decode_launch_key(
    lut,
    encoded_exponents,
    sign_mantissa,
    packed_gaps,
    output_positions,
    output,
    *,
    max_bits,
    threads_per_block,
):
    if not isinstance(max_bits, int) or not isinstance(threads_per_block, int):
        return None
    keys = (
        _tensor_launch_key(lut),
        _tensor_launch_key(encoded_exponents),
        _tensor_launch_key(sign_mantissa),
        _tensor_launch_key(packed_gaps),
        _tensor_launch_key(output_positions, track_contents=True),
        _tensor_launch_key(output),
    )
    if any(key is None for key in keys):
        return None
    return (max_bits, threads_per_block, *keys)


def _device_property(properties, name):
    if not isinstance(properties, Mapping):
        return None
    for key in (name, name.encode("ascii")):
        if key in properties:
            return int(properties[key])
    return None


def _require_tensor(name, tensor, *, dtypes):
    if not torch.is_tensor(tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor.")
    if tensor.dtype not in dtypes:
        expected = ", ".join(str(dtype) for dtype in dtypes)
        raise TypeError(f"{name} must have dtype {expected}; got {tensor.dtype}.")
    if tensor.dim() != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous.")


def _validate_decode_inputs(
    lut,
    encoded_exponents,
    sign_mantissa,
    packed_gaps,
    output_positions,
    output,
    *,
    max_bits,
    threads_per_block,
):
    if isinstance(max_bits, bool) or not isinstance(max_bits, int):
        raise TypeError("max_bits must be an integer.")
    if not 1 <= max_bits <= MAX_CODE_BITS:
        raise ValueError(f"max_bits must be in [1, {MAX_CODE_BITS}].")
    if isinstance(threads_per_block, bool) or not isinstance(threads_per_block, int):
        raise TypeError("threads_per_block must be an integer.")
    if not 1 <= threads_per_block <= 1024 or threads_per_block & (
        threads_per_block - 1
    ):
        raise ValueError("threads_per_block must be a power of two between 1 and 1024.")

    uint32 = getattr(torch, "uint32", None)
    position_dtypes = (torch.int32,)
    if uint32 is not None:
        position_dtypes += (uint32,)

    _require_tensor("lut", lut, dtypes=(torch.uint16,))
    _require_tensor("encoded_exponents", encoded_exponents, dtypes=(torch.uint8,))
    _require_tensor("sign_mantissa", sign_mantissa, dtypes=(torch.uint8,))
    _require_tensor("packed_gaps", packed_gaps, dtypes=(torch.uint8,))
    _require_tensor("output_positions", output_positions, dtypes=position_dtypes)
    _require_tensor("output", output, dtypes=(torch.bfloat16,))

    tensors = (
        ("encoded_exponents", encoded_exponents),
        ("sign_mantissa", sign_mantissa),
        ("packed_gaps", packed_gaps),
        ("output_positions", output_positions),
        ("output", output),
    )
    device = lut.device
    for name, tensor in tensors:
        if tensor.device != device:
            raise ValueError(f"{name} must be on the same CUDA device as lut.")
    if device.index is None:
        raise ValueError("CUDA tensors must have an explicit device index.")

    expected_lut_entries = 1 << max_bits
    if lut.numel() != expected_lut_entries:
        raise ValueError(
            f"lut must contain exactly 1 << max_bits ({expected_lut_entries}) entries."
        )

    n_bytes = encoded_exponents.numel()
    n_elements = output.numel()
    if n_bytes > _INT32_MAX:
        raise ValueError("encoded_exponents is too large for the CUDA kernel ABI.")
    if n_elements > _INT32_MAX:
        raise ValueError("output is too large for the CUDA kernel ABI.")
    if sign_mantissa.numel() != n_elements:
        raise ValueError(
            "sign_mantissa and output must contain the same number of elements."
        )
    if (n_bytes == 0) != (n_elements == 0):
        raise ValueError(
            "encoded and output must either both be empty or both be non-empty."
        )

    grid = (
        (n_bytes + threads_per_block * _BYTES_PER_THREAD - 1)
        // (threads_per_block * _BYTES_PER_THREAD)
        if n_bytes
        else 0
    )
    expected_gap_bytes = (grid * threads_per_block * GAP_BITS + 7) // 8
    if packed_gaps.numel() != expected_gap_bytes:
        raise ValueError(
            f"packed_gaps must contain exactly {expected_gap_bytes} bytes "
            "for this launch."
        )
    if output_positions.numel() != grid + 1:
        raise ValueError(f"output_positions must contain exactly {grid + 1} entries.")

    positions = _position_values(output_positions)
    if not positions or positions[0] != 0:
        raise ValueError("output_positions must start at zero.")
    if positions[-1] != n_elements:
        raise ValueError("output_positions must end at the output element count.")
    if any(
        current < 0 or current > n_elements or current > following
        for current, following in pairwise(positions)
    ):
        raise ValueError(
            "output_positions must be non-negative, bounded, and non-decreasing."
        )

    max_block_elements = (
        max((following - current) for current, following in pairwise(positions))
        if grid
        else 0
    )
    warp_count = (threads_per_block + 31) // 32
    shared_memory = warp_count * 4 + max_block_elements * 2
    return _DecodeLaunch(
        device_index=int(device.index),
        grid=grid,
        block=threads_per_block,
        shared_memory=shared_memory,
        n_bytes=n_bytes,
        n_elements=n_elements,
        kernel_grid=(grid,),
        kernel_block=(threads_per_block,),
        kernel_args=(
            np.uint64(lut.data_ptr()),
            np.int32(max_bits),
            np.uint64(encoded_exponents.data_ptr()),
            np.uint64(sign_mantissa.data_ptr()),
            np.uint64(packed_gaps.data_ptr()),
            np.uint64(output_positions.data_ptr()),
            np.uint64(output.data_ptr()),
            np.int32(n_bytes),
        ),
    )


class CuPyBF16Decoder:
    """Lazily compile and launch NeuralZip's CUDA BF16 decoder."""

    def __init__(self, kernel_path=None):
        self.kernel_path = Path(kernel_path or _DEFAULT_KERNEL_PATH)
        self._kernels = {}
        self._launches = OrderedDict()
        self._external_streams = OrderedDict()
        self._compile_lock = threading.Lock()
        self._cupy_lock = threading.Lock()
        self._launch_lock = threading.Lock()
        self._stream_lock = threading.Lock()
        self._cupy = None

    def _cupy_for_launch(self):
        cupy = self._cupy
        if cupy is not None:
            return cupy
        with self._cupy_lock:
            cupy = self._cupy
            if cupy is None:
                cupy = _load_cupy()
                self._cupy = cupy
            return cupy

    def _kernel_for_device(self, cupy, device_index):
        compiled = self._kernels.get(device_index)
        if compiled is not None:
            return compiled
        with self._compile_lock:
            compiled = self._kernels.get(device_index)
            if compiled is not None:
                return compiled
            try:
                source = self.kernel_path.read_text(encoding="utf-8")
            except OSError as error:
                raise RuntimeError(
                    f"Cannot read NeuralZip CUDA kernel: {self.kernel_path}"
                ) from error
            try:
                with cupy.cuda.Device(device_index):
                    module = cupy.RawModule(
                        code=source,
                        options=("--std=c++14",),
                        name_expressions=(_KERNEL_NAME, _BATCH_KERNEL_NAME),
                    )
                    kernel = module.get_function(_KERNEL_NAME)
                    batch_kernel = module.get_function(_BATCH_KERNEL_NAME)
            except Exception as error:
                raise RuntimeError(
                    "Failed to compile NeuralZip's BF16 CUDA decoder with "
                    "CuPy/NVRTC. Check that CuPy, the NVIDIA driver, and the "
                    "CUDA runtime are compatible."
                ) from error
            # Retain the module alongside the function for the lifetime of its
            # CUDA context.
            compiled = _CompiledKernel(
                module=module,
                function=kernel,
                batch_function=batch_kernel,
            )
            self._kernels[device_index] = compiled
            return compiled

    def _external_stream_for(self, cupy, torch_stream, device_index):
        key = (device_index, int(torch_stream.cuda_stream))
        with self._stream_lock:
            stream = self._external_streams.get(key)
            if stream is not None:
                self._external_streams.move_to_end(key)
                return stream
            stream = _cupy_external_stream(cupy, torch_stream)
            self._external_streams[key] = stream
            self._external_streams.move_to_end(key)
            while len(self._external_streams) > _MAX_CACHED_STREAMS:
                self._external_streams.popitem(last=False)
            return stream

    def _launch_for_inputs(
        self,
        lut,
        encoded_exponents,
        sign_mantissa,
        packed_gaps,
        output_positions,
        output,
        *,
        max_bits,
        threads_per_block,
    ):
        key = _decode_launch_key(
            lut,
            encoded_exponents,
            sign_mantissa,
            packed_gaps,
            output_positions,
            output,
            max_bits=max_bits,
            threads_per_block=threads_per_block,
        )
        if key is not None:
            with self._launch_lock:
                launch = self._launches.get(key)
                if launch is not None:
                    self._launches.move_to_end(key)
                    return launch

        launch = _validate_decode_inputs(
            lut,
            encoded_exponents,
            sign_mantissa,
            packed_gaps,
            output_positions,
            output,
            max_bits=max_bits,
            threads_per_block=threads_per_block,
        )
        if key is not None:
            with self._launch_lock:
                self._launches[key] = launch
                self._launches.move_to_end(key)
                while len(self._launches) > _MAX_CACHED_LAUNCHES:
                    self._launches.popitem(last=False)
        return launch

    def _ensure_shared_memory(
        self,
        cupy,
        compiled,
        device_index,
        required_bytes,
        *,
        kernel_name,
    ):
        default_limit = compiled.default_shared_memory
        configured = compiled.configured_shared_memory.get(kernel_name, 0)
        if default_limit is not None and (
            required_bytes <= default_limit or required_bytes <= configured
        ):
            return

        with compiled.configuration_lock:
            default_limit = compiled.default_shared_memory
            configured = compiled.configured_shared_memory.get(kernel_name, 0)
            if default_limit is not None and (
                required_bytes <= default_limit or required_bytes <= configured
            ):
                return
            if default_limit is None:
                try:
                    properties = cupy.cuda.runtime.getDeviceProperties(device_index)
                except Exception as error:
                    raise RuntimeError(
                        "Cannot query CUDA shared-memory limits for the "
                        "NeuralZip decoder."
                    ) from error
                default_limit = _device_property(properties, "sharedMemPerBlock")
                optin_limit = _device_property(properties, "sharedMemPerBlockOptin")
                if default_limit is None:
                    raise RuntimeError(
                        "CuPy did not report CUDA's sharedMemPerBlock limit."
                    )
                compiled.default_shared_memory = default_limit
                compiled.optin_shared_memory = optin_limit or default_limit

            if required_bytes <= default_limit:
                return
            if required_bytes > compiled.optin_shared_memory:
                raise ValueError(
                    "The NeuralZip decode launch requires "
                    f"{required_bytes} bytes of dynamic shared memory, but CUDA "
                    f"reports a {compiled.optin_shared_memory}-byte per-block "
                    "limit. Reduce threads_per_block or split the encoded group."
                )

            function = (
                compiled.function
                if kernel_name == _KERNEL_NAME
                else compiled.batch_function
            )
            try:
                function.max_dynamic_shared_size_bytes = required_bytes
            except (AttributeError, RuntimeError, TypeError) as error:
                raise RuntimeError(
                    "This CuPy version cannot opt the NeuralZip kernel into the "
                    f"required {required_bytes} bytes of dynamic shared memory. "
                    "Upgrade CuPy or split the encoded group."
                ) from error
            compiled.configured_shared_memory[kernel_name] = required_bytes

    def prepare_batch(
        self,
        groups: Iterable[BF16DecodeGroup],
        *,
        threads_per_block=_DEFAULT_THREADS_PER_BLOCK,
    ) -> PreparedBF16Batch:
        """Validate and upload immutable descriptors for one batched launch.

        Preparation may synchronize while validating output-position metadata;
        it is intended to run once during setup, never in the forward hook.
        """

        groups = tuple(groups)
        if not groups:
            raise ValueError("A BF16 decode batch must contain at least one group.")
        if len(groups) > _INT32_MAX:
            raise ValueError("The BF16 decode batch exceeds the CUDA kernel ABI.")

        launches = []
        for index, group in enumerate(groups):
            if not isinstance(group, BF16DecodeGroup):
                raise TypeError(
                    "groups must contain BF16DecodeGroup values; "
                    f"entry {index} has type {type(group).__name__}."
                )
            launch = _validate_decode_inputs(
                group.lut,
                group.encoded_exponents,
                group.sign_mantissa,
                group.packed_gaps,
                group.output_positions,
                group.output,
                max_bits=group.max_bits,
                threads_per_block=threads_per_block,
            )
            if launch.grid == 0:
                raise ValueError("Batched BF16 decode groups must be non-empty.")
            launches.append(launch)

        device_index = launches[0].device_index
        if any(launch.device_index != device_index for launch in launches):
            raise ValueError("Every group in a BF16 decode batch must share a device.")
        grid = sum(launch.grid for launch in launches)
        if grid > _INT32_MAX:
            raise ValueError("The BF16 decode batch has too many CUDA blocks.")

        device = groups[0].lut.device

        def pointers(attribute):
            return torch.tensor(
                [getattr(group, attribute).data_ptr() for group in groups],
                dtype=torch.uint64,
                device=device,
            )

        def integers(values):
            return torch.tensor(tuple(values), dtype=torch.int32, device=device)

        block_group_ids = []
        block_local_ids = []
        for group_id, launch in enumerate(launches):
            block_group_ids.extend([group_id] * launch.grid)
            block_local_ids.extend(range(launch.grid))

        lut_ptrs = pointers("lut")
        code_ptrs = pointers("encoded_exponents")
        sign_mantissa_ptrs = pointers("sign_mantissa")
        gap_ptrs = pointers("packed_gaps")
        output_position_ptrs = pointers("output_positions")
        output_ptrs = pointers("output")
        max_bits = integers(group.max_bits for group in groups)
        n_bytes = integers(launch.n_bytes for launch in launches)
        block_group_ids = integers(block_group_ids)
        block_local_ids = integers(block_local_ids)
        kernel_args = tuple(
            np.uint64(tensor.data_ptr())
            for tensor in (
                lut_ptrs,
                code_ptrs,
                sign_mantissa_ptrs,
                gap_ptrs,
                output_position_ptrs,
                output_ptrs,
                max_bits,
                n_bytes,
                block_group_ids,
                block_local_ids,
            )
        )

        return PreparedBF16Batch(
            groups=groups,
            lut_ptrs=lut_ptrs,
            code_ptrs=code_ptrs,
            sign_mantissa_ptrs=sign_mantissa_ptrs,
            gap_ptrs=gap_ptrs,
            output_position_ptrs=output_position_ptrs,
            output_ptrs=output_ptrs,
            max_bits=max_bits,
            n_bytes=n_bytes,
            block_group_ids=block_group_ids,
            block_local_ids=block_local_ids,
            device_index=device_index,
            grid=grid,
            block=threads_per_block,
            shared_memory=max(launch.shared_memory for launch in launches),
            kernel_grid=(grid,),
            kernel_block=(threads_per_block,),
            kernel_args=kernel_args,
            _outputs=tuple(group.output for group in groups),
        )

    def decode_batch(
        self,
        prepared: PreparedBF16Batch,
    ) -> tuple[torch.Tensor, ...]:
        """Launch all prepared streams with one kernel on PyTorch's stream."""

        if not isinstance(prepared, PreparedBF16Batch):
            raise TypeError("prepared must be a PreparedBF16Batch.")
        cupy = self._cupy_for_launch()
        compiled = self._kernel_for_device(cupy, prepared.device_index)

        device = prepared.groups[0].lut.device
        torch_stream = torch.cuda.current_stream(device=device)
        with cupy.cuda.Device(prepared.device_index):
            self._ensure_shared_memory(
                cupy,
                compiled,
                prepared.device_index,
                prepared.shared_memory,
                kernel_name=_BATCH_KERNEL_NAME,
            )
            stream = self._external_stream_for(
                cupy,
                torch_stream,
                prepared.device_index,
            )
            compiled.batch_function(
                prepared.kernel_grid,
                prepared.kernel_block,
                prepared.kernel_args,
                shared_mem=prepared.shared_memory,
                stream=stream,
            )
        return prepared.outputs

    def preflight(self, prepared_batches: Iterable[PreparedBF16Batch]) -> None:
        """Compile and validate real batches before model mutation."""

        prepared_batches = tuple(prepared_batches)
        if not prepared_batches:
            raise ValueError("preflight requires at least one prepared batch.")
        for prepared in prepared_batches:
            self.decode_batch(prepared)
        device = prepared_batches[0].groups[0].lut.device
        torch.cuda.synchronize(device=device)

    def decode(
        self,
        lut,
        encoded_exponents,
        sign_mantissa,
        packed_gaps,
        output_positions,
        output,
        *,
        max_bits,
        threads_per_block=_DEFAULT_THREADS_PER_BLOCK,
    ):
        """Decode one continuous exponent stream into an existing BF16 tensor.

        The launch is asynchronous on PyTorch's current CUDA stream. All input
        tensors must remain alive until work on that stream completes.
        """
        launch = self._launch_for_inputs(
            lut,
            encoded_exponents,
            sign_mantissa,
            packed_gaps,
            output_positions,
            output,
            max_bits=max_bits,
            threads_per_block=threads_per_block,
        )
        if launch.n_elements == 0:
            return output

        cupy = self._cupy_for_launch()
        compiled = self._kernel_for_device(cupy, launch.device_index)

        torch_stream = torch.cuda.current_stream(device=lut.device)
        with cupy.cuda.Device(launch.device_index):
            self._ensure_shared_memory(
                cupy,
                compiled,
                launch.device_index,
                launch.shared_memory,
                kernel_name=_KERNEL_NAME,
            )
            stream = self._external_stream_for(
                cupy,
                torch_stream,
                launch.device_index,
            )
            compiled.function(
                launch.kernel_grid,
                launch.kernel_block,
                launch.kernel_args,
                shared_mem=launch.shared_memory,
                stream=stream,
            )
        return output


_DEFAULT_DECODER = CuPyBF16Decoder()


def default_decoder() -> CuPyBF16Decoder:
    """Return the process-wide lazy decoder used by NeuralZip runtimes."""

    return _DEFAULT_DECODER


def decode_bf16(
    lut,
    encoded_exponents,
    sign_mantissa,
    packed_gaps,
    output_positions,
    output,
    *,
    max_bits,
    threads_per_block=_DEFAULT_THREADS_PER_BLOCK,
):
    """Decode using the 512-thread layout emitted by the GPU exporter."""
    return _DEFAULT_DECODER.decode(
        lut,
        encoded_exponents,
        sign_mantissa,
        packed_gaps,
        output_positions,
        output,
        max_bits=max_bits,
        threads_per_block=threads_per_block,
    )


def prepare_bf16_batch(
    groups: Iterable[BF16DecodeGroup],
    *,
    threads_per_block=_DEFAULT_THREADS_PER_BLOCK,
) -> PreparedBF16Batch:
    """Prepare immutable descriptors for one batched decode launch."""

    return _DEFAULT_DECODER.prepare_batch(
        groups,
        threads_per_block=threads_per_block,
    )


def decode_bf16_batch(
    prepared: PreparedBF16Batch,
) -> tuple[torch.Tensor, ...]:
    """Decode a previously prepared batch in one CUDA launch."""

    return _DEFAULT_DECODER.decode_batch(prepared)
