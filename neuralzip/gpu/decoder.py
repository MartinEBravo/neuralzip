"""CuPy launcher for NeuralZip's batched BF16 CUDA decoder."""

import importlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .export import BYTES_PER_THREAD, THREADS_PER_BLOCK

KERNEL_NAME = "decode_bf16_batched"
KERNEL_PATH = Path(__file__).with_name("cuda") / "decode_bf16.cu"
DESCRIPTOR_NAMES = (
    "lut",
    "encoded_exponents",
    "sign_mantissa",
    "packed_gaps",
    "output_positions",
    "output",
)


@dataclass(frozen=True)
class BF16DecodeGroup:
    lut: torch.Tensor
    encoded_exponents: torch.Tensor
    sign_mantissa: torch.Tensor
    packed_gaps: torch.Tensor
    output_positions: torch.Tensor
    output: torch.Tensor
    max_bits: int
    max_block_elements: int


@dataclass(frozen=True)
class PreparedBF16Batch:
    descriptor_tensors: tuple[torch.Tensor, ...] = field(repr=False)
    device: torch.device
    shared_memory: int
    kernel_grid: tuple[int, ...]
    kernel_args: tuple[object, ...] = field(repr=False)
    outputs: tuple[torch.Tensor, ...] = field(repr=False)


@dataclass
class _Kernel:
    module: object
    function: object
    default_shared_memory: int | None = None
    optin_shared_memory: int | None = None
    configured_shared_memory: int = 0


def _load_cupy():
    try:
        return importlib.import_module("cupy")
    except ImportError as error:
        raise ImportError(
            "NeuralZip GPU inference requires a CuPy wheel matching CUDA."
        ) from error


def _device_property(properties, name):
    return int(properties.get(name, properties.get(name.encode("ascii"), 0)))


class CuPyBF16Decoder:
    def __init__(self, kernel_path=KERNEL_PATH):
        self.kernel_path = Path(kernel_path)
        self._cupy = None
        self._kernels = {}

    @property
    def cupy(self):
        if self._cupy is None:
            self._cupy = _load_cupy()
        return self._cupy

    def _kernel(self, device_index):
        if device_index not in self._kernels:
            cupy = self.cupy
            with cupy.cuda.Device(device_index):
                module = cupy.RawModule(
                    code=self.kernel_path.read_text(encoding="utf-8"),
                    options=("--std=c++14",),
                    name_expressions=(KERNEL_NAME,),
                )
                self._kernels[device_index] = _Kernel(
                    module,
                    module.get_function(KERNEL_NAME),
                )
        return self._kernels[device_index]

    def _ensure_shared_memory(self, kernel, device_index, required):
        if required <= kernel.configured_shared_memory:
            return
        if kernel.default_shared_memory is None:
            properties = self.cupy.cuda.runtime.getDeviceProperties(device_index)
            kernel.default_shared_memory = _device_property(
                properties, "sharedMemPerBlock"
            )
            kernel.optin_shared_memory = (
                _device_property(properties, "sharedMemPerBlockOptin")
                or kernel.default_shared_memory
            )
        if required <= kernel.default_shared_memory:
            return
        if required > kernel.optin_shared_memory:
            raise ValueError("The BF16 decode group exceeds CUDA shared memory.")
        kernel.function.max_dynamic_shared_size_bytes = required
        kernel.configured_shared_memory = required

    def prepare_batch(self, groups):
        groups = tuple(groups)
        if not groups:
            raise ValueError("A decode batch must contain at least one group.")
        device = groups[0].lut.device
        if device.type != "cuda" or any(group.lut.device != device for group in groups):
            raise ValueError("A decode batch must use one CUDA device.")

        grids = [
            (group.encoded_exponents.numel() + THREADS_PER_BLOCK * BYTES_PER_THREAD - 1)
            // (THREADS_PER_BLOCK * BYTES_PER_THREAD)
            for group in groups
        ]
        if not all(grids):
            raise ValueError("GPU exponent streams must be non-empty.")

        def pointers(name):
            return torch.tensor(
                [getattr(group, name).data_ptr() for group in groups],
                dtype=torch.uint64,
                device=device,
            )

        def integers(values):
            return torch.tensor(tuple(values), dtype=torch.int32, device=device)

        group_ids = []
        local_ids = []
        for group_id, grid in enumerate(grids):
            group_ids.extend([group_id] * grid)
            local_ids.extend(range(grid))

        descriptors = tuple(pointers(name) for name in DESCRIPTOR_NAMES)
        descriptors += (
            integers(group.max_bits for group in groups),
            integers(group.encoded_exponents.numel() for group in groups),
            integers(group_ids),
            integers(local_ids),
        )
        shared_memory = 4 * ((THREADS_PER_BLOCK + 31) // 32) + 2 * max(
            group.max_block_elements for group in groups
        )
        return PreparedBF16Batch(
            descriptor_tensors=descriptors,
            device=device,
            shared_memory=shared_memory,
            kernel_grid=(sum(grids),),
            kernel_args=tuple(np.uint64(tensor.data_ptr()) for tensor in descriptors),
            outputs=tuple(group.output for group in groups),
        )

    def decode_batch(self, prepared):
        cupy = self.cupy
        device_index = prepared.device.index
        kernel = self._kernel(device_index)
        self._ensure_shared_memory(kernel, device_index, prepared.shared_memory)
        torch_stream = torch.cuda.current_stream(device=prepared.device)
        with cupy.cuda.Device(device_index):
            kernel.function(
                prepared.kernel_grid,
                (THREADS_PER_BLOCK,),
                prepared.kernel_args,
                shared_mem=prepared.shared_memory,
                stream=cupy.cuda.ExternalStream(int(torch_stream.cuda_stream)),
            )
        return prepared.outputs


_DECODER = CuPyBF16Decoder()


def default_decoder():
    return _DECODER
