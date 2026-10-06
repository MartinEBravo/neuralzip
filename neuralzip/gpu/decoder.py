from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch

try:
    import cupy
except ModuleNotFoundError as error:
    if error.name != "cupy":
        raise
    cupy = None

KERNEL_PATH = Path(__file__).with_name("cuda") / "decode.cu"
BYTES_PER_THREAD = 8
THREADS_PER_BLOCK = 512
DESCRIPTOR_NAMES = (
    "lut",
    "encoded_symbols",
    "raw_streams",
    "packed_gaps",
    "output_positions",
    "output",
)


@dataclass(frozen=True)
class PreparedBatch:
    descriptor_tensors: tuple[torch.Tensor, ...] = field(repr=False)
    device: torch.device
    layout: tuple[int, bool, int, int, int, bool, bool]
    stage_symbols: bool
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


def _device_property(properties, name):
    return int(properties.get(name, properties.get(name.encode("ascii"), 0)))


def shared_memory_bytes(element_size, max_elements):
    return 4 * ((THREADS_PER_BLOCK + 31) // 32) + element_size * max_elements


class CuPyDecoder:
    def __init__(self, kernel_path=KERNEL_PATH):
        self.kernel_path = Path(kernel_path)
        self._kernels = {}

    @property
    def cupy(self):
        if cupy is None:
            raise ImportError(
                "NeuralZip GPU inference requires a CuPy wheel matching CUDA."
            )
        return cupy

    def _kernel(self, device_index, layout, stage_symbols):
        key = device_index, layout, stage_symbols
        if key not in self._kernels:
            width, reorder, mantissa_bits, exponent_bits, stream_id, mixed, long_codes = layout
            word = {2: "uint16_t", 4: "uint32_t"}[width]
            name = f"decode_batched<{word}, {int(reorder)}, {mantissa_bits}, {exponent_bits}, {stream_id}, {int(mixed)}, {int(long_codes)}, {int(stage_symbols)}>"
            cupy = self.cupy
            with cupy.cuda.Device(device_index):
                module = cupy.RawModule(
                    code=self.kernel_path.read_text(encoding="utf-8"),
                    options=("--std=c++14",),
                    name_expressions=(name,),
                )
                self._kernels[key] = _Kernel(
                    module,
                    module.get_function(name),
                )
        return self._kernels[key]

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
            raise ValueError("The decode group exceeds CUDA shared memory.")
        kernel.function.max_dynamic_shared_size_bytes = required
        kernel.configured_shared_memory = required

    def prepare_kernel(self, device_index, layout, shared_memory, stage_symbols=False):
        kernel = self._kernel(device_index, layout, stage_symbols)
        self._ensure_shared_memory(kernel, device_index, shared_memory)
        return kernel

    def _staging(self, device_index, max_elements):
        required = shared_memory_bytes(1, max_elements)
        properties = self.cupy.cuda.runtime.getDeviceProperties(device_index)
        limit = (_device_property(properties, "sharedMemPerBlockOptin")
                 or _device_property(properties, "sharedMemPerBlock"))
        staged = max_elements > 0 and required <= limit
        return staged, required if staged else shared_memory_bytes(1, 0)

    def prepare_group(self, device_index, group):
        layout = group.layout
        positions = group.encoded.streams[layout[4]].output_positions
        max_elements = 0 if not group.codebook.max_bits else max(
            (right - left for left, right in pairwise(positions)), default=0
        )
        staged, shared_memory = self._staging(device_index, max_elements)
        return self.prepare_kernel(device_index, layout, shared_memory, staged)

    def prepare_batch(self, groups):
        groups = tuple(groups)
        if not groups:
            raise ValueError("A decode batch must contain at least one group.")
        device = groups[0].lut.device
        if device.type != "cuda" or any(group.lut.device != device for group in groups):
            raise ValueError("A decode batch must use one CUDA device.")
        layout = groups[0].layout
        if any(group.layout != layout for group in groups):
            raise ValueError("A decode batch must use one layout.")

        grids = [
            (group.encoded_symbols.numel() + THREADS_PER_BLOCK * BYTES_PER_THREAD - 1)
            // (THREADS_PER_BLOCK * BYTES_PER_THREAD)
            for group in groups
        ]
        if not all(grids):
            raise ValueError("GPU symbol streams must be non-empty.")

        def pointers(name):
            values = (
                [
                    stream.data_ptr() if stream is not None else 0
                    for group in groups
                    for stream in group.raw_streams
                ]
                if name == "raw_streams"
                else [getattr(group, name).data_ptr() for group in groups]
            )
            return torch.tensor(
                values,
                dtype=torch.uint64,
                device=device,
            )

        def integers(values):
            return torch.tensor(tuple(values), dtype=torch.int32, device=device)

        group_ids = [
            group_id for group_id, grid in enumerate(grids) for _ in range(grid)
        ]
        local_ids = [local_id for grid in grids for local_id in range(grid)]

        descriptors = tuple(pointers(name) for name in DESCRIPTOR_NAMES)
        descriptors += (
            integers(group.max_bits for group in groups),
            integers(group.encoded_symbols.numel() for group in groups),
            integers(group_ids),
            integers(local_ids),
        )
        staged, shared_memory = self._staging(
            device.index, max(group.max_block_elements for group in groups))
        return PreparedBatch(
            descriptor_tensors=descriptors,
            device=device,
            layout=layout,
            stage_symbols=staged,
            shared_memory=shared_memory,
            kernel_grid=(sum(grids),),
            kernel_args=tuple(np.uint64(tensor.data_ptr()) for tensor in descriptors),
            outputs=tuple(group.output for group in groups),
        )

    def decode_batch(self, prepared):
        cupy = self.cupy
        device_index = prepared.device.index
        kernel = self.prepare_kernel(
            device_index, prepared.layout, prepared.shared_memory, prepared.stage_symbols
        )
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


_DECODER = CuPyDecoder()


def default_decoder():
    return _DECODER
