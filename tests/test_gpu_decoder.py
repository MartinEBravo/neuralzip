import inspect
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from neuralzip import gpu_decoder
from neuralzip.gpu_export import GPUCodebook, encode_bfloat16


class _FakeCudaTensor:
    _next_pointer = 0x1000

    def __init__(self, dtype, size, *, device="cuda:0", contiguous=True):
        self.dtype = dtype
        self._size = size
        self.device = torch.device(device)
        self.is_cuda = True
        self._contiguous = contiguous
        self._pointer = self._next_pointer
        type(self)._next_pointer += 0x1000

    def dim(self):
        return 1

    def is_contiguous(self):
        return self._contiguous

    def numel(self):
        return self._size

    def data_ptr(self):
        return self._pointer


def _valid_inputs(*, n_bytes=17, n_elements=20, max_bits=3, threads=2):
    grid = (n_bytes + threads * 8 - 1) // (threads * 8) if n_bytes else 0
    gap_bytes = (grid * threads * 4 + 7) // 8
    return (
        _FakeCudaTensor(torch.uint16, 1 << max_bits),
        _FakeCudaTensor(torch.uint8, n_bytes),
        _FakeCudaTensor(torch.uint8, n_elements),
        _FakeCudaTensor(torch.uint8, gap_bytes),
        _FakeCudaTensor(torch.int32, grid + 1),
        _FakeCudaTensor(torch.bfloat16, n_elements),
    )


class GPUDecoderTests(unittest.TestCase):
    def test_public_defaults_match_exported_512_thread_layout(self):
        function_default = (
            inspect.signature(gpu_decoder.decode_bf16)
            .parameters["threads_per_block"]
            .default
        )
        method_default = (
            inspect.signature(gpu_decoder.CuPyBF16Decoder.decode)
            .parameters["threads_per_block"]
            .default
        )

        self.assertEqual(function_default, 512)
        self.assertEqual(method_default, 512)

    def test_cpu_input_fails_before_optional_cupy_import(self):
        decoder = gpu_decoder.CuPyBF16Decoder()
        tensors = (
            torch.zeros(4, dtype=torch.uint16),
            torch.zeros(1, dtype=torch.uint8),
            torch.zeros(1, dtype=torch.uint8),
            torch.zeros(1, dtype=torch.uint8),
            torch.tensor([0, 1], dtype=torch.int32),
            torch.zeros(1, dtype=torch.bfloat16),
        )

        with (
            mock.patch.object(gpu_decoder, "_load_cupy") as load_cupy,
            self.assertRaisesRegex(ValueError, "lut must be a CUDA tensor"),
        ):
            decoder.decode(*tensors, max_bits=2)

        load_cupy.assert_not_called()

    def test_missing_cupy_has_actionable_error(self):
        with (
            mock.patch.object(
                gpu_decoder.importlib,
                "import_module",
                side_effect=ModuleNotFoundError("No module named 'cupy'"),
            ),
            self.assertRaisesRegex(ImportError, "cupy-cuda12x"),
        ):
            gpu_decoder._load_cupy()

    def test_launch_uses_current_torch_stream_and_caches_compilation(self):
        tensors = _valid_inputs()
        decoder = gpu_decoder.CuPyBF16Decoder()
        kernel = mock.Mock()
        module = mock.Mock()
        module.get_function.return_value = kernel
        cupy = mock.Mock()
        cupy.cuda.Device.side_effect = lambda _index: nullcontext()
        stream = object()
        cupy.cuda.Stream.from_external.return_value = stream
        cupy.cuda.runtime.getDeviceProperties.return_value = {
            "sharedMemPerBlock": 48 << 10,
            "sharedMemPerBlockOptin": 96 << 10,
        }
        cupy.RawModule.return_value = module
        torch_stream = SimpleNamespace(cuda_stream=0xCAFE)

        with (
            mock.patch.object(gpu_decoder.torch, "is_tensor", return_value=True),
            mock.patch.object(
                gpu_decoder, "_position_values", return_value=[0, 10, 20]
            ) as position_values,
            mock.patch.object(
                gpu_decoder,
                "_load_cupy",
                return_value=cupy,
            ) as load_cupy,
            mock.patch.object(
                gpu_decoder.torch.cuda,
                "current_stream",
                return_value=torch_stream,
            ) as current_stream,
        ):
            first = decoder.decode(*tensors, max_bits=3, threads_per_block=2)
            second = decoder.decode(*tensors, max_bits=3, threads_per_block=2)

        self.assertIs(first, tensors[-1])
        self.assertIs(second, tensors[-1])
        cupy.RawModule.assert_called_once()
        position_values.assert_called_once_with(tensors[4])
        self.assertEqual(
            cupy.RawModule.call_args.kwargs["name_expressions"],
            ("decode_bf16", "decode_bf16_batched"),
        )
        self.assertEqual(
            module.get_function.call_args_list,
            [mock.call("decode_bf16"), mock.call("decode_bf16_batched")],
        )
        self.assertEqual(kernel.call_count, 2)
        load_cupy.assert_called_once_with()
        cupy.cuda.Stream.from_external.assert_called_once()
        current_stream.assert_called_with(device=tensors[0].device)
        adapter = cupy.cuda.Stream.from_external.call_args.args[0]
        self.assertEqual(adapter.__cuda_stream__(), (0, 0xCAFE))

        positional, keyword = kernel.call_args
        self.assertEqual(positional[0], (2,))
        self.assertEqual(positional[1], (2,))
        kernel_args = positional[2]
        self.assertEqual(len(kernel_args), 8)
        for index in (0, 2, 3, 4, 5, 6):
            self.assertIsInstance(kernel_args[index], np.uint64)
        for index in (1, 7):
            self.assertIsInstance(kernel_args[index], np.int32)
        self.assertEqual(kernel_args[0], tensors[0].data_ptr())
        self.assertIsInstance(kernel_args[1], np.int32)
        self.assertEqual(kernel_args[1], 3)
        self.assertEqual(kernel_args[-1], 17)
        self.assertIs(
            kernel.call_args_list[0].args[2],
            kernel.call_args_list[1].args[2],
        )
        self.assertEqual(keyword["shared_mem"], 24)
        self.assertIs(keyword["stream"], stream)

    def test_prepared_batch_uses_one_kernel_launch_for_multiple_groups(self):
        first = _valid_inputs()
        second = _valid_inputs(n_bytes=1, n_elements=3)
        groups = (
            gpu_decoder.BF16DecodeGroup(*first, max_bits=3),
            gpu_decoder.BF16DecodeGroup(*second, max_bits=3),
        )
        decoder = gpu_decoder.CuPyBF16Decoder()

        def descriptor(values, *, dtype, device):
            return _FakeCudaTensor(dtype, len(values), device=str(device))

        with (
            mock.patch.object(gpu_decoder.torch, "is_tensor", return_value=True),
            mock.patch.object(
                gpu_decoder,
                "_position_values",
                side_effect=([0, 10, 20], [0, 3]),
            ),
            mock.patch.object(gpu_decoder.torch, "tensor", side_effect=descriptor),
        ):
            prepared = decoder.prepare_batch(groups, threads_per_block=2)

        self.assertEqual(prepared.grid, 3)
        self.assertEqual(prepared.block, 2)
        self.assertEqual(prepared.shared_memory, 24)
        self.assertEqual(prepared.block_group_ids.numel(), 3)
        self.assertEqual(prepared.block_local_ids.numel(), 3)

        single_kernel = mock.Mock()
        batch_kernel = mock.Mock()
        module = mock.Mock()
        module.get_function.side_effect = {
            "decode_bf16": single_kernel,
            "decode_bf16_batched": batch_kernel,
        }.__getitem__
        cupy = mock.Mock()
        cupy.cuda.Device.side_effect = lambda _index: nullcontext()
        stream = object()
        cupy.cuda.Stream.from_external.return_value = stream
        cupy.cuda.runtime.getDeviceProperties.return_value = {
            "sharedMemPerBlock": 48 << 10,
            "sharedMemPerBlockOptin": 96 << 10,
        }
        cupy.RawModule.return_value = module

        with (
            mock.patch.object(gpu_decoder, "_load_cupy", return_value=cupy),
            mock.patch.object(
                gpu_decoder.torch.cuda,
                "current_stream",
                return_value=SimpleNamespace(cuda_stream=0xBEEF),
            ),
        ):
            outputs = decoder.decode_batch(prepared)

        self.assertEqual(outputs, (first[-1], second[-1]))
        single_kernel.assert_not_called()
        batch_kernel.assert_called_once()
        positional, keyword = batch_kernel.call_args
        self.assertEqual(positional[0], (3,))
        self.assertEqual(positional[1], (2,))
        self.assertIs(positional[2], prepared.kernel_args)
        self.assertEqual(len(positional[2]), 10)
        self.assertTrue(all(isinstance(value, np.uint64) for value in positional[2]))
        self.assertEqual(keyword["shared_mem"], 24)
        self.assertIs(keyword["stream"], stream)

    def test_large_dynamic_shared_memory_opts_in_or_fails_clearly(self):
        tensors = _valid_inputs(n_elements=40000)
        positions = [0, 20000, 40000]

        def run_with_limit(optin_limit):
            decoder = gpu_decoder.CuPyBF16Decoder()
            kernel = mock.Mock()
            module = mock.Mock()
            module.get_function.return_value = kernel
            cupy = mock.Mock()
            cupy.cuda.Device.side_effect = lambda _index: nullcontext()
            cupy.cuda.Stream.from_external.return_value = object()
            cupy.cuda.runtime.getDeviceProperties.return_value = {
                "sharedMemPerBlock": 32 << 10,
                "sharedMemPerBlockOptin": optin_limit,
            }
            cupy.RawModule.return_value = module
            contexts = (
                mock.patch.object(gpu_decoder.torch, "is_tensor", return_value=True),
                mock.patch.object(
                    gpu_decoder, "_position_values", return_value=positions
                ),
                mock.patch.object(gpu_decoder, "_load_cupy", return_value=cupy),
                mock.patch.object(
                    gpu_decoder.torch.cuda,
                    "current_stream",
                    return_value=SimpleNamespace(cuda_stream=1),
                ),
            )
            return decoder, kernel, contexts

        decoder, kernel, contexts = run_with_limit(64 << 10)
        with contexts[0], contexts[1], contexts[2], contexts[3]:
            decoder.decode(*tensors, max_bits=3, threads_per_block=2)
        self.assertEqual(kernel.max_dynamic_shared_size_bytes, 40004)
        kernel.assert_called_once()

        decoder, kernel, contexts = run_with_limit(39000)
        with (
            contexts[0],
            contexts[1],
            contexts[2],
            contexts[3],
            self.assertRaisesRegex(ValueError, "39000-byte per-block limit"),
        ):
            decoder.decode(*tensors, max_bits=3, threads_per_block=2)
        kernel.assert_not_called()

    def test_empty_decode_is_a_noop_without_cupy(self):
        tensors = _valid_inputs(n_bytes=0, n_elements=0)
        decoder = gpu_decoder.CuPyBF16Decoder()

        with (
            mock.patch.object(gpu_decoder.torch, "is_tensor", return_value=True),
            mock.patch.object(gpu_decoder, "_position_values", return_value=[0]),
            mock.patch.object(gpu_decoder, "_load_cupy") as load_cupy,
        ):
            result = decoder.decode(*tensors, max_bits=3, threads_per_block=2)

        self.assertIs(result, tensors[-1])
        load_cupy.assert_not_called()

    def test_launch_metadata_is_strictly_validated(self):
        tensors = _valid_inputs()
        decoder = gpu_decoder.CuPyBF16Decoder()

        with self.assertRaisesRegex(ValueError, r"\[1, 12\]"):
            decoder.decode(*tensors, max_bits=13, threads_per_block=2)

        with self.assertRaisesRegex(ValueError, "power of two"):
            decoder.decode(*tensors, max_bits=3, threads_per_block=3)

        bad_lut = list(tensors)
        bad_lut[0] = _FakeCudaTensor(torch.uint16, 7)
        with (
            mock.patch.object(gpu_decoder.torch, "is_tensor", return_value=True),
            self.assertRaisesRegex(ValueError, "1 << max_bits"),
        ):
            decoder.decode(*bad_lut, max_bits=3, threads_per_block=2)

        bad_gaps = list(tensors)
        bad_gaps[3] = _FakeCudaTensor(torch.uint8, 1)
        with (
            mock.patch.object(gpu_decoder.torch, "is_tensor", return_value=True),
            self.assertRaisesRegex(ValueError, "packed_gaps"),
        ):
            decoder.decode(*bad_gaps, max_bits=3, threads_per_block=2)

        with (
            mock.patch.object(gpu_decoder.torch, "is_tensor", return_value=True),
            mock.patch.object(
                gpu_decoder, "_position_values", return_value=[0, 10, 19]
            ),
            self.assertRaisesRegex(ValueError, "output element count"),
        ):
            decoder.decode(*tensors, max_bits=3, threads_per_block=2)


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    "requires a BF16-capable CUDA GPU",
)
class GPUDecoderCUDAIntegrationTests(unittest.TestCase):
    def test_single_and_batched_kernels_are_bitwise_exact(self):
        try:
            __import__("cupy")
        except ImportError as error:  # pragma: no cover - CUDA CI only.
            self.skipTest(str(error))

        lengths = [0] * 256
        values = [0] * 256
        for symbol, length, value in (
            (7, 1, 0b0),
            (127, 2, 0b10),
            (240, 3, 0b110),
            (255, 3, 0b111),
        ):
            lengths[symbol] = length
            values[symbol] = value
        codebook = GPUCodebook.from_codes(lengths, values)
        device = torch.device("cuda:0")
        decoder = gpu_decoder.CuPyBF16Decoder()

        def make_group(numel, seed):
            symbols = (7, 127, 240, 255)
            raw = bytearray()
            for index in range(numel):
                exponent = symbols[(index * 5 + seed) % len(symbols)]
                other = (index * 73 + seed * 19) & 0xFF
                raw.extend(
                    (
                        (other & 0x7F) | ((exponent & 1) << 7),
                        (other & 0x80) | (exponent >> 1),
                    )
                )
            encoded = encode_bfloat16(raw, codebook)

            def upload(data):
                return (
                    torch.frombuffer(bytearray(data), dtype=torch.uint8)
                    .clone()
                    .to(device)
                )

            group = gpu_decoder.BF16DecodeGroup(
                lut=torch.tensor(
                    codebook.decode_lut,
                    dtype=torch.uint16,
                    device=device,
                ),
                encoded_exponents=upload(encoded.exponents.bitstream),
                sign_mantissa=upload(encoded.sign_mantissa),
                packed_gaps=upload(encoded.exponents.packed_gaps),
                output_positions=torch.tensor(
                    encoded.exponents.output_positions,
                    dtype=torch.int32,
                    device=device,
                ),
                output=torch.empty(numel, dtype=torch.bfloat16, device=device),
                max_bits=codebook.max_bits,
            )
            return group, bytes(raw)

        first, first_raw = make_group(4_097, 1)
        second, second_raw = make_group(20_003, 2)
        decoder.decode(
            first.lut,
            first.encoded_exponents,
            first.sign_mantissa,
            first.packed_gaps,
            first.output_positions,
            first.output,
            max_bits=first.max_bits,
        )
        prepared = decoder.prepare_batch((first, second))
        decoder.decode_batch(prepared)
        torch.cuda.synchronize(device)

        def raw_output(group):
            return group.output.view(torch.uint8).cpu().numpy().tobytes()

        self.assertEqual(raw_output(first), first_raw)
        self.assertEqual(raw_output(second), second_raw)

    def test_maximum_restart_gap_is_bitwise_exact(self):
        try:
            __import__("cupy")
        except ImportError as error:  # pragma: no cover - CUDA CI only.
            self.skipTest(str(error))

        lengths = [0] * 256
        values = [0] * 256
        for length in range(1, 12):
            lengths[length - 1] = length
            values[length - 1] = (1 << length) - 2
        lengths[11] = lengths[12] = 12
        values[11], values[12] = 0xFFE, 0xFFF
        codebook = GPUCodebook.from_codes(lengths, values)
        exponents = bytes((6, 7, 11, 11, 11, 11, 11, 0))
        other = bytes((0xA5,)) * len(exponents)
        raw = bytearray()
        for exponent, lane in zip(exponents, other, strict=True):
            raw.extend(
                (
                    (lane & 0x7F) | ((exponent & 1) << 7),
                    (lane & 0x80) | (exponent >> 1),
                )
            )
        encoded = encode_bfloat16(raw, codebook)
        self.assertEqual(encoded.exponents.gaps[1], 11)

        device = torch.device("cuda:0")

        def upload(data):
            return (
                torch.frombuffer(bytearray(data), dtype=torch.uint8).clone().to(device)
            )

        group = gpu_decoder.BF16DecodeGroup(
            lut=torch.tensor(codebook.decode_lut, dtype=torch.uint16, device=device),
            encoded_exponents=upload(encoded.exponents.bitstream),
            sign_mantissa=upload(encoded.sign_mantissa),
            packed_gaps=upload(encoded.exponents.packed_gaps),
            output_positions=torch.tensor(
                encoded.exponents.output_positions,
                dtype=torch.int32,
                device=device,
            ),
            output=torch.empty(len(exponents), dtype=torch.bfloat16, device=device),
            max_bits=codebook.max_bits,
        )
        decoder = gpu_decoder.CuPyBF16Decoder()
        decoder.decode(
            group.lut,
            group.encoded_exponents,
            group.sign_mantissa,
            group.packed_gaps,
            group.output_positions,
            group.output,
            max_bits=group.max_bits,
        )
        decoder.decode_batch(decoder.prepare_batch((group,)))
        torch.cuda.synchronize(device)

        self.assertEqual(group.output.view(torch.uint8).cpu().numpy().tobytes(), raw)


if __name__ == "__main__":
    unittest.main()
