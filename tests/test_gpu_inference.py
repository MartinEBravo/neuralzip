import gc
import threading
import unittest
import weakref
from unittest import mock

import torch
from torch import nn

from neuralzip import huff0
from neuralzip.compressor import Compressor
from neuralzip.gpu_inference import (
    NeuralZipModel,
    NeuralZipRuntime,
    export_model_for_gpu,
)
from neuralzip.neuralzip_codec import NeuralZipCodec


class _ToyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
        self.fc2 = nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)

    def forward(self, inputs):
        return self.fc2(torch.tanh(self.fc1(inputs)))


class _NestedExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = nn.ModuleList(
            [nn.Linear(4, 4, bias=False, dtype=torch.bfloat16) for _ in range(2)]
        )

    def forward(self, inputs):
        return self.experts[0](inputs) + self.experts[1](inputs)


class _ToyModel(nn.Module):
    def __init__(self, *, nested=False):
        super().__init__()
        block_type = _NestedExperts if nested else _ToyBlock
        self.blocks = nn.ModuleList([block_type(), block_type()])

    def forward(self, inputs):
        for block in self.blocks:
            inputs = block(inputs)
        return inputs


class _EmbeddingBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokens = nn.Embedding(16, 4, dtype=torch.bfloat16)
        self.projection = nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)

    def forward(self, indices):
        return self.projection(self.tokens(indices))


class _EmbeddingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([_EmbeddingBlock()])

    def forward(self, indices):
        return self.blocks[0](indices)


def _initialize_model(model):
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            values = torch.arange(parameter.numel(), dtype=torch.float32)
            values = ((values + index * 3) % 17 - 8) / 4
            parameter.copy_(values.view_as(parameter).to(torch.bfloat16))
    return model.eval()


def _make_compressor(model, modes):
    exponent_samples = bytearray()
    for name, parameter in model.named_parameters():
        if not name.endswith(".weight") or parameter.dtype != torch.bfloat16:
            continue
        raw = (
            parameter.detach()
            .cpu()
            .contiguous()
            .view(torch.uint8)
            .reshape(-1)
            .numpy()
            .tobytes()
        )
        for index in range(0, len(raw), 2):
            low, high = raw[index], raw[index + 1]
            exponent_samples.append(((high & 0x7F) << 1) | (low >> 7))

    codebook_zero = huff0.build_codebook(exponent_samples, prior_count=1)
    codebook_one = huff0.build_codebook(
        bytes(reversed(exponent_samples)), prior_count=1
    )
    codec = NeuralZipCodec()
    payloads = (
        codec._candidate_payload(
            ("zero", "bfloat16", 1),
            {
                "kind": "shared_huff0",
                "codebook": codebook_zero,
                "frame_4x": True,
            },
            codebook_id=0,
        ),
        codec._candidate_payload(
            ("one", "bfloat16", 1),
            {
                "kind": "shared_huff0",
                "codebook": codebook_one,
                "frame_4x": True,
            },
            codebook_id=1,
        ),
    )
    codec._clusters_prepared = True
    codec._codebook_payloads = list(payloads)
    codec._codebook_payload_by_id = {
        payload.codebook_id: payload for payload in payloads
    }
    for tensor_name, mode in modes.items():
        codec._mode_by_tensor_stream[(tensor_name, "bfloat16", 0)] = {
            "kind": "raw",
            "fixed_cost": 0,
        }
        if mode == "raw":
            selected = {"kind": "raw", "fixed_cost": 0}
        elif mode == "packed_huff0":
            selected = {
                "kind": "packed_huff0",
                "codebook_id": 0,
                "ngram_n": 2,
            }
        else:
            selected = {
                "kind": "shared_huff0",
                "codebook_id": int(mode),
            }
        codec._mode_by_tensor_stream[(tensor_name, "bfloat16", 1)] = selected
    compressor = Compressor("neuralzip")
    compressor.codec = codec
    compressor._setup_ready = True
    compressor._reuse_setup = True
    return compressor


class _ReferenceDecoder:
    def __init__(self):
        self.calls = []

    def __call__(
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
        self.calls.append(
            {
                "workspace_pointer": output.untyped_storage().data_ptr(),
                "output_id": id(output),
                "output_pointer": output.data_ptr(),
                "numel": output.numel(),
                "max_bits": max_bits,
                "threads_per_block": threads_per_block,
                "gap_bytes": packed_gaps.numel(),
                "positions": tuple(output_positions.tolist()),
            }
        )
        bitstream = bytes(encoded_exponents.tolist())
        signs = bytes(sign_mantissa.tolist())
        entries = lut.tolist()
        padded = bitstream + b"\x00\x00\x00"
        window_bytes = (7 + max_bits + 7) // 8
        window_bits = window_bytes * 8
        mask = (1 << max_bits) - 1
        offset = 0
        words = []
        for sign_mantissa_value in signs:
            byte_offset, intra_byte = divmod(offset, 8)
            window = int.from_bytes(
                padded[byte_offset : byte_offset + window_bytes], "big"
            )
            prefix = (window >> (window_bits - intra_byte - max_bits)) & mask
            entry = entries[prefix]
            exponent = entry & 0xFF
            offset += entry >> 8
            words.append(
                ((sign_mantissa_value & 0x80) << 8)
                | (exponent << 7)
                | (sign_mantissa_value & 0x7F)
            )
        output.view(torch.uint16).copy_(torch.tensor(words, dtype=torch.uint16))
        return output


class _BatchedReferenceDecoder(_ReferenceDecoder):
    def __init__(self):
        super().__init__()
        self.prepared = []
        self.batch_calls = []

    def decode(self, *args, **kwargs):
        return self(*args, **kwargs)

    def prepare_batch(self, groups, *, threads_per_block):
        prepared = (tuple(groups), threads_per_block)
        self.prepared.append(prepared)
        return prepared

    def decode_batch(self, prepared):
        groups, threads_per_block = prepared
        self.batch_calls.append(prepared)
        for group in groups:
            self.decode(
                group.lut,
                group.encoded_exponents,
                group.sign_mantissa,
                group.packed_gaps,
                group.output_positions,
                group.output,
                max_bits=group.max_bits,
                threads_per_block=threads_per_block,
            )
        return tuple(group.output for group in groups)


class _FailingPreflightDecoder(_BatchedReferenceDecoder):
    def preflight(self, _prepared):
        raise RuntimeError("decoder preflight failed")


class GPUInferenceExportTests(unittest.TestCase):
    def test_export_groups_by_outer_block_and_codebook_with_exact_stats(self):
        model = _initialize_model(_ToyModel())
        modes = {
            "blocks.0.fc1.weight": 0,
            "blocks.0.fc2.weight": 1,
            "blocks.1.fc1.weight": 0,
            "blocks.1.fc2.weight": "raw",
        }
        compressor = _make_compressor(model, modes)

        exported = export_model_for_gpu(model, compressor)

        self.assertEqual(
            [(group.block_name, group.codebook_id) for group in exported.groups],
            [("blocks.0", 0), ("blocks.0", 1), ("blocks.1", 0)],
        )
        self.assertEqual(exported.block_order, ("blocks.0", "blocks.1"))
        self.assertEqual(exported.compressed_weight_count, 3)
        self.assertEqual(exported.resident_weight_count, 1)
        self.assertEqual(exported.resident_weight_names, ("blocks.1.fc2.weight",))
        self.assertEqual(exported.raw_bytes, 3 * 16 * 2)
        self.assertEqual(exported.resident_raw_bytes, 16 * 2)
        self.assertEqual(exported.workspace_numel, 32)
        self.assertEqual(exported.workspace_bytes, 64)
        self.assertEqual(
            exported.payload_bytes,
            sum(group.payload_bytes for group in exported.groups),
        )
        self.assertEqual(
            exported.metadata_bytes,
            sum(group.metadata_bytes for group in exported.groups),
        )
        self.assertEqual(
            exported.compressed_bytes,
            exported.payload_bytes + exported.metadata_bytes + exported.codebook_bytes,
        )
        expected_blocks = sum(
            group.encoded.exponents.block_count for group in exported.groups
        )
        self.assertEqual(
            exported.launch_descriptor_bytes,
            56 * len(exported.groups) + 8 * expected_blocks,
        )
        self.assertEqual(
            exported.runtime_candidate_bytes,
            exported.compressed_bytes
            + exported.launch_descriptor_bytes
            + exported.workspace_bytes
            + exported.resident_raw_bytes,
        )
        self.assertEqual(exported.groups[0].bindings[0].workspace_offset, 0)
        self.assertEqual(exported.groups[1].bindings[0].workspace_offset, 16)
        self.assertEqual(exported.groups[2].bindings[0].workspace_offset, 0)

    def test_export_remaps_the_complete_model_after_a_partial_target(self):
        model = _initialize_model(_ToyModel())
        modes = {name: 0 for name, _parameter in model.named_parameters()}
        compressor = _make_compressor(model, modes)
        first = next(iter(model.named_parameters()))
        compressor.codec.prepare_direct_transfer_target((first,))
        self.assertEqual(len(compressor.codec._transfer_cluster_by_name), 1)

        exported = export_model_for_gpu(model, compressor)

        compressed_names = {
            binding.tensor_name
            for group in exported.groups
            for binding in group.bindings
        }
        self.assertEqual(compressed_names, set(modes))
        self.assertEqual(
            set(compressor.codec._transfer_cluster_by_name),
            set(modes),
        )

    def test_nested_expert_modulelists_belong_to_the_outer_blocks(self):
        model = _initialize_model(_ToyModel(nested=True))
        modes = {
            f"blocks.{block}.experts.{expert}.weight": 0
            for block in range(2)
            for expert in range(2)
        }

        exported = export_model_for_gpu(model, _make_compressor(model, modes))

        self.assertEqual(exported.block_order, ("blocks.0", "blocks.1"))
        self.assertEqual(len(exported.groups), 2)
        self.assertEqual(
            [binding.module_name for binding in exported.groups[0].bindings],
            ["blocks.0.experts.0", "blocks.0.experts.1"],
        )

    def test_embedding_adapter_exports_and_executes_exactly(self):
        model = _initialize_model(_EmbeddingModel())
        modes = {name: 0 for name, _parameter in model.named_parameters()}
        compressor = _make_compressor(model, modes)
        inputs = torch.tensor([[1, 3, 5]], dtype=torch.long)
        with torch.no_grad():
            expected = model(inputs).clone()

        exported = export_model_for_gpu(model, compressor, include_embeddings=True)
        kinds = {
            binding.tensor_name: binding.module_kind for binding in exported.bindings
        }
        self.assertEqual(kinds["blocks.0.tokens.weight"], "embedding")
        self.assertEqual(kinds["blocks.0.projection.weight"], "linear")

        NeuralZipRuntime.attach(model, exported, decoder=_ReferenceDecoder())
        with torch.no_grad():
            actual = model(inputs)
        self.assertTrue(
            torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
        )

    def test_noncontiguous_linear_weight_remains_resident(self):
        model = _initialize_model(_ToyModel())
        original = model.blocks[0].fc1.weight.detach().clone()
        model.blocks[0].fc1.weight = nn.Parameter(original.T)
        self.assertFalse(model.blocks[0].fc1.weight.is_contiguous())
        modes = {name: 0 for name, _parameter in model.named_parameters()}

        exported = export_model_for_gpu(model, _make_compressor(model, modes))
        compressed_names = {
            binding.tensor_name
            for group in exported.groups
            for binding in group.bindings
        }

        self.assertIn("blocks.0.fc1.weight", exported.resident_weight_names)
        self.assertNotIn("blocks.0.fc1.weight", compressed_names)
        self.assertEqual(model.blocks[0].fc1.weight.stride(), (1, 4))

    def test_packed_exponents_are_rejected_without_mutating_the_model(self):
        model = _initialize_model(_ToyModel())
        original = model.blocks[0].fc1.weight
        modes = {
            "blocks.0.fc1.weight": "packed_huff0",
            "blocks.0.fc2.weight": "raw",
            "blocks.1.fc1.weight": "raw",
            "blocks.1.fc2.weight": "raw",
        }

        with self.assertRaisesRegex(ValueError, "does not yet decode packed"):
            export_model_for_gpu(model, _make_compressor(model, modes))

        self.assertIs(model.blocks[0].fc1.weight, original)

    def test_native_encoder_is_preferred(self):
        model = _initialize_model(_ToyModel())
        modes = {
            name: 0
            for name, _parameter in model.named_parameters()
            if name.endswith("weight")
        }
        compressor = _make_compressor(model, modes)

        with mock.patch.object(
            huff0,
            "encode_gpu_bfloat16",
            wraps=huff0.encode_gpu_bfloat16,
        ) as native:
            exported = export_model_for_gpu(model, compressor)

        self.assertEqual(native.call_count, len(exported.groups))


class GPUInferenceRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.model = _initialize_model(_ToyModel())
        self.modes = {
            "blocks.0.fc1.weight": 0,
            "blocks.0.fc2.weight": 1,
            "blocks.1.fc1.weight": 0,
            "blocks.1.fc2.weight": "raw",
        }
        self.compressor = _make_compressor(self.model, self.modes)
        self.inputs = torch.tensor([[-1.0, -0.5, 0.5, 1.0]], dtype=torch.bfloat16)
        with torch.no_grad():
            self.expected = self.model(self.inputs).clone()
        self.original_weights = {
            name: parameter.detach().clone()
            for name, parameter in self.model.named_parameters()
        }

    def test_from_model_returns_same_model_and_reuses_persistent_weight_views(self):
        decoder = _ReferenceDecoder()

        returned = NeuralZipModel.from_model(
            self.model,
            self.compressor,
            decoder=decoder,
        )

        self.assertIs(returned, self.model)
        runtime = self.model.neuralzip_runtime
        self.assertIsInstance(runtime, NeuralZipRuntime)
        parameter_names = dict(self.model.named_parameters())
        self.assertNotIn("blocks.0.fc1.weight", parameter_names)
        self.assertNotIn("blocks.0.fc2.weight", parameter_names)
        self.assertNotIn("blocks.1.fc1.weight", parameter_names)
        self.assertIn("blocks.1.fc2.weight", parameter_names)
        weight_view_ids = {
            name: id(self.model.get_submodule(name.rsplit(".", 1)[0]).weight)
            for name in self.original_weights
            if name != "blocks.1.fc2.weight"
        }
        self.assertTrue(
            all(
                not isinstance(
                    self.model.get_submodule(name.rsplit(".", 1)[0]).weight,
                    nn.Parameter,
                )
                for name in weight_view_ids
            )
        )

        with torch.no_grad():
            first = self.model(self.inputs)
            second = self.model(self.inputs)

        self.assertTrue(
            torch.equal(first.view(torch.uint8), self.expected.view(torch.uint8))
        )
        self.assertTrue(
            torch.equal(second.view(torch.uint8), self.expected.view(torch.uint8))
        )
        self.assertEqual(len(decoder.calls), 6)
        self.assertEqual(
            {call["workspace_pointer"] for call in decoder.calls},
            {runtime.workspace.untyped_storage().data_ptr()},
        )
        self.assertEqual(
            [call["output_id"] for call in decoder.calls[:3]],
            [call["output_id"] for call in decoder.calls[3:]],
        )
        self.assertTrue(all(call["threads_per_block"] == 512 for call in decoder.calls))
        self.assertEqual(
            {
                name: id(self.model.get_submodule(name.rsplit(".", 1)[0]).weight)
                for name in weight_view_ids
            },
            weight_view_ids,
        )

    def test_precompressed_attach_never_requires_target_parameters(self):
        exported = export_model_for_gpu(self.model, self.compressor)
        for binding in exported.bindings:
            module = self.model.get_submodule(binding.module_name)
            delattr(module, "weight")

        runtime = NeuralZipRuntime.attach(
            self.model,
            exported,
            decoder=_ReferenceDecoder(),
            precompressed=True,
        )

        with torch.no_grad():
            actual = self.model(self.inputs)
        self.assertTrue(
            torch.equal(actual.view(torch.uint8), self.expected.view(torch.uint8))
        )
        self.assertNotIn("blocks.0.fc1.weight", dict(self.model.named_parameters()))

        restored = runtime.restore()
        self.assertIs(restored, self.model)
        self.assertIn("blocks.0.fc1.weight", dict(restored.named_parameters()))
        for name, expected in self.original_weights.items():
            actual_weight = dict(restored.named_parameters())[name]
            self.assertTrue(
                torch.equal(actual_weight.view(torch.uint8), expected.view(torch.uint8))
            )

    def test_precompressed_attach_rejects_live_target_parameters(self):
        exported = export_model_for_gpu(self.model, self.compressor)

        with self.assertRaisesRegex(RuntimeError, "must be absent"):
            NeuralZipRuntime.attach(
                self.model,
                exported,
                decoder=_ReferenceDecoder(),
                precompressed=True,
            )

        self.assertFalse(hasattr(self.model, "neuralzip_runtime"))
        for name, expected in self.original_weights.items():
            actual_weight = dict(self.model.named_parameters())[name]
            self.assertTrue(
                torch.equal(actual_weight.view(torch.uint8), expected.view(torch.uint8))
            )

    def test_cpu_offload_keeps_payloads_on_host_and_decodes_each_block(self):
        decoder = _BatchedReferenceDecoder()
        exported = export_model_for_gpu(self.model, self.compressor)
        runtime = NeuralZipRuntime.attach(
            self.model,
            exported,
            decoder=decoder,
            cpu_offload=True,
            pin_memory=False,
        )

        self.assertEqual(runtime._offloaded_blocks, {"blocks.0", "blocks.1"})
        self.assertEqual(decoder.prepared, [])
        self.assertTrue(
            all(group.payload_offloaded for group in runtime._uploaded_groups)
        )
        self.assertTrue(
            all(
                group.encoded_exponents.device.type == "cpu"
                and group.sign_mantissa.device.type == "cpu"
                for group in runtime._uploaded_groups
            )
        )

        with torch.no_grad():
            first = self.model(self.inputs)
            second = self.model(self.inputs)

        self.assertTrue(
            torch.equal(first.view(torch.uint8), self.expected.view(torch.uint8))
        )
        self.assertTrue(
            torch.equal(second.view(torch.uint8), self.expected.view(torch.uint8))
        )
        self.assertEqual(len(decoder.prepared), 4)
        self.assertEqual(len(decoder.batch_calls), 4)

    def test_cpu_offload_block_limit_and_argument_validation(self):
        exported = export_model_for_gpu(self.model, self.compressor)
        decoder = _BatchedReferenceDecoder()
        runtime = NeuralZipRuntime.attach(
            self.model,
            exported,
            decoder=decoder,
            cpu_offload=True,
            cpu_offload_blocks=1,
            pin_memory=False,
        )

        self.assertEqual(runtime._offloaded_blocks, {"blocks.0"})
        self.assertEqual(len(decoder.prepared), 1)
        runtime.restore()

        with self.assertRaisesRegex(ValueError, "requires cpu_offload"):
            NeuralZipRuntime.attach(
                self.model,
                exported,
                decoder=_ReferenceDecoder(),
                cpu_offload_blocks=1,
            )

    def test_batched_decoder_prepares_once_and_launches_once_per_block(self):
        decoder = _BatchedReferenceDecoder()
        NeuralZipModel.from_model(self.model, self.compressor, decoder=decoder)

        self.assertEqual([len(groups) for groups, _threads in decoder.prepared], [2, 1])
        self.assertTrue(all(threads == 512 for _groups, threads in decoder.prepared))

        with torch.no_grad():
            first = self.model(self.inputs)
            second = self.model(self.inputs)

        self.assertTrue(
            torch.equal(first.view(torch.uint8), self.expected.view(torch.uint8))
        )
        self.assertTrue(
            torch.equal(second.view(torch.uint8), self.expected.view(torch.uint8))
        )
        self.assertEqual(len(decoder.batch_calls), 4)
        self.assertEqual(len(decoder.calls), 6)

    def test_materialize_is_exact_and_restore_returns_an_ordinary_model(self):
        decoder = _ReferenceDecoder()
        NeuralZipModel.from_model(self.model, self.compressor, decoder=decoder)
        runtime = self.model.neuralzip_runtime

        with self.assertRaisesRegex(RuntimeError, r"restore\(\).+state_dict"):
            self.model.state_dict()

        tensors = runtime.materialize()

        self.assertEqual(
            set(tensors), set(self.original_weights) - {"blocks.1.fc2.weight"}
        )
        for name, tensor in tensors.items():
            self.assertTrue(
                torch.equal(
                    tensor.view(torch.uint8),
                    self.original_weights[name].view(torch.uint8),
                )
            )
        restored = runtime.restore()

        self.assertIs(restored, self.model)
        self.assertFalse(hasattr(self.model, "neuralzip_runtime"))
        self.assertEqual(
            set(dict(self.model.named_parameters())), set(self.original_weights)
        )
        for name, parameter in self.model.named_parameters():
            self.assertTrue(
                torch.equal(
                    parameter.view(torch.uint8),
                    self.original_weights[name].view(torch.uint8),
                )
            )
            self.assertEqual(parameter.requires_grad, True)
        with torch.no_grad():
            actual = self.model(self.inputs)
        self.assertTrue(
            torch.equal(actual.view(torch.uint8), self.expected.view(torch.uint8))
        )
        self.assertEqual(set(self.model.state_dict()), set(self.original_weights))

    def test_accelerate_dispatched_runtime_rejects_restore_and_resave(self):
        NeuralZipModel.from_model(
            self.model,
            self.compressor,
            decoder=_ReferenceDecoder(),
        )
        runtime = self.model.neuralzip_runtime
        self.model.hf_device_map = {"": "cpu"}
        try:
            with self.assertRaisesRegex(RuntimeError, "re-save.+Accelerate"):
                runtime.resident_state_dict()
            with self.assertRaisesRegex(RuntimeError, "restore.+Accelerate"):
                runtime.restore()
        finally:
            del self.model.hf_device_map

        self.assertIs(runtime.restore(), self.model)

    def test_restore_releases_runtime_owned_tensor_state(self):
        decoder = _ReferenceDecoder()
        NeuralZipModel.from_model(self.model, self.compressor, decoder=decoder)
        runtime = self.model.neuralzip_runtime
        tensor_references = [weakref.ref(runtime.workspace)]
        for uploaded in runtime._uploaded_groups:
            tensor_references.extend(
                weakref.ref(tensor)
                for tensor in (
                    uploaded.lut,
                    uploaded.encoded_exponents,
                    uploaded.sign_mantissa,
                    uploaded.packed_gaps,
                    uploaded.output_positions,
                    uploaded.output,
                )
            )
        del uploaded

        runtime.restore()
        gc.collect()

        self.assertIsNone(runtime.workspace)
        self.assertEqual(runtime._uploaded_groups, ())
        self.assertEqual(runtime._groups_by_block, {})
        self.assertEqual(runtime._prepared_batches, {})
        self.assertEqual(runtime._cuda_tensors, ())
        self.assertIsNone(runtime._attach_event)
        self.assertIsNone(runtime._decoder_backend)
        self.assertIsNone(runtime._decoder)
        self.assertIsNone(runtime._prepare_batch)
        self.assertIsNone(runtime._decode_batch)
        self.assertTrue(all(reference() is None for reference in tensor_references))

    def test_runtime_rejects_training_and_grad_enabled_execution(self):
        decoder = _ReferenceDecoder()
        self.model.train()
        with self.assertRaisesRegex(ValueError, r"model\.eval"):
            NeuralZipModel.from_model(self.model, self.compressor, decoder=decoder)

        self.model.eval()
        NeuralZipModel.from_model(self.model, self.compressor, decoder=decoder)
        with self.assertRaisesRegex(RuntimeError, "inference-only"):
            self.model(self.inputs)

        self.model.train()
        with (
            torch.no_grad(),
            self.assertRaisesRegex(RuntimeError, "evaluation mode"),
        ):
            self.model(self.inputs)

    def test_root_hook_resets_after_failure_and_direct_blocks_are_rejected(self):
        decoder = _ReferenceDecoder()
        NeuralZipModel.from_model(self.model, self.compressor, decoder=decoder)
        runtime = self.model.neuralzip_runtime

        with (
            torch.no_grad(),
            self.assertRaisesRegex(RuntimeError, "root model"),
        ):
            self.model.blocks[0](self.inputs)

        original_forward = self.model.blocks[0].forward
        self.model.blocks[0].forward = mock.Mock(
            side_effect=RuntimeError("deliberate failure")
        )
        try:
            with (
                torch.no_grad(),
                self.assertRaisesRegex(RuntimeError, "deliberate failure"),
            ):
                self.model(self.inputs)
        finally:
            self.model.blocks[0].forward = original_forward

        self.assertIsNone(runtime._active_block)
        with torch.no_grad():
            actual = self.model(self.inputs)
        self.assertTrue(
            torch.equal(actual.view(torch.uint8), self.expected.view(torch.uint8))
        )

    def test_rejected_concurrent_forward_does_not_release_valid_owner(self):
        decoder = _ReferenceDecoder()
        NeuralZipModel.from_model(self.model, self.compressor, decoder=decoder)
        runtime = self.model.neuralzip_runtime
        entered = threading.Event()
        release = threading.Event()
        outcome = {}
        original_forward = self.model.blocks[0].forward

        def blocking_forward(inputs):
            entered.set()
            if not release.wait(timeout=5):
                raise RuntimeError("test timed out")
            return original_forward(inputs)

        self.model.blocks[0].forward = blocking_forward

        def run_first():
            try:
                with torch.no_grad():
                    outcome["output"] = self.model(self.inputs)
            except RuntimeError as error:  # pragma: no cover - asserted below.
                outcome["error"] = error

        worker = threading.Thread(target=run_first)
        worker.start()
        self.assertTrue(entered.wait(timeout=5))
        try:
            with (
                torch.no_grad(),
                self.assertRaisesRegex(RuntimeError, "Concurrent or re-entrant"),
            ):
                self.model(self.inputs)
        finally:
            release.set()
            worker.join(timeout=5)
            self.model.blocks[0].forward = original_forward

        self.assertFalse(worker.is_alive())
        self.assertNotIn("error", outcome)
        self.assertTrue(
            torch.equal(
                outcome["output"].view(torch.uint8),
                self.expected.view(torch.uint8),
            )
        )
        self.assertIsNone(runtime._active_block)

    def test_restore_failure_rolls_back_to_attached_runtime(self):
        decoder = _ReferenceDecoder()
        NeuralZipModel.from_model(self.model, self.compressor, decoder=decoder)
        runtime = self.model.neuralzip_runtime
        failing_module = self.model.blocks[0].fc2

        with (
            mock.patch.object(
                failing_module,
                "register_parameter",
                side_effect=RuntimeError("deliberate restore failure"),
            ),
            self.assertRaisesRegex(RuntimeError, "deliberate restore failure"),
        ):
            runtime.restore()

        self.assertIs(self.model.neuralzip_runtime, runtime)
        parameter_names = set(dict(self.model.named_parameters()))
        for binding in runtime.export.bindings:
            self.assertNotIn(binding.tensor_name, parameter_names)
            module = self.model.get_submodule(binding.module_name)
            self.assertIsInstance(module.weight, torch.Tensor)
            self.assertNotIsInstance(module.weight, nn.Parameter)

        with torch.no_grad():
            actual = self.model(self.inputs)
        self.assertTrue(
            torch.equal(actual.view(torch.uint8), self.expected.view(torch.uint8))
        )
        self.assertIs(runtime.restore(), self.model)

    def test_upload_failure_leaves_every_parameter_registered(self):
        exported = export_model_for_gpu(self.model, self.compressor)
        expected_names = set(dict(self.model.named_parameters()))

        with (
            mock.patch(
                "neuralzip.gpu_inference._upload_group",
                side_effect=RuntimeError("upload failed"),
            ),
            self.assertRaisesRegex(RuntimeError, "upload failed"),
        ):
            NeuralZipRuntime.attach(
                self.model,
                exported,
                decoder=_ReferenceDecoder(),
            )

        self.assertEqual(set(dict(self.model.named_parameters())), expected_names)
        self.assertFalse(hasattr(self.model, "neuralzip_runtime"))

    def test_decoder_preflight_failure_leaves_model_untouched(self):
        expected_names = set(dict(self.model.named_parameters()))

        with self.assertRaisesRegex(RuntimeError, "decoder preflight failed"):
            NeuralZipModel.from_model(
                self.model,
                self.compressor,
                decoder=_FailingPreflightDecoder(),
            )

        self.assertEqual(set(dict(self.model.named_parameters())), expected_names)
        self.assertFalse(hasattr(self.model, "neuralzip_runtime"))

    def test_offloaded_decoder_preflight_failure_leaves_model_untouched(self):
        expected_names = set(dict(self.model.named_parameters()))

        with self.assertRaisesRegex(RuntimeError, "decoder preflight failed"):
            NeuralZipModel.from_model(
                self.model,
                self.compressor,
                decoder=_FailingPreflightDecoder(),
                cpu_offload=True,
                pin_memory=False,
            )

        self.assertEqual(set(dict(self.model.named_parameters())), expected_names)
        self.assertFalse(hasattr(self.model, "neuralzip_runtime"))

    def test_all_raw_export_is_rejected_without_mutating_model(self):
        compressor = _make_compressor(
            self.model,
            {name: "raw" for name in self.original_weights},
        )
        expected_names = set(dict(self.model.named_parameters()))

        with self.assertRaisesRegex(ValueError, "no eligible shared BF16"):
            NeuralZipModel.from_model(
                self.model,
                compressor,
                decoder=_ReferenceDecoder(),
            )

        self.assertEqual(set(dict(self.model.named_parameters())), expected_names)
        self.assertFalse(hasattr(self.model, "neuralzip_runtime"))


if __name__ == "__main__":
    unittest.main()
