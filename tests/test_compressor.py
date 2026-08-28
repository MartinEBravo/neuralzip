import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from neuralzip import Compressor
from neuralzip.compressor import _named_input_tensors


class NestedModelModule(torch.nn.Module):
    def __init__(self, *, tied=False):
        super().__init__()
        self.root_parameter = torch.nn.Parameter(torch.arange(2, dtype=torch.float32))
        self.model = torch.nn.Linear(2, 2, bias=False)
        self.head = torch.nn.Linear(2, 2, bias=False)
        if tied:
            self.head.weight = self.model.weight
        self.register_buffer("running", torch.ones(2, dtype=torch.float32))
        self.register_buffer("counter", torch.ones((), dtype=torch.int64))


class CompressorInputContractTests(unittest.TestCase):
    def test_module_with_model_child_is_not_unwrapped(self):
        module = NestedModelModule()
        expected = list(module.named_parameters())

        direct = list(_named_input_tensors(module))
        wrapped = list(_named_input_tensors(SimpleNamespace(model=module)))

        self.assertEqual(
            [(name, id(value)) for name, value in direct],
            [(name, id(value)) for name, value in expected],
        )
        self.assertEqual(
            [(name, id(value)) for name, value in wrapped],
            [(name, id(value)) for name, value in expected],
        )
        self.assertIn("model.weight", dict(direct))
        self.assertIn("head.weight", dict(direct))
        self.assertIn("root_parameter", dict(direct))

    def test_module_ties_are_deduplicated_but_mapping_aliases_are_preserved(self):
        module = NestedModelModule(tied=True)
        module_parameters = list(_named_input_tensors(module))
        shared_parameter = module.model.weight

        self.assertEqual(
            sum(value is shared_parameter for _, value in module_parameters),
            1,
        )

        shared_tensor = torch.ones(4)
        mapping = {"model": {"weight": shared_tensor}, "head": shared_tensor}
        direct = list(_named_input_tensors(mapping))
        wrapped = list(_named_input_tensors(SimpleNamespace(model=mapping)))

        self.assertEqual([name for name, _ in direct], ["model.weight", "head"])
        self.assertEqual([name for name, _ in wrapped], ["model.weight", "head"])
        self.assertIs(direct[0][1], direct[1][1])

    def test_invalid_wrapper_sources_are_rejected(self):
        for value in (object(), SimpleNamespace(model=object())):
            with (
                self.subTest(value=type(value).__name__),
                self.assertRaisesRegex(TypeError, "tensor mapping"),
            ):
                list(_named_input_tensors(value))

    def test_module_and_state_dict_have_explicit_floating_tensor_scopes(self):
        module = NestedModelModule()
        parameter_names = [name for name, _ in module.named_parameters()]
        floating_state_names = [
            name
            for name, value in module.state_dict().items()
            if value.is_floating_point()
        ]

        compressor = Compressor("zipnn-local", keep_payloads=True, verify=True)
        module_result = compressor.compress(module)
        mapping_result = compressor.compress(module.state_dict())

        self.assertEqual(
            [item["name"] for item in module_result["tensor_results"]],
            parameter_names,
        )
        self.assertEqual(
            [item["name"] for item in mapping_result["tensor_results"]],
            floating_state_names,
        )
        self.assertNotIn("running", parameter_names)
        self.assertIn("running", floating_state_names)
        self.assertNotIn("counter", floating_state_names)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "floating-state.nzip"
            Compressor.save(mapping_result, path)
            restored = compressor.decompress_file(path)

        self.assertEqual(tuple(restored), tuple(floating_state_names))
        for name in floating_state_names:
            self.assertTrue(
                torch.equal(
                    restored[name].view(torch.uint8),
                    module.state_dict()[name].contiguous().view(torch.uint8),
                )
            )

    def test_hugging_face_model_is_accepted_directly(self):
        from transformers import BertConfig, BertModel

        model = BertModel(
            BertConfig(
                vocab_size=32,
                hidden_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                intermediate_size=32,
            )
        )
        result = Compressor("zipnn-local", verify=True).compress(model)

        self.assertEqual(result["num_tensors"], len(list(model.named_parameters())))
        self.assertTrue(result["roundtrip_ok"])


class ZipNNCompressorContractTests(unittest.TestCase):
    @mock.patch("neuralzip.compressor.ZipNNSourceCodec")
    def test_source_frontend_uses_the_official_adapter(self, adapter_class):
        compressor = Compressor(
            "zipnn",
            threads=3,
            chunk_size=4096,
            compression_threshold=0.8,
            zipnn_method="huffman",
        )

        self.assertIs(compressor.codec, adapter_class.return_value)
        adapter_class.assert_called_once_with(
            method="HUFFMAN",
            threads=3,
            compression_threshold=0.8,
            chunk_size=4096,
        )

    def test_legacy_source_names_are_not_public_methods(self):
        for method in ("zipnn-src", "zipnn_src", "zipnn src"):
            with (
                self.subTest(method=method),
                self.assertRaisesRegex(ValueError, "Unsupported compression method"),
            ):
                Compressor(method)

    @mock.patch("neuralzip.compressor.ZipNNSourceCodec")
    def test_source_decode_is_model_wide_after_all_encodes(self, adapter_class):
        codec = adapter_class.return_value
        tensors = {
            "first": torch.arange(8, dtype=torch.bfloat16),
            "second": torch.arange(12, dtype=torch.bfloat16),
        }
        payloads = {"first": b"first", "second": b"second"}
        restored = {payloads[name]: tensor.clone() for name, tensor in tensors.items()}
        events = []
        encoded_names = iter(tensors)

        def encode_tensor(tensor, *, verify):
            del tensor
            name = next(encoded_names)
            events.append(("encode", name, verify))
            return SimpleNamespace(
                payload=payloads[name],
                input_copy_seconds=0.1,
                codec_encode_seconds=0.2,
                encode_seconds=0.3,
            )

        def decode(payload):
            events.append(("decode", payload.decode()))
            return restored[payload]

        codec.encode_tensor.side_effect = encode_tensor
        codec.decode.side_effect = decode
        model = SimpleNamespace(model=tensors)

        result = Compressor("zipnn", verify=True).compress(model)

        self.assertEqual(
            events,
            [
                ("encode", "first", False),
                ("encode", "second", False),
                ("decode", "first"),
                ("decode", "second"),
            ],
        )
        self.assertTrue(result["roundtrip_ok"])
        self.assertGreaterEqual(result["decode_seconds"], 0.0)

    @mock.patch("neuralzip.compressor.ZipNNSourceCodec")
    def test_source_model_io_is_rejected_before_payload_or_file_access(
        self, adapter_class
    ):
        del adapter_class
        compressor = Compressor("zipnn")
        results = tuple(
            {
                "method": method,
                "tensor_results": (
                    [] if index == 0 else [{"name": "weight", "payload": b"ZNframe"}]
                ),
            }
            for index, method in enumerate(
                ("zipnn", "zipnn-src", "zipnn_src", "zipnn src")
            )
        )
        for result in results:
            with (
                self.subTest(has_payload=bool(result["tensor_results"])),
                self.assertRaisesRegex(
                    NotImplementedError, "opaque per-tensor benchmark frames"
                ),
            ):
                compressor.decompress(result)

        local_compressor = Compressor("zipnn-local")
        with self.assertRaisesRegex(
            NotImplementedError, "opaque per-tensor benchmark frames"
        ):
            local_compressor.decompress(results[0])

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "official-frames.nzip"
            path.write_bytes(b"unchanged")
            with self.assertRaisesRegex(
                NotImplementedError, "opaque per-tensor benchmark frames"
            ):
                Compressor.save(results[1], path)
            self.assertEqual(path.read_bytes(), b"unchanged")

            missing = Path(directory) / "does-not-exist.nzip"
            with self.assertRaisesRegex(
                NotImplementedError, "opaque per-tensor benchmark frames"
            ):
                compressor.decompress_file(missing)
            self.assertFalse(missing.exists())

    def test_load_rejects_an_official_frame_as_not_a_neuralzip_archive(self):
        with self.assertRaisesRegex(ValueError, "Not a NeuralZip archive"):
            Compressor.load(io.BytesIO(b"ZNframe"))


if __name__ == "__main__":
    unittest.main()
