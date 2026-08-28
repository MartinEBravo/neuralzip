import json
import os
import shutil
import sys
import types
from pathlib import Path
from unittest import mock

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM, ViTConfig, ViTModel
from transformers.cache_utils import DynamicCache

from neuralzip import huff0
from neuralzip.gpu_checkpoint import (
    DEFAULT_FILENAME as GPU_EXPORT_FILENAME,
)
from neuralzip.gpu_checkpoint import (
    load_gpu_export,
)
from neuralzip.gpu_inference import NeuralZipRuntime
from neuralzip.gpu_pretrained import (
    MANIFEST_FILENAME,
    RESIDENT_FILENAME,
    GPUPretrainedError,
    _block_payload_bytes,
    _cpu_runtime_reservation_bytes,
    _deployment_sizing_skeleton,
    _gpu_runtime_reservation_bytes,
    load_gpu_pretrained,
    save_gpu_pretrained,
)
from neuralzip.gpu_resident import iter_resident_shards, load_resident_state
from neuralzip.plan import NeuralZipPlan


class _CPUReferenceDecoder:
    """Decode the GPU ABI on CPU so the pretrained path needs no CUDA."""

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
        del packed_gaps, output_positions, threads_per_block
        bitstream = bytes(encoded_exponents.tolist()) + b"\x00\x00\x00"
        signs = bytes(sign_mantissa.tolist())
        entries = lut.tolist()
        window_bytes = (7 + max_bits + 7) // 8
        window_bits = window_bytes * 8
        prefix_mask = (1 << max_bits) - 1
        bit_offset = 0
        words = []
        for sign_mantissa_value in signs:
            byte_offset, intra_byte = divmod(bit_offset, 8)
            window = int.from_bytes(
                bitstream[byte_offset : byte_offset + window_bytes], "big"
            )
            prefix = (window >> (window_bits - intra_byte - max_bits)) & prefix_mask
            entry = entries[prefix]
            exponent = entry & 0xFF
            bit_offset += entry >> 8
            words.append(
                ((sign_mantissa_value & 0x80) << 8)
                | (exponent << 7)
                | (sign_mantissa_value & 0x7F)
            )
        output.view(torch.uint16).copy_(torch.tensor(words, dtype=torch.uint16))
        return output


class _FakeDiffusionBlock(torch.nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.projection = torch.nn.Linear(
            width,
            width,
            bias=False,
            dtype=torch.bfloat16,
        )

    def forward(self, inputs):
        return torch.tanh(self.projection(inputs))


class _FakeDiffusionModel(torch.nn.Module):
    """Minimal Diffusers-shaped component for an offline format test."""

    def __init__(self, width: int = 64):
        super().__init__()
        self.config = {"width": width}
        self.lookup = torch.nn.Embedding(8, width, dtype=torch.bfloat16)
        self.output = torch.nn.Linear(
            width,
            8,
            bias=False,
            dtype=torch.bfloat16,
        )
        self.output.weight = self.lookup.weight
        self.blocks = torch.nn.ModuleList(
            [_FakeDiffusionBlock(width), _FakeDiffusionBlock(width)]
        )
        self.register_buffer("output_scale", torch.tensor(0.5, dtype=torch.bfloat16))

    def forward(self, inputs):
        for block in self.blocks:
            inputs = block(inputs)
        return inputs * self.output_scale

    def save_config(self, directory):
        path = Path(directory) / "config.json"
        path.write_text(
            json.dumps(self.config, sort_keys=True) + "\n", encoding="utf-8"
        )

    @classmethod
    def load_config(cls, directory):
        return json.loads((Path(directory) / "config.json").read_text(encoding="utf-8"))

    @classmethod
    def from_config(cls, config):
        return cls(width=int(config["width"]))


# The loader deliberately accepts only classes exported by the Diffusers
# namespace. No real Diffusers dependency or network access is needed here.
_FakeDiffusionModel.__module__ = "diffusers.models.fake"


@pytest.fixture(scope="module", autouse=True)
def _offline_transformers():
    names = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    previous = {name: os.environ.get(name) for name in names}
    os.environ.update({name: "1" for name in names})
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _tiny_llama() -> LlamaForCausalLM:
    torch.manual_seed(1234)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        bos_token_id=1,
        eos_token_id=63,
        pad_token_id=0,
        tie_word_embeddings=False,
    )
    return LlamaForCausalLM(config).to(dtype=torch.bfloat16).eval()


@pytest.fixture(scope="module")
def tiny_llama_artifact(tmp_path_factory):
    model = _tiny_llama()
    model.generation_config.max_length = 37
    input_ids = torch.tensor([[1, 5, 7, 9]], dtype=torch.long)
    next_token = torch.tensor([[11]], dtype=torch.long)
    with torch.inference_mode():
        expected = model(input_ids, use_cache=True)
        expected_logits = expected.logits.clone()
        expected_cached_logits = model(
            next_token,
            past_key_values=expected.past_key_values,
            use_cache=True,
        ).logits.clone()
        expected_generation = model.generate(
            input_ids,
            max_new_tokens=3,
            min_new_tokens=3,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
        ).sequences.clone()

    expected_state_names = set(model.state_dict())
    plan = NeuralZipPlan.fit(
        dict(model.named_parameters()),
        threads=1,
        ngram_values=(),
        packed_source_budget_ratio=0.0,
        packed_time_budget_ratio=0.0,
        max_codebook_sample_bytes=1 << 20,
    )
    directory = tmp_path_factory.mktemp("tiny-llama-neuralzip")
    returned = save_gpu_pretrained(model, plan, directory, max_shard_size="8KB")
    assert returned == directory
    return {
        "directory": directory,
        "input_ids": input_ids,
        "next_token": next_token,
        "expected_logits": expected_logits,
        "expected_cached_logits": expected_cached_logits,
        "expected_generation": expected_generation,
        "expected_state_names": expected_state_names,
    }


def _load(directory: Path):
    return load_gpu_pretrained(
        directory,
        device="cpu",
        decoder=_CPUReferenceDecoder(),
        pin_memory=False,
    )


def _canonical_manifest_write(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="ascii",
    )


def _copied_artifact(source: Path, target: Path) -> Path:
    destination = target / "checkpoint"
    shutil.copytree(source, destination)
    return destination


def test_archive_partitions_resident_and_compressed_state(tiny_llama_artifact):
    directory = tiny_llama_artifact["directory"]
    exported = load_gpu_export(directory / GPU_EXPORT_FILENAME)
    compressed_names = {binding.tensor_name for binding in exported.bindings}
    resident = load_resident_state(directory)
    resident_names = set(resident.tensor_names)
    physical_names = {
        name for _path, shard in iter_resident_shards(resident) for name in shard
    }
    assert directory / RESIDENT_FILENAME == resident.index_path
    assert physical_names == set(resident.descriptors)
    assert len(resident.shard_paths) > 1

    assert compressed_names
    assert compressed_names.isdisjoint(resident_names)
    assert (
        compressed_names | resident_names == tiny_llama_artifact["expected_state_names"]
    )
    assert "model.layers.0.self_attn.q_proj.weight" in compressed_names
    assert "model.layers.1.mlp.down_proj.weight" in compressed_names
    assert "model.embed_tokens.weight" in resident_names
    assert "lm_head.weight" in resident_names
    manifest = json.loads((directory / MANIFEST_FILENAME).read_text(encoding="ascii"))
    assert manifest["auto_class"] == "AutoModelForCausalLM"
    assert manifest["library"] == "transformers"
    assert manifest["model_class"] == "LlamaForCausalLM"


def test_auto_placement_uses_reversible_compressed_payload_skeleton(
    tiny_llama_artifact,
):
    model = _tiny_llama()
    exported = load_gpu_export(tiny_llama_artifact["directory"] / GPU_EXPORT_FILENAME)
    originals = {
        binding.tensor_name: model.get_submodule(binding.module_name).weight
        for binding in exported.bindings
    }
    payloads = _block_payload_bytes(exported)

    with _deployment_sizing_skeleton(
        model,
        exported,
        cpu_offload=False,
        cpu_offload_blocks=None,
    ) as placeholder_bytes:
        assert placeholder_bytes == payloads
        for binding in exported.bindings:
            module = model.get_submodule(binding.module_name)
            assert "weight" not in module._parameters
        for block_name in exported.block_order:
            placeholder = model.get_submodule(block_name)._neuralzip_placement_payload
            assert placeholder.device.type == "meta"
            assert placeholder.dtype == torch.uint8
            assert placeholder.numel() == payloads[block_name]
        assert not any(
            name.endswith("_neuralzip_placement_payload") for name in model.state_dict()
        )

    for binding in exported.bindings:
        module = model.get_submodule(binding.module_name)
        assert module.weight is originals[binding.tensor_name]
    assert all(
        not hasattr(model.get_submodule(name), "_neuralzip_placement_payload")
        for name in exported.block_order
    )


def test_auto_placement_skeleton_restores_after_failure_and_sizes_offload(
    tiny_llama_artifact,
):
    model = _tiny_llama()
    exported = load_gpu_export(tiny_llama_artifact["directory"] / GPU_EXPORT_FILENAME)
    originals = tuple(
        model.get_submodule(binding.module_name).weight for binding in exported.bindings
    )

    with (
        pytest.raises(RuntimeError, match="injected"),
        _deployment_sizing_skeleton(
            model,
            exported,
            cpu_offload=True,
            cpu_offload_blocks=1,
        ) as placeholder_bytes,
    ):
        assert placeholder_bytes[exported.block_order[0]] == 1
        assert all(
            placeholder_bytes[name] == _block_payload_bytes(exported)[name]
            for name in exported.block_order[1:]
        )
        raise RuntimeError("injected")

    restored = tuple(
        model.get_submodule(binding.module_name).weight for binding in exported.bindings
    )
    assert all(actual is expected for actual, expected in zip(restored, originals))
    assert all(
        not hasattr(model.get_submodule(name), "_neuralzip_placement_payload")
        for name in exported.block_order
    )


def test_auto_placement_sizing_replaces_raw_weights_with_payloads(
    tiny_llama_artifact,
):
    from accelerate.utils import compute_module_sizes

    model = _tiny_llama()
    exported = load_gpu_export(tiny_llama_artifact["directory"] / GPU_EXPORT_FILENAME)
    ordinary_bytes = int(compute_module_sizes(model, dtype=torch.bfloat16)[""])
    with _deployment_sizing_skeleton(
        model,
        exported,
        cpu_offload=False,
        cpu_offload_blocks=None,
    ) as placeholder_bytes:
        deployment_bytes = int(compute_module_sizes(model, dtype=torch.bfloat16)[""])

    assert ordinary_bytes - deployment_bytes == (
        exported.raw_bytes - sum(placeholder_bytes.values())
    )
    assert deployment_bytes < ordinary_bytes
    assert _gpu_runtime_reservation_bytes(exported) == (
        exported.workspace_bytes
        + exported.metadata_bytes
        + exported.codebook_bytes
        + exported.launch_descriptor_bytes
        + max(_block_payload_bytes(exported).values())
    )
    assert _cpu_runtime_reservation_bytes(exported) == (
        exported.compressed_bytes + exported.payload_bytes
    )


def test_load_is_precompressed_and_does_not_reencode(tiny_llama_artifact):
    observed = {}
    original_attach = NeuralZipRuntime.attach

    def checked_attach(model, exported, **kwargs):
        observed["precompressed"] = kwargs.get("precompressed")
        observed["absent_before_attach"] = all(
            model.get_submodule(binding.module_name)._parameters.get("weight") is None
            and "weight" not in model.get_submodule(binding.module_name).__dict__
            for binding in exported.bindings
        )
        return original_attach(model, exported, **kwargs)

    with (
        mock.patch(
            "neuralzip.gpu_pretrained.export_model_for_gpu",
            side_effect=AssertionError("load attempted to re-export the model"),
        ),
        mock.patch.object(
            huff0,
            "encode_gpu_bfloat16",
            side_effect=AssertionError("load attempted to re-encode weights"),
        ),
        mock.patch.object(NeuralZipRuntime, "attach", side_effect=checked_attach),
    ):
        model = _load(tiny_llama_artifact["directory"])

    assert observed == {"precompressed": True, "absent_before_attach": True}
    assert model.neuralzip_runtime._precompressed is True
    compressed_names = {
        binding.tensor_name for binding in model.neuralzip_runtime.export.bindings
    }
    assert compressed_names.isdisjoint(dict(model.named_parameters()))
    for binding in model.neuralzip_runtime.export.bindings:
        weight = model.get_submodule(binding.module_name).weight
        assert torch.is_tensor(weight)
        assert not isinstance(weight, torch.nn.Parameter)


def test_loaded_tiny_llama_logits_are_bitwise_exact(tiny_llama_artifact):
    model = _load(tiny_llama_artifact["directory"])
    with torch.inference_mode():
        first = model(tiny_llama_artifact["input_ids"], use_cache=True).logits
        second = model(tiny_llama_artifact["input_ids"], use_cache=True).logits

    expected = tiny_llama_artifact["expected_logits"]
    assert torch.equal(first.view(torch.uint8), expected.view(torch.uint8))
    assert torch.equal(second.view(torch.uint8), expected.view(torch.uint8))


def test_generate_uses_kv_cache_and_matches_reference(tiny_llama_artifact):
    model = _load(tiny_llama_artifact["directory"])
    with torch.inference_mode():
        prefill = model(tiny_llama_artifact["input_ids"], use_cache=True)
        cached = model(
            tiny_llama_artifact["next_token"],
            past_key_values=prefill.past_key_values,
            use_cache=True,
        )
        generated = model.generate(
            tiny_llama_artifact["input_ids"],
            max_new_tokens=3,
            min_new_tokens=3,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
        )

    assert torch.equal(
        cached.logits.view(torch.uint8),
        tiny_llama_artifact["expected_cached_logits"].view(torch.uint8),
    )
    assert torch.equal(generated.sequences, tiny_llama_artifact["expected_generation"])
    assert isinstance(generated.past_key_values, DynamicCache)
    assert generated.past_key_values.get_seq_length() == (
        tiny_llama_artifact["input_ids"].shape[1] + 3 - 1
    )
    assert model.generation_config.max_length == 37


def test_cpu_device_map_offloads_resident_and_compressed_payloads(
    tiny_llama_artifact,
):
    model = load_gpu_pretrained(
        tiny_llama_artifact["directory"],
        device_map={"": "cpu"},
        decoder=_CPUReferenceDecoder(),
        pin_memory=False,
    )

    runtime = model.neuralzip_runtime
    assert runtime._offloaded_blocks == set(runtime.export.block_order)
    assert runtime.devices == (torch.device("cpu"),)
    with torch.inference_mode():
        actual = model(tiny_llama_artifact["input_ids"], use_cache=True).logits
    assert torch.equal(
        actual.view(torch.uint8),
        tiny_llama_artifact["expected_logits"].view(torch.uint8),
    )


def test_attached_model_can_be_saved_without_materializing(
    tiny_llama_artifact, tmp_path
):
    attached = _load(tiny_llama_artifact["directory"])
    destination = tmp_path / "attached-copy"

    save_gpu_pretrained(attached, destination)
    restored = _load(destination)

    with torch.inference_mode():
        actual = restored(tiny_llama_artifact["input_ids"], use_cache=True).logits
    assert torch.equal(
        actual.view(torch.uint8),
        tiny_llama_artifact["expected_logits"].view(torch.uint8),
    )
    assert hasattr(attached, "neuralzip_runtime")


def test_base_transformers_automodel_round_trip(tmp_path):
    torch.manual_seed(2468)
    model = (
        ViTModel(
            ViTConfig(
                image_size=16,
                patch_size=4,
                num_channels=3,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
            )
        )
        .to(dtype=torch.bfloat16)
        .eval()
    )
    pixel_values = torch.randn(1, 3, 16, 16, dtype=torch.bfloat16)
    with torch.inference_mode():
        expected = model(pixel_values).last_hidden_state.clone()
    plan = NeuralZipPlan.fit(
        dict(model.named_parameters()),
        threads=1,
        ngram_values=(),
        packed_source_budget_ratio=0.0,
        packed_time_budget_ratio=0.0,
        max_codebook_sample_bytes=1 << 20,
    )
    directory = tmp_path / "vit-base-model"
    save_gpu_pretrained(model, plan, directory, max_shard_size="8KB")

    manifest = json.loads((directory / MANIFEST_FILENAME).read_text(encoding="ascii"))
    assert manifest["auto_class"] == "AutoModel"
    restored = load_gpu_pretrained(
        directory,
        device="cpu",
        decoder=_CPUReferenceDecoder(),
        pin_memory=False,
    )
    with torch.inference_mode():
        actual = restored(pixel_values).last_hidden_state
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


def test_diffusers_component_round_trip_without_real_dependency(tmp_path, monkeypatch):
    torch.manual_seed(4321)
    model = _FakeDiffusionModel().eval()
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            values = torch.arange(parameter.numel(), dtype=torch.float32)
            values = ((values + index * 3) % 17 - 8) / 4
            parameter.copy_(values.view_as(parameter).to(torch.bfloat16))
    inputs = torch.randn(2, 64, dtype=torch.bfloat16)
    with torch.inference_mode():
        expected = model(inputs).clone()

    plan = NeuralZipPlan.fit(
        dict(model.named_parameters()),
        threads=1,
        ngram_values=(),
        packed_source_budget_ratio=0.0,
        packed_time_budget_ratio=0.0,
        max_codebook_sample_bytes=1 << 20,
    )
    directory = tmp_path / "diffusion-component"
    save_gpu_pretrained(model, plan, directory, max_shard_size="8KB")

    manifest = json.loads((directory / MANIFEST_FILENAME).read_text(encoding="ascii"))
    assert manifest["library"] == "diffusers"
    assert manifest["model_class"] == "_FakeDiffusionModel"
    assert manifest["auto_class"] is None

    fake_diffusers = types.ModuleType("diffusers")
    fake_diffusers._FakeDiffusionModel = _FakeDiffusionModel
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)
    restored = load_gpu_pretrained(
        directory,
        device="cpu",
        decoder=_CPUReferenceDecoder(),
        pin_memory=False,
    )
    with torch.inference_mode():
        actual = restored(inputs)

    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    assert restored.output_scale.item() == model.output_scale.item()
    assert restored.lookup.weight is restored.output.weight


def test_load_rejects_corrupted_resident_archive(tiny_llama_artifact, tmp_path):
    directory = _copied_artifact(tiny_llama_artifact["directory"], tmp_path)
    resident = directory / RESIDENT_FILENAME
    payload = bytearray(resident.read_bytes())
    payload[-1] ^= 1
    resident.write_bytes(payload)

    with pytest.raises(GPUPretrainedError, match="SHA-256"):
        _load(directory)


def test_load_rejects_manifest_export_disagreement(tiny_llama_artifact, tmp_path):
    directory = _copied_artifact(tiny_llama_artifact["directory"], tmp_path)
    manifest_path = directory / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    manifest["compressed_weight_names"] = manifest["compressed_weight_names"][1:]
    _canonical_manifest_write(manifest_path, manifest)

    with pytest.raises(GPUPretrainedError, match="disagree"):
        _load(directory)


def test_load_rejects_incompatible_model_architecture(tiny_llama_artifact):
    incompatible = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            tie_word_embeddings=False,
        )
    ).to(dtype=torch.bfloat16)

    with pytest.raises(GPUPretrainedError, match="architecture|no module"):
        load_gpu_pretrained(
            tiny_llama_artifact["directory"],
            model=incompatible,
            device="cpu",
            decoder=_CPUReferenceDecoder(),
            pin_memory=False,
        )
