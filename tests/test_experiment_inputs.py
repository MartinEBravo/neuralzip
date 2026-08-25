import io
import json
import sys
from pathlib import Path

import pytest
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import examples.run_setup_transfer_suite as family_suite
from examples.model_catalog import MODEL_SPECS, QUANTIZED_MODEL_SPECS
from examples.model_loading import MappingModel
from examples.run_quantized_nibble_compression import (
    _bytes_from_nibbles,
    _encode_local,
    _encode_shared,
    _hf_model_files,
    _nibble_stream,
    _prepare_shared_plan,
    _verify_archive,
)
from examples.run_setup_transfer_suite import (
    FAMILY_SPECS,
    FamilySpec,
    canonical_tensor_name,
)


class _FakeHfFileSystem:
    def __init__(self, model_id, files, contents=None, *, revision_in_path=False):
        self.model_id = model_id
        self.files = dict(files)
        self.contents = dict(contents or {})
        self.revision_in_path = revision_in_path

    def ls(self, model_id, *, detail, revision):
        assert model_id == self.model_id
        assert detail is True
        assert revision
        prefix = f"{model_id}@{revision}" if self.revision_in_path else model_id
        return [
            {
                "type": "file",
                "name": f"{prefix}/{path}",
                "size": size,
            }
            for path, size in self.files.items()
        ]

    def open(self, path, mode, *, revision, block_size):
        assert mode == "rb"
        assert revision
        assert block_size > 0
        prefix = f"{self.model_id}/"
        assert path.startswith(prefix)
        return io.BytesIO(self.contents[path[len(prefix) :]])


def test_family_transfer_uses_complete_floating_state_once():
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, kernel_size=3),
        torch.nn.BatchNorm2d(4),
    )

    view = MappingModel("test", dict(model.state_dict(keep_vars=True)), {})
    state = family_suite.named_tensors(view, "resnet18")
    names = [name for name, _ in state]

    assert len(names) == len(set(names))
    assert "1.running_mean" in names
    assert "1.running_var" in names
    assert "1.num_batches_tracked" not in names


def test_family_transfer_suite_has_the_planned_coverage():
    assert len(FAMILY_SPECS) == 11
    assert sum(len(spec.target_keys) for spec in FAMILY_SPECS.values()) == 20
    assert (
        len(
            {
                key
                for spec in FAMILY_SPECS.values()
                for key in (spec.source_key, *spec.target_keys)
            }
        )
        == 31
    )
    assert all(
        spec.source_key not in spec.target_keys for spec in FAMILY_SPECS.values()
    )


def test_family_transfer_canonicalizes_gemma_text_backbones():
    multimodal = "language_model.model.layers.0.mlp.up_proj.weight"
    text_only = "model.layers.0.mlp.up_proj.weight"

    assert canonical_tensor_name("gemma3-12b", multimodal) == text_only
    assert canonical_tensor_name("gemma3-4b", multimodal) == text_only
    assert canonical_tensor_name("gemma3-1b", text_only) == text_only
    assert canonical_tensor_name("gemma3-12b", "vision_tower.weight") is None


def test_family_audit_reports_targets_independently(monkeypatch, tmp_path):
    family = FamilySpec("test", "Test", "source", ("target-a", "target-b"))
    names = {
        "source": {"a", "b", "c"},
        "target-a": {"a", "b"},
        "target-b": {"a", "d"},
    }
    calls = []

    def fake_names(key, **_):
        calls.append(key)
        return names[key]

    monkeypatch.setattr(family_suite, "checkpoint_names", fake_names)
    result = family_suite.audit_family(
        family,
        cache_dir=tmp_path,
        allow_download=False,
    )

    assert calls == ["source", "target-a", "target-b"]
    assert result["targets"][0]["matched_tensors"] == 2
    assert result["targets"][0]["missing_tensors"] == []
    assert result["targets"][1]["matched_tensors"] == 1
    assert result["targets"][1]["missing_tensors"] == ["d"]


def test_named_tensors_applies_gemma_scope_before_transfer():
    model = MappingModel(
        "gemma",
        {
            "language_model.model.weight": torch.ones(4, dtype=torch.bfloat16),
            "language_model.empty": torch.empty(0, dtype=torch.bfloat16),
            "vision_tower.weight": torch.ones(4, dtype=torch.bfloat16),
            "language_model.counter": torch.ones(1, dtype=torch.int64),
        },
        {},
    )

    tensors = family_suite.named_tensors(model, "gemma3-12b")

    assert [name for name, _ in tensors] == ["model.weight"]


def test_frozen_setup_detects_in_place_mode_mutation():
    tensor = torch.ones(32768, dtype=torch.bfloat16)
    codec, _ = family_suite.setup_neuralzip(
        [("layer.weight", tensor)],
        threads=2,
    )
    fingerprint = family_suite.frozen_setup_fingerprint(codec)
    first_mode = next(iter(codec._mode_by_tensor_stream.values()))
    first_mode["kind"] = "corrupted"

    with pytest.raises(RuntimeError, match="changed the reusable source setup"):
        family_suite.assert_frozen_setup(codec, fingerprint)


def test_run_family_builds_one_source_setup_for_multiple_targets(
    monkeypatch,
    tmp_path,
):
    family = FamilySpec(
        "test",
        "Test",
        "resnet34",
        ("resnet18", "resnet50"),
    )
    source = MappingModel(
        "source",
        {"weight": torch.ones(8, dtype=torch.bfloat16)},
        {"revision": "source-revision"},
    )

    class FakeCodec:
        pass

    codec = FakeCodec()
    setup_calls = []
    target_calls = []

    monkeypatch.setattr(family_suite, "load_model", lambda *_, **__: source)

    def fake_setup(tensors, *, threads):
        setup_calls.append((list(tensors), threads))
        return codec, {"setup_seconds": 1.0}

    def fake_target(_family, received_codec, _source_names, target_key, **kwargs):
        target_calls.append((target_key, received_codec, kwargs["setup_fingerprint"]))
        return {"target_key": target_key}

    monkeypatch.setattr(family_suite, "setup_neuralzip", fake_setup)
    monkeypatch.setattr(family_suite, "frozen_setup_fingerprint", lambda _: "frozen")
    monkeypatch.setattr(family_suite, "benchmark_target", fake_target)
    monkeypatch.setattr(family_suite, "host_provenance", dict)
    monkeypatch.setattr(family_suite, "code_provenance", dict)

    result = family_suite.run_family(
        family,
        cache_dir=tmp_path,
        allow_download=False,
        threads=12,
    )

    assert len(setup_calls) == 1
    assert [target for target, _, _ in target_calls] == ["resnet18", "resnet50"]
    assert all(received is codec for _, received, _ in target_calls)
    assert all(fingerprint == "frozen" for _, _, fingerprint in target_calls)
    assert [row["target_key"] for row in result["targets"]] == [
        "resnet18",
        "resnet50",
    ]
    assert result["target_histograms_computed"] is False
    assert result["target_js_matching"] is False
    assert result["target_mode_probing"] is False


def test_quantized_nibble_backends_roundtrip_exactly():
    patterned = bytes(range(16)) * 4096
    repeated = b"\x00\x11\x22\x33" * 4096
    samples = (patterned + repeated, repeated + patterned)

    assert _bytes_from_nibbles(_nibble_stream(samples[0])) == samples[0]
    plan = _prepare_shared_plan(
        samples,
        raw_chunk_bytes=32 << 10,
        js_threshold=0.05,
        max_cluster_sample_bytes=1 << 20,
    )
    local = _encode_local(samples, raw_chunk_bytes=32 << 10, threads=2)
    shared = _encode_shared(
        samples,
        plan,
        raw_chunk_bytes=32 << 10,
        threads=2,
    )

    _verify_archive(local)
    _verify_archive(shared, shared_plan=plan)
    assert local.huffman_chunks > 0
    assert shared.huffman_chunks > 0
    assert local.compressed_bytes < sum(map(len, samples))
    assert shared.compressed_bytes < sum(map(len, samples))


def test_quantized_exact_selector_never_mixes_gguf_variants_or_bin_files():
    spec = {
        "model_id": "owner/latam",
        "artifact_selector": {
            "kind": "exact-files",
            "files": ("latamgpt-Q4_K_M.gguf",),
        },
    }
    fs = _FakeHfFileSystem(
        spec["model_id"],
        {
            "latamgpt-Q4_K_M.gguf": 100,
            "latamgpt-Q5_K_M.gguf": 120,
            "latamgpt-IQ2_M.gguf": 60,
            "training_args.bin": 10,
        },
    )

    selected = _hf_model_files(fs, spec, "a" * 40)

    assert [row["relative_path"] for row in selected] == ["latamgpt-Q4_K_M.gguf"]


def test_quantized_index_selects_only_authoritative_safetensors_shards():
    spec = {
        "model_id": "owner/model",
        "artifact_selector": {
            "kind": "safetensors-index",
            "index": "model.safetensors.index.json",
        },
    }
    index = {
        "weight_map": {
            "a.weight": "model-00001-of-00002.safetensors",
            "b.weight": "model-00002-of-00002.safetensors",
        }
    }
    fs = _FakeHfFileSystem(
        spec["model_id"],
        {
            "model.safetensors.index.json": 100,
            "model-00001-of-00002.safetensors": 1_000,
            "model-00002-of-00002.safetensors": 2_000,
            "alternate.safetensors": 3_000,
            "training_args.bin": 10,
        },
        {"model.safetensors.index.json": json.dumps(index).encode()},
    )

    selected = _hf_model_files(fs, spec, "b" * 40)

    assert [row["relative_path"] for row in selected] == [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]


def test_quantized_index_accepts_revision_qualified_hf_filesystem_paths():
    spec = {
        "model_id": "owner/model",
        "artifact_selector": {
            "kind": "safetensors-index",
            "index": "model.safetensors.index.json",
        },
    }
    index = {"weight_map": {"a.weight": "model.safetensors"}}
    fs = _FakeHfFileSystem(
        spec["model_id"],
        {
            "model.safetensors.index.json": 100,
            "model.safetensors": 1_000,
        },
        {"model.safetensors.index.json": json.dumps(index).encode()},
        revision_in_path=True,
    )

    selected = _hf_model_files(fs, spec, "d" * 40)

    assert [row["relative_path"] for row in selected] == ["model.safetensors"]


def test_quantized_index_rejects_non_safetensors_weight_artifact():
    spec = {
        "model_id": "owner/model",
        "artifact_selector": {
            "kind": "safetensors-index",
            "index": "model.safetensors.index.json",
        },
    }
    index = {"weight_map": {"a.weight": "pytorch_model.bin"}}
    fs = _FakeHfFileSystem(
        spec["model_id"],
        {
            "model.safetensors.index.json": 100,
            "pytorch_model.bin": 1_000,
        },
        {"model.safetensors.index.json": json.dumps(index).encode()},
    )

    with pytest.raises(ValueError, match="allowed suffix"):
        _hf_model_files(fs, spec, "c" * 40)


def test_quantized_index_rejects_an_empty_remote_shard():
    spec = {
        "model_id": "owner/model",
        "artifact_selector": {
            "kind": "safetensors-index",
            "index": "model.safetensors.index.json",
        },
    }
    index = {"weight_map": {"a.weight": "model.safetensors"}}
    fs = _FakeHfFileSystem(
        spec["model_id"],
        {
            "model.safetensors.index.json": 100,
            "model.safetensors": 0,
        },
        {"model.safetensors.index.json": json.dumps(index).encode()},
    )

    with pytest.raises(ValueError, match="references empty shard"):
        _hf_model_files(fs, spec, "c" * 40)


def test_every_quantized_model_has_an_explicit_artifact_selector():
    quantized = QUANTIZED_MODEL_SPECS
    assert quantized
    assert all(spec.get("artifact_selector") for spec in quantized)
    latam = next(spec for spec in quantized if "LatamGPT" in spec["name"])
    assert latam["artifact_selector"] == {
        "kind": "exact-files",
        "files": ("latamgpt-Q4_K_M.gguf",),
    }


def test_quantized_suite_has_versioned_format_and_distillation_coverage():
    quantized = QUANTIZED_MODEL_SPECS
    distilled = [spec for spec in MODEL_SPECS if "distilled" in spec.suites]

    assert len(quantized) == 6
    assert {spec["representation"] for spec in quantized} >= {
        "4-bit quantized",
        "4-bit AWQ",
        "4-bit GPTQ",
    }
    assert len(distilled) == 4
    assert {
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    } <= {spec.source for spec in distilled}
    assert not any("adapter" in spec.source.lower() for spec in distilled)
