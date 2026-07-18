import io
import json
import sys
from pathlib import Path

import pytest
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from examples.config import LOSSY_LOSSLESS_MODEL_SUITE
from examples.run_precompute_family_transfer import (
    floating_parameters,
    select_snapshot_weights,
)
from examples.run_quantized_nibble_compression import (
    _bytes_from_nibbles,
    _encode_local,
    _encode_shared,
    _hf_model_files,
    _nibble_stream,
    _prepare_shared_plan,
    _verify_archive,
)


class _FakeHfFileSystem:
    def __init__(self, model_id, files, contents=None):
        self.model_id = model_id
        self.files = dict(files)
        self.contents = dict(contents or {})

    def ls(self, model_id, *, detail, revision):
        assert model_id == self.model_id
        assert detail is True
        assert revision
        return [
            {
                "type": "file",
                "name": f"{model_id}/{path}",
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

    state = floating_parameters(model)
    names = [name for name, _ in state]

    assert len(names) == len(set(names))
    assert "1.running_mean" in names
    assert "1.running_var" in names
    assert "1.num_batches_tracked" not in names


def test_family_snapshot_rejects_an_index_with_a_missing_shard(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    index = {
        "weight_map": {
            "a.weight": "model-00001-of-00002.safetensors",
            "b.weight": "model-00002-of-00002.safetensors",
        }
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"first")

    with pytest.raises(FileNotFoundError, match="missing or empty shards"):
        select_snapshot_weights(tmp_path)


def test_family_snapshot_uses_only_shards_declared_by_the_index(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    index = {
        "weight_map": {
            "a.weight": "model-00001-of-00002.safetensors",
            "b.weight": "model-00002-of-00002.safetensors",
        }
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    for name in (
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "alternate.safetensors",
    ):
        (tmp_path / name).write_bytes(name.encode())

    selection = select_snapshot_weights(tmp_path)

    assert [path.name for path in selection.files] == [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]
    assert selection.expected_keys == frozenset({"a.weight", "b.weight"})


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

    assert [row["relative_path"] for row in selected] == [
        "latamgpt-Q4_K_M.gguf"
    ]


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

    with pytest.raises(ValueError, match="safetensors or GGUF"):
        _hf_model_files(fs, spec, "c" * 40)


def test_every_quantized_model_has_an_explicit_artifact_selector():
    quantized = [
        spec
        for spec in LOSSY_LOSSLESS_MODEL_SUITE
        if spec.get("category") == "quantized"
    ]
    assert quantized
    assert all(spec.get("artifact_selector") for spec in quantized)
    latam = next(spec for spec in quantized if "LatamGPT" in spec["name"])
    assert latam["artifact_selector"] == {
        "kind": "exact-files",
        "files": ("latamgpt-Q4_K_M.gguf",),
    }
