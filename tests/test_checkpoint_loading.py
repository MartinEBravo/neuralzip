import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from neuralzip.checkpoint_loading import (
    download_safetensors_snapshot,
    iter_safetensors_shards,
    load_torch_checkpoint,
    select_safetensors_weights,
)
from neuralzip.model_experiments import _load_safetensors_mapping


def _indexed_snapshot(path: Path) -> None:
    index = {
        "weight_map": {
            "a.weight": "model-00001-of-00002.safetensors",
            "b.weight": "model-00002-of-00002.safetensors",
        }
    }
    (path / "model.safetensors.index.json").write_text(json.dumps(index))
    save_file(
        {"a.weight": torch.arange(3, dtype=torch.float32)},
        path / "model-00001-of-00002.safetensors",
    )
    save_file(
        {"b.weight": torch.arange(4, dtype=torch.float32)},
        path / "model-00002-of-00002.safetensors",
    )


def test_index_is_authoritative_and_shards_are_validated(tmp_path):
    _indexed_snapshot(tmp_path)
    save_file({"alternate": torch.ones(1)}, tmp_path / "alternate.safetensors")

    selection = select_safetensors_weights(tmp_path)
    loaded = {
        name
        for _, state in iter_safetensors_shards(selection)
        for name in state
    }

    assert [path.name for path in selection.files] == [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]
    assert loaded == {"a.weight", "b.weight"}


def test_snapshot_selection_rejects_unsafe_or_missing_index_shards(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "../outside.safetensors"}})
    )
    with pytest.raises(ValueError, match="Unsafe checkpoint shard path"):
        select_safetensors_weights(tmp_path)

    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "missing.safetensors"}})
    )
    with pytest.raises(FileNotFoundError, match="missing or empty shards"):
        select_safetensors_weights(tmp_path)


def test_snapshot_download_uses_one_strict_selector(tmp_path, monkeypatch):
    _indexed_snapshot(tmp_path)
    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        return str(tmp_path)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    snapshot, selection = download_safetensors_snapshot(
        "owner/model",
        revision="a" * 40,
        local_files_only=False,
    )

    assert snapshot == tmp_path
    assert len(selection.files) == 2
    assert calls == [
        {
            "repo_id": "owner/model",
            "local_files_only": False,
            "allow_patterns": (
                "*.safetensors",
                "*.safetensors.index.json",
            ),
            "revision": "a" * 40,
        }
    ]


def test_model_mapping_loader_uses_authoritative_selection(tmp_path, monkeypatch):
    _indexed_snapshot(tmp_path)
    save_file({"alternate": torch.ones(1)}, tmp_path / "alternate.safetensors")
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda **_kwargs: str(tmp_path),
    )

    tensors = _load_safetensors_mapping(
        "owner/model",
        dtype=torch.bfloat16,
        local_files_only=True,
    )

    assert set(tensors) == {"a.weight", "b.weight"}
    assert all(tensor.dtype == torch.bfloat16 for tensor in tensors.values())


def test_torch_checkpoint_loader_rejects_non_mapping_payload(tmp_path):
    valid = tmp_path / "valid.pt"
    invalid = tmp_path / "invalid.pt"
    torch.save({"epoch": 1}, valid)
    torch.save(torch.arange(3), invalid)

    assert load_torch_checkpoint(valid)["epoch"] == 1
    with pytest.raises(TypeError, match="not a mapping"):
        load_torch_checkpoint(invalid)
