import pytest
import torch

from neuralzip.checkpoint_loading import (
    load_torch_checkpoint,
    resolve_weight_map,
)


def test_weight_map_resolver_is_strict_and_format_agnostic():
    resolved = resolve_weight_map(
        {
            "b.weight": "nested/model-00002-of-00002.bin",
            "a.weight": "nested/model-00001-of-00002.bin",
        },
        available_files={
            "nested/model-00001-of-00002.bin": 10,
            "nested/model-00002-of-00002.bin": 20,
            "unrelated.bin": 30,
        },
        allowed_suffixes=(".bin",),
    )

    assert resolved.files == (
        "nested/model-00001-of-00002.bin",
        "nested/model-00002-of-00002.bin",
    )
    assert resolved.expected_keys == frozenset({"a.weight", "b.weight"})

    with pytest.raises(ValueError, match="Unsafe checkpoint shard path"):
        resolve_weight_map(
            {"weight": "../outside.bin"},
            available_files={"../outside.bin": 10},
            allowed_suffixes=(".bin",),
        )
    with pytest.raises(ValueError, match="allowed suffix"):
        resolve_weight_map(
            {"weight": "model.pt"},
            available_files={"model.pt": 10},
            allowed_suffixes=(".bin",),
        )
    with pytest.raises(ValueError, match="empty shard"):
        resolve_weight_map(
            {"weight": "model.bin"},
            available_files={"model.bin": 0},
            allowed_suffixes=(".bin",),
        )


def test_torch_checkpoint_loader_rejects_non_mapping_payload(tmp_path):
    valid = tmp_path / "valid.pt"
    invalid = tmp_path / "invalid.pt"
    torch.save({"epoch": 1}, valid)
    torch.save(torch.arange(3), invalid)

    assert load_torch_checkpoint(valid)["epoch"] == 1
    with pytest.raises(TypeError, match="not a mapping"):
        load_torch_checkpoint(invalid)
