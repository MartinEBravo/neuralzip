import json

import pytest
import torch
from safetensors import safe_open

from neuralzip.gpu_resident import (
    INDEX_FILENAME,
    ResidentStateError,
    iter_resident_shards,
    load_resident_state,
    save_resident_state,
)


def _canonical_write(path, value):
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


def test_roundtrip_deduplicates_only_exact_storage_aliases(tmp_path):
    tied = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
    same_values = tied.clone()
    state = {
        "head.weight": tied.detach(),
        "model.embed.weight": tied,
        "model.norm.weight": same_values,
    }

    manifest = save_resident_state(state, tmp_path)
    loaded = load_resident_state(tmp_path)
    shards = list(iter_resident_shards(loaded))

    assert manifest.index_path == tmp_path / INDEX_FILENAME
    assert manifest.aliases == {"model.embed.weight": "head.weight"}
    assert set(manifest.descriptors) == {"head.weight", "model.norm.weight"}
    assert manifest.tensor_names == (
        "head.weight",
        "model.embed.weight",
        "model.norm.weight",
    )
    assert loaded.index_sha256 == manifest.index_sha256
    assert len(shards) == 1
    path, tensors = shards[0]
    assert path == manifest.shard_paths[0]
    assert set(tensors) == set(manifest.descriptors)
    assert torch.equal(tensors["head.weight"], tied)
    assert torch.equal(tensors["model.norm.weight"], same_values)


def test_different_views_of_one_storage_are_not_aliases(tmp_path):
    storage = torch.arange(16, dtype=torch.float32)

    manifest = save_resident_state(
        {"left.weight": storage[:8], "right.weight": storage[8:]},
        tmp_path,
    )

    assert not manifest.aliases
    assert set(manifest.descriptors) == {"left.weight", "right.weight"}


def test_empty_views_of_the_same_storage_are_aliases(tmp_path):
    tensor = torch.empty(0, dtype=torch.float32)

    manifest = save_resident_state(
        {"first.weight": tensor, "second.weight": tensor.detach()},
        tmp_path,
    )

    assert manifest.aliases == {"second.weight": "first.weight"}
    assert set(manifest.descriptors) == {"first.weight"}


def test_sharding_is_deterministic_and_iteration_is_one_shard_at_a_time(tmp_path):
    state = {
        f"layers.{index}.weight": torch.full((8,), index, dtype=torch.float32)
        for index in range(3)
    }

    manifest = save_resident_state(state, tmp_path, max_shard_size=40)

    assert [path.name for path in manifest.shard_paths] == [
        "neuralzip-resident-00001-of-00003.safetensors",
        "neuralzip-resident-00002-of-00003.safetensors",
        "neuralzip-resident-00003-of-00003.safetensors",
    ]
    iterator = iter_resident_shards(manifest)
    first_path, first = next(iterator)
    assert first_path == manifest.shard_paths[0]
    assert set(first) == {"layers.0.weight"}
    assert set(next(iterator)[1]) == {"layers.1.weight"}
    assert set(next(iterator)[1]) == {"layers.2.weight"}
    with pytest.raises(StopIteration):
        next(iterator)


def test_empty_state_still_has_a_canonical_index(tmp_path):
    manifest = save_resident_state({}, tmp_path, max_shard_size="1KiB")

    assert manifest.index_path.is_file()
    assert manifest.shard_paths == ()
    assert not manifest.descriptors
    assert not manifest.aliases
    assert manifest.total_size == 0
    assert list(iter_resident_shards(manifest)) == []
    source = manifest.index_path.read_text(encoding="ascii")
    assert (
        source
        == json.dumps(
            json.loads(source),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    )


def test_shard_payload_corruption_is_detected(tmp_path):
    manifest = save_resident_state(
        {"layer.weight": torch.arange(32, dtype=torch.uint8)},
        tmp_path,
    )
    shard = manifest.shard_paths[0]
    payload = bytearray(shard.read_bytes())
    payload[-1] ^= 1
    shard.write_bytes(payload)

    with pytest.raises(ResidentStateError, match="SHA-256"):
        load_resident_state(tmp_path)
    with pytest.raises(ResidentStateError, match="changed"):
        next(iter_resident_shards(manifest))


def test_index_schema_and_canonical_encoding_are_strict(tmp_path):
    manifest = save_resident_state(
        {"layer.weight": torch.ones(2, dtype=torch.float32)},
        tmp_path,
    )
    index = json.loads(manifest.index_path.read_text(encoding="ascii"))
    manifest.index_path.write_text(json.dumps(index, indent=2), encoding="ascii")
    with pytest.raises(ResidentStateError, match="not canonical"):
        load_resident_state(tmp_path)

    index["unexpected"] = True
    _canonical_write(manifest.index_path, index)
    with pytest.raises(ResidentStateError, match="unexpected"):
        load_resident_state(tmp_path)


def test_forged_alias_and_tensor_shape_are_rejected(tmp_path):
    manifest = save_resident_state(
        {"layer.weight": torch.ones((2, 3), dtype=torch.float32)},
        tmp_path,
    )
    index = json.loads(manifest.index_path.read_text(encoding="ascii"))
    index["aliases"] = {"other.weight": "missing.weight"}
    _canonical_write(manifest.index_path, index)
    with pytest.raises(ResidentStateError, match="Alias"):
        load_resident_state(tmp_path)

    index["aliases"] = {}
    index["tensors"]["layer.weight"]["shape"] = [3, 3]
    _canonical_write(manifest.index_path, index)
    with pytest.raises(ResidentStateError, match="byte size"):
        load_resident_state(tmp_path)


@pytest.mark.parametrize(
    ("value", "expected_shards"),
    [("64B", 1), ("0.064KB", 1), ("0.0625KiB", 1), (64, 1)],
)
def test_max_shard_size_parser_accepts_decimal_and_binary_units(
    tmp_path, value, expected_shards
):
    manifest = save_resident_state(
        {"layer.weight": torch.zeros(64, dtype=torch.uint8)},
        tmp_path / str(value).replace("/", "-"),
        max_shard_size=value,
    )
    assert len(manifest.shard_paths) == expected_shards


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "", "12XB", "0B"])
def test_invalid_max_shard_sizes_are_rejected(tmp_path, value):
    with pytest.raises((TypeError, ValueError)):
        save_resident_state({}, tmp_path / "state", max_shard_size=value)


def test_invalid_state_and_paths_are_rejected(tmp_path):
    with pytest.raises(ResidentStateError, match="non-empty string"):
        save_resident_state({1: torch.ones(1)}, tmp_path / "integer-name")
    with pytest.raises(ResidentStateError, match="dotted tensor path"):
        save_resident_state({"../weight": torch.ones(1)}, tmp_path / "bad-name")
    with pytest.raises(ResidentStateError, match="cannot be meta"):
        save_resident_state(
            {"layer.weight": torch.empty(1, device="meta")},
            tmp_path / "meta",
        )
    with pytest.raises(TypeError, match="torch.Tensor"):
        save_resident_state({"layer.weight": object()}, tmp_path / "not-tensor")


def test_shard_headers_match_the_index(tmp_path):
    manifest = save_resident_state(
        {"layer.weight": torch.ones((2, 3), dtype=torch.bfloat16)},
        tmp_path,
    )
    with safe_open(
        str(manifest.shard_paths[0]), framework="pt", device="cpu"
    ) as archive:
        assert archive.metadata() == {
            "format": "neuralzip-resident-state",
            "version": "1",
        }
        assert set(archive.keys()) == {"layer.weight"}
