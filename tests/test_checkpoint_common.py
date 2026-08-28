import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from examples.experiments.checkpoint_common import (
    atomic_torch_save,
    atomic_write_csv,
    checkpoint_files,
    find_complete_local_checkpoint_dir,
    find_external_checkpoint_dir,
    limited,
    read_csv,
    restore_rng_state,
    rng_state,
    set_seed,
    upsert_model_epoch_row,
)
from neuralzip.experiment_io import atomic_write_json


def test_upsert_preserves_identity_schema_and_stable_order(tmp_path):
    path = tmp_path / "results.csv"
    fields = ("model", "epoch", "compressor", "value")
    rows = (
        {"model": "second", "epoch": 1, "compressor": "ZipNN", "value": "b"},
        {"model": "first", "epoch": 2, "compressor": "NeuralZip", "value": "c"},
        {"model": "first", "epoch": 1, "compressor": "ZipNN", "value": "old"},
        {"model": "first", "epoch": 1, "compressor": "NeuralZip", "value": "a"},
        {"model": "first", "epoch": 1, "compressor": "ZipNN", "value": "new"},
    )

    for row in rows:
        upsert_model_epoch_row(
            path,
            row,
            fields=fields,
            model_order=("first", "second"),
            include_compressor=True,
            lineterminator="\n",
        )

    assert read_csv(path) == [
        {"model": "first", "epoch": "1", "compressor": "NeuralZip", "value": "a"},
        {"model": "first", "epoch": "1", "compressor": "ZipNN", "value": "new"},
        {"model": "first", "epoch": "2", "compressor": "NeuralZip", "value": "c"},
        {"model": "second", "epoch": "1", "compressor": "ZipNN", "value": "b"},
    ]
    assert b"\r\n" not in path.read_bytes()
    assert not path.with_suffix(".csv.tmp").exists()


def test_atomic_writers_preserve_default_csv_dialect_and_replace_temp_files(tmp_path):
    csv_path = tmp_path / "nested" / "rows.csv"
    json_path = tmp_path / "nested" / "value.json"
    torch_path = tmp_path / "nested" / "value.pt"

    atomic_write_csv(csv_path, ("name",), ({"name": "value"},))
    atomic_write_json(json_path, {"path": tmp_path})
    atomic_torch_save(torch_path, {"epoch": 3, "tensor": torch.arange(4)})

    assert csv_path.read_bytes() == b"name\r\nvalue\r\n"
    assert json_path.read_text().endswith("\n")
    loaded = torch.load(torch_path, map_location="cpu", weights_only=False)
    assert loaded["epoch"] == 3
    assert torch.equal(loaded["tensor"], torch.arange(4))
    assert not list((tmp_path / "nested").glob("*.tmp"))


def test_rng_state_roundtrip_restores_all_cpu_generators():
    original_generator = torch.Generator()
    original_state = rng_state(original_generator)
    loader_generator = torch.Generator().manual_seed(99)
    try:
        set_seed(17)
        saved = rng_state(loader_generator)
        expected = (
            random.random(),
            float(np.random.random()),
            float(torch.rand(())),
            float(torch.rand((), generator=loader_generator)),
        )

        set_seed(1234)
        torch.rand((), generator=loader_generator)
        restore_rng_state(saved, loader_generator)
        actual = (
            random.random(),
            float(np.random.random()),
            float(torch.rand(())),
            float(torch.rand((), generator=loader_generator)),
        )

        assert actual == expected
    finally:
        restore_rng_state(original_state, original_generator)


def test_checkpoint_discovery_filters_and_finds_complete_trajectories(tmp_path):
    external = tmp_path / "external" / "model" / "checkpoints"
    external.mkdir(parents=True)
    atomic_torch_save(external / "model_epoch_000.pt", {"epoch": 0})
    atomic_torch_save(external / "model_epoch_001.pt", {"epoch": 1})
    atomic_torch_save(external / "latest_training_state.pt", {"epoch": 99})
    atomic_torch_save(external / "broken.pt", {})

    assert checkpoint_files(external, 1) == {
        0: external / "model_epoch_000.pt",
        1: external / "model_epoch_001.pt",
    }
    assert find_external_checkpoint_dir(tmp_path / "external", "model", 1) == external
    assert find_external_checkpoint_dir(tmp_path / "external", "model", 2) is None

    local = tmp_path / "output" / "model" / "checkpoints"
    local.mkdir(parents=True)
    atomic_torch_save(local / "model_epoch_000.pt", {"epoch": 0})
    atomic_torch_save(local / "model_epoch_001.pt", {"epoch": 1})
    assert find_complete_local_checkpoint_dir(tmp_path / "output", "model", 1) == local


def test_limited_keeps_existing_dataset_or_returns_prefix_subset():
    dataset = TensorDataset(torch.arange(5))

    assert limited(dataset, None) is dataset
    assert limited(dataset, 5) is dataset
    assert [int(item[0]) for item in limited(dataset, 3)] == [0, 1, 2]
    assert len(limited(dataset, -2)) == 0
