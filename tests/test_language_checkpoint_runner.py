import sys
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from examples.run_final_language_checkpoint_suite import (
    MODEL_SPECS,
    apply_mlm_mask,
    build_model,
    checkpoint_unique_state,
    expand_storage_aliases,
    smoke_token_blocks,
    unique_cpu_model_state,
)
import examples.checkpoint_common as checkpoint_common
import examples.checkpoint_compression as checkpoint_compression
import examples.run_final_checkpoint_suite as vision_runner
import examples.run_final_language_checkpoint_suite as language_runner
from neuralzip import experiment_io


class _TiedToyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(16, 8)
        self.output = torch.nn.Linear(8, 16, bias=False)
        self.output.weight = self.embedding.weight
        self.register_buffer("counter", torch.tensor(0, dtype=torch.int64))


def test_checkpoint_runners_share_task_agnostic_infrastructure():
    shared_names = (
        "atomic_torch_save",
        "checkpoint_files",
        "limited",
        "read_csv",
        "restore_rng_state",
        "rng_state",
        "set_seed",
    )

    for name in shared_names:
        common_function = getattr(checkpoint_common, name)
        assert getattr(vision_runner, name) is common_function
        assert getattr(language_runner, name) is common_function
    assert vision_runner.atomic_write_json is experiment_io.atomic_write_json
    assert language_runner.atomic_write_json is experiment_io.atomic_write_json


def test_checkpoint_runners_share_one_compression_protocol():
    shared_names = (
        "benchmark_neuralzip",
        "benchmark_zipnn_src",
        "load_codec_plan",
        "nonfloating_state",
        "save_codec_plan",
        "setup_neuralzip",
        "tensor_digest",
    )

    for name in shared_names:
        common_function = getattr(checkpoint_compression, name)
        assert getattr(vision_runner, name) is common_function
        assert getattr(language_runner, name) is common_function
    assert vision_runner.RESULT_FIELDS is checkpoint_compression.BASE_RESULT_FIELDS


def test_checkpoint_runner_upserts_preserve_schemas_keys_and_csv_dialects(tmp_path):
    vision_results = tmp_path / "vision-results.csv"
    vision_training = tmp_path / "vision-training.csv"
    language_results = tmp_path / "language-results.csv"
    language_training = tmp_path / "language-training.csv"

    for compressor, value in (("ZipNN src", 1), ("NeuralZip", 2), ("ZipNN src", 3)):
        vision_runner.upsert_result(
            vision_results,
            {
                "model": "resnet18",
                "epoch": 0,
                "compressor": compressor,
                "raw_bytes": value,
            },
        )
    vision_runner.upsert_training(
        vision_training,
        {"model": "resnet18", "epoch": 0, "train_loss": 1.0},
    )
    vision_runner.upsert_training(
        vision_training,
        {"model": "resnet18", "epoch": 0, "train_loss": 0.5},
    )

    language_runner.upsert_result(
        language_results,
        {"model": "gpt2", "epoch": 0, "compressor": "NeuralZip"},
    )
    language_runner.upsert_training(
        language_training,
        {"model": "gpt2", "epoch": 0, "train_loss": 1.0},
    )

    assert vision_results.read_bytes().splitlines()[0] == ",".join(
        vision_runner.RESULT_FIELDS
    ).encode()
    assert vision_training.read_bytes().splitlines()[0] == ",".join(
        vision_runner.TRAINING_FIELDS
    ).encode()
    assert language_results.read_bytes().splitlines()[0] == ",".join(
        language_runner.RESULT_FIELDS
    ).encode()
    assert language_training.read_bytes().splitlines()[0] == ",".join(
        language_runner.TRAINING_FIELDS
    ).encode()
    assert b"\r\n" in vision_results.read_bytes()
    assert b"\r\n" not in language_results.read_bytes()

    vision_rows = checkpoint_common.read_csv(vision_results)
    assert [(row["compressor"], row["raw_bytes"]) for row in vision_rows] == [
        ("NeuralZip", "2"),
        ("ZipNN src", "3"),
    ]
    assert checkpoint_common.read_csv(vision_training)[0]["train_loss"] == "0.5"


def test_vision_checkpoint_resume_rejects_legacy_encoder_rows(tmp_path):
    result_path = tmp_path / "results.csv"
    base = {
        "model": "resnet18",
        "epoch": 0,
        "compressor": "NeuralZip",
        "status": "ok",
        "roundtrip_exact": True,
        "input_unchanged": True,
    }
    vision_runner.upsert_result(result_path, base)
    assert vision_runner.completed_methods(result_path, "resnet18", 0) == set()
    assert not checkpoint_compression.is_current_codec_result(
        checkpoint_common.read_csv(result_path)[0]
    )

    vision_runner.upsert_result(
        result_path,
        {
            **base,
            "codec_protocol_version": checkpoint_compression.CODEC_PROTOCOL_VERSION,
            "encoder_path": "compress_tensors",
        },
    )
    assert vision_runner.completed_methods(result_path, "resnet18", 0) == {
        "NeuralZip"
    }
    assert checkpoint_compression.is_current_codec_result(
        checkpoint_common.read_csv(result_path)[0]
    )


def test_unique_state_records_and_reconstructs_tied_weight_aliases():
    model = _TiedToyModel()

    canonical, aliases = unique_cpu_model_state(model)

    assert len(canonical) == 2
    assert len(aliases) == 1
    assert aliases["output.weight"] == "embedding.weight"
    reconstructed = expand_storage_aliases(canonical, aliases)
    assert set(reconstructed) == set(model.state_dict())
    assert reconstructed["output.weight"] is reconstructed["embedding.weight"]

    restored = _TiedToyModel()
    restored.load_state_dict(reconstructed, strict=True)
    assert torch.equal(restored.embedding.weight, model.embedding.weight)


def test_checkpoint_unique_state_keeps_aliases_out_of_compressed_mapping():
    model = _TiedToyModel()
    canonical, aliases = unique_cpu_model_state(model)

    loaded, loaded_aliases = checkpoint_unique_state(
        {"model_state": canonical, "storage_aliases": aliases}
    )

    assert set(loaded) == {"embedding.weight", "counter"}
    assert loaded_aliases == {"output.weight": "embedding.weight"}


def test_mlm_mask_never_selects_special_tokens_and_uses_ignore_index():
    values = torch.arange(200, dtype=torch.long).remainder(128).reshape(10, 20)
    special = torch.zeros_like(values, dtype=torch.bool)
    special[:, 0] = True
    special[:, -1] = True

    masked, labels = apply_mlm_mask(
        values,
        special,
        mask_token_id=4,
        vocab_size=128,
        generator=torch.Generator().manual_seed(7),
    )

    assert labels[special].eq(-100).all()
    assert torch.equal(masked[special], values[special])
    selected = labels.ne(-100).sum().item()
    assert 10 <= selected <= 45


def test_smoke_models_use_only_synthetic_token_blocks_and_keep_ties():
    for slug in ("gpt2", "bert-base"):
        spec = MODEL_SPECS[slug]
        dataset = smoke_token_blocks(spec, seed=2026, train=True)
        model = build_model(spec, smoke=True)
        canonical, aliases = unique_cpu_model_state(model)

        assert len(dataset) == 8
        assert dataset[0]["input_ids"].shape == (16,)
        assert canonical
        assert aliases
