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


class _TiedToyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(16, 8)
        self.output = torch.nn.Linear(8, 16, bias=False)
        self.output.weight = self.embedding.weight
        self.register_buffer("counter", torch.tensor(0, dtype=torch.int64))


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
