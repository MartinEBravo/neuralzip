import gc
import sys
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from examples.run_final_federated_language_suite import (
    MODEL_BERT,
    MODEL_GPT2,
    ClientCorpus,
    TimingIsolation,
    TokenizerSpec,
    _bert_mask,
    _expand_canonical_state,
    _parse_cpu_set,
    _reconstruct_wikitext_documents,
    _snapshot_model_state,
    _state_layout,
    _synthetic_tokenizer_spec,
    _tokenize_partition,
    _unique_floating_state,
)
from examples.run_final_federated_suite import SuspendAutomaticGC


class _TiedToyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(16, 8)
        self.output = torch.nn.Linear(8, 16, bias=False)
        self.output.weight = self.embedding.weight
        self.register_buffer("counter", torch.tensor(0, dtype=torch.int64))


class _BertTokenizerWithoutLegacyBuilder:
    """Transformers 5.14-style contract used by the real BERT path."""

    cls_token_id = 101
    sep_token_id = 102

    def encode(self, document: str, *, add_special_tokens: bool):
        assert document == "document"
        assert not add_special_tokens
        return [11, 12, 13, 14, 15, 16, 17, 18]


def _bert_tokenizer_spec() -> TokenizerSpec:
    return TokenizerSpec(
        identifier="test",
        revision=None,
        vocab_sha256="test",
        vocab_size=128,
        special_token_ids=(0, 1, 2, 3),
        mask_token_id=3,
        tokenizer=None,
    )


def test_unique_storage_layout_elides_and_restores_tied_weight():
    model = _TiedToyModel()
    layout = _state_layout(model)
    state = _snapshot_model_state(model, layout)

    assert layout.aliases == {"output.weight": "embedding.weight"}
    assert state["output.weight"].data_ptr() == state["embedding.weight"].data_ptr()
    assert set(_unique_floating_state(state, layout)) == {"embedding.weight"}

    canonical = {name: state[name].clone() for name in layout.canonical_names}
    restored = _expand_canonical_state(canonical, layout)
    assert restored["output.weight"] is restored["embedding.weight"]


def test_wikitext_reconstruction_splits_only_top_level_headings():
    lines = [
        " = First article = ",
        "First paragraph.",
        " = = Section = = ",
        "Section paragraph.",
        " = Second article = ",
        "Second paragraph.",
    ]

    documents = _reconstruct_wikitext_documents(lines)

    assert len(documents) == 2
    assert "Section paragraph" in documents[0]
    assert documents[1].startswith(" = Second article = ".strip())


def test_bert_real_and_synthetic_tokenizer_contracts_do_not_need_legacy_builder():
    tokenizer = _BertTokenizerWithoutLegacyBuilder()
    assert not hasattr(tokenizer, "build_inputs_with_special_tokens")
    real_spec = TokenizerSpec(
        identifier="transformers-5.14-contract",
        revision=None,
        vocab_sha256="test",
        vocab_size=128,
        special_token_ids=(0, 101, 102, 103),
        mask_token_id=103,
        tokenizer=tokenizer,
    )

    sequences = _tokenize_partition(
        ["document"],
        model_name=MODEL_BERT,
        sequence_length=6,
        tokenizer_spec=real_spec,
    )

    assert sequences.dtype == torch.int32
    assert sequences.tolist() == [
        [101, 11, 12, 13, 14, 102],
        [101, 15, 16, 17, 18, 102],
    ]
    synthetic_spec = _synthetic_tokenizer_spec(MODEL_BERT)
    assert synthetic_spec.tokenizer is None
    assert {101, 102, 103}.issubset(synthetic_spec.special_token_ids)


def test_client_batches_are_deterministic_per_round_and_client():
    corpus = ClientCorpus(torch.arange(160).reshape(10, 16).to(torch.int32), 2)

    first = corpus.batch(
        batch_size=4,
        seed=2026,
        round_index=7,
        client_index=3,
        local_batch_index=0,
    )
    repeated = corpus.batch(
        batch_size=4,
        seed=2026,
        round_index=7,
        client_index=3,
        local_batch_index=0,
    )
    next_round = corpus.batch(
        batch_size=4,
        seed=2026,
        round_index=8,
        client_index=3,
        local_batch_index=0,
    )

    assert torch.equal(first, repeated)
    assert not torch.equal(first, next_round)


def test_bert_mask_is_deterministic_and_never_masks_special_tokens():
    input_ids = torch.arange(8 * 32, dtype=torch.long).remainder(124).add(4)
    input_ids = input_ids.reshape(8, 32)
    input_ids[:, 0] = 1
    input_ids[:, -1] = 2
    spec = _bert_tokenizer_spec()

    masked, labels = _bert_mask(input_ids, tokenizer_spec=spec, seed=17)
    repeated_masked, repeated_labels = _bert_mask(
        input_ids, tokenizer_spec=spec, seed=17
    )

    assert torch.equal(masked, repeated_masked)
    assert torch.equal(labels, repeated_labels)
    assert labels[:, 0].eq(-100).all()
    assert labels[:, -1].eq(-100).all()
    selected = int(labels.ne(-100).sum())
    assert 20 <= selected <= 60
    assert MODEL_BERT == "bert-base"
    assert MODEL_GPT2 == "gpt2"


def test_cpu_set_parser_and_disabled_timing_isolation():
    assert _parse_cpu_set("0-2,5,7-8") == (0, 1, 2, 5, 7, 8)
    isolation = TimingIsolation(
        lock_path=None,
        timing_cpus=None,
        background_cpus=None,
    )
    with isolation:
        assert not isolation.enabled


def test_encode_gc_guard_restores_the_callers_gc_state():
    gc.enable()
    with SuspendAutomaticGC():
        assert not gc.isenabled()
    assert gc.isenabled()

    gc.disable()
    try:
        with SuspendAutomaticGC():
            assert not gc.isenabled()
        assert not gc.isenabled()
    finally:
        gc.enable()
