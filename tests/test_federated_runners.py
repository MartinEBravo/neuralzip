import gc
import struct
import sys
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from examples import federated_common
from examples import run_federated_language as language_runner
from examples import run_federated_vision as vision_runner
from examples.run_federated_language import (
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

SuspendAutomaticGC = federated_common.SuspendAutomaticGC


def _zipnn_fp32_frame(*, byte_reorder: int = 220, bit_reorder: int = 1) -> bytes:
    original_bytes = 128
    chunk_exponent = 5
    stream_totals = (5, 7, 9, 17)
    chunk_count = original_bytes // (1 << chunk_exponent)

    header = bytearray(32)
    header[:2] = b"ZN"
    header[5] = int(byte_reorder)
    header[6] = int(bit_reorder)
    header[8] = 2
    header[14] = chunk_exponent
    header[16:24] = original_bytes.to_bytes(8, "little")
    shape = bytes((1, 2, 4, 0))
    types = bytes(4 * chunk_count)
    cumulative_sizes = b"".join(
        ((total * (chunk + 1)) // chunk_count).to_bytes(
            struct.calcsize("P"), sys.byteorder
        )
        for total in stream_totals
        for chunk in range(chunk_count)
    )
    payload = header + shape + types + cumulative_sizes + bytes(sum(stream_totals))
    payload[24:32] = len(payload).to_bytes(8, "little")
    return bytes(payload)


class _TiedToyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(16, 8)
        self.output = torch.nn.Linear(8, 16, bias=False)
        self.output.weight = self.embedding.weight
        self.register_buffer("counter", torch.tensor(0, dtype=torch.int64))


def test_federated_runners_share_one_codec_and_reporting_implementation():
    shared_names = (
        "COMPRESSOR_NEURALZIP",
        "COMPRESSOR_SRC",
        "DTYPE_NAME",
        "FederatedCodecProtocol",
        "_write_csv_atomic",
        "_write_json_atomic",
        "build_round_rows",
        "build_summary_rows",
        "new_cumulative_metrics",
    )
    for name in shared_names:
        shared = getattr(federated_common, name)
        assert getattr(vision_runner, name) is shared
        assert getattr(language_runner, name) is shared


def test_federated_zipnn_wrapper_preserves_fp32_exponent_accounting():
    raw_bytes, compressed_bytes = (
        federated_common.ZipNNSrcStateCodec._fp32_exponent_sizes(_zipnn_fp32_frame())
    )

    assert raw_bytes == 32
    assert compressed_bytes == 17 + 4 * (1 + struct.calcsize("P"))


def test_federated_zipnn_wrapper_rejects_non_fp32_reorder_layouts():
    for payload in (
        _zipnn_fp32_frame(byte_reorder=10),
        _zipnn_fp32_frame(bit_reorder=0),
    ):
        try:
            federated_common.ZipNNSrcStateCodec._fp32_exponent_sizes(payload)
        except ValueError as error:
            assert "FP32 byte/bit-reorder layout" in str(error)
        else:
            raise AssertionError("The federated ZipNN adapter accepted a wrong layout.")


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


class _IdentityStateCodec:
    def __init__(self, encode_seconds: float):
        self.encode_seconds = encode_seconds

    def encode(self, tensors):
        return federated_common.EncodedState(
            payloads={name: tensor.clone() for name, tensor in tensors.items()},
            encode_seconds=self.encode_seconds,
            exponent_raw_bytes=100,
            exponent_compressed_bytes=40,
            archive_bytes=400,
            raw_floating_bytes=400,
        )

    def decode(self, encoded):
        return {name: tensor.clone() for name, tensor in encoded.payloads.items()}


def _identity_protocol():
    protocol = federated_common.FederatedCodecProtocol.__new__(
        federated_common.FederatedCodecProtocol
    )
    protocol.codecs = {
        federated_common.COMPRESSOR_SRC: _IdentityStateCodec(0.2),
        federated_common.COMPRESSOR_NEURALZIP: _IdentityStateCodec(0.1),
    }
    protocol.setup_seconds = {
        federated_common.COMPRESSOR_SRC: 0.0,
        federated_common.COMPRESSOR_NEURALZIP: 3.0,
    }
    protocol.setup_bytes = {
        federated_common.COMPRESSOR_SRC: 0,
        federated_common.COMPRESSOR_NEURALZIP: 8,
    }
    return protocol


def test_shared_federated_protocol_alternates_and_discards_verified_payloads():
    protocol = _identity_protocol()
    tensors = {"weight": torch.arange(8, dtype=torch.float32)}

    assert protocol.encoding_order(1) == (
        federated_common.COMPRESSOR_SRC,
        federated_common.COMPRESSOR_NEURALZIP,
    )
    assert protocol.encoding_order(2) == (
        federated_common.COMPRESSOR_NEURALZIP,
        federated_common.COMPRESSOR_SRC,
    )
    assert protocol.encoding_order(2, client_index=1) == protocol.encoding_order(1)

    measurements, decoded = protocol.encode_verified(
        tensors,
        order=protocol.encoding_order(1),
        context="toy round 1",
        phase="uplink",
    )

    assert torch.equal(decoded["weight"], tensors["weight"])
    assert all(not encoded.payloads for encoded in measurements.values())


def test_shared_federated_rows_use_one_accumulator_and_setup_accounting():
    protocol = _identity_protocol()
    tensors = {"weight": torch.arange(8, dtype=torch.float32)}
    downlinks, _ = protocol.encode_verified(
        tensors,
        order=protocol.encoding_order(1),
        context="toy round 1",
        phase="downlink",
    )
    uplink, _ = protocol.encode_verified(
        tensors,
        order=protocol.encoding_order(1),
        context="toy round 1 client 0",
        phase="uplink",
    )
    cumulative = federated_common.new_cumulative_metrics()

    rows = federated_common.build_round_rows(
        prefix={"architecture": "toy", "dtype": "float32"},
        suffix={"roundtrip_ok": True},
        downlinks=downlinks,
        uplinks={name: [encoded] for name, encoded in uplink.items()},
        clients=1,
        cumulative=cumulative,
        setup_seconds=protocol.setup_seconds,
        setup_bytes=protocol.setup_bytes,
    )
    summaries = federated_common.build_summary_rows(
        prefix={"architecture": "toy"},
        dimensions={"completed_rounds": 1},
        clients=1,
        cumulative=cumulative,
        setup_seconds=protocol.setup_seconds,
        setup_bytes=protocol.setup_bytes,
    )

    assert [row["compressor"] for row in rows] == list(federated_common.COMPRESSORS)
    assert rows[0]["downlink_encode_seconds"] == 0.2
    assert rows[1]["downlink_encode_seconds"] == 0.1
    neuralzip = summaries[1]
    assert neuralzip["downlink_encode_seconds"] == 0.1
    assert neuralzip["exponent_compressed_bytes"] == 80
    assert neuralzip["exponent_compressed_plus_setup_bytes"] == 88
    assert neuralzip["exponent_ratio_with_setup_pct"] == 44.0
