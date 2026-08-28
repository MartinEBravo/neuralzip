from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from examples.experiments import run_dfloat11_comparison as runner


class _Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
        self.k_proj = nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
        self.v_proj = nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
        self.o_proj = nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)


class _MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
        self.up_proj = nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
        self.down_proj = nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _Attention()
        self.mlp = _MLP()


class _StandardModel(nn.Module):
    def __init__(self, blocks: int = 2) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(_Block() for _ in range(blocks))


class _NestedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.layers = nn.ModuleList([_Block()])


def _target_with_numel(numel: int) -> runner.TargetScope:
    weight = runner.TargetWeight(
        block_name="model.layers.0",
        module_name="model.layers.0.self_attn.q_proj",
        tensor_name="model.layers.0.self_attn.q_proj.weight",
        numel=numel,
        raw_bytes=2 * numel,
    )
    return runner.TargetScope(
        layout=runner.STANDARD_TEXT_LAYOUT,
        block_names=("model.layers.0",),
        weights=(weight,),
    )


def _suite_args(tmp_path):
    return SimpleNamespace(
        device="cuda:0",
        batch_size=1,
        sequence_length=128,
        warmup=20,
        repetitions=50,
        trials=5,
        threads=0,
        exponent_prior_count=1,
        max_codebook_sample_bytes=8 << 20,
        cache_dir=str(tmp_path / "cache"),
    )


def _orchestrator_args(tmp_path, models=("qwen25-05b",), *, force=False):
    args = _suite_args(tmp_path)
    args.output_jsonl = str(tmp_path / "results.jsonl")
    args.output_csv = str(tmp_path / "results.csv")
    args.suite_model = list(models)
    args.revision = None
    args.force_suite = force
    return args


def _synthetic_result(
    method: str,
    digest: str = "abc",
    resolved: runner.ResolvedModel | None = None,
) -> dict[str, object]:
    resolved = resolved or runner._resolve_model("qwen25-05b")
    result = {
        "method": method,
        "model": resolved.model_id,
        "revision": resolved.revision,
        "target_layout": runner.STANDARD_TEXT_LAYOUT.name,
        "target_block_pattern": runner.STANDARD_TEXT_LAYOUT.block_pattern,
        "target_block_names": ["model.layers.0"],
        "target_weight_names": ["model.layers.0.self_attn.q_proj.weight"],
        "target_block_count": 1,
        "target_weight_count": 1,
        "target_raw_bytes": 200,
        "target_raw_exponent_bytes": 100,
        "batch_size": 1,
        "sequence_length": 128,
        "use_cache": False,
        "warmup": 20,
        "repetitions": 50,
        "trials": 5,
        "exponent_ratio": 0.4,
        "exponent_ratio_pct": 40.0,
        "steady_gpu_allocated_bytes": 1024,
        "inference_ms": 2.0,
        "logit_sha256": digest,
    }
    if method == "neuralzip":
        result["post_setup_encode_seconds"] = 1.0
        result["bitwise_exact_logits"] = True
        result["reference_logit_sha256"] = digest
    elif method == "dfloat11":
        result["encoding_seconds"] = 2.0
        result["bitwise_exact_logits"] = True
        result["reference_logit_sha256"] = digest
    return result


def _fingerprint(digest: str = "fingerprint") -> dict[str, object]:
    return {"digest": digest, "code": "code", "runtime": {}, "config": {}}


def test_catalog_has_nine_public_defaults_and_optional_gated_models() -> None:
    assert len(runner.DEFAULT_SUITE_MODEL_KEYS) == 9
    assert len(set(runner.DEFAULT_SUITE_MODEL_KEYS)) == 9
    assert set(runner.DEFAULT_SUITE_MODEL_KEYS) <= set(runner.MODEL_CATALOG)
    assert "llama32-1b" in runner.MODEL_CATALOG
    assert "gemma3-1b" in runner.MODEL_CATALOG
    for key, spec in runner.MODEL_CATALOG.items():
        assert spec.key == key
        assert spec.revision is not None
        assert len(spec.revision) == 40
        assert spec.layouts[0].relative_modules == runner.SEVEN_LINEAR_MODULES


def test_resolve_catalog_key_and_custom_model() -> None:
    pinned = runner._resolve_model("qwen3-06b")
    assert pinned.model_id == "Qwen/Qwen3-0.6B"
    assert pinned.revision == "c1899de289a04d12100db370d81485cdf75e47ca"

    custom = runner._resolve_model("owner/custom", "revision")
    assert custom.model_id == "owner/custom"
    assert custom.revision == "revision"
    assert custom.layouts == runner.SUPPORTED_BLOCK_LAYOUTS


def test_target_scope_discovers_and_orders_standard_blocks() -> None:
    scope = runner._target_scope(_StandardModel())
    assert scope.layout == runner.STANDARD_TEXT_LAYOUT
    assert scope.block_names == ("model.layers.0", "model.layers.1")
    assert len(scope.weights) == 14
    assert scope.raw_bytes == 112
    assert scope.raw_exponent_bytes == 56
    assert scope.dfloat_pattern == {
        runner.STANDARD_TEXT_LAYOUT.block_pattern: list(runner.SEVEN_LINEAR_MODULES)
    }


def test_target_scope_supports_nested_text_stack() -> None:
    scope = runner._target_scope(_NestedModel())
    assert scope.layout == runner.NESTED_TEXT_LAYOUT
    assert scope.block_names == ("model.language_model.layers.0",)
    assert len(scope.weights) == 7


def test_target_scope_rejects_missing_linear() -> None:
    model = _StandardModel(blocks=1)
    del model.model.layers[0].mlp.down_proj
    with pytest.raises(RuntimeError, match="missing target mlp.down_proj"):
        runner._target_scope(model)


def test_target_scope_rejects_non_contiguous_block_indices() -> None:
    model = nn.Module()
    model.model = nn.Module()
    model.model.layers = nn.ModuleDict({"1": _Block()})
    with pytest.raises(RuntimeError, match="not contiguous from zero"):
        runner._target_scope(model)


def test_target_scope_rejects_ambiguous_stacks() -> None:
    model = _StandardModel(blocks=1)
    model.model.language_model = nn.Module()
    model.model.language_model.layers = nn.ModuleList([_Block()])
    with pytest.raises(RuntimeError, match="ambiguous"):
        runner._target_scope(model)


def test_neuralzip_exponent_ratio_excludes_sign_mantissa() -> None:
    exported = SimpleNamespace(
        groups=(
            SimpleNamespace(exponent_bytes=20, metadata_bytes=3),
            SimpleNamespace(exponent_bytes=10, metadata_bytes=2),
        ),
        codebook_bytes=5,
        resident_raw_bytes=20,
    )
    metrics = runner._neuralzip_exponent_metrics(exported, _target_with_numel(100))
    assert metrics == {
        "raw_exponent_bytes": 100,
        "exponent_stream_bytes": 30,
        "exponent_metadata_bytes": 5,
        "exponent_codebook_bytes": 5,
        "raw_fallback_exponent_bytes": 10,
        "compressed_exponent_bytes": 50,
        "exponent_ratio": 0.5,
        "exponent_ratio_pct": 50.0,
    }


def test_dfloat_exponent_ratio_excludes_sign_mantissa_and_host_metadata() -> None:
    scope = runner.DFloatScope(
        raw_bytes=200,
        exponent_stream_bytes=30,
        sign_mantissa_bytes=100,
        exponent_metadata_bytes=5,
        exponent_codebook_bytes=5,
        split_positions_host_bytes=800,
        blocks=1,
    )
    assert scope.raw_exponent_bytes == 100
    assert scope.exponent_compressed_bytes == 40
    assert scope.exponent_ratio == pytest.approx(0.4)
    assert scope.compressed_bytes == 140


def test_child_command_preserves_pinned_model_and_separate_reference(tmp_path) -> None:
    args = _suite_args(tmp_path)
    resolved = runner._resolve_model("qwen25-05b")
    command = runner._child_command(
        args,
        "neuralzip",
        resolved,
        tmp_path / "result.json",
        reference_logit_sha256="digest",
    )
    assert command[2] == "neuralzip"
    assert command[command.index("--model") + 1] == "qwen25-05b"
    assert command[command.index("--revision") + 1] == resolved.revision
    assert command[command.index("--reference-logit-sha256") + 1] == "digest"
    assert "--dfloat-directory" not in command


def test_suite_validation_builds_one_complete_row(tmp_path) -> None:
    args = _suite_args(tmp_path)
    resolved = runner._resolve_model("qwen25-05b")
    configuration = runner._suite_configuration(args)
    row = runner._validate_suite_results(
        resolved,
        configuration,
        _synthetic_result("bf16"),
        _synthetic_result("neuralzip"),
        _synthetic_result("dfloat11"),
        fingerprint=_fingerprint(),
    )
    assert row["status"] == "complete"
    assert row["bitwise_exact_logits"] is True
    assert row["metrics"]["neuralzip"]["encoding_seconds"] == 1.0
    assert row["metrics"]["dfloat11"]["encoding_seconds"] == 2.0
    assert (
        runner._complete_row_for([row], resolved, configuration, "fingerprint") is row
    )


def test_suite_validation_rejects_digest_or_scope_mismatch(tmp_path) -> None:
    args = _suite_args(tmp_path)
    resolved = runner._resolve_model("qwen25-05b")
    configuration = runner._suite_configuration(args)
    dfloat = _synthetic_result("dfloat11", digest="different")
    with pytest.raises(RuntimeError, match="bitwise exact"):
        runner._validate_suite_results(
            resolved,
            configuration,
            _synthetic_result("bf16"),
            _synthetic_result("neuralzip"),
            dfloat,
            fingerprint=_fingerprint(),
        )


def test_jsonl_and_csv_are_restart_safe(tmp_path) -> None:
    args = _suite_args(tmp_path)
    resolved = runner._resolve_model("qwen25-05b")
    configuration = runner._suite_configuration(args)
    row = runner._validate_suite_results(
        resolved,
        configuration,
        _synthetic_result("bf16"),
        _synthetic_result("neuralzip"),
        _synthetic_result("dfloat11"),
        fingerprint=_fingerprint(),
    )
    jsonl = tmp_path / "results.jsonl"
    csv_path = tmp_path / "results.csv"
    rows = runner._upsert_jsonl(jsonl, [], row)
    runner._write_csv(csv_path, rows)

    assert len(rows) == 1
    assert (
        runner._complete_row_for(rows, resolved, configuration, "fingerprint")
        is not None
    )
    csv_text = csv_path.read_text(encoding="utf-8")
    assert "neuralzip_encoding_seconds" in csv_text
    assert "Qwen2.5 0.5B" in csv_text


def test_suite_resume_skips_a_complete_matching_row(tmp_path, monkeypatch) -> None:
    args = _orchestrator_args(tmp_path)
    resolved = runner._resolve_model("qwen25-05b")
    configuration = runner._suite_configuration(args)
    row = runner._validate_suite_results(
        resolved,
        configuration,
        _synthetic_result("bf16"),
        _synthetic_result("neuralzip"),
        _synthetic_result("dfloat11"),
        fingerprint=_fingerprint(),
    )
    runner._upsert_jsonl(tmp_path / "results.jsonl", [], row)
    monkeypatch.setattr(runner, "_suite_fingerprint", lambda *_: _fingerprint())

    def unexpected_child(*_args, **_kwargs):
        raise AssertionError("A completed model must not launch another process.")

    monkeypatch.setattr(runner, "_run_child_method", unexpected_child)
    runner._run_suite(args)
    assert (tmp_path / "results.csv").is_file()


def test_partial_method_checkpoint_resumes_remaining_methods(
    tmp_path, monkeypatch
) -> None:
    args = _orchestrator_args(tmp_path)
    resolved = runner._resolve_model("qwen25-05b")
    configuration = runner._suite_configuration(args)
    fingerprint = _fingerprint()
    bf16 = _synthetic_result("bf16")
    record = runner._suite_record(
        "method_result",
        resolved,
        configuration,
        fingerprint,
        status="complete",
        method="bf16",
        result=bf16,
    )
    runner._upsert_jsonl(tmp_path / "results.jsonl", [], record)
    calls = []

    def child(_args, method, model, **_kwargs):
        calls.append(method)
        return _synthetic_result(runner._result_method_name(method), resolved=model)

    monkeypatch.setattr(runner, "_suite_fingerprint", lambda *_: fingerprint)
    monkeypatch.setattr(runner, "_run_child_method", child)
    runner._run_suite(args)

    assert calls == ["neuralzip", "dfloat"]
    rows = runner._read_jsonl(tmp_path / "results.jsonl")
    assert sum(runner._record_type(row) == "method_result" for row in rows) == 3
    assert sum(runner._record_type(row) == "complete" for row in rows) == 1


def test_failure_is_recorded_next_model_runs_and_exit_is_nonzero(
    tmp_path, monkeypatch
) -> None:
    args = _orchestrator_args(tmp_path, ("qwen25-05b", "qwen3-06b"))
    first = runner._resolve_model("qwen25-05b")
    fingerprint = _fingerprint()

    def child(_args, method, model, **_kwargs):
        if model == first and method == "bf16":
            raise RuntimeError("synthetic failure")
        return _synthetic_result(runner._result_method_name(method), resolved=model)

    monkeypatch.setattr(runner, "_suite_fingerprint", lambda *_: fingerprint)
    monkeypatch.setattr(runner, "_run_child_method", child)
    with pytest.raises(SystemExit) as raised:
        runner._run_suite(args)
    assert raised.value.code == 1

    rows = runner._read_jsonl(tmp_path / "results.jsonl")
    assert any(
        runner._record_type(row) == "failure" and row["model"] == first.model_id
        for row in rows
    )
    assert any(
        runner._record_type(row) == "complete"
        and row["model"] == runner.MODEL_CATALOG["qwen3-06b"].model_id
        for row in rows
    )


def test_retry_only_reruns_failed_method(tmp_path, monkeypatch) -> None:
    args = _orchestrator_args(tmp_path)
    fingerprint = _fingerprint()
    calls = []

    def child(_args, method, model, **_kwargs):
        calls.append(method)
        if method == "neuralzip" and calls.count("neuralzip") == 1:
            raise RuntimeError("retry me")
        return _synthetic_result(runner._result_method_name(method), resolved=model)

    monkeypatch.setattr(runner, "_suite_fingerprint", lambda *_: fingerprint)
    monkeypatch.setattr(runner, "_run_child_method", child)
    with pytest.raises(SystemExit):
        runner._run_suite(args)
    runner._run_suite(args)

    assert calls == ["bf16", "neuralzip", "dfloat", "neuralzip"]
    rows = runner._read_jsonl(tmp_path / "results.jsonl")
    assert not any(runner._record_type(row) == "failure" for row in rows)
    assert sum(runner._record_type(row) == "complete" for row in rows) == 1


def test_force_suite_replaces_equivalent_records(tmp_path, monkeypatch) -> None:
    args = _orchestrator_args(tmp_path)
    fingerprint = _fingerprint()

    def child(_args, method, model, **_kwargs):
        return _synthetic_result(runner._result_method_name(method), resolved=model)

    monkeypatch.setattr(runner, "_suite_fingerprint", lambda *_: fingerprint)
    monkeypatch.setattr(runner, "_run_child_method", child)
    runner._run_suite(args)
    args.force_suite = True
    runner._run_suite(args)

    rows = runner._read_jsonl(tmp_path / "results.jsonl")
    assert len(rows) == 4
    assert sum(runner._record_type(row) == "method_result" for row in rows) == 3
    assert sum(runner._record_type(row) == "complete" for row in rows) == 1


def test_resume_key_rejects_stale_fingerprint_or_configuration(tmp_path) -> None:
    args = _orchestrator_args(tmp_path)
    resolved = runner._resolve_model("qwen25-05b")
    configuration = runner._suite_configuration(args)
    record = runner._suite_record(
        "method_result",
        resolved,
        configuration,
        _fingerprint("old"),
        status="complete",
        method="bf16",
        result=_synthetic_result("bf16"),
    )
    assert (
        runner._method_result_for([record], resolved, configuration, "new", "bf16")
        is None
    )
    changed = {**configuration, "sequence_length": 256}
    assert runner._method_result_for([record], resolved, changed, "old", "bf16") is None

    complete = runner._validate_suite_results(
        resolved,
        configuration,
        _synthetic_result("bf16"),
        _synthetic_result("neuralzip"),
        _synthetic_result("dfloat11"),
        fingerprint=_fingerprint("old"),
    )
    runner._write_csv(tmp_path / "current.csv", [complete], "new")
    assert len((tmp_path / "current.csv").read_text().splitlines()) == 1


def test_fingerprint_covers_code_runtime_and_configuration(
    tmp_path, monkeypatch
) -> None:
    args = _orchestrator_args(tmp_path)
    configuration = runner._suite_configuration(args)
    first = runner._suite_fingerprint(args, configuration)
    changed_config = runner._suite_fingerprint(
        args, {**configuration, "sequence_length": 256}
    )
    monkeypatch.setattr(
        runner,
        "_version_fields",
        lambda: {
            "dfloat11_version": "changed",
            "torch_version": "x",
            "transformers_version": "y",
        },
    )
    changed_runtime = runner._suite_fingerprint(args, configuration)
    assert len(first["code"]) == 64
    assert (
        len({first["digest"], changed_config["digest"], changed_runtime["digest"]}) == 3
    )


def test_inexact_direct_result_is_never_written(tmp_path) -> None:
    destination = tmp_path / "inexact.json"
    args = SimpleNamespace(result_file=str(destination), no_print_result=True)
    with pytest.raises(SystemExit):
        runner._emit_verified_result(
            args,
            {"method": "neuralzip", "bitwise_exact_logits": False},
        )
    assert not destination.exists()


@pytest.mark.parametrize(
    "extra",
    (
        ("--sequence-length", "0"),
        ("--threads", "-1"),
        ("--suite-model", "qwen25-05b", "--suite-model", "qwen25-05b"),
        ("--output-csv", "results.jsonl"),
    ),
)
def test_suite_arguments_are_validated(extra) -> None:
    parser = runner._build_parser()
    args = parser.parse_args(["suite", "--output-jsonl", "results.jsonl", *extra])
    with pytest.raises(SystemExit) as raised:
        runner._validate_args(parser, args)
    assert raised.value.code == 2


def test_parser_uses_fixed_suite_defaults() -> None:
    args = runner._build_parser().parse_args(
        ["suite", "--output-jsonl", "results.jsonl"]
    )
    assert args.batch_size == 1
    assert args.sequence_length == 128
    assert args.warmup == 20
    assert args.repetitions == 50
    assert args.trials == 5
    assert args.dfloat_directory is None
