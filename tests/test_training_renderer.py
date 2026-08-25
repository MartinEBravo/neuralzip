from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples" / "render_four_model_training_suite.py"
SPEC = importlib.util.spec_from_file_location(
    "render_four_model_training_suite", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def _write(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _checkpoint_rows(models: tuple[str, ...]) -> list[dict[str, object]]:
    return [
        {
            "model": model,
            "epoch": epoch,
            "compressor": compressor,
            "exponent_ratio_pct": 30.0 + epoch,
            "encode_seconds": 0.1 + epoch,
            "roundtrip_exact": True,
            "input_unchanged": True,
            "status": "ok",
        }
        for model in models
        for epoch in (0, 1)
        for compressor in module.COMPRESSORS
    ]


def _federated_rows(model: str) -> list[dict[str, object]]:
    return [
        {
            "architecture": model,
            "round": round_index,
            "compressor": compressor,
            "exponent_ratio_pct": 31.0 + round_index,
            "downlink_encode_seconds": 0.1 + round_index,
            "roundtrip_ok": True,
        }
        for round_index in (1, 2)
        for compressor in module.COMPRESSORS
    ]


def _args(tmp_path: Path) -> SimpleNamespace:
    language_checkpoints = tmp_path / "language-checkpoints.csv"
    vision_checkpoints = tmp_path / "vision-checkpoints.csv"
    checkpoint_fields = sorted(module.CHECKPOINT_REQUIRED_COLUMNS)
    _write(
        language_checkpoints,
        checkpoint_fields,
        _checkpoint_rows(("gpt2", "bert-base")),
    )
    _write(
        vision_checkpoints,
        checkpoint_fields,
        _checkpoint_rows(("resnet18", "resnet34")),
    )

    language_federated = tmp_path / "language-federated"
    vision_federated = tmp_path / "vision-federated"
    federated_fields = sorted(module.FEDERATED_REQUIRED_COLUMNS)
    for root, models in (
        (language_federated, ("gpt2", "bert-base")),
        (vision_federated, ("resnet18", "resnet34")),
    ):
        for model in models:
            _write(
                root / model / "federated_round_results.csv",
                federated_fields,
                _federated_rows(model),
            )
    return SimpleNamespace(
        language_checkpoint_csv=language_checkpoints,
        vision_checkpoint_csv=vision_checkpoints,
        language_federated_root=language_federated,
        vision_federated_root=vision_federated,
        output_dir=tmp_path / "output",
        language_checkpoint_epochs=1,
        vision_checkpoint_epochs=1,
        federated_rounds=2,
    )


def test_renders_both_complete_four_model_figures(tmp_path: Path) -> None:
    args = _args(tmp_path)

    checkpoint_outputs = module.build_checkpoint_outputs(args)
    federated_outputs = module.build_federated_outputs(args)

    for output in (*checkpoint_outputs, *federated_outputs):
        assert output.is_file()
        assert output.stat().st_size > 0

    with checkpoint_outputs[0].open(newline="", encoding="utf-8") as handle:
        checkpoint_rows = list(csv.DictReader(handle))
    with federated_outputs[0].open(newline="", encoding="utf-8") as handle:
        federated_rows = list(csv.DictReader(handle))
    assert len(checkpoint_rows) == 4 * 2 * 2
    assert len(federated_rows) == 4 * 2 * 2
    assert [row["model"] for row in checkpoint_rows[::4]] == list(module.MODEL_ORDER)


def test_rejects_missing_compressor_before_render(tmp_path: Path) -> None:
    args = _args(tmp_path)
    rows, fields = module._read_rows(args.language_checkpoint_csv)
    rows.pop()
    _write(args.language_checkpoint_csv, sorted(fields), rows)

    with pytest.raises(ValueError, match="missing="):
        module.build_checkpoint_outputs(args)

    assert not (args.output_dir / "figures" / "checkpoint_training.pdf").exists()
