import json
import sys
from pathlib import Path

import pytest
from huggingface_hub.file_download import repo_folder_name

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from examples.experiments.model_catalog import ModelSpec
from examples.experiments.model_loading import snapshot_weight_files


def _snapshot(
    cache_dir: Path,
    repo_id: str,
    revision: str,
    *,
    ref: str | None = None,
) -> Path:
    repo_cache = cache_dir / repo_folder_name(repo_id=repo_id, repo_type="model")
    snapshot = repo_cache / "snapshots" / revision
    snapshot.mkdir(parents=True)
    if ref is not None:
        ref_path = repo_cache / "refs" / ref
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        ref_path.write_text(f"{revision}\n")
    return snapshot


def _spec(repo_id: str = "owner/model") -> ModelSpec:
    return ModelSpec(
        key="model",
        label="Model",
        family="Family",
        suites=("model-storing",),
        backend="hf",
        source=repo_id,
    )


def _write_indexed_checkpoint(
    snapshot: Path,
    *,
    complete: bool = True,
    empty: bool = False,
) -> None:
    index = {
        "weight_map": {
            "a.weight": "model-00001-of-00002.safetensors",
            "b.weight": "model-00002-of-00002.safetensors",
        }
    }
    (snapshot / "model.safetensors.index.json").write_text(json.dumps(index))
    (snapshot / "model-00001-of-00002.safetensors").write_bytes(b"first")
    if complete:
        (snapshot / "model-00002-of-00002.safetensors").write_bytes(
            b"" if empty else b"second"
        )


def test_offline_snapshot_uses_cached_canonical_index_without_snapshot_download(
    tmp_path, monkeypatch
):
    repo_id = "mistralai/Mistral-Nemo-Base-2407"
    revision = "a" * 40
    snapshot = _snapshot(tmp_path, repo_id, revision, ref="main")
    _write_indexed_checkpoint(snapshot)

    def unexpected_snapshot_download(**_kwargs):
        raise AssertionError("offline resolution must not call snapshot_download")

    monkeypatch.setattr(
        "huggingface_hub.snapshot_download", unexpected_snapshot_download
    )

    selection = snapshot_weight_files(
        _spec(repo_id),
        cache_dir=tmp_path,
        allow_download=False,
    )

    assert selection.snapshot == snapshot
    assert selection.revision == revision
    assert selection.index_path == snapshot / "model.safetensors.index.json"
    assert [path.name for path in selection.files] == [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]
    assert selection.expected_keys == frozenset({"a.weight", "b.weight"})


def test_offline_snapshot_honors_a_pinned_commit_instead_of_moving_main(tmp_path):
    repo_id = "owner/model"
    main_revision = "a" * 40
    pinned_revision = "b" * 40
    main_snapshot = _snapshot(tmp_path, repo_id, main_revision, ref="main")
    pinned_snapshot = _snapshot(tmp_path, repo_id, pinned_revision)
    (main_snapshot / "model.safetensors").write_bytes(b"main")
    (pinned_snapshot / "model.safetensors").write_bytes(b"pinned")

    selection = snapshot_weight_files(
        _spec(repo_id),
        cache_dir=tmp_path,
        allow_download=False,
        revision=pinned_revision,
    )

    assert selection.snapshot == pinned_snapshot
    assert selection.revision == pinned_revision
    assert selection.files == (pinned_snapshot / "model.safetensors",)


def test_offline_snapshot_rejects_an_unpinned_ambiguous_cache(tmp_path):
    repo_id = "owner/model"
    _snapshot(tmp_path, repo_id, "a" * 40)
    _snapshot(tmp_path, repo_id, "b" * 40)

    with pytest.raises(ValueError, match="cannot be resolved unambiguously"):
        snapshot_weight_files(
            _spec(repo_id),
            cache_dir=tmp_path,
            allow_download=False,
        )


def test_offline_snapshot_rejects_an_index_with_a_missing_cached_shard(tmp_path):
    repo_id = "owner/model"
    snapshot = _snapshot(tmp_path, repo_id, "a" * 40, ref="main")
    _write_indexed_checkpoint(snapshot, complete=False)

    with pytest.raises(FileNotFoundError, match="references missing shard"):
        snapshot_weight_files(
            _spec(repo_id),
            cache_dir=tmp_path,
            allow_download=False,
        )


def test_offline_snapshot_rejects_an_empty_indexed_shard(tmp_path):
    repo_id = "owner/model"
    snapshot = _snapshot(tmp_path, repo_id, "a" * 40, ref="main")
    _write_indexed_checkpoint(snapshot, empty=True)

    with pytest.raises(ValueError, match="references empty shard"):
        snapshot_weight_files(
            _spec(repo_id),
            cache_dir=tmp_path,
            allow_download=False,
        )


def test_offline_snapshot_rejects_a_ref_to_a_missing_snapshot(tmp_path):
    repo_id = "owner/model"
    repo_cache = tmp_path / repo_folder_name(repo_id=repo_id, repo_type="model")
    (repo_cache / "snapshots").mkdir(parents=True)
    (repo_cache / "refs").mkdir()
    (repo_cache / "refs" / "main").write_text(f"{'a' * 40}\n")

    with pytest.raises(FileNotFoundError, match="points to missing snapshot"):
        snapshot_weight_files(
            _spec(repo_id),
            cache_dir=tmp_path,
            allow_download=False,
        )


def test_online_snapshot_keeps_using_snapshot_download(tmp_path, monkeypatch):
    downloaded = tmp_path / "downloaded"
    downloaded.mkdir()
    (downloaded / "model.safetensors").write_bytes(b"weights")
    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        return str(downloaded)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)

    selection = snapshot_weight_files(
        _spec(),
        cache_dir=tmp_path / "cache",
        allow_download=True,
        revision="release",
    )

    assert selection.files == (downloaded / "model.safetensors",)
    assert calls == [
        {
            "repo_id": "owner/model",
            "revision": "release",
            "cache_dir": tmp_path / "cache",
            "allow_patterns": (
                "*.safetensors",
                "*.safetensors.index.json",
                "*.bin",
                "*.bin.index.json",
                "*.pt",
                "config.json",
            ),
            "local_files_only": False,
        }
    ]
