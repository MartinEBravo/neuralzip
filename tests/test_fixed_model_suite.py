import json

import pytest

from examples.run_fixed_model_suite import DEFAULT_MANIFEST, load_manifest


def test_bundled_manifest_pins_every_fixed_model():
    manifest = load_manifest(DEFAULT_MANIFEST)

    assert manifest["model_count"] == 50
    assert len({row["model_key"] for row in manifest["models"]}) == 50
    assert all(row["revision"] for row in manifest["models"])
    assert all(len(row["weights_fingerprint"]) == 64 for row in manifest["models"])
    assert len(manifest["fingerprint"]) == 64


def test_manifest_rejects_unknown_models(tmp_path):
    source = json.loads(DEFAULT_MANIFEST.read_text())
    source["models"][0]["model_key"] = "unknown-model"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(source))

    with pytest.raises(ValueError, match="unknown model"):
        load_manifest(path)
