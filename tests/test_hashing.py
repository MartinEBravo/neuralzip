import hashlib
import json

import pytest

from neuralzip.hashing import canonical_digest, canonical_json_bytes, sha256_file


def test_canonical_json_matches_legacy_serialization_and_is_order_independent():
    left = {"b": [2, 1], "a": {"value": 3}}
    right = {"a": {"value": 3}, "b": [2, 1]}
    expected = json.dumps(
        left,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")

    assert canonical_json_bytes(left) == expected
    assert canonical_json_bytes(right) == expected
    assert canonical_digest(left) == hashlib.sha256(expected).hexdigest()


def test_strict_canonical_json_rejects_unknown_objects():
    with pytest.raises(TypeError):
        canonical_json_bytes(object(), stringify_unknown=False)
    assert canonical_json_bytes(object()).startswith(b'"<object object at ')


def test_sha256_file_uses_the_same_digest_for_any_block_size(tmp_path):
    path = tmp_path / "payload.bin"
    path.write_bytes(bytes(range(256)) * 3)
    expected = hashlib.sha256(path.read_bytes()).hexdigest()

    assert sha256_file(path, block_bytes=1) == expected
    assert sha256_file(path, block_bytes=257) == expected
