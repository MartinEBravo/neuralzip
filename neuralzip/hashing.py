"""Deterministic hashing helpers for experiment provenance."""

import hashlib
import json
from pathlib import Path


def canonical_json_bytes(value, *, stringify_unknown=True, ensure_ascii=True):
    options = {
        "sort_keys": True,
        "separators": (",", ":"),
        "ensure_ascii": bool(ensure_ascii),
    }
    if stringify_unknown:
        options["default"] = str
    return json.dumps(value, **options).encode("utf-8")


def canonical_digest(value, *, stringify_unknown=True, ensure_ascii=True):
    return hashlib.sha256(
        canonical_json_bytes(
            value,
            stringify_unknown=stringify_unknown,
            ensure_ascii=ensure_ascii,
        )
    ).hexdigest()


def sha256_file(path, *, block_bytes=8 << 20):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(int(block_bytes)):
            digest.update(block)
    return digest.hexdigest()
