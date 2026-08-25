from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import io
import json
import os
import platform
import struct
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
from huggingface_hub import HfApi, HfFileSystem

try:
    from config import LOSSY_LOSSLESS_MODEL_SUITE
except ModuleNotFoundError:  # Imported as examples.run_quantized_nibble_compression.
    from examples.config import LOSSY_LOSSLESS_MODEL_SUITE

from neuralzip import huff0
from neuralzip.checkpoint_loading import resolve_weight_map
from neuralzip.experiment_io import atomic_write_json
from neuralzip.hashing import (
    canonical_digest as _shared_canonical_digest,
    sha256_file as _shared_sha256_file,
)
from neuralzip.metrics import jensen_shannon_distance as _js_distance
from neuralzip.payloads import (
    RAW_CHUNK_METADATA_BYTES as CHUNK_HEADER_BYTES,
    SHARED_CHUNK_METADATA_BYTES as SHARED_CHUNK_HEADER_BYTES,
)
from neuralzip.system import cpu_affinity


DEFAULT_OUTPUT_DIR = Path(
    "examples/outputs/lossy_lossless/quantized_nibbles_final"
)
ALLOWED_ARTIFACT_EXTENSIONS = (".safetensors", ".gguf")
RAW_CHUNK_BYTES = 256 << 10
MAX_SAMPLE_BYTES_PER_FILE = 64 << 20
CODEBOOK_HEADER_BYTES = 8 + len("nibble".encode("utf-8")) + 8
CLUSTER_JS_THRESHOLD = 0.05
MAX_CLUSTER_SAMPLE_BYTES = 16 << 20
SAMPLE_MANIFEST_VERSION = 2
RESULT_FORMAT_VERSION = 1


@dataclass(frozen=True)
class EncodedBlock:
    mode: str
    source: bytes
    payload: bytes
    cluster_index: int | None = None


@dataclass(frozen=True)
class EncodedArchive:
    blocks: tuple[EncodedBlock, ...]
    compressed_bytes: int
    huffman_chunks: int
    raw_chunks: int
    referenced_codebooks: int
    codebook_metadata_bytes: int


@dataclass(frozen=True)
class SharedPlan:
    codebooks: tuple[object, ...]
    assignments: tuple[int, ...]
    clusters: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reproducible nibble-compression probe for real quantized model "
            "artifacts. Remote I/O is materialized before any encode timing."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--sample-cache-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "sample-cache",
    )
    parser.add_argument(
        "--models",
        default="all",
        help="Comma-separated model names, ids, or slugs; default: all.",
    )
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument(
        "--revision",
        action="append",
        default=[],
        metavar="MODEL=REVISION",
        help=(
            "Optional pinned Hugging Face revision. MODEL may be a name, id, "
            "or slug. The resolved commit is always stored in the manifest."
        ),
    )
    parser.add_argument(
        "--max-sample-bytes-per-file",
        type=int,
        default=MAX_SAMPLE_BYTES_PER_FILE,
    )
    parser.add_argument("--raw-chunk-bytes", type=int, default=RAW_CHUNK_BYTES)
    parser.add_argument(
        "--cluster-js-threshold",
        type=float,
        default=CLUSTER_JS_THRESHOLD,
    )
    parser.add_argument(
        "--max-cluster-sample-bytes",
        type=int,
        default=MAX_CLUSTER_SAMPLE_BYTES,
    )
    phase = parser.add_mutually_exclusive_group()
    phase.add_argument(
        "--prepare-only",
        action="store_true",
        help="Resolve revisions and materialize real samples; do not encode.",
    )
    phase.add_argument(
        "--run-only",
        action="store_true",
        help="Require and validate cached samples; perform no network access.",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Re-download samples even if a valid manifest exists.",
    )
    args = parser.parse_args()
    if args.threads <= 0:
        parser.error("--threads must be positive; use 12 for final results.")
    if args.max_sample_bytes_per_file <= 0:
        parser.error("--max-sample-bytes-per-file must be positive.")
    if args.raw_chunk_bytes <= 0:
        parser.error("--raw-chunk-bytes must be positive.")
    if args.max_cluster_sample_bytes <= 0:
        parser.error("--max-cluster-sample-bytes must be positive.")
    if args.cluster_js_threshold < 0:
        parser.error("--cluster-js-threshold must be non-negative.")
    if args.run_only and args.refresh_cache:
        parser.error("--refresh-cache cannot be combined with --run-only.")
    return args


def _slug(value: str) -> str:
    output = []
    for character in value.lower():
        output.append(character if character.isalnum() else "-")
    return "-".join(filter(None, "".join(output).split("-")))


def _quantized_specs() -> list[dict]:
    return [
        dict(spec)
        for spec in LOSSY_LOSSLESS_MODEL_SUITE
        if spec.get("category") == "quantized"
    ]


def _spec_aliases(spec: dict) -> set[str]:
    return {
        str(spec["name"]),
        str(spec["model_id"]),
        _slug(str(spec["name"])),
    }


def _select_specs(selection: str) -> list[dict]:
    specs = _quantized_specs()
    if selection.strip().lower() == "all":
        return specs
    requested = [item.strip() for item in selection.split(",") if item.strip()]
    selected = []
    unknown = []
    for item in requested:
        match = next((spec for spec in specs if item in _spec_aliases(spec)), None)
        if match is None:
            unknown.append(item)
        elif match not in selected:
            selected.append(match)
    if unknown:
        raise ValueError(f"Unknown quantized model selectors: {unknown}")
    if not selected:
        raise ValueError("No quantized models selected.")
    return selected


def _revision_overrides(values: Iterable[str], specs: list[dict]) -> dict[str, str]:
    aliases = {
        alias: str(spec["model_id"])
        for spec in specs
        for alias in _spec_aliases(spec)
    }
    revisions = {}
    for value in values:
        selector, separator, revision = value.partition("=")
        if not separator or not selector.strip() or not revision.strip():
            raise ValueError(f"Invalid --revision value: {value!r}")
        model_id = aliases.get(selector.strip())
        if model_id is None:
            raise ValueError(f"Unknown revision selector: {selector!r}")
        revisions[model_id] = revision.strip()
    return revisions


def _safe_repository_path(value: object) -> str:
    path = str(value)
    candidate = Path(path)
    if (
        not path
        or candidate.is_absolute()
        or ".." in candidate.parts
        or path.endswith("/")
    ):
        raise ValueError(f"Unsafe repository artifact path: {path!r}")
    if not path.lower().endswith(ALLOWED_ARTIFACT_EXTENSIONS):
        raise ValueError(
            "Quantized artifacts must be explicit safetensors or GGUF files; "
            f"got {path!r}. Generic .bin/.pt discovery is forbidden."
        )
    return path


def _artifact_selector(spec: dict) -> dict:
    raw = spec.get("artifact_selector")
    if not isinstance(raw, dict):
        raise ValueError(
            f"{spec.get('model_id')} has no explicit artifact_selector."
        )
    kind = str(raw.get("kind") or "")
    if kind == "safetensors-index":
        index = str(raw.get("index") or "")
        candidate = Path(index)
        if (
            not index.endswith(".safetensors.index.json")
            or candidate.is_absolute()
            or ".." in candidate.parts
        ):
            raise ValueError(f"Invalid safetensors index selector: {index!r}")
        return {"kind": kind, "index": index}
    if kind == "exact-files":
        files = tuple(_safe_repository_path(path) for path in raw.get("files", ()))
        if not files or len(files) != len(set(files)):
            raise ValueError(
                f"exact-files selector must be non-empty and unique: {files!r}"
            )
        return {"kind": kind, "files": list(files)}
    raise ValueError(f"Unsupported artifact selector kind: {kind!r}")


def _canonical_digest(value: object) -> str:
    return _shared_canonical_digest(value, stringify_unknown=False)


def _sha256_file(path: Path, *, block_bytes: int = 4 << 20) -> str:
    return _shared_sha256_file(path, block_bytes=block_bytes)


def _write_json(path: Path, value: object) -> None:
    atomic_write_json(path, value, sort_keys=True)


def _read_exact(handle, size: int) -> bytes:
    data = handle.read(int(size))
    if len(data) != int(size):
        raise ValueError(f"Unexpected EOF: requested {size}, received {len(data)}")
    return data


def _gguf_string(handle) -> str:
    length = struct.unpack("<Q", _read_exact(handle, 8))[0]
    return _read_exact(handle, length).decode("utf-8", errors="replace")


_GGUF_SCALARS = {
    0: ("<B", 1),
    1: ("<b", 1),
    2: ("<H", 2),
    3: ("<h", 2),
    4: ("<I", 4),
    5: ("<i", 4),
    6: ("<f", 4),
    7: ("<?", 1),
    10: ("<Q", 8),
    11: ("<q", 8),
    12: ("<d", 8),
}


def _gguf_value(handle, value_type: int):
    if value_type in _GGUF_SCALARS:
        format_string, size = _GGUF_SCALARS[value_type]
        return struct.unpack(format_string, _read_exact(handle, size))[0]
    if value_type == 8:
        return _gguf_string(handle)
    if value_type == 9:
        element_type = struct.unpack("<I", _read_exact(handle, 4))[0]
        count = struct.unpack("<Q", _read_exact(handle, 8))[0]
        if element_type in _GGUF_SCALARS:
            handle.seek(_GGUF_SCALARS[element_type][1] * count, io.SEEK_CUR)
        else:
            for _ in range(count):
                _gguf_value(handle, element_type)
        return None
    raise ValueError(f"Unsupported GGUF metadata value type: {value_type}")


def _gguf_data_offset(handle, file_size: int) -> int:
    handle.seek(0)
    if _read_exact(handle, 4) != b"GGUF":
        raise ValueError("Invalid GGUF magic.")
    version = struct.unpack("<I", _read_exact(handle, 4))[0]
    if version not in {2, 3}:
        raise ValueError(f"Unsupported GGUF version: {version}")
    tensor_count = struct.unpack("<Q", _read_exact(handle, 8))[0]
    metadata_count = struct.unpack("<Q", _read_exact(handle, 8))[0]
    alignment = 32
    for _ in range(metadata_count):
        key = _gguf_string(handle)
        value_type = struct.unpack("<I", _read_exact(handle, 4))[0]
        value = _gguf_value(handle, value_type)
        if key == "general.alignment" and value is not None:
            alignment = int(value)
    if alignment <= 0 or alignment > (1 << 20):
        raise ValueError(f"Invalid GGUF alignment: {alignment}")
    for _ in range(tensor_count):
        _gguf_string(handle)
        dimensions = struct.unpack("<I", _read_exact(handle, 4))[0]
        handle.seek(8 * dimensions + 4 + 8, io.SEEK_CUR)
    position = int(handle.tell())
    offset = ((position + alignment - 1) // alignment) * alignment
    if not 0 <= offset < int(file_size):
        raise ValueError(
            f"Invalid GGUF tensor-data offset {offset} for {file_size} bytes."
        )
    return offset


def _data_offset(path: str, handle, file_size: int) -> tuple[int, str]:
    suffix = str(path).lower()
    if suffix.endswith(".safetensors"):
        handle.seek(0)
        header_size = int.from_bytes(_read_exact(handle, 8), "little")
        offset = 8 + header_size
        if not 8 <= offset < int(file_size):
            raise ValueError(
                f"Invalid safetensors data offset {offset} for {file_size} bytes."
            )
        return offset, "safetensors-data"
    if suffix.endswith(".gguf"):
        return _gguf_data_offset(handle, file_size), "gguf-tensor-data"
    return 0, "container-start"


def _remote_relative_path(model_id: str, path: str) -> str:
    prefix = f"{model_id}/"
    if path.startswith(prefix):
        return path[len(prefix) :]
    revision_prefix = f"{model_id}@"
    if path.startswith(revision_prefix):
        _, separator, relative_path = path[len(revision_prefix) :].partition("/")
        if separator and relative_path:
            return relative_path
    return path


def _hf_model_files(
    fs: HfFileSystem,
    spec: dict,
    revision: str,
) -> list[dict]:
    model_id = str(spec["model_id"])
    selector = _artifact_selector(spec)
    repository_files = {}
    for item in fs.ls(model_id, detail=True, revision=revision):
        if item.get("type") != "file":
            continue
        path = str(item["name"])
        relative_path = _remote_relative_path(model_id, path)
        if relative_path in repository_files:
            raise ValueError(
                f"Duplicate repository artifact path: {relative_path}"
            )
        repository_files[relative_path] = item

    if selector["kind"] == "safetensors-index":
        index_path = str(selector["index"])
        if index_path not in repository_files:
            raise FileNotFoundError(
                f"Authoritative weight index is missing: {model_id}/{index_path}"
            )
        with fs.open(
            f"{model_id}/{index_path}",
            "rb",
            revision=revision,
            block_size=1 << 20,
        ) as handle:
            index = json.load(handle)
        resolved = resolve_weight_map(
            index.get("weight_map"),
            available_files={
                path: int(item.get("size") or 0)
                for path, item in repository_files.items()
            },
            allowed_suffixes=(".safetensors",),
        )
        selected_paths = list(resolved.files)
    else:
        selected_paths = list(selector["files"])

    files = []
    for relative_path in selected_paths:
        item = repository_files.get(relative_path)
        if item is None:
            raise FileNotFoundError(
                f"Selected weight artifact is missing: {model_id}/{relative_path}"
            )
        size = int(item.get("size") or 0)
        if size <= 0:
            raise ValueError(f"Missing file size for {model_id}/{relative_path}")
        files.append(
            {
                "path": str(item["name"]),
                "relative_path": relative_path,
                "size": size,
            }
        )
    return files


def _manifest_payload(manifest: dict) -> dict:
    return {
        "format_version": manifest["format_version"],
        "model": manifest["model"],
        "model_id": manifest["model_id"],
        "resolved_revision": manifest["resolved_revision"],
        "artifact_selector": manifest["artifact_selector"],
        "selected_artifacts": manifest["selected_artifacts"],
        "max_sample_bytes_per_file": manifest[
            "max_sample_bytes_per_file"
        ],
        "files": [
            {
                "relative_path": row["relative_path"],
                "repository_file_size": row["repository_file_size"],
                "data_offset": row["data_offset"],
                "offset_policy": row["offset_policy"],
                "sample_bytes": row["sample_bytes"],
                "sample_sha256": row["sample_sha256"],
            }
            for row in manifest["files"]
        ],
    }


def _validate_manifest(
    manifest_path: Path,
    *,
    expected_spec: dict,
    max_sample_bytes_per_file: int,
    requested_revision: str | None = None,
) -> dict:
    manifest = json.loads(manifest_path.read_text())
    if int(manifest.get("format_version", -1)) != SAMPLE_MANIFEST_VERSION:
        raise ValueError(f"Unsupported sample manifest: {manifest_path}")
    if manifest.get("model_id") != expected_spec["model_id"]:
        raise ValueError(f"Model mismatch in {manifest_path}")
    expected_selector = _artifact_selector(expected_spec)
    if manifest.get("artifact_selector") != expected_selector:
        raise ValueError(f"Artifact selector mismatch in {manifest_path}")
    if int(manifest.get("max_sample_bytes_per_file", -1)) != int(
        max_sample_bytes_per_file
    ):
        raise ValueError(f"Sample-size mismatch in {manifest_path}")
    if requested_revision and manifest.get("requested_revision") != requested_revision:
        raise ValueError(
            f"Revision mismatch in {manifest_path}; refresh the sample cache."
        )
    files = manifest.get("files", [])
    selected_artifacts = [str(row.get("relative_path")) for row in files]
    if selected_artifacts != manifest.get("selected_artifacts"):
        raise ValueError(f"Selected artifact list mismatch in {manifest_path}")
    if len(selected_artifacts) != len(set(selected_artifacts)):
        raise ValueError(f"Duplicate selected artifacts in {manifest_path}")
    selector = expected_selector
    if selector["kind"] == "exact-files" and selected_artifacts != selector["files"]:
        raise ValueError(f"Exact artifact selection changed in {manifest_path}")
    if selector["kind"] == "safetensors-index" and any(
        not path.endswith(".safetensors") for path in selected_artifacts
    ):
        raise ValueError(f"Index selected a non-safetensors file in {manifest_path}")
    for row in files:
        sample_path = manifest_path.parent / str(row["cache_file"])
        if not sample_path.is_file():
            raise FileNotFoundError(f"Missing cached sample: {sample_path}")
        if sample_path.stat().st_size != int(row["sample_bytes"]):
            raise ValueError(f"Cached sample size changed: {sample_path}")
        if _sha256_file(sample_path) != row["sample_sha256"]:
            raise ValueError(f"Cached sample digest changed: {sample_path}")
    if not files:
        raise ValueError(f"No supported model files in {manifest_path}")
    fingerprint = _canonical_digest(_manifest_payload(manifest))
    if fingerprint != manifest.get("sample_fingerprint"):
        raise ValueError(f"Sample fingerprint mismatch in {manifest_path}")
    return manifest


def _materialize_model(
    spec: dict,
    *,
    cache_root: Path,
    fs: HfFileSystem,
    api: HfApi,
    requested_revision: str | None,
    max_sample_bytes_per_file: int,
    refresh: bool,
) -> dict:
    model_dir = cache_root / _slug(str(spec["name"]))
    manifest_path = model_dir / "manifest.json"
    if manifest_path.is_file() and not refresh:
        return _validate_manifest(
            manifest_path,
            expected_spec=spec,
            max_sample_bytes_per_file=max_sample_bytes_per_file,
            requested_revision=requested_revision,
        )

    info = api.model_info(
        str(spec["model_id"]),
        revision=requested_revision or "main",
    )
    resolved_revision = str(info.sha)
    if len(resolved_revision) < 12:
        raise ValueError(
            f"Hugging Face did not resolve a commit for {spec['model_id']}."
        )
    artifact_selector = _artifact_selector(spec)
    files = _hf_model_files(fs, spec, resolved_revision)
    if not files:
        raise ValueError(f"No supported files found for {spec['model_id']}")

    model_dir.mkdir(parents=True, exist_ok=True)
    manifest_files = []
    for file_index, item in enumerate(files):
        cache_name = f"{file_index:04d}.sample.bin"
        destination = model_dir / cache_name
        temporary = destination.with_suffix(destination.suffix + ".part")
        digest = hashlib.sha256()
        with fs.open(
            item["path"],
            "rb",
            revision=resolved_revision,
            block_size=4 << 20,
        ) as source:
            offset, offset_policy = _data_offset(
                item["relative_path"], source, int(item["size"])
            )
            source.seek(offset)
            target = min(
                int(max_sample_bytes_per_file),
                max(0, int(item["size"]) - offset),
            )
            written = 0
            with temporary.open("wb") as output:
                while written < target:
                    data = source.read(min(4 << 20, target - written))
                    if not data:
                        break
                    output.write(data)
                    digest.update(data)
                    written += len(data)
        if written != target:
            temporary.unlink(missing_ok=True)
            raise ValueError(
                f"Short sample for {item['path']}: {written}/{target} bytes."
            )
        temporary.replace(destination)
        manifest_files.append(
            {
                "remote_path": item["path"],
                "relative_path": item["relative_path"],
                "repository_file_size": int(item["size"]),
                "data_offset": offset,
                "offset_policy": offset_policy,
                "cache_file": cache_name,
                "sample_bytes": written,
                "sample_sha256": digest.hexdigest(),
            }
        )

    manifest = {
        "format_version": SAMPLE_MANIFEST_VERSION,
        "model": str(spec["name"]),
        "model_id": str(spec["model_id"]),
        "requested_revision": requested_revision,
        "resolved_revision": resolved_revision,
        "artifact_selector": artifact_selector,
        "selected_artifacts": [row["relative_path"] for row in manifest_files],
        "max_sample_bytes_per_file": int(max_sample_bytes_per_file),
        "files": manifest_files,
    }
    manifest["sample_fingerprint"] = _canonical_digest(
        _manifest_payload(manifest)
    )
    _write_json(manifest_path, manifest)
    return _validate_manifest(
        manifest_path,
        expected_spec=spec,
        max_sample_bytes_per_file=max_sample_bytes_per_file,
        requested_revision=requested_revision,
    )


def _load_samples(manifest: dict, model_dir: Path) -> tuple[bytes, ...]:
    samples = tuple(
        (model_dir / str(row["cache_file"])).read_bytes()
        for row in manifest["files"]
    )
    if sum(map(len, samples)) != sum(
        int(row["sample_bytes"]) for row in manifest["files"]
    ):
        raise RuntimeError("Loaded sample length does not match the manifest.")
    return samples


def _iter_raw_chunks(samples: Iterable[bytes], raw_chunk_bytes: int):
    for sample in samples:
        for start in range(0, len(sample), raw_chunk_bytes):
            chunk = sample[start : start + raw_chunk_bytes]
            if chunk:
                yield chunk


def _nibble_stream(data: bytes) -> bytes:
    values = np.frombuffer(data, dtype=np.uint8)
    nibbles = np.empty(values.size * 2, dtype=np.uint8)
    nibbles[0::2] = values & 0x0F
    nibbles[1::2] = values >> 4
    return nibbles.tobytes()


def _bytes_from_nibbles(data: bytes) -> bytes:
    nibbles = np.frombuffer(data, dtype=np.uint8)
    if nibbles.size % 2 or np.any(nibbles > 0x0F):
        raise ValueError("Invalid nibble stream.")
    values = nibbles[0::2] | (nibbles[1::2] << 4)
    return values.astype(np.uint8, copy=False).tobytes()


def _nibble_counts(data: bytes) -> np.ndarray:
    values = np.frombuffer(data, dtype=np.uint8)
    counts = np.bincount(values & 0x0F, minlength=16)
    counts += np.bincount(values >> 4, minlength=16)
    return counts.astype(np.float64)


def _distribution(counts: np.ndarray) -> np.ndarray:
    total = float(counts.sum())
    if total <= 0.0:
        return np.full(16, 1.0 / 16.0, dtype=np.float64)
    return counts / total


def _nibble_source_chunks(data: bytes) -> tuple[tuple[bytes, bytes], ...]:
    source_bytes = max(1, huff0.max_block_size() // 2)
    return tuple(
        (part, _nibble_stream(part))
        for start in range(0, len(data), source_bytes)
        if (part := data[start : start + source_bytes])
    )


def _raw_chunk_cost(data: bytes) -> int:
    return CHUNK_HEADER_BYTES + len(data)


def _prepare_shared_plan(
    samples: tuple[bytes, ...],
    *,
    raw_chunk_bytes: int,
    js_threshold: float,
    max_cluster_sample_bytes: int,
) -> SharedPlan:
    clusters: list[dict] = []
    assignments = []
    for data in _iter_raw_chunks(samples, raw_chunk_bytes):
        counts = _nibble_counts(data)
        distribution = _distribution(counts)
        assigned = None
        for cluster_index, cluster in enumerate(clusters):
            if _js_distance(distribution, cluster["distribution"]) <= js_threshold:
                assigned = cluster_index
                break
        if assigned is None:
            clusters.append(
                {
                    "counts": counts.copy(),
                    "distribution": distribution.copy(),
                    "sample": bytearray(),
                    "sample_bytes": 0,
                }
            )
            assigned = len(clusters) - 1
        else:
            cluster = clusters[assigned]
            cluster["counts"] += counts
            cluster["distribution"] = _distribution(cluster["counts"])
        assignments.append(assigned)

        cluster = clusters[assigned]
        remaining = max_cluster_sample_bytes - int(cluster["sample_bytes"])
        if remaining > 0:
            sample = data[:remaining]
            cluster["sample"].extend(_nibble_stream(sample))
            cluster["sample_bytes"] += len(sample)

    if not clusters:
        raise ValueError("No sampled data.")
    codebooks = tuple(
        huff0.build_codebook(bytes(cluster["sample"]), prior_count=1)
        for cluster in clusters
    )
    return SharedPlan(
        codebooks=codebooks,
        assignments=tuple(assignments),
        clusters=len(clusters),
    )


def _encode_local(
    samples: tuple[bytes, ...],
    *,
    raw_chunk_bytes: int,
    threads: int,
) -> EncodedArchive:
    source_blocks = [
        pair
        for data in _iter_raw_chunks(samples, raw_chunk_bytes)
        for pair in _nibble_source_chunks(data)
    ]
    payloads = huff0.compress_many(
        (nibbles for _, nibbles in source_blocks),
        min_saving_bytes=0,
        threads=threads,
    )
    blocks = []
    compressed_bytes = 0
    huffman_chunks = 0
    raw_chunks = 0
    for (source, _), payload in zip(source_blocks, payloads, strict=True):
        raw_cost = _raw_chunk_cost(source)
        encoded_cost = CHUNK_HEADER_BYTES + len(payload) if payload else raw_cost
        if payload is not None and encoded_cost < raw_cost:
            blocks.append(EncodedBlock("local_huff0", source, payload))
            compressed_bytes += encoded_cost
            huffman_chunks += 1
        else:
            blocks.append(EncodedBlock("raw", source, source))
            compressed_bytes += raw_cost
            raw_chunks += 1
    return EncodedArchive(
        blocks=tuple(blocks),
        compressed_bytes=compressed_bytes,
        huffman_chunks=huffman_chunks,
        raw_chunks=raw_chunks,
        referenced_codebooks=0,
        codebook_metadata_bytes=0,
    )


def _encode_shared(
    samples: tuple[bytes, ...],
    plan: SharedPlan,
    *,
    raw_chunk_bytes: int,
    threads: int,
) -> EncodedArchive:
    raw_chunks = tuple(_iter_raw_chunks(samples, raw_chunk_bytes))
    if len(raw_chunks) != len(plan.assignments):
        raise RuntimeError("Shared plan does not cover every raw chunk.")
    source_blocks = []
    for data, cluster_index in zip(raw_chunks, plan.assignments, strict=True):
        source_blocks.extend(
            (source, nibbles, cluster_index)
            for source, nibbles in _nibble_source_chunks(data)
        )

    encoded_by_index = [None] * len(source_blocks)
    for cluster_index, codebook in enumerate(plan.codebooks):
        indices = [
            index
            for index, (_, _, assigned) in enumerate(source_blocks)
            if assigned == cluster_index
        ]
        if not indices:
            continue
        payloads = huff0.encode_many_with_codebook(
            (source_blocks[index][1] for index in indices),
            codebook,
            min_saving_bytes=0,
            compression_threshold=1.0,
            metadata_bytes=SHARED_CHUNK_HEADER_BYTES,
            threads=threads,
        )
        for index, payload in zip(indices, payloads, strict=True):
            encoded_by_index[index] = payload

    blocks = []
    payload_bytes = 0
    huffman_chunks = 0
    raw_fallback_chunks = 0
    referenced_clusters = set()
    for (source, _, cluster_index), payload in zip(
        source_blocks, encoded_by_index, strict=True
    ):
        raw_cost = _raw_chunk_cost(source)
        encoded_cost = (
            SHARED_CHUNK_HEADER_BYTES + len(payload) if payload else raw_cost
        )
        if payload is not None and encoded_cost < raw_cost:
            blocks.append(
                EncodedBlock("shared_huff0", source, payload, cluster_index)
            )
            payload_bytes += encoded_cost
            huffman_chunks += 1
            referenced_clusters.add(cluster_index)
        else:
            blocks.append(EncodedBlock("raw", source, source))
            payload_bytes += raw_cost
            raw_fallback_chunks += 1

    codebook_bytes = sum(
        CODEBOOK_HEADER_BYTES + len(plan.codebooks[index].codebook)
        for index in sorted(referenced_clusters)
    )
    return EncodedArchive(
        blocks=tuple(blocks),
        compressed_bytes=payload_bytes + codebook_bytes,
        huffman_chunks=huffman_chunks,
        raw_chunks=raw_fallback_chunks,
        referenced_codebooks=len(referenced_clusters),
        codebook_metadata_bytes=codebook_bytes,
    )


def _verify_archive(
    archive: EncodedArchive,
    *,
    shared_plan: SharedPlan | None = None,
) -> None:
    decodebooks = None
    if shared_plan is not None:
        decodebooks = tuple(
            huff0.build_decodebook(codebook.codebook)
            for codebook in shared_plan.codebooks
        )
    for block_index, block in enumerate(archive.blocks):
        if block.mode == "raw":
            restored = block.payload
        elif block.mode == "local_huff0":
            nibble_bytes = huff0.decompress_block(
                block.payload, 2 * len(block.source)
            )
            restored = _bytes_from_nibbles(nibble_bytes)
        elif block.mode == "shared_huff0":
            if decodebooks is None or block.cluster_index is None:
                raise RuntimeError("Shared block is missing its decodebook.")
            codebook = shared_plan.codebooks[block.cluster_index]
            payload = huff0.Huff0Payload(
                block.payload,
                codebook.codebook,
                2 * len(block.source),
            )
            nibble_bytes = huff0.decompress_with_decodebook(
                payload, decodebooks[block.cluster_index]
            )
            restored = _bytes_from_nibbles(nibble_bytes)
        else:
            raise RuntimeError(f"Unknown encoded mode: {block.mode}")
        if restored != block.source:
            raise RuntimeError(f"Round-trip failed for block {block_index}.")


def _measure_encoder(encoder, *, shared_plan=None) -> tuple[EncodedArchive, float]:
    warmup = encoder()
    del warmup
    gc.collect()
    started = time.perf_counter()
    archive = encoder()
    encode_seconds = time.perf_counter() - started
    _verify_archive(archive, shared_plan=shared_plan)
    return archive, encode_seconds


def _result_row(
    *,
    spec: dict,
    manifest: dict,
    compressor: str,
    archive: EncodedArchive,
    precompute_seconds: float,
    encode_seconds: float,
    clusters: int,
    threads: int,
    raw_chunk_bytes: int,
    max_cluster_sample_bytes: int,
    js_threshold: float,
    measurement_order: int,
) -> dict:
    raw_bytes = sum(int(row["sample_bytes"]) for row in manifest["files"])
    available_tensor_bytes = sum(
        max(0, int(row["repository_file_size"]) - int(row["data_offset"]))
        for row in manifest["files"]
    )
    ratio = 100.0 * archive.compressed_bytes / raw_bytes if raw_bytes else None
    return {
        "experiment": "lossy-lossless-quantized-nibbles-final",
        "model": str(spec["name"]),
        "model_id": str(spec["model_id"]),
        "representation": str(spec["representation"]),
        "resolved_revision": manifest["resolved_revision"],
        "sample_fingerprint": manifest["sample_fingerprint"],
        "compressor": compressor,
        "status": "ok",
        "files": len(manifest["files"]),
        "sampled_files": len(manifest["files"]),
        "sampled_bytes": raw_bytes,
        "available_tensor_bytes": available_tensor_bytes,
        "sample_coverage_pct": (
            100.0 * raw_bytes / available_tensor_bytes
            if available_tensor_bytes
            else None
        ),
        "full_tensor_payload_covered": raw_bytes == available_tensor_bytes,
        "raw_bytes": raw_bytes,
        "compressed_bytes": archive.compressed_bytes,
        "ratio_pct": ratio,
        "compressed_pct": ratio,
        "precompute_seconds": precompute_seconds,
        "encoding_seconds": encode_seconds,
        "encode_seconds": encode_seconds,
        "huffman_chunks": archive.huffman_chunks,
        "raw_chunks": archive.raw_chunks,
        "codebooks": archive.referenced_codebooks,
        "codebook_metadata_bytes": archive.codebook_metadata_bytes,
        "clusters": clusters,
        "threads": threads,
        "warmup_repetitions": 1,
        "measured_repetitions": 1,
        "measurement_order": measurement_order,
        "roundtrip_ok": True,
        "raw_chunk_bytes": raw_chunk_bytes,
        "max_sample_bytes_per_file": manifest["max_sample_bytes_per_file"],
        "max_cluster_sample_bytes": max_cluster_sample_bytes,
        "cluster_js_threshold": js_threshold if clusters else None,
    }


def _run_model(
    spec: dict,
    manifest: dict,
    *,
    model_dir: Path,
    threads: int,
    raw_chunk_bytes: int,
    max_cluster_sample_bytes: int,
    js_threshold: float,
    model_index: int,
) -> list[dict]:
    samples = _load_samples(manifest, model_dir)
    precompute_started = time.perf_counter()
    shared_plan = _prepare_shared_plan(
        samples,
        raw_chunk_bytes=raw_chunk_bytes,
        js_threshold=js_threshold,
        max_cluster_sample_bytes=max_cluster_sample_bytes,
    )
    precompute_seconds = time.perf_counter() - precompute_started

    encoders = {
        "zipnn-nibble-local": (
            lambda: _encode_local(
                samples,
                raw_chunk_bytes=raw_chunk_bytes,
                threads=threads,
            ),
            None,
        ),
        "neuralzip-nibble-js": (
            lambda: _encode_shared(
                samples,
                shared_plan,
                raw_chunk_bytes=raw_chunk_bytes,
                threads=threads,
            ),
            shared_plan,
        ),
    }
    order = list(encoders)
    if model_index % 2:
        order.reverse()

    rows = []
    for measurement_order, compressor in enumerate(order, start=1):
        print(f"  compressor={compressor}", flush=True)
        encoder, verification_plan = encoders[compressor]
        archive, encode_seconds = _measure_encoder(
            encoder,
            shared_plan=verification_plan,
        )
        rows.append(
            _result_row(
                spec=spec,
                manifest=manifest,
                compressor=compressor,
                archive=archive,
                precompute_seconds=(
                    precompute_seconds
                    if compressor == "neuralzip-nibble-js"
                    else 0.0
                ),
                encode_seconds=encode_seconds,
                clusters=(
                    shared_plan.clusters
                    if compressor == "neuralzip-nibble-js"
                    else 0
                ),
                threads=threads,
                raw_chunk_bytes=raw_chunk_bytes,
                max_cluster_sample_bytes=max_cluster_sample_bytes,
                js_threshold=js_threshold,
                measurement_order=measurement_order,
            )
        )
        del archive
        gc.collect()
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _backend_fingerprint() -> dict:
    extension = Path(huff0._huff0_cpp.__file__).resolve()
    config = Path(__file__).resolve().with_name("config.py")
    return {
        "huff0_extension": str(extension),
        "huff0_extension_sha256": _sha256_file(extension),
        "runner_sha256": _sha256_file(Path(__file__).resolve()),
        "config_sha256": _sha256_file(config),
    }


def _host_metadata(threads: int) -> dict:
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "logical_cpus": os.cpu_count(),
        "cpu_affinity": cpu_affinity(),
        "threads": threads,
    }


def main() -> int:
    args = parse_args()
    specs = _select_specs(args.models)
    revisions = _revision_overrides(args.revision, specs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.sample_cache_dir.mkdir(parents=True, exist_ok=True)

    manifests = {}
    if args.run_only:
        for spec in specs:
            model_dir = args.sample_cache_dir / _slug(str(spec["name"]))
            manifests[spec["model_id"]] = _validate_manifest(
                model_dir / "manifest.json",
                expected_spec=spec,
                max_sample_bytes_per_file=args.max_sample_bytes_per_file,
                requested_revision=revisions.get(spec["model_id"]),
            )
    else:
        fs = HfFileSystem()
        api = HfApi()
        for spec in specs:
            print(f"prepare model={spec['name']}", flush=True)
            manifests[spec["model_id"]] = _materialize_model(
                spec,
                cache_root=args.sample_cache_dir,
                fs=fs,
                api=api,
                requested_revision=revisions.get(spec["model_id"]),
                max_sample_bytes_per_file=args.max_sample_bytes_per_file,
                refresh=args.refresh_cache,
            )

    prepared = {
        "format_version": RESULT_FORMAT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "models": [manifests[spec["model_id"]] for spec in specs],
    }
    _write_json(args.output_dir / "prepared_samples.json", prepared)
    if args.prepare_only:
        print(f"Prepared {len(specs)} real model samples.", flush=True)
        return 0

    rows = []
    for model_index, spec in enumerate(specs):
        print(f"run model={spec['name']}", flush=True)
        model_dir = args.sample_cache_dir / _slug(str(spec["name"]))
        rows.extend(
            _run_model(
                spec,
                manifests[spec["model_id"]],
                model_dir=model_dir,
                threads=args.threads,
                raw_chunk_bytes=args.raw_chunk_bytes,
                max_cluster_sample_bytes=args.max_cluster_sample_bytes,
                js_threshold=args.cluster_js_threshold,
                model_index=model_index,
            )
        )
        _write_csv(args.output_dir / "quantized_nibble_compression.csv", rows)
        _write_json(args.output_dir / "quantized_nibble_compression.json", rows)

    metadata = {
        "format_version": RESULT_FORMAT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "lossy-lossless-quantized-nibbles-final",
        "protocol": {
            "input": (
                "real repository bytes from the tensor-data region, sampled "
                "up to the configured limit per weight shard"
            ),
            "coverage_reporting": (
                "each row reports sampled bytes, available tensor-data bytes, "
                "coverage percentage, and whether coverage is complete"
            ),
            "io_timing": "all remote and local file I/O occurs before encode",
            "warmup_repetitions": 1,
            "measured_repetitions": 1,
            "threads": args.threads,
            "raw_chunk_bytes": args.raw_chunk_bytes,
            "max_sample_bytes_per_file": args.max_sample_bytes_per_file,
            "cluster_js_threshold": args.cluster_js_threshold,
            "max_cluster_sample_bytes": args.max_cluster_sample_bytes,
            "ratio_scope": "symbols + raw fallback + referenced codebooks + metadata",
            "roundtrip": "every measured block reconstructed byte-for-byte",
        },
        "host": _host_metadata(args.threads),
        "backend": _backend_fingerprint(),
        "sample_fingerprints": {
            str(spec["model_id"]): manifests[spec["model_id"]][
                "sample_fingerprint"
            ]
            for spec in specs
        },
        "selected_specs_sha256": _canonical_digest(specs),
        "rows": len(rows),
    }
    _write_json(args.output_dir / "run_metadata.json", metadata)
    print(
        f"Wrote {len(rows)} rows to "
        f"{args.output_dir / 'quantized_nibble_compression.csv'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
