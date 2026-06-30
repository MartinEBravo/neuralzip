from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import HfFileSystem

from config import LOSSY_LOSSLESS_MODEL_SUITE
from neuralzip import huffman


OUTPUT_DIR = Path("examples/outputs/lossy_lossless/quantized_nibbles")
OUTPUT_CSV = OUTPUT_DIR / "quantized_nibble_compression.csv"

EXTENSIONS = (".safetensors", ".bin", ".pt", ".gguf")
RAW_CHUNK_BYTES = 256 << 10
MAX_SAMPLE_BYTES_PER_FILE = 64 << 20
CHUNK_HEADER_BYTES = 1 + 8 + 8
CODEBOOK_HEADER_BYTES = 8 + len("nibble".encode("utf-8")) + 8
CLUSTER_JS_THRESHOLD = 0.05
MAX_CLUSTER_SAMPLE_BYTES = 16 << 20


def _ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _data_offset(path: str, handle, file_size: int) -> int:
    suffix = str(path).lower()
    if suffix.endswith(".safetensors"):
        header_size_bytes = handle.read(8)
        if len(header_size_bytes) == 8:
            return 8 + int.from_bytes(header_size_bytes, "little")
        return 0
    if suffix.endswith(".gguf"):
        return min(4096, int(file_size))
    return 0


def _hf_model_files(fs: HfFileSystem, model_id: str) -> list[dict]:
    files = []
    for item in fs.ls(model_id, detail=True):
        if item.get("type") != "file":
            continue
        name = item["name"]
        if str(name).lower().endswith(EXTENSIONS):
            files.append({"path": name, "size": int(item.get("size") or 0)})
    return files


def _iter_sampled_chunks(fs: HfFileSystem, files: list[dict]):
    for file_index, item in enumerate(files):
        path = item["path"]
        size = int(item["size"])
        with fs.open(path, "rb") as handle:
            offset = _data_offset(path, handle, size)
            try:
                handle.seek(offset)
            except Exception:
                if offset:
                    handle.read(offset)

            target = min(MAX_SAMPLE_BYTES_PER_FILE, max(0, size - offset))
            sampled = 0
            chunk_index = 0
            while sampled < target:
                data = handle.read(min(RAW_CHUNK_BYTES, target - sampled))
                if not data:
                    break
                yield file_index, chunk_index, data
                sampled += len(data)
                chunk_index += 1


def _nibble_stream(data: bytes) -> bytes:
    values = np.frombuffer(data, dtype=np.uint8)
    nibbles = np.empty(values.size * 2, dtype=np.uint8)
    nibbles[0::2] = values & 0x0F
    nibbles[1::2] = values >> 4
    return nibbles.tobytes()


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


def _js_distance(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    middle = 0.5 * (left + right)
    mask_left = left > 0
    mask_right = right > 0
    divergence = 0.0
    divergence += 0.5 * np.sum(left[mask_left] * np.log2(left[mask_left] / middle[mask_left]))
    divergence += 0.5 * np.sum(right[mask_right] * np.log2(right[mask_right] / middle[mask_right]))
    return float(math.sqrt(max(divergence, 0.0)))


def _raw_chunk_cost(data: bytes) -> int:
    return CHUNK_HEADER_BYTES + len(data)


def _local_huffman_cost(data: bytes) -> tuple[int, bool]:
    nibbles = _nibble_stream(data)
    encoded = huffman.compress(nibbles, min_saving_bytes=0)
    raw_cost = _raw_chunk_cost(data)
    if encoded is None:
        return raw_cost, False
    encoded_cost = CHUNK_HEADER_BYTES + len(encoded.code_lengths) + len(encoded.payload)
    if encoded_cost >= raw_cost:
        return raw_cost, False
    return encoded_cost, True


def _shared_huffman_cost(data: bytes, codebook) -> tuple[int, bool]:
    raw_cost = _raw_chunk_cost(data)
    max_payload_bytes = max(0, raw_cost - CHUNK_HEADER_BYTES)
    encoded = huffman.encode_with_codebook(
        _nibble_stream(data),
        codebook,
        min_saving_bytes=0,
        max_output_bytes=max_payload_bytes,
    )
    if encoded is None:
        return raw_cost, False
    encoded_cost = CHUNK_HEADER_BYTES + len(encoded.payload)
    if encoded_cost >= raw_cost:
        return raw_cost, False
    return encoded_cost, True


def _common_row(
    *,
    model_name: str,
    model_id: str,
    method: str,
    raw_bytes: int,
    compressed_bytes: int,
    sampled_files: int,
    files: int,
    elapsed: float,
    huffman_chunks: int,
    raw_chunks: int,
    codebooks: int | None,
    clusters: int | None,
    precompute_seconds: float = 0.0,
    encode_seconds: float | None = None,
) -> dict:
    return {
        "experiment": "lossy-lossless-quantized-nibbles",
        "model": model_name,
        "model_id": model_id,
        "method": method,
        "status": "ok",
        "files": files,
        "sampled_files": sampled_files,
        "sampled_bytes": raw_bytes,
        "raw_bytes": raw_bytes,
        "compressed_bytes": compressed_bytes,
        "compressed_pct": 100.0 * compressed_bytes / raw_bytes if raw_bytes else None,
        "precompute_seconds": precompute_seconds,
        "encode_seconds": elapsed if encode_seconds is None else encode_seconds,
        "cold_encode_seconds": elapsed,
        "huffman_chunks": huffman_chunks,
        "raw_chunks": raw_chunks,
        "codebooks": codebooks,
        "clusters": clusters,
        "raw_chunk_bytes": RAW_CHUNK_BYTES,
        "max_sample_bytes_per_file": MAX_SAMPLE_BYTES_PER_FILE,
        "cluster_js_threshold": CLUSTER_JS_THRESHOLD if clusters is not None else None,
    }


def _compress_zipnn_local(fs: HfFileSystem, files: list[dict], model_name: str, model_id: str) -> dict:
    start = time.perf_counter()
    raw_bytes = 0
    compressed_bytes = 0
    sampled_files = set()
    huffman_chunks = 0
    raw_chunks = 0
    for file_index, _, data in _iter_sampled_chunks(fs, files):
        sampled_files.add(file_index)
        raw_bytes += len(data)
        cost, used_huffman = _local_huffman_cost(data)
        compressed_bytes += cost
        huffman_chunks += int(used_huffman)
        raw_chunks += int(not used_huffman)
    elapsed = time.perf_counter() - start
    return _common_row(
        model_name=model_name,
        model_id=model_id,
        method="zipnn-nibble-local",
        raw_bytes=raw_bytes,
        compressed_bytes=compressed_bytes,
        sampled_files=len(sampled_files),
        files=len(files),
        elapsed=elapsed,
        huffman_chunks=huffman_chunks,
        raw_chunks=raw_chunks,
        codebooks=0,
        clusters=0,
    )


def _compress_global_huffman(fs: HfFileSystem, files: list[dict], model_name: str, model_id: str) -> dict:
    precompute_start = time.perf_counter()
    representative = None
    for _, _, data in _iter_sampled_chunks(fs, files):
        if data:
            representative = _nibble_stream(data)
            break
    if representative is None:
        raise ValueError("No sampled data.")
    codebook = huffman.build_codebook(representative, prior_count=1)
    precompute_seconds = time.perf_counter() - precompute_start

    encode_start = time.perf_counter()
    raw_bytes = 0
    compressed_bytes = CODEBOOK_HEADER_BYTES + len(codebook.code_lengths)
    sampled_files = set()
    huffman_chunks = 0
    raw_chunks = 0
    for file_index, _, data in _iter_sampled_chunks(fs, files):
        sampled_files.add(file_index)
        raw_bytes += len(data)
        cost, used_huffman = _shared_huffman_cost(data, codebook)
        compressed_bytes += cost
        huffman_chunks += int(used_huffman)
        raw_chunks += int(not used_huffman)
    encode_seconds = time.perf_counter() - encode_start
    return _common_row(
        model_name=model_name,
        model_id=model_id,
        method="global-huffman-nibble",
        raw_bytes=raw_bytes,
        compressed_bytes=compressed_bytes,
        sampled_files=len(sampled_files),
        files=len(files),
        elapsed=precompute_seconds + encode_seconds,
        precompute_seconds=precompute_seconds,
        encode_seconds=encode_seconds,
        huffman_chunks=huffman_chunks,
        raw_chunks=raw_chunks,
        codebooks=1,
        clusters=None,
    )


def _prepare_neuralzip_clusters(fs: HfFileSystem, files: list[dict]):
    clusters: list[dict] = []
    raw_bytes = 0
    sampled_files = set()
    for file_index, _, data in _iter_sampled_chunks(fs, files):
        sampled_files.add(file_index)
        raw_bytes += len(data)
        counts = _nibble_counts(data)
        distribution = _distribution(counts)
        assigned = None
        for cluster_index, cluster in enumerate(clusters):
            if _js_distance(distribution, cluster["distribution"]) <= CLUSTER_JS_THRESHOLD:
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

        cluster = clusters[assigned]
        remaining = MAX_CLUSTER_SAMPLE_BYTES - int(cluster["sample_bytes"])
        if remaining > 0:
            sample = data[:remaining]
            cluster["sample"].extend(_nibble_stream(sample))
            cluster["sample_bytes"] += len(sample)

    return clusters, raw_bytes, sampled_files


def _nearest_cluster(clusters: list[dict], data: bytes) -> int:
    distribution = _distribution(_nibble_counts(data))
    return min(
        range(len(clusters)),
        key=lambda index: _js_distance(distribution, clusters[index]["distribution"]),
    )


def _compress_neuralzip(fs: HfFileSystem, files: list[dict], model_name: str, model_id: str) -> dict:
    precompute_start = time.perf_counter()
    clusters, raw_bytes_seen, sampled_files_seen = _prepare_neuralzip_clusters(fs, files)
    if not clusters:
        raise ValueError("No sampled data.")
    codebooks = [
        huffman.build_codebook(bytes(cluster["sample"]), prior_count=1)
        for cluster in clusters
    ]
    precompute_seconds = time.perf_counter() - precompute_start

    encode_start = time.perf_counter()
    raw_bytes = 0
    compressed_bytes = sum(CODEBOOK_HEADER_BYTES + len(book.code_lengths) for book in codebooks)
    sampled_files = set()
    huffman_chunks = 0
    raw_chunks = 0
    for file_index, _, data in _iter_sampled_chunks(fs, files):
        sampled_files.add(file_index)
        raw_bytes += len(data)
        cluster_index = _nearest_cluster(clusters, data)
        cost, used_huffman = _shared_huffman_cost(data, codebooks[cluster_index])
        compressed_bytes += cost
        huffman_chunks += int(used_huffman)
        raw_chunks += int(not used_huffman)
    encode_seconds = time.perf_counter() - encode_start

    if raw_bytes != raw_bytes_seen or sampled_files != sampled_files_seen:
        raise RuntimeError("Precompute and encode passes sampled different data.")

    return _common_row(
        model_name=model_name,
        model_id=model_id,
        method="neuralzip-nibble-js",
        raw_bytes=raw_bytes,
        compressed_bytes=compressed_bytes,
        sampled_files=len(sampled_files),
        files=len(files),
        elapsed=precompute_seconds + encode_seconds,
        precompute_seconds=precompute_seconds,
        encode_seconds=encode_seconds,
        huffman_chunks=huffman_chunks,
        raw_chunks=raw_chunks,
        codebooks=len(codebooks),
        clusters=len(clusters),
    )


def _append_rows(rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)


def main() -> None:
    _ensure_output_dir()
    fs = HfFileSystem()
    rows = []
    for spec in LOSSY_LOSSLESS_MODEL_SUITE:
        if spec.get("category") != "quantized":
            continue
        model_name = spec["name"]
        model_id = spec["model_id"]
        print(f"model={model_name}", flush=True)
        try:
            files = _hf_model_files(fs, model_id)
            if not files:
                raise ValueError("No supported files found.")
            for compressor in (
                _compress_global_huffman,
                _compress_zipnn_local,
                _compress_neuralzip,
            ):
                print(f"  {compressor.__name__}", flush=True)
                rows.append(compressor(fs, files, model_name, model_id))
                _append_rows(rows)
        except Exception as error:
            rows.append(
                {
                    "experiment": "lossy-lossless-quantized-nibbles",
                    "model": model_name,
                    "model_id": model_id,
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            _append_rows(rows)
    _append_rows(rows)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Wrote {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
