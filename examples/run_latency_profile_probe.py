#!/usr/bin/env python3
"""Diagnose NeuralZip post-setup latency against official ZipNN.

The probe loads one checkpoint at a time and compares four production paths:

* official ZipNN source;
* NeuralZip constrained to local/raw modes;
* NeuralZip with shared codebooks but no packed exponents; and
* the complete NeuralZip selector.

Each NeuralZip setup is built once.  A verified warm-up is followed by
interleaved measured encodes that reuse the frozen setup. ZipNN's required
defensive input copy remains in its reported encoding time, matching the
thesis protocol.
"""

from __future__ import annotations

import argparse
import gc
import os
import platform
import statistics
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch

from neuralzip import Compressor
from neuralzip.archive import serialized_codebooks_size
from neuralzip.codec_protocol import measure_exponent_streams
from neuralzip.experiment_io import atomic_write_json
from neuralzip.tensors import bitwise_equal

try:
    from .run_general_codec_rerun import (
        DEFAULT_CHUNK_SIZE,
        DEFAULT_COMPRESSION_THRESHOLD,
        SPECS_BY_KEY,
        iter_float_tensors,
        load_model,
        parse_cpu_set,
        zipnn_src_pass,
    )
except ImportError:  # Direct execution from the examples directory.
    from run_general_codec_rerun import (
        DEFAULT_CHUNK_SIZE,
        DEFAULT_COMPRESSION_THRESHOLD,
        SPECS_BY_KEY,
        iter_float_tensors,
        load_model,
        parse_cpu_set,
        zipnn_src_pass,
    )


PROFILE_OPTIONS = {
    "local-only": {
        "enable_shared_codebooks": False,
        "ngram_values": (),
        "packed_source_budget_ratio": 0.0,
        "packed_time_budget_ratio": 0.0,
    },
    "sharing-only": {
        "ngram_values": (),
        "packed_source_budget_ratio": 0.0,
        "packed_time_budget_ratio": 0.0,
    },
    "full": {},
}

VARIANTS = ("zipnn-src", *PROFILE_OPTIONS)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: object) -> None:
    atomic_write_json(
        path,
        value,
        sort_keys=True,
        trailing_newline=False,
    )


def selected_mode_counts(codec) -> dict[str, int]:
    selected = codec.selected_modes
    if all(isinstance(value, int) for value in selected.values()):
        return {str(key): int(value) for key, value in sorted(selected.items())}
    counts = Counter()
    for mode in selected.values():
        kind = mode.get("kind", "unknown") if isinstance(mode, dict) else mode
        counts[str(kind)] += 1
    return dict(sorted(counts.items()))


def exponent_metrics(codec, payloads) -> dict[str, object]:
    payloads = tuple(
        payload for payload in payloads if str(payload.dtype) == "bfloat16"
    )
    codebooks = tuple(codec.export_codebooks())
    sizes = measure_exponent_streams(
        payloads,
        codebooks,
        codebook_scope="stream",
    )
    data_bytes = 0
    local_chunks = 0
    shared_chunks = 0
    packed_chunks = 0
    raw_chunks = 0
    zero_chunks = 0
    hits = 0
    escapes = 0

    for payload in payloads:
        if not payload.groups:
            continue
        group = payload.groups[int(payload.element_size) - 1]
        data_bytes += int(group.payload_bytes)
        local_chunks += int(group.local_huff0_chunks)
        shared_chunks += int(group.shared_huff0_chunks)
        packed_chunks += int(group.packed_ngram_chunks)
        raw_chunks += int(group.raw_chunks)
        zero_chunks += int(group.zero_chunks)
        hits += int(group.ngram_hits)
        escapes += int(group.ngram_escapes)

    # The archive serializer stores all codebook definitions together.  Report
    # both the exact total and the exponent-only definition bytes used by the
    # thesis tables.
    all_codebook_bytes = int(serialized_codebooks_size(codebooks))
    group_metadata_bytes = sizes.group_bytes - data_bytes
    return {
        "exponent_raw_bytes": sizes.raw_bytes,
        "exponent_data_bytes": data_bytes,
        "exponent_group_metadata_bytes": group_metadata_bytes,
        "exponent_codebook_metadata_bytes": sizes.codebook_bytes,
        "all_codebook_metadata_bytes": all_codebook_bytes,
        "exponent_compressed_bytes": sizes.compressed_bytes,
        "exponent_ratio_pct": sizes.ratio_pct if sizes.raw_bytes else None,
        "local_huff0_chunks": local_chunks,
        "shared_huff0_chunks": shared_chunks,
        "packed_ngram_chunks": packed_chunks,
        "raw_chunks": raw_chunks,
        "zero_chunks": zero_chunks,
        "ngram_hits": hits,
        "ngram_escapes": escapes,
    }


def prepare_profile(
    model,
    name: str,
    threads: int,
    *,
    verify_warmup: bool = True,
    direct_batch_bytes: int | None = None,
):
    compressor = Compressor(
        "neuralzip",
        verify=False,
        keep_payloads=False,
        threads=threads,
        chunk_size=DEFAULT_CHUNK_SIZE,
        compression_threshold=DEFAULT_COMPRESSION_THRESHOLD,
        **PROFILE_OPTIONS[name],
    )
    codec = compressor.codec
    if direct_batch_bytes is not None:
        if direct_batch_bytes <= 0:
            raise ValueError("direct_batch_bytes must be positive")
        codec.direct_batch_bytes = int(direct_batch_bytes)
    parameters = list(iter_float_tensors(model))
    codec.reset_compression()
    codec.reset_decompression()

    started = time.perf_counter()
    codec.prepare_clusters(parameters)
    clustering_seconds = time.perf_counter() - started
    started = time.perf_counter()
    codec.build_codebooks_and_select_modes(parameters)
    codebook_seconds = time.perf_counter() - started

    started = time.perf_counter()
    warmup_payloads = tuple(codec.compress_tensors(parameters))
    warmup_seconds = time.perf_counter() - started
    if len(warmup_payloads) != len(parameters):
        raise RuntimeError(f"{name}: warm-up returned the wrong tensor count")
    if verify_warmup:
        for (tensor_name, tensor), payload in zip(
            parameters, warmup_payloads, strict=True
        ):
            restored = codec.decompress_tensor(payload)
            if not bitwise_equal(tensor, restored):
                raise RuntimeError(f"{name}: round trip failed for {tensor_name}")

    warmup_metrics = exponent_metrics(codec, warmup_payloads)
    del warmup_payloads
    if verify_warmup:
        del restored
    gc.collect()
    return {
        "name": name,
        "compressor": compressor,
        "parameters": parameters,
        "clustering_seconds": clustering_seconds,
        "codebook_setup_seconds": codebook_seconds,
        "setup_seconds": clustering_seconds + codebook_seconds,
        "warmup_seconds": warmup_seconds,
        "verified_warmup": bool(verify_warmup),
        "warmup_metrics": warmup_metrics,
        "selected_mode_streams": selected_mode_counts(codec),
        "selection_stats": dict(codec.selection_stats),
    }


def measure_profile(profile) -> dict[str, object]:
    codec = profile["compressor"].codec
    started = time.perf_counter()
    payloads = tuple(codec.compress_tensors(profile["parameters"]))
    encode_seconds = time.perf_counter() - started
    metrics = exponent_metrics(codec, payloads)
    del payloads
    gc.collect()
    return {
        "encode_seconds": encode_seconds,
        "codec_encode_seconds": encode_seconds,
        "input_copy_seconds": 0.0,
        **metrics,
    }


def rotated_order(round_index: int) -> tuple[str, ...]:
    base = VARIANTS if round_index % 2 == 0 else tuple(reversed(VARIANTS))
    offset = (round_index // 2) % len(base)
    return base[offset:] + base[:offset]


def summarize(samples: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    keys = sorted({(row["model_key"], row["variant"]) for row in samples})
    for model_key, variant in keys:
        group = [
            row
            for row in samples
            if row["model_key"] == model_key and row["variant"] == variant
        ]
        times = [float(row["encode_seconds"]) for row in group]
        ratios = [float(row["exponent_ratio_pct"]) for row in group]
        rows.append(
            {
                "model_key": model_key,
                "model": group[0]["model"],
                "variant": variant,
                "repetitions": len(group),
                "encode_median_seconds": statistics.median(times),
                "encode_min_seconds": min(times),
                "encode_max_seconds": max(times),
                "exponent_ratio_pct": statistics.median(ratios),
                "ratio_stable": len(set(ratios)) == 1,
                "verified_warmup": bool(group[0].get("verified_warmup")),
                "roundtrip_ok": group[0].get("roundtrip_ok"),
                "local_huff0_chunks": group[0].get("local_huff0_chunks"),
                "shared_huff0_chunks": group[0].get("shared_huff0_chunks"),
                "packed_ngram_chunks": group[0].get("packed_ngram_chunks"),
                "raw_chunks": group[0].get("raw_chunks"),
                "zero_chunks": group[0].get("zero_chunks"),
            }
        )
    return rows


def cached_revision(spec, cache_dir: Path) -> str | None:
    if spec.backend != "hf":
        return None
    repository = cache_dir / f"models--{spec.source.replace('/', '--')}"
    main_ref = repository / "refs" / "main"
    if main_ref.is_file():
        return main_ref.read_text().strip()
    snapshots = sorted(
        path.name
        for path in (repository / "snapshots").iterdir()
        if path.is_dir()
    ) if (repository / "snapshots").is_dir() else []
    if len(snapshots) == 1:
        return snapshots[0]
    if not snapshots:
        return None
    raise RuntimeError(
        f"{spec.source} has multiple cached revisions and no main ref: {snapshots}"
    )


def run_model(spec, args, state: dict[str, object]) -> None:
    print(f"load={spec.key}", flush=True)
    revision = cached_revision(spec, args.cache_dir)
    model = load_model(
        spec,
        cache_dir=args.cache_dir,
        allow_download=args.allow_download,
        revision=revision,
    )
    model_meta = {
        "model_key": spec.key,
        "model": spec.label,
        "source": spec.source,
        "revision": model.provenance.get("revision"),
        "weights_fingerprint": model.provenance.get("weights_fingerprint"),
        "floating_numel": model.provenance.get("floating_numel"),
    }

    print(f"warmup={spec.key}:zipnn-src", flush=True)
    zipnn_warmup = zipnn_src_pass(model, threads=args.threads, verify=True)
    if zipnn_warmup.get("roundtrip_ok") is not True:
        raise RuntimeError("ZipNN warm-up failed its round trip")
    state["setups"].append(
        {
            **model_meta,
            "variant": "zipnn-src",
            "setup_seconds": 0.0,
            "warmup_seconds": zipnn_warmup["encode_seconds"],
            "roundtrip_ok": True,
        }
    )
    del zipnn_warmup
    gc.collect()

    profiles = {}
    for name in args.profiles:
        print(f"prepare={spec.key}:{name}", flush=True)
        profile = prepare_profile(
            model,
            name,
            args.threads,
            verify_warmup=not args.skip_profile_verification,
        )
        profiles[name] = profile
        state["setups"].append(
            {
                **model_meta,
                "variant": name,
                "clustering_seconds": profile["clustering_seconds"],
                "codebook_setup_seconds": profile["codebook_setup_seconds"],
                "setup_seconds": profile["setup_seconds"],
                "warmup_seconds": profile["warmup_seconds"],
                "verified_warmup": profile["verified_warmup"],
                "roundtrip_ok": True if profile["verified_warmup"] else None,
                "selected_mode_streams": profile["selected_mode_streams"],
                "selection_stats": profile["selection_stats"],
                **profile["warmup_metrics"],
            }
        )
        atomic_json(args.output, state)

    for repetition in range(args.repetitions):
        order = tuple(
            variant
            for variant in rotated_order(repetition)
            if variant == "zipnn-src" or variant in profiles
        )
        for variant in order:
            print(
                f"measure={spec.key}:{variant}:repeat-{repetition + 1}",
                flush=True,
            )
            if variant == "zipnn-src":
                metrics = zipnn_src_pass(model, threads=args.threads, verify=False)
            else:
                metrics = measure_profile(profiles[variant])
            state["samples"].append(
                {
                    **model_meta,
                    "variant": variant,
                    "repetition": repetition + 1,
                    "measured_at": utc_now(),
                    "verified_warmup": (
                        True
                        if variant == "zipnn-src"
                        else profiles[variant]["verified_warmup"]
                    ),
                    "roundtrip_ok": (
                        True
                        if variant == "zipnn-src"
                        or profiles[variant]["verified_warmup"]
                        else None
                    ),
                    **metrics,
                }
            )
            state["summary"] = summarize(state["samples"])
            atomic_json(args.output, state)

    del profiles, model
    gc.collect()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        default="resnet18,qwen25-15b,qwen25-7b,mixtral-8x7b",
        help="Comma-separated keys from run_general_codec_rerun.py.",
    )
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--cpu-set", default="0-11")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument(
        "--profiles",
        default=",".join(PROFILE_OPTIONS),
        help="Comma-separated NeuralZip profiles to run.",
    )
    parser.add_argument(
        "--skip-profile-verification",
        action="store_true",
        help="Do not decompress NeuralZip warm-ups (ZipNN remains verified).",
    )
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.profiles = tuple(
        value.strip() for value in args.profiles.split(",") if value.strip()
    )
    unknown_profiles = [name for name in args.profiles if name not in PROFILE_OPTIONS]
    if unknown_profiles:
        raise SystemExit(f"Unknown profiles: {', '.join(unknown_profiles)}")
    keys = [value.strip() for value in args.models.split(",") if value.strip()]
    unknown = [key for key in keys if key not in SPECS_BY_KEY]
    if unknown:
        raise SystemExit(f"Unknown model keys: {', '.join(unknown)}")
    cpus = parse_cpu_set(args.cpu_set)
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, cpus)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    state: dict[str, object] = {
        "status": "running",
        "created_at": utc_now(),
        "host": platform.node(),
        "pid": os.getpid(),
        "models": keys,
        "variants": ["zipnn-src", *args.profiles],
        "threads": args.threads,
        "cpu_set": sorted(cpus),
        "chunk_size": DEFAULT_CHUNK_SIZE,
        "compression_threshold": DEFAULT_COMPRESSION_THRESHOLD,
        "warmup_runs": 1,
        "measured_runs": args.repetitions,
        "profile_warmups_verified": not args.skip_profile_verification,
        "setups": [],
        "samples": [],
        "summary": [],
    }
    atomic_json(args.output, state)
    for key in keys:
        run_model(SPECS_BY_KEY[key], args, state)
    state["status"] = "complete"
    state["finished_at"] = utc_now()
    state["summary"] = summarize(state["samples"])
    atomic_json(args.output, state)
    print(f"complete={args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
