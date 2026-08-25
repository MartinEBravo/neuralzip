#!/usr/bin/env python3
"""Reproducible post-setup ZipNN/NeuralZip latency suite.

The suite intentionally measures NeuralZip in the repeated-use regime: one
setup and one verified, unmeasured warm-up are reused by every measured encode
of a checkpoint.  Exact-target materialization remains disabled.  Each model
runs in its own child process so a failure cannot contaminate later models and
all intermediate state is written atomically for safe overnight resumption.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch

from neuralzip.hashing import canonical_digest, sha256_file
from neuralzip.experiment_io import atomic_write_text as atomic_text

from examples.run_general_codec_rerun import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_COMPRESSION_THRESHOLD,
    SPECS_BY_KEY,
    host_provenance,
    load_model,
    parse_cpu_set,
    zipnn_src_pass,
)
from examples.run_latency_profile_probe import (
    atomic_json,
    measure_profile,
    prepare_profile,
)


MIB = 1 << 20
GIB = 1 << 30
PROTOCOL_VERSION = 1

# Small-to-large order for every fixed BF16 checkpoint in the direct,
# model-storage, MoE, and distilled tables. GPT-2 and BERT are included because
# they also appear in the setup/direct-encoding table.
MODEL_ORDER = (
    "efficientnet-b0",
    "resnet18",
    "efficientnet-b3",
    "resnet34",
    "resnet50",
    "convnext-tiny",
    "resnet101",
    "convnext-small",
    "resnet152",
    "efficientnet-b7",
    "gpt2",
    "bert-base",
    "convnext-base",
    "convnext-large",
    "sd15-unet",
    "qwen35-08b-base",
    "gemma3-1b",
    "llama32-1b",
    "olmo2-1b",
    "qwen25-15b",
    "qwen35-2b-base",
    "qwen25-3b",
    "llama32-3b",
    "gemma3-4b",
    "olmoe-1b-7b-0125",
    "mistral7b",
    "olmo2-7b",
    "deepseek-r1-distill-qwen-7b",
    "qwen25-7b",
    "qwen25-7b-distilled",
    "deepseek-r1-distill-llama-8b",
    "llama31-nemotron-8b",
    "lfm25-8b-a1b",
    "jetmoe-8b",
    "zaya1-8b",
    "gemma2-9b-distilled",
    "gemma3-12b",
    "mistral-nemo-12b",
    "olmo2-13b",
    "qwen15-moe-a27b",
    "phi4-14b",
    "qwen25-14b",
    "deepseek-moe-16b",
    "gemma4-26b-a4b",
    "qwen35-27b",
    "qwen3-30b-a3b",
    "gemma4-31b",
    "qwen35-35b-a3b",
    "mixtral-8x7b",
    "latamgpt-70b",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def code_provenance() -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    paths = (
        Path(__file__).resolve(),
        root / "examples" / "run_general_codec_rerun.py",
        root / "examples" / "run_latency_profile_probe.py",
        root / "neuralzip" / "compressor.py",
        root / "neuralzip" / "neuralzip_codec.py",
        root / "neuralzip" / "huff0.py",
        root / "neuralzip" / "archive.py",
        root / "neuralzip" / "_huff0_cpp.cpp",
    )
    extensions = tuple((root / "neuralzip").glob("_huff0_cpp*.so"))
    files = {
        str(path.relative_to(root)): sha256_file(path)
        for path in (*paths, *extensions)
        if path.is_file()
    }
    for module_name in ("zipnn", "zipnn_core"):
        module = importlib.import_module(module_name)
        module_path = Path(str(module.__file__)).resolve()
        module_paths = (
            [module_path]
            if module_path.is_file() and module_path.suffix != ".py"
            else sorted(module_path.parent.glob("*.py"))
        )
        for path in module_paths:
            if path.is_file():
                files[f"external/{module_name}/{path.name}"] = sha256_file(path)
    value = {
        "files": files,
        "packages": {
            name: package_version(name)
            for name in ("torch", "torchvision", "neuralzip", "zipnn")
        },
        "runtime_versions": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
        },
    }
    value["digest"] = canonical_digest(value)
    return value


def protocol(args: argparse.Namespace) -> dict[str, object]:
    value = {
        "version": PROTOCOL_VERSION,
        "regime": "post-setup repeated direct encoding",
        "dtype": "bfloat16",
        "chunk_size": DEFAULT_CHUNK_SIZE,
        "compression_threshold": DEFAULT_COMPRESSION_THRESHOLD,
        "threads": args.threads,
        "cpu_set": sorted(parse_cpu_set(args.cpu_set)),
        "direct_batch_mib": args.batch_mib,
        "warmup_runs": 1,
        "warmup_roundtrip_verified": True,
        "steady_warmup_max_runs": args.steady_warmup_max,
        "steady_warmup_relative_tolerance": args.steady_warmup_tolerance,
        "steady_warmup_absolute_ms": args.steady_warmup_absolute_ms,
        "small_steady_warmup_relative_tolerance": max(
            0.20, args.steady_warmup_tolerance
        ),
        "small_steady_warmup_absolute_ms": max(
            20.0, args.steady_warmup_absolute_ms
        ),
        "repetitions": args.repetitions,
        "small_repetitions": args.small_repetitions,
        "small_bf16_gib": args.small_bf16_gib,
        "summary_statistic": "median",
        "measurement_order": (
            "independent warmed phases: ZipNN warmup and repetitions, then "
            "NeuralZip setup, warmup, and repetitions"
        ),
        "zipnn_time": "input clone plus official codec encode",
        "neuralzip_time": "compress_tensors on the same setup and codec",
        "setup_in_encode": False,
        "decode_in_encode": False,
        "metrics_in_encode": False,
        "exact_target_cache": False,
        "profiles": ["full"],
    }
    value["code"] = code_provenance()
    value["fingerprint"] = canonical_digest(value)
    return value


def reference_rows(paths: list[Path]) -> dict[str, dict[str, object]]:
    selected: dict[str, dict[str, object]] = {}
    for path in paths:
        payload = json.loads(path.read_text())
        if isinstance(payload, dict):
            payload = payload.get("results", payload.get("rows", []))
        if not isinstance(payload, list):
            raise ValueError(f"Reference results are not a row list: {path}")
        grouped: dict[str, list[dict[str, object]]] = {}
        for row in payload:
            if not isinstance(row, dict) or row.get("status") != "ok":
                continue
            key = str(row.get("model_key") or "")
            if key in SPECS_BY_KEY:
                grouped.setdefault(key, []).append(row)
        for key, rows in grouped.items():
            revisions = {str(row.get("revision")) for row in rows}
            fingerprints = {str(row.get("weights_fingerprint")) for row in rows}
            floating = {int(row.get("floating_numel") or 0) for row in rows}
            if len(revisions) != 1 or len(fingerprints) != 1 or len(floating) != 1:
                raise ValueError(f"Reference rows disagree for {key} in {path}")
            row = rows[0]
            selected[key] = {
                "revision": row.get("revision"),
                "weights_fingerprint": row.get("weights_fingerprint"),
                "floating_numel": int(row.get("floating_numel") or 0),
                "reference_results": str(path),
            }
    return selected


def build_manifest(args: argparse.Namespace) -> int:
    if not args.reference_json:
        raise SystemExit("--build-manifest requires --reference-json")
    references = [path.resolve() for path in args.reference_json]
    rows = reference_rows(references)
    requested = (
        [value.strip() for value in args.models.split(",") if value.strip()]
        if args.models
        else list(MODEL_ORDER)
    )
    unknown = [key for key in requested if key not in SPECS_BY_KEY]
    missing = [key for key in requested if key not in rows]
    if unknown:
        raise SystemExit(f"Unknown models: {', '.join(unknown)}")
    if missing:
        raise SystemExit(f"Models missing pinned provenance: {', '.join(missing)}")
    models = []
    for index, key in enumerate(requested):
        spec = SPECS_BY_KEY[key]
        models.append(
            {
                "index": index,
                "model_key": key,
                "model": spec.label,
                "family": spec.family,
                "suites": list(spec.suites),
                "spec": asdict(spec),
                **rows[key],
            }
        )
    manifest = {
        "created_at": utc_now(),
        "models": models,
        "model_count": len(models),
        "reference_results": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in references
        ],
    }
    manifest["fingerprint"] = canonical_digest(manifest)
    atomic_json(args.output.resolve(), manifest)
    print(f"manifest={args.output.resolve()} models={len(models)}", flush=True)
    return 0


def manifest_entry(manifest: dict[str, object], key: str) -> dict[str, object]:
    for entry in manifest["models"]:
        if entry["model_key"] == key:
            return entry
    raise KeyError(key)


def hf_cache_root(
    spec,
    revision: str,
    roots: list[Path],
) -> Path:
    if spec.backend != "hf":
        return roots[0]
    repository_name = f"models--{spec.source.replace('/', '--')}"
    checked = []
    for root in roots:
        snapshot = root / repository_name / "snapshots" / revision
        checked.append(str(snapshot))
        if snapshot.is_dir():
            return root
    raise FileNotFoundError(
        f"No pinned snapshot for {spec.key} revision {revision}; checked: "
        + ", ".join(checked)
    )


def median_summary(samples: list[dict[str, object]]) -> dict[str, object]:
    methods = {}
    for variant in ("zipnn-src", "neuralzip"):
        group = [row for row in samples if row["variant"] == variant]
        times = [float(row["encode_seconds"]) for row in group]
        ratios = [float(row["exponent_ratio_pct"]) for row in group]
        compressed = [int(row["exponent_compressed_bytes"]) for row in group]
        raw = [int(row["exponent_raw_bytes"]) for row in group]
        median = statistics.median(times)
        relative_mad = (
            statistics.median(abs(value - median) for value in times) / median
            if median
            else 0.0
        )
        variation_threshold = 0.20 if median < 0.5 else 0.10
        methods[variant] = {
            "repetitions": len(group),
            "encode_median_seconds": median,
            "encode_min_seconds": min(times),
            "encode_max_seconds": max(times),
            "relative_range": (max(times) - min(times)) / median if median else 0.0,
            "relative_mad": relative_mad,
            "variation_threshold": variation_threshold,
            "time_stable": relative_mad <= variation_threshold,
            "exponent_ratio_pct": statistics.median(ratios),
            "ratio_stable": len(set(ratios)) == 1,
            "payload_stable": len(set(compressed)) == 1,
            "raw_stable": len(set(raw)) == 1,
            "exponent_compressed_bytes": compressed[0],
            "exponent_raw_bytes": raw[0],
        }
        if variant == "zipnn-src":
            codec_times = [float(row["codec_encode_seconds"]) for row in group]
            copy_times = [float(row["input_copy_seconds"]) for row in group]
            methods[variant]["codec_median_seconds"] = statistics.median(codec_times)
            methods[variant]["input_copy_median_seconds"] = statistics.median(
                copy_times
            )
    zipnn = methods["zipnn-src"]
    neuralzip = methods["neuralzip"]
    if zipnn["exponent_raw_bytes"] != neuralzip["exponent_raw_bytes"]:
        raise RuntimeError("ZipNN and NeuralZip exponent scopes differ")
    return {
        "methods": methods,
        "speedup_vs_zipnn": (
            zipnn["encode_median_seconds"] / neuralzip["encode_median_seconds"]
        ),
        "ratio_delta_vs_zipnn_pp": (
            neuralzip["exponent_ratio_pct"] - zipnn["exponent_ratio_pct"]
        ),
        "high_time_variation": any(
            not value["time_stable"] for value in methods.values()
        ),
    }


def assert_uncached(codec) -> None:
    cache_bytes = int(getattr(codec, "prepacked_cache_bytes", 0))
    cache_stats = dict(getattr(codec, "prepacked_cache_stats", {}) or {})
    if cache_bytes or int(cache_stats.get("entries", 0)):
        raise RuntimeError(
            f"Exact-target cache is not empty: bytes={cache_bytes}, stats={cache_stats}"
        )


def warm_until_stable(
    label: str,
    callback,
    *,
    maximum_runs: int,
    relative_tolerance: float,
    absolute_tolerance_seconds: float,
) -> dict[str, object]:
    """Discard complete encodes until two consecutive timings converge."""

    samples = []
    converged = False
    for run in range(1, maximum_runs + 1):
        print(f"steady-warmup={label}:run-{run}", flush=True)
        metrics = callback()
        seconds = float(metrics["encode_seconds"])
        samples.append(
            {
                "run": run,
                "encode_seconds": seconds,
                "codec_encode_seconds": float(metrics["codec_encode_seconds"]),
                "input_copy_seconds": float(metrics["input_copy_seconds"]),
                "exponent_compressed_bytes": int(
                    metrics["exponent_compressed_bytes"]
                ),
                "exponent_ratio_pct": float(metrics["exponent_ratio_pct"]),
            }
        )
        gc.collect()
        if len(samples) < 2:
            continue
        previous = float(samples[-2]["encode_seconds"])
        scale = statistics.median((previous, seconds))
        tolerance = max(
            absolute_tolerance_seconds,
            relative_tolerance * scale,
        )
        if abs(seconds - previous) <= tolerance:
            converged = True
            break
    return {
        "converged": converged,
        "runs": samples,
        "maximum_runs": maximum_runs,
        "relative_tolerance": relative_tolerance,
        "absolute_tolerance_seconds": absolute_tolerance_seconds,
    }


def worker(args: argparse.Namespace) -> int:
    manifest = json.loads(args.manifest.read_text())
    entry = manifest_entry(manifest, args.worker_model)
    spec = SPECS_BY_KEY[args.worker_model]
    output = args.output_dir / "models" / f"{spec.key}.json"
    proto = protocol(args)
    state: dict[str, object] = {
        "status": "running",
        "created_at": utc_now(),
        "host": platform.node(),
        "host_provenance": host_provenance(),
        "pid": os.getpid(),
        "manifest_fingerprint": manifest["fingerprint"],
        "protocol": proto,
        "model_key": spec.key,
        "model": spec.label,
        "family": spec.family,
        "suites": list(spec.suites),
        "expected_revision": entry["revision"],
        "expected_weights_fingerprint": entry["weights_fingerprint"],
        "samples": [],
    }
    atomic_json(output, state)
    try:
        cpus = parse_cpu_set(args.cpu_set)
        if hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, cpus)
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        roots = [path.resolve() for path in args.cache_root]
        cache_root = hf_cache_root(spec, str(entry["revision"]), roots)
        load_started = time.perf_counter()
        model = load_model(
            spec,
            cache_dir=cache_root,
            allow_download=False,
            revision=str(entry["revision"]),
        )
        state["load_seconds"] = time.perf_counter() - load_started
        state["cache_root"] = str(cache_root)
        state["revision"] = model.provenance.get("revision")
        state["weights_fingerprint"] = model.provenance.get("weights_fingerprint")
        state["floating_numel"] = int(model.provenance.get("floating_numel") or 0)
        if state["revision"] != entry["revision"]:
            raise RuntimeError(
                f"Revision mismatch: {state['revision']} != {entry['revision']}"
            )
        if state["weights_fingerprint"] != entry["weights_fingerprint"]:
            raise RuntimeError("Checkpoint fingerprint differs from the pinned manifest")
        if state["floating_numel"] != int(entry["floating_numel"]):
            raise RuntimeError("Floating parameter count differs from the pinned manifest")
        bf16_gib = int(state["floating_numel"]) * 2 / GIB
        is_small = bf16_gib <= args.small_bf16_gib
        repetitions = args.small_repetitions if is_small else args.repetitions
        warmup_relative_tolerance = (
            max(0.20, args.steady_warmup_tolerance)
            if is_small
            else args.steady_warmup_tolerance
        )
        warmup_absolute_seconds = (
            max(20.0, args.steady_warmup_absolute_ms) / 1000.0
            if is_small
            else args.steady_warmup_absolute_ms / 1000.0
        )
        state["measured_repetitions"] = repetitions
        state["bf16_gib"] = bf16_gib
        state["small_timing_regime"] = is_small
        atomic_json(output, state)

        print(f"warmup={spec.key}:zipnn-src", flush=True)
        zipnn_warmup = zipnn_src_pass(model, threads=args.threads, verify=True)
        if zipnn_warmup.get("roundtrip_ok") is not True:
            raise RuntimeError("ZipNN verified warm-up failed")
        state["zipnn_warmup"] = zipnn_warmup
        gc.collect()

        # Verification performs decoding and may leave allocator/page state
        # unlike the repeated encode workload.  Discard one immediate encode
        # before collecting the steady ZipNN samples.
        state["zipnn_steady_warmups"] = warm_until_stable(
            f"{spec.key}:zipnn-src",
            lambda: zipnn_src_pass(
                model,
                threads=args.threads,
                verify=False,
            ),
            maximum_runs=args.steady_warmup_max,
            relative_tolerance=warmup_relative_tolerance,
            absolute_tolerance_seconds=warmup_absolute_seconds,
        )
        if not state["zipnn_steady_warmups"]["converged"]:
            raise RuntimeError("ZipNN steady warm-up did not converge")

        for repetition in range(1, repetitions + 1):
            print(
                f"measure={spec.key}:zipnn-src:repeat-{repetition}",
                flush=True,
            )
            metrics = zipnn_src_pass(
                model,
                threads=args.threads,
                verify=False,
            )
            state["samples"].append(
                {
                    "variant": "zipnn-src",
                    "repetition": repetition,
                    "measured_at": utc_now(),
                    **metrics,
                }
            )
            atomic_json(output, state)
            gc.collect()

        print(f"setup={spec.key}:neuralzip", flush=True)
        profile = prepare_profile(
            model,
            "full",
            args.threads,
            verify_warmup=True,
            direct_batch_bytes=args.batch_mib * MIB,
        )
        codec = profile["compressor"].codec
        assert_uncached(codec)
        state["neuralzip_setup"] = {
            "clustering_seconds": profile["clustering_seconds"],
            "codebook_setup_seconds": profile["codebook_setup_seconds"],
            "setup_seconds": profile["setup_seconds"],
            "warmup_seconds": profile["warmup_seconds"],
            "warmup_roundtrip_ok": profile["verified_warmup"],
            "warmup_metrics": profile["warmup_metrics"],
            "selected_mode_streams": profile["selected_mode_streams"],
            "selection_stats": profile["selection_stats"],
            "direct_batch_bytes": int(codec.direct_batch_bytes),
            "prepacked_cache_bytes": int(
                getattr(codec, "prepacked_cache_bytes", 0)
            ),
            "prepacked_cache_stats": dict(
                getattr(codec, "prepacked_cache_stats", {}) or {}
            ),
        }
        atomic_json(output, state)

        # The verified NeuralZip warm-up is followed by decoding, so discard a
        # second encode immediately before the measured repeated-use phase.
        def neuralzip_warmup():
            assert_uncached(codec)
            metrics = measure_profile(profile)
            assert_uncached(codec)
            return metrics

        state["neuralzip_steady_warmups"] = warm_until_stable(
            f"{spec.key}:neuralzip",
            neuralzip_warmup,
            maximum_runs=args.steady_warmup_max,
            relative_tolerance=warmup_relative_tolerance,
            absolute_tolerance_seconds=warmup_absolute_seconds,
        )
        if not state["neuralzip_steady_warmups"]["converged"]:
            raise RuntimeError("NeuralZip steady warm-up did not converge")
        for repetition in range(1, repetitions + 1):
            print(
                f"measure={spec.key}:neuralzip:repeat-{repetition}",
                flush=True,
            )
            assert_uncached(codec)
            metrics = measure_profile(profile)
            assert_uncached(codec)
            state["samples"].append(
                {
                    "variant": "neuralzip",
                    "repetition": repetition,
                    "measured_at": utc_now(),
                    **metrics,
                }
            )
            atomic_json(output, state)
            gc.collect()

        state["summary"] = median_summary(state["samples"])
        for method in state["summary"]["methods"].values():
            if not method["ratio_stable"] or not method["payload_stable"]:
                raise RuntimeError("Repeated payload or ratio changed")
        if state["summary"]["high_time_variation"]:
            raise RuntimeError("Measured timings remained unstable after warm-up")
        state["status"] = "complete"
        state["finished_at"] = utc_now()
        atomic_json(output, state)
        print(
            f"complete={spec.key} speedup={state['summary']['speedup_vs_zipnn']:.4f}x",
            flush=True,
        )
        return 0
    except BaseException as error:
        state["status"] = "failed"
        state["finished_at"] = utc_now()
        state["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        atomic_json(output, state)
        traceback.print_exc()
        return 1


def result_is_reusable(
    path: Path,
    *,
    manifest_fingerprint: str,
    protocol_fingerprint: str,
) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    return bool(
        payload.get("status") == "complete"
        and payload.get("manifest_fingerprint") == manifest_fingerprint
        and payload.get("protocol", {}).get("fingerprint") == protocol_fingerprint
    )


def write_suite_summary(
    output_dir: Path,
    manifest: dict[str, object],
    proto: dict[str, object],
    selected_keys: set[str] | None = None,
) -> dict[str, object]:
    completed = []
    failed = []
    table_rows = []
    for entry in manifest["models"]:
        if selected_keys is not None and entry["model_key"] not in selected_keys:
            continue
        path = output_dir / "models" / f"{entry['model_key']}.json"
        if not path.is_file():
            failed.append({"model_key": entry["model_key"], "reason": "missing"})
            continue
        result = json.loads(path.read_text())
        if result.get("status") != "complete":
            failed.append(
                {
                    "model_key": entry["model_key"],
                    "reason": result.get("error", {}).get("message", result.get("status")),
                }
            )
            continue
        completed.append(entry["model_key"])
        summary = result["summary"]
        for variant, method in summary["methods"].items():
            table_rows.append(
                {
                    "model_key": entry["model_key"],
                    "model": entry["model"],
                    "family": entry["family"],
                    "compressor": "ZipNN" if variant == "zipnn-src" else "NeuralZip",
                    "repetitions": method["repetitions"],
                    "encode_median_seconds": method["encode_median_seconds"],
                    "encode_min_seconds": method["encode_min_seconds"],
                    "encode_max_seconds": method["encode_max_seconds"],
                    "relative_range": method["relative_range"],
                    "relative_mad": method["relative_mad"],
                    "variation_threshold": method["variation_threshold"],
                    "time_stable": method["time_stable"],
                    "exponent_ratio_pct": method["exponent_ratio_pct"],
                    "speedup_vs_zipnn": summary["speedup_vs_zipnn"],
                    "ratio_delta_vs_zipnn_pp": summary["ratio_delta_vs_zipnn_pp"],
                    "setup_seconds": (
                        0.0
                        if variant == "zipnn-src"
                        else result["neuralzip_setup"]["setup_seconds"]
                    ),
                    "high_time_variation": summary["high_time_variation"],
                }
            )
    summary_payload = {
        "updated_at": utc_now(),
        "status": "complete" if not failed else "incomplete",
        "manifest_fingerprint": manifest["fingerprint"],
        "protocol": proto,
        "total_models": (
            len(selected_keys) if selected_keys is not None else len(manifest["models"])
        ),
        "completed_models": completed,
        "failed_models": failed,
        "rows": table_rows,
    }
    atomic_json(output_dir / "results.json", summary_payload)
    columns = (
        list(table_rows[0])
        if table_rows
        else ["model_key", "model", "compressor"]
    )
    from io import StringIO

    stream = StringIO()
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows(table_rows)
    atomic_text(output_dir / "results.csv", stream.getvalue())
    return summary_payload


def parent(args: argparse.Namespace) -> int:
    manifest = json.loads(args.manifest.read_text())
    proto = protocol(args)
    output_dir = args.output_dir.resolve()
    (output_dir / "models").mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "manifest.json", manifest)
    atomic_json(output_dir / "protocol.json", proto)
    requested = (
        {value.strip() for value in args.models.split(",") if value.strip()}
        if args.models
        else None
    )
    entries = [
        entry
        for entry in manifest["models"]
        if requested is None or entry["model_key"] in requested
    ]
    selected_keys = {entry["model_key"] for entry in entries}
    if requested is not None:
        missing = requested - {entry["model_key"] for entry in entries}
        if missing:
            raise SystemExit(f"Models absent from manifest: {', '.join(sorted(missing))}")
    progress = {
        "status": "running",
        "created_at": utc_now(),
        "host": platform.node(),
        "pid": os.getpid(),
        "models": [entry["model_key"] for entry in entries],
        "completed": [],
        "failed": [],
        "current": None,
    }
    atomic_json(output_dir / "progress.json", progress)
    for entry in entries:
        key = entry["model_key"]
        result_path = output_dir / "models" / f"{key}.json"
        if result_is_reusable(
            result_path,
            manifest_fingerprint=manifest["fingerprint"],
            protocol_fingerprint=proto["fingerprint"],
        ):
            print(f"skip={key}:complete", flush=True)
            progress["completed"].append(key)
            continue
        progress["current"] = key
        atomic_json(output_dir / "progress.json", progress)
        success = False
        for attempt in range(1, args.max_attempts + 1):
            print(f"start={key}:attempt-{attempt}", flush=True)
            log_path = output_dir / "logs" / f"{key}__attempt-{attempt}.log"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--manifest",
                str(args.manifest.resolve()),
                "--output-dir",
                str(output_dir),
                "--worker-model",
                key,
                "--threads",
                str(args.threads),
                "--cpu-set",
                args.cpu_set,
                "--batch-mib",
                str(args.batch_mib),
                "--repetitions",
                str(args.repetitions),
                "--small-repetitions",
                str(args.small_repetitions),
                "--small-bf16-gib",
                str(args.small_bf16_gib),
                "--steady-warmup-max",
                str(args.steady_warmup_max),
                "--steady-warmup-tolerance",
                str(args.steady_warmup_tolerance),
                "--steady-warmup-absolute-ms",
                str(args.steady_warmup_absolute_ms),
            ]
            for root in args.cache_root:
                command.extend(("--cache-root", str(root.resolve())))
            environment = os.environ.copy()
            environment.update(
                {
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                    "TOKENIZERS_PARALLELISM": "false",
                    "PYTHONHASHSEED": "0",
                    "OMP_NUM_THREADS": str(args.threads),
                    "MKL_NUM_THREADS": str(args.threads),
                }
            )
            with log_path.open("w") as log:
                result = subprocess.run(
                    command,
                    cwd=Path(__file__).resolve().parents[1],
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            if result.returncode == 0:
                success = True
                break
            print(f"failed={key}:attempt-{attempt}:exit-{result.returncode}", flush=True)
        if success:
            progress["completed"].append(key)
        else:
            progress["failed"].append(key)
        progress["current"] = None
        progress["updated_at"] = utc_now()
        atomic_json(output_dir / "progress.json", progress)
        write_suite_summary(output_dir, manifest, proto, selected_keys)
    progress["status"] = "complete" if not progress["failed"] else "incomplete"
    progress["finished_at"] = utc_now()
    atomic_json(output_dir / "progress.json", progress)
    final = write_suite_summary(output_dir, manifest, proto, selected_keys)
    print(
        f"suite={final['status']} completed={len(final['completed_models'])}/"
        f"{final['total_models']}",
        flush=True,
    )
    return 0 if not progress["failed"] else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-manifest", action="store_true")
    parser.add_argument("--reference-json", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cache-root", type=Path, action="append", default=[])
    parser.add_argument("--models", default="")
    parser.add_argument("--worker-model", choices=tuple(SPECS_BY_KEY))
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--cpu-set", default="0-11")
    parser.add_argument("--batch-mib", type=int, default=512)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--small-repetitions", type=int, default=9)
    parser.add_argument("--small-bf16-gib", type=float, default=2.0)
    parser.add_argument("--steady-warmup-max", type=int, default=6)
    parser.add_argument("--steady-warmup-tolerance", type=float, default=0.05)
    parser.add_argument("--steady-warmup-absolute-ms", type=float, default=1.0)
    parser.add_argument("--max-attempts", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.build_manifest:
        if args.output is None:
            raise SystemExit("--build-manifest requires --output")
        return build_manifest(args)
    if args.manifest is None or args.output_dir is None:
        raise SystemExit("A run requires --manifest and --output-dir")
    if not args.cache_root:
        raise SystemExit("A run requires at least one --cache-root")
    if (
        args.batch_mib <= 0
        or args.repetitions <= 0
        or args.small_repetitions <= 0
        or args.steady_warmup_max < 2
        or args.steady_warmup_tolerance <= 0
        or args.steady_warmup_absolute_ms < 0
    ):
        raise SystemExit("Batch size and repetition counts must be positive")
    if args.worker_model:
        return worker(args)
    return parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
