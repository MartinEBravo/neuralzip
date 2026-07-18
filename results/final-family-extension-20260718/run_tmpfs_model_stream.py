#!/usr/bin/env python3
"""Stream HF checkpoints through tmpfs and run the general codec workers.

This wrapper deliberately lives outside the Codebase.  It uses the existing
runner's workers and provenance/manifest helpers without changing experiment
semantics.  Only one checkpoint is resident in the Hugging Face cache at a
time; durable output is limited to JSON receipts, job rows, and logs.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace


MODEL_KEYS = (
    "qwen35-08b-base",
    "llama32-1b",
    "gemma3-1b",
    "qwen35-2b-base",
    "llama32-3b",
    "gemma3-4b",
    "qwen25-7b",
    "olmo2-7b",
    "llama31-nemotron-8b",
    "mistral-nemo-12b",
    "gemma3-12b",
    "qwen25-14b",
    "olmo2-13b",
    "olmoe-1b-7b-0125",
    "qwen15-moe-a27b",
    "deepseek-moe-16b",
    "gemma4-26b-a4b",
    "qwen3-30b-a3b",
    "qwen35-35b-a3b",
    "mixtral-8x7b",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--ram-root",
        type=Path,
        default=Path("/dev/shm/mbravo-neuralzip-family-stream"),
    )
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--cpu-set", default="0-11")
    parser.add_argument("--retries", type=int, default=1)
    return parser.parse_args()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def safe_rmtree(path: Path, ram_root: Path) -> None:
    resolved_root = ram_root.resolve()
    resolved_path = path.resolve()
    if resolved_path == resolved_root or resolved_root not in resolved_path.parents:
        raise RuntimeError(f"Refusing unsafe tmpfs cleanup: {resolved_path}")
    if path.exists():
        shutil.rmtree(path)


def stable_artifacts(provenance: dict[str, object]) -> list[dict[str, object]]:
    keys = ("repository_path", "size_bytes", "digest_algorithm", "digest")
    return [
        {key: artifact.get(key) for key in keys}
        for artifact in provenance.get("weight_artifacts", [])
    ]


def stable_index(provenance: dict[str, object]) -> dict[str, object] | None:
    artifact = provenance.get("index_artifact")
    if not isinstance(artifact, dict):
        return None
    keys = ("repository_path", "size_bytes", "digest_algorithm", "digest")
    return {key: artifact.get(key) for key in keys}


def main() -> int:
    args = parse_args()
    args.code_root = args.code_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.ram_root = args.ram_root.resolve()
    if not str(args.ram_root).startswith("/dev/shm/"):
        raise SystemExit("--ram-root must be below /dev/shm")
    if args.threads < 1 or args.retries < 0:
        raise SystemExit("threads must be positive and retries non-negative")

    # These must be set before importing huggingface_hub/hf_xet.
    args.ram_root.mkdir(parents=True, exist_ok=True)
    xet_root = args.ram_root / "xet"
    tmp_root = args.ram_root / "tmp"
    xet_root.mkdir(exist_ok=True)
    tmp_root.mkdir(exist_ok=True)
    os.environ.update(
        {
            "HF_XET_CACHE": str(xet_root),
            "HF_XET_CHUNK_CACHE_SIZE_BYTES": "0",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HF_HUB_DOWNLOAD_TIMEOUT": "3600",
            "HF_HUB_ETAG_TIMEOUT": "60",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            "TMPDIR": str(tmp_root),
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": str(args.threads),
            "MKL_NUM_THREADS": str(args.threads),
            "OPENBLAS_NUM_THREADS": str(args.threads),
        }
    )

    sys.path.insert(0, str(args.code_root))
    from huggingface_hub import HfApi, hf_hub_download
    from examples import run_general_codec_rerun as runner

    specs = [runner.SPECS_BY_KEY[key] for key in MODEL_KEYS]
    if any(spec.backend != "hf" for spec in specs):
        raise RuntimeError("This streaming wrapper accepts only HF checkpoints.")

    output = args.output_dir
    job_dir = output / "jobs"
    receipt_dir = output / "receipts"
    log_dir = output / "logs"
    for directory in (job_dir, receipt_dir, log_dir):
        directory.mkdir(parents=True, exist_ok=True)
    lock_handle = (output / "stream.lock").open("w")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(f"Another streaming run owns {output}") from error
    for aggregate in (output / "results.json", output / "results.csv"):
        if aggregate.exists() and not (output / "manifest.json").is_file():
            aggregate.unlink()

    host = platform.node()
    host_fp = str(runner.host_provenance()["fingerprint"])
    api = HfApi()
    pin_path = output / "pinned_revisions.json"
    if pin_path.is_file():
        pins_payload = json.loads(pin_path.read_text())
        if pins_payload.get("model_keys") != list(MODEL_KEYS):
            raise RuntimeError("Pinned model order differs from this run.")
        expected_sources = {spec.key: spec.source for spec in specs}
        if pins_payload.get("sources") != expected_sources:
            raise RuntimeError("Pinned model sources differ from the current specs.")
        pins = {str(key): str(value) for key, value in pins_payload["revisions"].items()}
    else:
        pins = {}
        for spec in specs:
            info = api.model_info(spec.source, revision="main", token=True)
            if not info.sha:
                raise RuntimeError(f"Hub did not resolve a commit for {spec.source}")
            pins[spec.key] = str(info.sha)
            print(f"pin={spec.key}:{info.sha}", flush=True)
        atomic_json(
            pin_path,
            {
                "format_version": 1,
                "created_at": runner.utc_now(),
                "model_keys": list(MODEL_KEYS),
                "sources": {spec.key: spec.source for spec in specs},
                "revisions": pins,
            },
        )

    if set(pins) != set(MODEL_KEYS):
        raise RuntimeError("Pinned revision set is incomplete.")

    manifest_args = SimpleNamespace(
        threads=args.threads,
        cpu_set=args.cpu_set,
        smoke=False,
    )
    manifest_path = output / "manifest.json"
    created_at = runner.utc_now()
    if manifest_path.is_file():
        created_at = str(json.loads(manifest_path.read_text()).get("created_at") or created_at)

    def update_progress(*, complete: bool = False) -> None:
        rows = runner.read_job_rows(job_dir)
        runner.atomic_write_json(
            output / "progress.json", runner.completed_job_provenance(rows)
        )
        runner.atomic_write_json(
            manifest_path,
            runner.manifest_payload(
                manifest_args,
                specs,
                [],
                host=host,
                created_at=created_at,
                rows=rows,
                complete=complete,
            ),
        )

    update_progress(complete=False)

    def valid_row(row: dict[str, object], spec, method: str, fingerprint: str) -> bool:
        configuration = runner.configuration_payload(
            spec,
            method,
            threads=args.threads,
            cpu_set=args.cpu_set,
            smoke=False,
        )
        expected = runner.job_identity(
            host=host,
            configuration=configuration,
            weights_fingerprint=fingerprint,
        )
        return bool(
            row.get("status") == "ok"
            and row.get("host") == host
            and row.get("host_fingerprint") == host_fp
            and row.get("model_key") == spec.key
            and row.get("compressor") == runner.COMPRESSOR_LABELS[method]
            and row.get("source") == spec.source
            and row.get("revision") == pins[spec.key]
            and row.get("weights_fingerprint") == fingerprint
            and runner.prior_matches_expected(row, expected, host=host)
            and row.get("warmup_runs") == 1
            and row.get("measured_runs") == 1
            and row.get("verified_warmup") is True
            and row.get("roundtrip_ok") is True
        )

    def receipt_valid(spec) -> tuple[bool, dict[str, object] | None]:
        path = receipt_dir / f"{spec.key}.json"
        if not path.is_file():
            return False, None
        try:
            receipt = json.loads(path.read_text())
            digest = receipt.pop("receipt_digest")
            valid_digest = digest == runner.canonical_digest(receipt)
            receipt["receipt_digest"] = digest
        except Exception:
            return False, None
        if not valid_digest or receipt.get("revision") != pins[spec.key]:
            return False, receipt
        fingerprint = str(receipt.get("weights_fingerprint"))
        for method in runner.METHODS:
            job = job_dir / f"{spec.key}__{method}.json"
            if not job.is_file():
                return False, receipt
            try:
                row = json.loads(job.read_text())
            except Exception:
                return False, receipt
            if not valid_row(row, spec, method, fingerprint):
                return False, receipt
        return True, receipt

    def repo_inventory(spec, revision: str) -> tuple[set[str], dict[str, int]]:
        info = api.model_info(
            spec.source,
            revision=revision,
            files_metadata=True,
            token=True,
        )
        sizes = {
            str(item.rfilename): int(item.size or 0)
            for item in info.siblings
        }
        return set(sizes), sizes

    def download_exact(spec, revision: str, hub_cache: Path) -> tuple[list[str], int]:
        names, sizes = repo_inventory(spec, revision)
        prefix = f"{spec.subfolder}/" if spec.subfolder else ""
        if spec.weight_file:
            index_name = None
            files = [spec.weight_file]
        else:
            index_candidates = (
                f"{prefix}model.safetensors.index.json",
                f"{prefix}pytorch_model.bin.index.json",
            )
            index_name = next((name for name in index_candidates if name in names), None)
            if index_name:
                index_path = Path(
                    hf_hub_download(
                        spec.source,
                        index_name,
                        revision=revision,
                        cache_dir=hub_cache,
                        token=True,
                    )
                )
                index = json.loads(index_path.read_text())
                weight_map = index.get("weight_map")
                if not isinstance(weight_map, dict) or not weight_map:
                    raise RuntimeError(f"Empty weight map: {spec.source}/{index_name}")
                files = sorted({str(value) for value in weight_map.values()})
                files.insert(0, index_name)
            else:
                candidates = (
                    f"{prefix}model.safetensors",
                    f"{prefix}diffusion_pytorch_model.safetensors",
                    f"{prefix}pytorch_model.bin",
                )
                selected = next((name for name in candidates if name in names), None)
                if selected is None:
                    raise RuntimeError(
                        f"No canonical indexed or monolithic checkpoint in {spec.source}"
                    )
                files = [selected]

        missing = [name for name in files if name not in names]
        unsafe = [
            name
            for name in files
            if Path(name).is_absolute() or ".." in Path(name).parts
        ]
        if missing or unsafe:
            raise RuntimeError(
                f"Invalid canonical selection for {spec.source}: missing={missing}, unsafe={unsafe}"
            )
        selected_bytes = sum(sizes.get(name, 0) for name in files)
        free = shutil.disk_usage(args.ram_root).free
        margin = max(16 << 30, selected_bytes // 4)
        if selected_bytes and free < selected_bytes + margin:
            raise RuntimeError(
                f"tmpfs lacks headroom for {spec.key}: need "
                f"{(selected_bytes + margin)/(1<<30):.1f} GiB, have {free/(1<<30):.1f} GiB"
            )
        meminfo = Path("/proc/meminfo")
        if selected_bytes and meminfo.is_file():
            fields = {}
            for line in meminfo.read_text().splitlines():
                if ":" in line:
                    name, value = line.split(":", 1)
                    fields[name] = int(value.strip().split()[0]) * 1024
            available_memory = fields.get("MemAvailable")
            required_memory = 3 * selected_bytes + (32 << 30)
            if available_memory is not None and available_memory < required_memory:
                raise RuntimeError(
                    f"RAM lacks conservative headroom for {spec.key}: need "
                    f"{required_memory/(1<<30):.1f} GiB, have "
                    f"{available_memory/(1<<30):.1f} GiB"
                )
        print(
            f"download-plan={spec.key} files={len(files)} "
            f"selected_gib={selected_bytes/(1<<30):.2f} free_gib={free/(1<<30):.2f}",
            flush=True,
        )
        for position, name in enumerate(files, 1):
            if index_name and name == index_name:
                continue
            for attempt in range(1, 4):
                try:
                    print(
                        f"fetch={spec.key}:{position}/{len(files)}:{name}:attempt-{attempt}",
                        flush=True,
                    )
                    hf_hub_download(
                        spec.source,
                        name,
                        revision=revision,
                        cache_dir=hub_cache,
                        token=True,
                    )
                    break
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(5 * attempt)
        return files, selected_bytes

    failed_models: list[str] = []
    for index, spec in enumerate(specs):
        already_valid, _ = receipt_valid(spec)
        if already_valid:
            print(f"skip-complete={spec.key}", flush=True)
            continue

        model_root = args.ram_root / "models" / spec.key
        safe_rmtree(model_root, args.ram_root)
        safe_rmtree(xet_root, args.ram_root)
        safe_rmtree(tmp_root, args.ram_root)
        model_root.mkdir(parents=True)
        xet_root.mkdir(parents=True)
        tmp_root.mkdir(parents=True)
        hub_cache = model_root / "hub"
        hub_cache.mkdir()
        revision = pins[spec.key]
        print(f"model-start={index+1}/{len(specs)}:{spec.key}:{revision}", flush=True)

        try:
            selected_files, selected_bytes = download_exact(spec, revision, hub_cache)
            selection = runner.snapshot_weight_files(
                spec,
                cache_dir=hub_cache,
                allow_download=False,
                revision=revision,
            )
            actual_files = [str(path.relative_to(selection.snapshot)) for path in selection.files]
            expected_weights = [name for name in selected_files if not name.endswith(".index.json")]
            if actual_files != sorted(expected_weights):
                raise RuntimeError(
                    f"Runner selection differs for {spec.key}: {actual_files} != {sorted(expected_weights)}"
                )
            provenance = runner.hf_provenance(spec, selection)
            fingerprint = str(provenance["weights_fingerprint"])

            for method in runner.methods_for_model(index):
                job_path = job_dir / f"{spec.key}__{method}.json"
                prior = {}
                if job_path.is_file():
                    try:
                        prior = json.loads(job_path.read_text())
                    except Exception:
                        pass
                if valid_row(prior, spec, method, fingerprint):
                    print(f"skip-job={spec.key}:{method}", flush=True)
                    continue

                command = [
                    sys.executable,
                    str(args.code_root / "examples" / "run_general_codec_rerun.py"),
                    "--worker-model",
                    spec.key,
                    "--worker-method",
                    method,
                    "--worker-output",
                    str(job_path),
                    "--worker-revision",
                    revision,
                    "--cache-dir",
                    str(hub_cache),
                    "--threads",
                    str(args.threads),
                    "--cpu-set",
                    args.cpu_set,
                ]
                succeeded = False
                for attempt in range(1, args.retries + 2):
                    log_path = log_dir / f"{spec.key}__{method}__attempt-{attempt}.log"
                    print(f"worker-start={spec.key}:{method}:attempt-{attempt}", flush=True)
                    environment = os.environ.copy()
                    environment["HF_HUB_CACHE"] = str(hub_cache)
                    existing_pythonpath = environment.get("PYTHONPATH")
                    environment["PYTHONPATH"] = str(args.code_root) + (
                        os.pathsep + existing_pythonpath if existing_pythonpath else ""
                    )
                    with log_path.open("w") as log:
                        completed = subprocess.run(
                            command,
                            cwd=args.code_root,
                            env=environment,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            check=False,
                        )
                    try:
                        row = json.loads(job_path.read_text())
                    except Exception:
                        row = {}
                    succeeded = completed.returncode == 0 and valid_row(
                        row, spec, method, fingerprint
                    )
                    print(
                        f"worker-end={spec.key}:{method}:attempt-{attempt}:"
                        f"returncode={completed.returncode}:valid={int(succeeded)}",
                        flush=True,
                    )
                    update_progress(complete=False)
                    if succeeded:
                        break
                if not succeeded:
                    raise RuntimeError(f"Worker failed validation: {spec.key}:{method}")

            rows = runner.read_job_rows(job_dir)
            runner.validate_completed_pairs(
                rows,
                [spec],
                SimpleNamespace(
                    cache_dir=hub_cache,
                    smoke=False,
                    threads=args.threads,
                    cpu_set=args.cpu_set,
                ),
                host=host,
                pinned_revisions={spec.key: revision},
            )
            receipt = {
                "format_version": 1,
                "validated_at": runner.utc_now(),
                "model_key": spec.key,
                "source": spec.source,
                "revision": revision,
                "weights_fingerprint": fingerprint,
                "weight_artifacts": stable_artifacts(provenance),
                "index_artifact": stable_index(provenance),
                "selected_bytes": selected_bytes,
                "host": host,
                "host_fingerprint": host_fp,
                "code_digest": runner.code_provenance()["digest"],
                "job_fingerprints": {
                    method: json.loads(
                        (job_dir / f"{spec.key}__{method}.json").read_text()
                    )["job_fingerprint"]
                    for method in runner.METHODS
                },
            }
            receipt["receipt_digest"] = runner.canonical_digest(receipt)
            atomic_json(receipt_dir / f"{spec.key}.json", receipt)
            print(f"model-valid={spec.key}:{fingerprint}", flush=True)
        except Exception as error:
            failed_models.append(spec.key)
            print(f"model-failed={spec.key}:{type(error).__name__}:{error}", flush=True)
        finally:
            safe_rmtree(model_root, args.ram_root)
            safe_rmtree(xet_root, args.ram_root)
            safe_rmtree(tmp_root, args.ram_root)
            xet_root.mkdir(parents=True)
            tmp_root.mkdir(parents=True)
            update_progress(complete=False)
            print(
                f"cache-cleared={spec.key}:tmpfs_free_gib="
                f"{shutil.disk_usage(args.ram_root).free/(1<<30):.2f}",
                flush=True,
            )

    # Offline aggregate validation: every row was checked natively while its
    # canonical snapshot was present; receipts bind that validation to revision,
    # weight fingerprint, code, host, and both job fingerprints.
    rows = runner.read_job_rows(job_dir)
    allowed = {
        (spec.key, runner.COMPRESSOR_LABELS[method])
        for spec in specs
        for method in runner.METHODS
    }
    actual = {(str(row.get("model_key")), str(row.get("compressor"))) for row in rows}
    if len(rows) != len(allowed) or actual != allowed:
        failed_models.append("aggregate-missing-or-duplicate")
    for spec in specs:
        valid, receipt = receipt_valid(spec)
        if not valid:
            failed_models.append(spec.key)
            continue
        assert receipt is not None
        if (
            receipt.get("host") != host
            or receipt.get("host_fingerprint") != host_fp
            or receipt.get("code_digest") != runner.code_provenance()["digest"]
        ):
            failed_models.append(spec.key)

    if failed_models:
        unique_failures = list(dict.fromkeys(failed_models))
        update_progress(complete=False)
        atomic_json(
            output / "failures.json",
            {"updated_at": runner.utc_now(), "failed_models": unique_failures},
        )
        print(f"incomplete={','.join(unique_failures)}", flush=True)
        return 1

    runner.write_summary(output, expected_host=host)
    update_progress(complete=True)
    failure_path = output / "failures.json"
    if failure_path.exists():
        failure_path.unlink()
    print(f"complete=1 jobs={len(rows)} output={output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
