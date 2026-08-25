#!/usr/bin/env python3
"""Benchmark one reusable NeuralZip setup across multiple family members.

Each invocation builds the setup once from the configured source
checkpoint, releases the source tensors, and then processes every target in
the family sequentially.  Target values are not used for histogramming,
Jensen--Shannon matching, or mode probing.  The target-specific NeuralZip and
official ZipNN paths are retained as controls but do not affect the reused
setup.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from examples.direct_compression import zipnn_src_pass
from examples.model_catalog import SPECS_BY_KEY
from examples.model_loading import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_COMPRESSION_THRESHOLD,
    MappingModel,
    code_provenance,
    default_hf_cache,
    host_provenance,
    iter_float_tensors,
    load_model,
    snapshot_weight_files,
    utc_now,
)
from neuralzip import Compressor
from neuralzip.codec_protocol import (
    decode_tensor_batch,
    encode_tensor_batch,
    measure_exponent_streams,
    prepare_neuralzip_codec,
)
from neuralzip.experiment_io import atomic_write_json
from neuralzip.hashing import sha256_bytes, sha256_file
from neuralzip.system import cpu_affinity, parse_cpu_set
from neuralzip.tensors import bitwise_equal

DEFAULT_THREADS = 12
SHARED_MODES = {"shared_huff0", "packed_huff0"}
PACKED_MODES = {"packed_huff0"}
LOCAL_MODES = {"local_huff0"}


@dataclass(frozen=True)
class FamilySpec:
    key: str
    label: str
    source_key: str
    target_keys: tuple[str, ...]


MODELS_BY_KEY = SPECS_BY_KEY


FIXED_MODEL_MANIFEST = Path(__file__).with_name("fixed_model_manifest.json")
PINNED_REVISIONS = {
    str(row["model_key"]): str(row["revision"])
    for row in json.loads(FIXED_MODEL_MANIFEST.read_text())["models"]
}
PINNED_REVISIONS.update(
    {
        "whisper-large-v3": "06f233fe06e710322aca913c1bc4249a0d71fce1",
        "whisper-large-v3-turbo": "41f01f3fe87f28c78e2fbf8b568835947dd65ed9",
        "siglip2-giant": "a713301b217d38485fb2204c808367d10bc3cc40",
        "siglip2-so400m": "dd658faac399427308559e2c3ac1e99cbe43845d",
    }
)


FAMILY_SPECS = {
    spec.key: spec
    for spec in (
        FamilySpec(
            "resnet-basic",
            "ResNet BasicBlock",
            "resnet34",
            ("resnet18",),
        ),
        FamilySpec(
            "resnet-bottleneck",
            "ResNet Bottleneck",
            "resnet152",
            ("resnet101", "resnet50"),
        ),
        FamilySpec(
            "convnext",
            "ConvNeXt",
            "convnext-large",
            ("convnext-base", "convnext-small", "convnext-tiny"),
        ),
        FamilySpec(
            "efficientnet",
            "EfficientNet",
            "efficientnet-b7",
            ("efficientnet-b3", "efficientnet-b0"),
        ),
        FamilySpec(
            "qwen25",
            "Qwen 2.5",
            "qwen25-14b",
            ("qwen25-7b", "qwen25-3b", "qwen25-1p5b"),
        ),
        FamilySpec(
            "qwen35",
            "Qwen 3.5",
            "qwen35-27b",
            ("qwen35-2b-base", "qwen35-08b-base"),
        ),
        FamilySpec(
            "olmo2",
            "OLMo-2",
            "olmo2-13b",
            ("olmo2-7b", "olmo2-1b"),
        ),
        FamilySpec(
            "gemma3",
            "Gemma 3",
            "gemma3-12b",
            ("gemma3-4b", "gemma3-1b"),
        ),
        FamilySpec("llama32", "Llama 3.2", "llama32-3b", ("llama32-1b",)),
        FamilySpec(
            "whisper-v3",
            "Whisper v3",
            "whisper-large-v3",
            ("whisper-large-v3-turbo",),
        ),
        FamilySpec(
            "siglip2",
            "SigLIP 2",
            "siglip2-giant",
            ("siglip2-so400m",),
        ),
    )
}


GEMMA_TEXT_KEYS = {"gemma3-12b", "gemma3-4b", "gemma3-1b"}


def canonical_tensor_name(model_key: str, name: str) -> str | None:
    """Return the structural name used by the reusable family setup.

    Gemma 3 4B and 12B store the text backbone below ``language_model`` while
    the text-only 1B checkpoint stores the same module at the root.  Transfer
    is evaluated on that common language-model component, matching the
    ``Gemma3TextModel`` scope shared by the checkpoints.
    """

    name = str(name)
    if model_key in {"gemma3-12b", "gemma3-4b"}:
        prefix = "language_model."
        return name[len(prefix) :] if name.startswith(prefix) else None
    return name


def canonicalize_names(model_key: str, names) -> set[str]:
    canonical = {
        mapped
        for name in names
        if (mapped := canonical_tensor_name(model_key, str(name))) is not None
    }
    return canonical


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=tuple(FAMILY_SPECS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=default_hf_cache())
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--cpu-set", default="0-11")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument(
        "--audit-keys",
        action="store_true",
        help="Compare checkpoint tensor names without loading tensor data.",
    )
    return parser.parse_args()


def runner_sha256() -> str:
    return sha256_file(__file__)


def dependency_sha256() -> dict[str, str]:
    paths = (
        REPOSITORY_ROOT / "examples" / "model_loading.py",
        REPOSITORY_ROOT / "examples" / "direct_compression.py",
    )
    return {str(path.relative_to(REPOSITORY_ROOT)): sha256_file(path) for path in paths}


def model_revision(key: str) -> str | None:
    return PINNED_REVISIONS.get(key)


def checkpoint_names(
    key: str,
    *,
    cache_dir: Path,
    allow_download: bool,
) -> set[str]:
    spec = MODELS_BY_KEY[key]
    if spec.backend == "torchvision":
        from torchvision import models

        constructor = getattr(models, str(spec.constructor))
        module = constructor(weights=None)
        names = {
            str(name)
            for name, tensor in module.state_dict().items()
            if torch.is_tensor(tensor) and tensor.is_floating_point()
        }
        del module
        return canonicalize_names(key, names)

    selection = snapshot_weight_files(
        spec,
        cache_dir=cache_dir,
        allow_download=allow_download,
        revision=model_revision(key),
    )
    if selection.expected_keys is not None:
        return canonicalize_names(key, selection.expected_keys)

    names: set[str] = set()
    for path in selection.files:
        if path.suffix == ".safetensors":
            from safetensors import safe_open

            with safe_open(str(path), framework="pt", device="cpu") as handle:
                names.update(str(name) for name in handle)
            continue
        state = torch.load(path, map_location="meta", weights_only=True)
        names.update(str(name) for name in state)
        del state
    return canonicalize_names(key, names)


def audit_family(
    family: FamilySpec,
    *,
    cache_dir: Path,
    allow_download: bool,
) -> dict[str, object]:
    source_names = checkpoint_names(
        family.source_key,
        cache_dir=cache_dir,
        allow_download=allow_download,
    )
    targets = []
    for target_key in family.target_keys:
        target_names = checkpoint_names(
            target_key,
            cache_dir=cache_dir,
            allow_download=allow_download,
        )
        missing = sorted(target_names - source_names)
        targets.append(
            {
                "target_key": target_key,
                "target_tensors": len(target_names),
                "matched_tensors": len(target_names) - len(missing),
                "full_exact_name_mapping": not missing,
                "missing_tensors": missing,
            }
        )
    return {
        "family": asdict(family),
        "source_tensors": len(source_names),
        "targets": targets,
    }


def named_tensors(
    model: MappingModel,
    model_key: str,
) -> list[tuple[str, torch.Tensor]]:
    tensors = []
    for name, tensor in iter_float_tensors(model):
        mapped = canonical_tensor_name(model_key, str(name))
        if mapped is not None and tensor.numel() > 0:
            tensors.append((mapped, tensor))
    if not tensors:
        raise ValueError(f"{model.model_name} contains no non-empty floating tensors.")
    if len({name for name, _ in tensors}) != len(tensors):
        raise ValueError(f"{model.model_name} contains duplicate tensor names.")
    return tensors


def exponent_metrics(payloads, codebooks=()):
    payloads = tuple(payloads)
    sizes = measure_exponent_streams(payloads, tuple(codebooks))
    mode_symbols: dict[str, int] = {}
    mode_chunks: dict[str, int] = {}
    for payload in payloads:
        if not payload.groups:
            continue
        group = payload.groups[int(payload.element_size) - 1]
        for chunk in group.chunks:
            mode = str(chunk.mode)
            mode_symbols[mode] = mode_symbols.get(mode, 0) + int(chunk.symbol_count)
            mode_chunks[mode] = mode_chunks.get(mode, 0) + 1

    def symbol_pct(modes):
        symbols = sum(mode_symbols.get(mode, 0) for mode in modes)
        return 100.0 * symbols / sizes.raw_bytes if sizes.raw_bytes else 0.0

    return {
        "raw_exponents": sizes.raw_bytes,
        "compressed_exponent_bytes": sizes.compressed_bytes,
        "compressed_exponent_group_bytes": sizes.group_bytes,
        "exponent_codebook_metadata_bytes": sizes.codebook_bytes,
        "exponent_ratio_pct": sizes.ratio_pct,
        "shared_exponent_share_pct": symbol_pct(SHARED_MODES),
        "packed_exponent_share_pct": symbol_pct(PACKED_MODES),
        "local_exponent_share_pct": symbol_pct(LOCAL_MODES),
        "raw_exponent_share_pct": symbol_pct({"raw"}),
        "mode_symbols": mode_symbols,
        "mode_chunks": mode_chunks,
        "referenced_codebooks": sizes.referenced_codebooks,
    }


def compress_with_codec(codec, named_tensors, *, verify=False):
    named_tensors = list(named_tensors)
    encoded = encode_tensor_batch(codec, named_tensors)
    roundtrip_ok = None
    if verify:
        decoded, _ = decode_tensor_batch(codec, encoded.payloads)
        roundtrip_ok = all(
            name in decoded and bitwise_equal(tensor, decoded[name])
            for name, tensor in named_tensors
        )
    return {
        "encode_seconds": encoded.encode_seconds,
        "roundtrip_ok": roundtrip_ok,
        **exponent_metrics(
            encoded.payloads,
            codec.export_codebooks() if hasattr(codec, "export_codebooks") else (),
        ),
    }


def setup_neuralzip(named_tensors, *, threads):
    codec = Compressor(
        "neuralzip",
        verify=False,
        compression_threshold=DEFAULT_COMPRESSION_THRESHOLD,
        threads=threads,
    ).codec
    report = prepare_neuralzip_codec(codec, named_tensors)
    return codec, {
        "clustering_seconds": report.clustering_seconds,
        "codebook_setup_seconds": report.codebook_setup_seconds,
        "setup_seconds": report.setup_seconds,
        "clusters": int(codec.cluster_count),
        "codebooks": len(codec.export_codebooks()),
        "selected_modes": dict(codec.selected_modes),
    }


def _mode_map_signature(mapping) -> tuple[object, ...]:
    return tuple(
        sorted(
            (
                repr(key),
                (
                    str(mode.get("kind")),
                    int(mode.get("ngram_n", 1)),
                    bytes(getattr(mode.get("codebook"), "codebook", b"")),
                    bytes(mode.get("vocab", b"")),
                    bool(mode.get("frame_4x", False)),
                ),
                mode.get("codebook_id"),
            )
            for key, mode in mapping.items()
        )
    )


def frozen_setup_fingerprint(codec) -> tuple[object, ...]:
    """Capture immutable encoding-relevant state from the source setup."""

    return (
        codec.export_codebooks(),
        _mode_map_signature(codec._mode_by_key),
        _mode_map_signature(codec._mode_by_tensor_stream),
        tuple(sorted(codec.cluster_map.items())),
        tuple(sorted(codec._computed_cluster_map.items())),
    )


def frozen_setup_sha256(fingerprint) -> str:
    return sha256_bytes(repr(fingerprint).encode("utf-8"))


def assert_frozen_setup(codec, fingerprint) -> None:
    if frozen_setup_fingerprint(codec) != fingerprint:
        raise RuntimeError("Direct transfer changed the reusable source setup.")


def benchmark_target(
    family: FamilySpec,
    source_codec,
    source_names: set[str],
    target_key: str,
    *,
    cache_dir: Path,
    allow_download: bool,
    threads: int,
    setup_fingerprint,
) -> dict[str, object]:
    spec = MODELS_BY_KEY[target_key]
    print(f"[{family.key}] loading target {spec.label}", flush=True)
    target = load_model(
        spec,
        cache_dir=cache_dir,
        allow_download=allow_download,
        revision=model_revision(target_key),
    )
    tensors = named_tensors(target, target_key)
    # All controls must cover exactly the same evaluated component.  Replacing
    # the raw checkpoint mapping also releases any excluded multimodal tensors
    # before the compression passes begin.
    target.model = dict(tensors)
    target_names = {name for name, _ in tensors}
    missing = sorted(target_names - source_names)
    if missing:
        preview = ", ".join(missing[:5])
        raise RuntimeError(
            f"{spec.label} has {len(missing)} tensor name(s) outside the frozen "
            f"source setup, including: {preview}."
        )

    print(f"[{family.key}] mapping {spec.label}", flush=True)
    mapping_started = time.perf_counter()
    mapping = source_codec.prepare_direct_transfer_target(tensors)
    mapping_seconds = time.perf_counter() - mapping_started
    if int(mapping["matched_tensors"]) != len(tensors):
        raise RuntimeError(f"{spec.label} did not map every floating tensor.")
    for flag in (
        "target_histograms_computed",
        "target_js_matching",
        "target_mode_probing",
    ):
        if mapping.get(flag) is not False:
            raise RuntimeError(f"Direct transfer unexpectedly enabled {flag}.")
    assert_frozen_setup(source_codec, setup_fingerprint)

    print(f"[{family.key}] transferred warm-up {spec.label}", flush=True)
    transfer_warmup = compress_with_codec(source_codec, tensors, verify=True)
    if transfer_warmup["roundtrip_ok"] is not True:
        raise RuntimeError(f"Transferred round-trip failed for {spec.label}.")
    transferred = compress_with_codec(source_codec, tensors)
    assert_frozen_setup(source_codec, setup_fingerprint)

    print(f"[{family.key}] target-specific control {spec.label}", flush=True)
    native_codec, native_setup = setup_neuralzip(tensors, threads=threads)
    native_warmup = compress_with_codec(native_codec, tensors, verify=True)
    if native_warmup["roundtrip_ok"] is not True:
        raise RuntimeError(f"Target-specific round-trip failed for {spec.label}.")
    native = compress_with_codec(native_codec, tensors)

    print(f"[{family.key}] ZipNN source control {spec.label}", flush=True)
    zipnn_warmup = zipnn_src_pass(target, threads=threads, verify=True)
    if zipnn_warmup["roundtrip_ok"] is not True:
        raise RuntimeError(f"ZipNN source round-trip failed for {spec.label}.")
    zipnn = zipnn_src_pass(target, threads=threads, verify=False)

    output = {
        "target_key": target_key,
        "target_model": asdict(spec),
        "target_revision": target.provenance.get("revision"),
        "target_weights": target.provenance,
        "target_parameters": sum(int(tensor.numel()) for _, tensor in tensors),
        "tensor_scope": (
            "Gemma 3 text backbone"
            if target_key in GEMMA_TEXT_KEYS
            else "complete floating checkpoint"
        ),
        "transfer_mapping": {"seconds": mapping_seconds, **mapping},
        "neuralzip_transferred_direct": {
            **transferred,
            "roundtrip_ok": True,
        },
        "neuralzip_target_specific_direct": {
            **native,
            "roundtrip_ok": True,
            "setup": native_setup,
        },
        "zipnn": {
            **zipnn,
            "roundtrip_ok": True,
        },
        "verification": {
            "neuralzip_transferred_direct": True,
            "neuralzip_target_specific_direct": True,
            "zipnn": True,
        },
        "comparison": {
            "direct_transfer_speedup_vs_zipnn": (
                float(zipnn["encode_seconds"]) / float(transferred["encode_seconds"])
            ),
            "direct_transfer_speedup_vs_target_specific": (
                float(native["encode_seconds"]) / float(transferred["encode_seconds"])
            ),
            "direct_transfer_ratio_delta_vs_zipnn_pp": (
                float(transferred["exponent_ratio_pct"])
                - float(zipnn["exponent_ratio_pct"])
            ),
            "direct_transfer_ratio_delta_vs_target_specific_pp": (
                float(transferred["exponent_ratio_pct"])
                - float(native["exponent_ratio_pct"])
            ),
        },
    }

    del native_codec, tensors, target
    gc.collect()
    return output


def run_family(
    family: FamilySpec,
    *,
    cache_dir: Path,
    allow_download: bool,
    threads: int,
    progress_path: Path | None = None,
) -> dict[str, object]:
    if threads <= 0:
        raise ValueError("threads must be positive.")
    started_at = utc_now()
    source_spec = MODELS_BY_KEY[family.source_key]
    print(f"[{family.key}] loading source {source_spec.label}", flush=True)
    source = load_model(
        source_spec,
        cache_dir=cache_dir,
        allow_download=allow_download,
        revision=model_revision(family.source_key),
    )
    source_tensors = named_tensors(source, family.source_key)
    source_names = {name for name, _ in source_tensors}
    source_parameters = sum(int(tensor.numel()) for _, tensor in source_tensors)

    print(f"[{family.key}] building one reusable source setup", flush=True)
    source_codec, source_setup = setup_neuralzip(source_tensors, threads=threads)
    setup_fingerprint = frozen_setup_fingerprint(source_codec)
    source_provenance = source.provenance
    del source_tensors, source
    gc.collect()

    result = {
        "schema_version": 1,
        "complete": False,
        "started_at": started_at,
        "completed_at": None,
        "family": asdict(family),
        "host": platform.node(),
        "host_provenance": host_provenance(),
        "code_provenance": code_provenance(),
        "dtype": "bfloat16",
        "threads": threads,
        "cpu_set": cpu_affinity(),
        "chunk_size": DEFAULT_CHUNK_SIZE,
        "compression_threshold": DEFAULT_COMPRESSION_THRESHOLD,
        "warmup_repetitions": 1,
        "measured_repetitions": 1,
        "transfer_mode": "direct",
        "runner_sha256": runner_sha256(),
        "dependency_sha256": dependency_sha256(),
        "source_model": asdict(source_spec),
        "source_revision": source_provenance.get("revision"),
        "source_weights": source_provenance,
        "source_parameters": source_parameters,
        "tensor_scope": (
            "Gemma 3 text backbone"
            if family.source_key in GEMMA_TEXT_KEYS
            else "complete floating checkpoint"
        ),
        "source_setup": source_setup,
        "source_setup_sha256": frozen_setup_sha256(setup_fingerprint),
        "target_histograms_computed": False,
        "target_js_matching": False,
        "target_mode_probing": False,
        "targets": [],
    }
    if progress_path is not None:
        atomic_write_json(progress_path, result)

    for target_key in family.target_keys:
        result["targets"].append(
            benchmark_target(
                family,
                source_codec,
                source_names,
                target_key,
                cache_dir=cache_dir,
                allow_download=allow_download,
                threads=threads,
                setup_fingerprint=setup_fingerprint,
            )
        )
        if progress_path is not None:
            atomic_write_json(progress_path, result)

    result["complete"] = True
    result["completed_at"] = utc_now()
    if progress_path is not None:
        atomic_write_json(progress_path, result)
    return result


def main() -> int:
    args = parse_args()
    if args.threads <= 0:
        raise SystemExit("--threads must be positive.")
    cpu_set = parse_cpu_set(args.cpu_set)
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, cpu_set)
    if len(cpu_set) < args.threads:
        raise SystemExit("--cpu-set must contain at least --threads CPUs.")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    family = FAMILY_SPECS[args.family]
    if args.audit_keys:
        result = audit_family(
            family,
            cache_dir=args.cache_dir,
            allow_download=args.allow_download,
        )
        exit_code = (
            0 if all(row["full_exact_name_mapping"] for row in result["targets"]) else 1
        )
    else:
        result = run_family(
            family,
            cache_dir=args.cache_dir,
            allow_download=args.allow_download,
            threads=args.threads,
            progress_path=args.output,
        )
        exit_code = 0
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, default=str), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
