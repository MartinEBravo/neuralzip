#!/usr/bin/env python3
"""Benchmark one frozen NeuralZip plan across multiple family members.

Each invocation builds the structural plan once from the configured source
checkpoint, releases the source tensors, and then processes every target in
the family sequentially.  Target values are not used for histogramming,
Jensen--Shannon matching, or mode probing.  The target-specific NeuralZip and
official ZipNN paths are retained as controls but do not affect the transferred
plan.
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

from neuralzip.experiment_io import atomic_write_json
from neuralzip.hashing import sha256_bytes, sha256_file
from neuralzip.system import cpu_affinity, parse_cpu_set

from examples.run_general_codec_rerun import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_COMPRESSION_THRESHOLD,
    MappingModel,
    ModelSpec,
    SPECS_BY_KEY,
    code_provenance,
    default_hf_cache,
    host_provenance,
    iter_float_tensors,
    load_model,
    snapshot_weight_files,
    utc_now,
    zipnn_src_pass,
)
from examples.run_precompute_family_transfer import (
    compress_with_codec,
    setup_neuralzip,
)


DEFAULT_THREADS = 12


@dataclass(frozen=True)
class FamilySpec:
    key: str
    label: str
    source_key: str
    target_keys: tuple[str, ...]


EXTRA_MODEL_SPECS = (
    ModelSpec(
        "whisper-large-v3",
        "Whisper large-v3",
        "Whisper v3",
        ("family-transfer",),
        "hf",
        "openai/whisper-large-v3",
        1.55,
    ),
    ModelSpec(
        "whisper-large-v3-turbo",
        "Whisper large-v3 Turbo",
        "Whisper v3",
        ("family-transfer",),
        "hf",
        "openai/whisper-large-v3-turbo",
        0.81,
    ),
    ModelSpec(
        "siglip2-giant",
        "SigLIP 2 Giant",
        "SigLIP 2",
        ("family-transfer",),
        "hf",
        "google/siglip2-giant-opt-patch16-384",
    ),
    ModelSpec(
        "siglip2-so400m",
        "SigLIP 2 So400m",
        "SigLIP 2",
        ("family-transfer",),
        "hf",
        "google/siglip2-so400m-patch16-384",
    ),
)


MODEL_SPECS = {
    **SPECS_BY_KEY,
    **{spec.key: spec for spec in EXTRA_MODEL_SPECS},
}


# These immutable revisions are the same checkpoints used by the final storage
# run.  The two additional task families use the revisions already pinned by
# the original pairwise transfer runner.
PINNED_REVISIONS = {
    "qwen25-15b": "8faed761d45a263340a0528343f099c05c9a4323",
    "qwen25-3b": "3aab1f1954e9cc14eb9509a215f9e5ca08227a9b",
    "qwen25-7b": "d149729398750b98c0af14eb82c78cfe92750796",
    "qwen25-14b": "97e1e76335b7017d8f67c08a19d103c0504298c9",
    "qwen35-08b-base": "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68",
    "qwen35-2b-base": "b1485b2fa6dfa1287294f269f5fb618e03d52d7c",
    "qwen35-27b": "fc05daec18b0a78c049392ed2e771dde82bdf654",
    "olmo2-1b": "a1847dff35000b4271fa70afc5db10fd29fedbdf",
    "olmo2-7b": "7df9a82518afdecae4e8c026b27adccc8c1f0032",
    "olmo2-13b": "3fefddc1bf18a30e1d9b91000271630718f2aa8b",
    "gemma3-1b": "fcf18a2a879aab110ca39f8bffbccd5d49d8eb29",
    "gemma3-4b": "cc012e0a6d0787b4adcc0fa2c4da74402494554d",
    "gemma3-12b": "295efb63d01a7017928f273a94ebb86105c9526f",
    "llama32-1b": "4e20de362430cd3b72f300e6b0f18e50e7166e08",
    "llama32-3b": "13afe5124825b4f3751f836b40dafda64c1ed062",
    "whisper-large-v3": "06f233fe06e710322aca913c1bc4249a0d71fce1",
    "whisper-large-v3-turbo": "41f01f3fe87f28c78e2fbf8b568835947dd65ed9",
    "siglip2-giant": "a713301b217d38485fb2204c808367d10bc3cc40",
    "siglip2-so400m": "dd658faac399427308559e2c3ac1e99cbe43845d",
}


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
            ("qwen25-7b", "qwen25-3b", "qwen25-15b"),
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
    """Return the structural name used by the frozen family plan.

    Gemma 3 4B and 12B store the text backbone below ``language_model`` while
    the text-only 1B checkpoint stores the same module at the root.  Transfer
    is evaluated on that common language-model component, matching the
    ``Gemma3TextModel`` scope used by the original pairwise experiment.
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
        REPOSITORY_ROOT / "examples" / "run_general_codec_rerun.py",
        REPOSITORY_ROOT / "examples" / "run_precompute_family_transfer.py",
    )
    return {
        str(path.relative_to(REPOSITORY_ROOT)): sha256_file(path)
        for path in paths
    }


def model_revision(key: str) -> str | None:
    return PINNED_REVISIONS.get(key)


def checkpoint_names(
    key: str,
    *,
    cache_dir: Path,
    allow_download: bool,
) -> set[str]:
    spec = MODEL_SPECS[key]
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
                names.update(str(name) for name in handle.keys())
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


def _mode_map_signature(codec, mapping) -> tuple[object, ...]:
    return tuple(
        sorted(
            (
                repr(key),
                codec._prepacked_mode_signature(mode),
                mode.get("codebook_id"),
            )
            for key, mode in mapping.items()
        )
    )


def frozen_plan_fingerprint(codec) -> tuple[object, ...]:
    """Capture immutable encoding-relevant state from the source plan."""

    return (
        codec.export_codebooks(),
        _mode_map_signature(codec, codec._mode_by_key),
        _mode_map_signature(codec, codec._mode_by_tensor_stream),
        tuple(sorted(codec.cluster_map.items())),
        tuple(sorted(codec._computed_cluster_map.items())),
    )


def frozen_plan_sha256(fingerprint) -> str:
    return sha256_bytes(repr(fingerprint).encode("utf-8"))


def assert_frozen_plan(codec, fingerprint) -> None:
    if frozen_plan_fingerprint(codec) != fingerprint:
        raise RuntimeError("Direct transfer changed the frozen source plan.")


def benchmark_target(
    family: FamilySpec,
    source_codec,
    source_names: set[str],
    target_key: str,
    *,
    cache_dir: Path,
    allow_download: bool,
    threads: int,
    plan_fingerprint,
) -> dict[str, object]:
    spec = MODEL_SPECS[target_key]
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
            f"source plan, including: {preview}."
        )

    print(f"[{family.key}] mapping {spec.label}", flush=True)
    mapping_started = time.perf_counter()
    mapping = source_codec.prepare_direct_transfer_target(
        tensors,
        require_full_match=True,
    )
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
    source_codec.clear_prepacked_cache()
    source_codec.enable_prepacked_cache = False
    assert_frozen_plan(source_codec, plan_fingerprint)

    print(f"[{family.key}] transferred warm-up {spec.label}", flush=True)
    transfer_warmup = compress_with_codec(source_codec, tensors, verify=True)
    if transfer_warmup["roundtrip_ok"] is not True:
        raise RuntimeError(f"Transferred round-trip failed for {spec.label}.")
    transferred = compress_with_codec(source_codec, tensors)
    if source_codec.prepacked_cache_bytes != 0:
        raise RuntimeError("Transferred encoding populated an exact-target cache.")
    assert_frozen_plan(source_codec, plan_fingerprint)

    print(f"[{family.key}] target-specific control {spec.label}", flush=True)
    native_codec, native_setup = setup_neuralzip(tensors, threads=threads)
    native_codec.clear_prepacked_cache()
    native_codec.enable_prepacked_cache = False
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
        "transfer_setup": {"seconds": mapping_seconds, **mapping},
        "neuralzip_transferred_direct": {
            **transferred,
            "roundtrip_ok": True,
            "roundtrip_verified_on": "uncached warm-up",
        },
        "neuralzip_target_specific_direct": {
            **native,
            "roundtrip_ok": True,
            "roundtrip_verified_on": "uncached warm-up",
            "setup": native_setup,
        },
        "zipnn_source": {
            **zipnn,
            "roundtrip_ok": True,
            "roundtrip_verified_on": "uncached warm-up",
        },
        "verification": {
            "neuralzip_transferred_direct": True,
            "neuralzip_target_specific_direct": True,
            "zipnn_source": True,
        },
        "comparison": {
            "direct_transfer_speedup_vs_zipnn_source": (
                float(zipnn["encode_seconds"])
                / float(transferred["encode_seconds"])
            ),
            "direct_transfer_speedup_vs_target_specific": (
                float(native["encode_seconds"])
                / float(transferred["encode_seconds"])
            ),
            "direct_transfer_ratio_delta_vs_zipnn_source_pp": (
                float(transferred["exponent_pct"])
                - float(zipnn["exponent_ratio_pct"])
            ),
            "direct_transfer_ratio_delta_vs_target_specific_pp": (
                float(transferred["exponent_pct"])
                - float(native["exponent_pct"])
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
    source_spec = MODEL_SPECS[family.source_key]
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

    print(f"[{family.key}] building one frozen source plan", flush=True)
    source_codec, source_setup = setup_neuralzip(source_tensors, threads=threads)
    source_codec.clear_prepacked_cache()
    source_codec.enable_prepacked_cache = False
    plan_fingerprint = frozen_plan_fingerprint(source_codec)
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
        "source_plan_sha256": frozen_plan_sha256(plan_fingerprint),
        "target_histograms_computed": False,
        "target_js_matching": False,
        "target_mode_probing": False,
        "exact_target_cache": False,
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
                plan_fingerprint=plan_fingerprint,
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
            0
            if all(
                row["full_exact_name_mapping"] for row in result["targets"]
            )
            else 1
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
