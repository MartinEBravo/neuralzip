#!/usr/bin/env python3
"""Incremental general rerun for official ZipNN versus NeuralZip.

This is deliberately independent from the checkpoint and federated-learning
runners.  Each job loads one pretrained checkpoint, performs one untimed
verified warmup, and then one measured encode with 12 threads by default.

ZipNN 0.5.4 mutates BF16 and FP32 tensor inputs while byte-reordering them.
Consequently, its defensive input clone is required and is part of the main
non-destructive encoding time.  The clone and codec portions are also emitted
separately for diagnostic use.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import torch

from neuralzip import Compressor


METHODS = ("zipnn-src", "neuralzip")
COMPRESSOR_LABELS = {
    "zipnn-src": "ZipNN src",
    "neuralzip": "NeuralZip",
}
DEFAULT_THREADS = 12
DEFAULT_CHUNK_SIZE = 256 << 10
DEFAULT_COMPRESSION_THRESHOLD = 0.95
MIN_PARAMETER_COVERAGE = 0.70
CODE_SUFFIXES = {".py", ".so", ".pyd", ".dylib", ".dll"}
MODEL_STORAGE_FAMILY_ORDER = {
    "ResNet": 0,
    "ConvNeXt": 1,
    "EfficientNet": 2,
    "Qwen2.5": 3,
    "OLMo 2": 4,
    "Mistral": 5,
    "Stable Diffusion": 6,
    "Phi 4": 7,
    "Qwen 3.5": 8,
    "Gemma 4": 9,
    "Llama 3": 10,
    "Llama 3 / LatamGPT": 11,
}


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    family: str
    suites: tuple[str, ...]
    backend: str
    source: str
    parameters_b: float | None = None
    constructor: str | None = None
    subfolder: str | None = None
    weight_file: str | None = None
    preferred_host: str = "relela-06"


MODEL_SPECS = (
    ModelSpec(
        "gpt2", "GPT-2", "GPT-2", ("phenomenon",), "hf", "gpt2", 0.124
    ),
    ModelSpec(
        "bert-base",
        "BERT Base",
        "BERT",
        ("phenomenon",),
        "hf",
        "google-bert/bert-base-uncased",
        0.109,
    ),
    ModelSpec(
        "resnet18",
        "ResNet-18",
        "ResNet",
        ("phenomenon", "vision", "model-storing"),
        "torchvision",
        "torchvision.models.resnet18",
        0.012,
        constructor="resnet18",
    ),
    ModelSpec(
        "resnet34",
        "ResNet-34",
        "ResNet",
        ("phenomenon", "vision", "model-storing"),
        "torchvision",
        "torchvision.models.resnet34",
        0.022,
        constructor="resnet34",
    ),
    ModelSpec(
        "resnet50",
        "ResNet-50",
        "ResNet",
        ("vision", "model-storing"),
        "torchvision",
        "torchvision.models.resnet50",
        0.026,
        constructor="resnet50",
    ),
    ModelSpec(
        "resnet101",
        "ResNet-101",
        "ResNet",
        ("vision", "model-storing"),
        "torchvision",
        "torchvision.models.resnet101",
        0.045,
        constructor="resnet101",
    ),
    ModelSpec(
        "resnet152",
        "ResNet-152",
        "ResNet",
        ("vision", "model-storing"),
        "torchvision",
        "torchvision.models.resnet152",
        0.060,
        constructor="resnet152",
    ),
    ModelSpec(
        "qwen25-15b",
        "Qwen2.5-1.5B",
        "Qwen2.5",
        ("model-storing",),
        "hf",
        "Qwen/Qwen2.5-1.5B",
        1.5,
    ),
    ModelSpec(
        "qwen25-3b",
        "Qwen2.5-3B",
        "Qwen2.5",
        ("model-storing",),
        "hf",
        "Qwen/Qwen2.5-3B",
        3.0,
    ),
    ModelSpec(
        "olmo2-1b",
        "OLMo-2-1B",
        "OLMo 2",
        ("model-storing",),
        "hf",
        "allenai/Olmo-2-0425-1B",
        1.0,
    ),
    ModelSpec(
        "olmo2-7b",
        "OLMo-2-7B",
        "OLMo 2",
        ("model-storing",),
        "hf",
        "allenai/OLMo-2-1124-7B",
        7.0,
    ),
    ModelSpec(
        "mistral7b",
        "Mistral-7B",
        "Mistral",
        ("model-storing",),
        "hf",
        "mistralai/Mistral-7B-v0.3",
        7.0,
    ),
    ModelSpec(
        "llama33-70b",
        "Llama 3.3 70B Instruct",
        "Llama 3",
        ("model-storing",),
        "hf",
        "meta-llama/Llama-3.3-70B-Instruct",
        70.0,
        preferred_host="godel",
    ),
    ModelSpec(
        "gemma4-31b",
        "Gemma 4 31B",
        "Gemma 4",
        ("model-storing",),
        "hf",
        "google/gemma-4-31B",
        31.0,
    ),
    ModelSpec(
        "phi4-14b",
        "Phi-4 14B",
        "Phi 4",
        ("model-storing",),
        "hf",
        "microsoft/phi-4",
        14.0,
    ),
    ModelSpec(
        "qwen35-27b",
        "Qwen3.5 27B",
        "Qwen 3.5",
        ("model-storing",),
        "hf",
        "Qwen/Qwen3.5-27B",
        27.0,
    ),
    ModelSpec(
        "latamgpt-70b",
        "LatamGPT-70B",
        "Llama 3 / LatamGPT",
        ("model-storing",),
        "hf",
        "latam-gpt/Llama-3.1-70B-LatamGPT-SFT-1.0",
        70.0,
        preferred_host="godel",
    ),
    ModelSpec(
        "convnext-tiny",
        "ConvNeXt-Tiny",
        "ConvNeXt",
        ("vision", "model-storing"),
        "torchvision",
        "torchvision.models.convnext_tiny",
        0.029,
        constructor="convnext_tiny",
    ),
    ModelSpec(
        "convnext-small",
        "ConvNeXt-Small",
        "ConvNeXt",
        ("vision", "model-storing"),
        "torchvision",
        "torchvision.models.convnext_small",
        0.050,
        constructor="convnext_small",
    ),
    ModelSpec(
        "convnext-base",
        "ConvNeXt-Base",
        "ConvNeXt",
        ("vision", "model-storing"),
        "torchvision",
        "torchvision.models.convnext_base",
        0.089,
        constructor="convnext_base",
    ),
    ModelSpec(
        "convnext-large",
        "ConvNeXt-Large",
        "ConvNeXt",
        ("vision", "model-storing"),
        "torchvision",
        "torchvision.models.convnext_large",
        0.198,
        constructor="convnext_large",
    ),
    ModelSpec(
        "efficientnet-b0",
        "EfficientNet-B0",
        "EfficientNet",
        ("vision", "model-storing"),
        "torchvision",
        "torchvision.models.efficientnet_b0",
        0.005,
        constructor="efficientnet_b0",
    ),
    ModelSpec(
        "efficientnet-b3",
        "EfficientNet-B3",
        "EfficientNet",
        ("vision", "model-storing"),
        "torchvision",
        "torchvision.models.efficientnet_b3",
        0.012,
        constructor="efficientnet_b3",
    ),
    ModelSpec(
        "efficientnet-b7",
        "EfficientNet-B7",
        "EfficientNet",
        ("vision", "model-storing"),
        "torchvision",
        "torchvision.models.efficientnet_b7",
        0.066,
        constructor="efficientnet_b7",
    ),
    ModelSpec(
        "sd15-unet",
        "Stable Diffusion v1.5 UNet",
        "Stable Diffusion",
        ("vision", "model-storing"),
        "hf",
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        0.860,
        subfolder="unet",
        weight_file="unet/diffusion_pytorch_model.safetensors",
    ),
    ModelSpec(
        "zaya1-8b",
        "ZAYA1-8B",
        "ZAYA",
        ("moe",),
        "hf",
        "Zyphra/ZAYA1-8B",
        8.0,
    ),
    ModelSpec(
        "lfm25-8b-a1b",
        "Liquid AI LFM2.5-8B-A1B",
        "LFM",
        ("moe",),
        "hf",
        "LiquidAI/LFM2.5-8B-A1B",
        8.0,
    ),
    ModelSpec(
        "jetmoe-8b",
        "JetMoE-8B",
        "JetMoE",
        ("moe",),
        "hf",
        "jetmoe/jetmoe-8b",
        8.0,
    ),
    ModelSpec(
        "llama3-8b-distilled",
        "Llama-3-8B-Instruct-Distilled",
        "Llama 3 distilled",
        ("distilled",),
        "hf",
        "JunchengXie/Meta-Llama-3-8B-Instruct-openwebtext-distilled-80k-gpt4",
        8.0,
    ),
    ModelSpec(
        "gemma2-9b-distilled",
        "Gemma-2-9B-Distilled",
        "Gemma 2 distilled",
        ("distilled",),
        "hf",
        "kazuyamaa/gemma-2-9b-opd-Distill-v001",
        9.0,
    ),
    ModelSpec(
        "qwen25-7b-distilled",
        "Qwen2.5-7B-Instruct-Distilled",
        "Qwen 2.5 distilled",
        ("distilled",),
        "hf",
        "RZ412/Qwen2.5-7B-Instruct-S1K-DeepSeek-Distilled",
        7.0,
    ),
)

SPECS_BY_KEY = {spec.key: spec for spec in MODEL_SPECS}


@dataclass
class MappingModel:
    model_name: str
    model: dict[str, torch.Tensor]
    provenance: dict[str, object]


@dataclass(frozen=True)
class CheckpointSelection:
    files: tuple[Path, ...]
    snapshot: Path
    revision: str
    index_path: Path | None = None
    expected_keys: frozenset[str] | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


_HOST_PROVENANCE: dict[str, object] | None = None


def host_provenance() -> dict[str, object]:
    global _HOST_PROVENANCE
    if _HOST_PROVENANCE is not None:
        return _HOST_PROVENANCE
    cpu_model = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(errors="replace").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                cpu_model = line.split(":", 1)[1].strip()
                break
    machine_id_digest = None
    machine_id = Path("/etc/machine-id")
    if machine_id.is_file():
        machine_id_digest = hashlib.sha256(machine_id.read_bytes()).hexdigest()
    fields = {
        "hostname": platform.node(),
        "machine": platform.machine(),
        "kernel": platform.release(),
        "cpu_model": cpu_model,
        "logical_cpus": os.cpu_count(),
        "machine_id_sha256": machine_id_digest,
    }
    _HOST_PROVENANCE = {**fields, "fingerprint": canonical_digest(fields)}
    return _HOST_PROVENANCE


def default_hf_cache() -> Path:
    explicit = os.environ.get("HF_HUB_CACHE")
    if explicit:
        return Path(explicit)
    home = os.environ.get("HF_HOME")
    if home:
        return Path(home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def parse_cpu_set(value: str) -> set[int]:
    cpus: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            first, last = part.split("-", 1)
            cpus.update(range(int(first), int(last) + 1))
        else:
            cpus.add(int(part))
    if not cpus:
        raise ValueError("CPU set cannot be empty.")
    return cpus


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def canonical_digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


_ARTIFACT_CACHE: dict[tuple[str, int, int], dict[str, object]] = {}
_CODE_PROVENANCE: dict[str, object] | None = None


def file_artifact(path: Path, *, snapshot: Path) -> dict[str, object]:
    path = path.absolute()
    stat = path.stat()
    resolved = path.resolve()
    cache_key = (str(resolved), int(stat.st_size), int(stat.st_mtime_ns))
    cached = _ARTIFACT_CACHE.get(cache_key)
    if cached is not None:
        artifact = dict(cached)
    else:
        blob_name = resolved.name.lower()
        if re.fullmatch(r"[0-9a-f]{64}", blob_name):
            algorithm = "sha256"
            digest = blob_name
            digest_source = "huggingface-content-addressed-cache"
        elif re.fullmatch(r"[0-9a-f]{40}", blob_name):
            algorithm = "git-blob-sha1"
            digest = blob_name
            digest_source = "huggingface-content-addressed-cache"
        else:
            algorithm = "sha256"
            digest = sha256_file(path)
            digest_source = "computed"
        artifact = {
            "cache_path": str(path),
            "resolved_path": str(resolved),
            "size_bytes": int(stat.st_size),
            "digest_algorithm": algorithm,
            "digest": digest,
            "digest_source": digest_source,
        }
        _ARTIFACT_CACHE[cache_key] = dict(artifact)
    try:
        artifact["repository_path"] = str(path.relative_to(snapshot))
    except ValueError:
        artifact["repository_path"] = path.name
    return artifact


def code_provenance() -> dict[str, object]:
    global _CODE_PROVENANCE
    if _CODE_PROVENANCE is not None:
        return _CODE_PROVENANCE

    roots: dict[str, Path] = {"runner": Path(__file__).resolve()}
    for module_name in ("neuralzip", "zipnn", "zipnn_core"):
        module = importlib.import_module(module_name)
        module_path = Path(str(module.__file__)).resolve()
        roots[module_name] = (
            module_path.parent
            if module_path.name.startswith("__init__.")
            else module_path
        )

    artifacts = []
    for label, root in roots.items():
        paths = [root] if root.is_file() else sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in CODE_SUFFIXES
        )
        for path in paths:
            relative = path.name if root.is_file() else str(path.relative_to(root))
            artifacts.append(
                {
                    "component": label,
                    "path": relative,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    versions = {
        name: package_version(name)
        for name in ("torch", "torchvision", "neuralzip", "zipnn")
    }
    _CODE_PROVENANCE = {
        "digest": canonical_digest({"artifacts": artifacts, "versions": versions}),
        "artifacts": artifacts,
        "versions": versions,
    }
    return _CODE_PROVENANCE


def configuration_payload(
    spec: ModelSpec,
    method: str,
    *,
    threads: int,
    cpu_set: str,
    smoke: bool,
) -> dict[str, object]:
    return {
        "model": asdict(spec),
        "compressor": COMPRESSOR_LABELS[method],
        "compressor_key": method,
        "dtype": "bfloat16",
        "threads": int(threads),
        "cpu_set": sorted(parse_cpu_set(cpu_set)),
        "chunk_size": DEFAULT_CHUNK_SIZE,
        "compression_threshold": DEFAULT_COMPRESSION_THRESHOLD,
        "warmup_runs": 1,
        "measured_runs": 1,
        "smoke": bool(smoke),
    }


def job_identity(
    *,
    host: str,
    configuration: dict[str, object],
    weights_fingerprint: str,
) -> dict[str, str]:
    configuration_fingerprint = canonical_digest(configuration)
    code_digest = str(code_provenance()["digest"])
    host_fingerprint = str(host_provenance()["fingerprint"])
    job_fingerprint = canonical_digest(
        {
            "host": host,
            "host_fingerprint": host_fingerprint,
            "configuration_fingerprint": configuration_fingerprint,
            "code_digest": code_digest,
            "weights_fingerprint": weights_fingerprint,
        }
    )
    return {
        "configuration_fingerprint": configuration_fingerprint,
        "code_digest": code_digest,
        "host_fingerprint": host_fingerprint,
        "job_fingerprint": job_fingerprint,
    }


def iter_float_tensors(model: MappingModel) -> Iterable[tuple[str, torch.Tensor]]:
    for name, tensor in model.model.items():
        if torch.is_tensor(tensor) and tensor.is_floating_point():
            yield name, tensor


def floating_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Return every floating state entry, including BatchNorm running buffers."""

    return {
        str(name): tensor.detach().cpu().contiguous()
        for name, tensor in module.state_dict().items()
        if torch.is_tensor(tensor) and tensor.is_floating_point()
    }


def flatten_state(value, prefix: str = "") -> Iterable[tuple[str, torch.Tensor]]:
    if torch.is_tensor(value):
        yield prefix, value
        return
    if isinstance(value, dict):
        for key, child in value.items():
            name = str(key) if not prefix else f"{prefix}.{key}"
            yield from flatten_state(child, name)


def build_weight_provenance(
    *,
    source: str,
    snapshot_path: Path,
    snapshot_label: str,
    revision: str,
    files: Iterable[Path],
    index_path: Path | None = None,
) -> dict[str, object]:
    artifacts = [
        file_artifact(path, snapshot=snapshot_path)
        for path in sorted(files, key=lambda item: str(item))
    ]
    index_artifact = (
        file_artifact(index_path, snapshot=snapshot_path)
        if index_path is not None
        else None
    )
    stable_artifacts = [
        {
            "repository_path": artifact["repository_path"],
            "size_bytes": artifact["size_bytes"],
            "digest_algorithm": artifact["digest_algorithm"],
            "digest": artifact["digest"],
        }
        for artifact in artifacts
    ]
    stable_index = None
    if index_artifact is not None:
        stable_index = {
            "repository_path": index_artifact["repository_path"],
            "size_bytes": index_artifact["size_bytes"],
            "digest_algorithm": index_artifact["digest_algorithm"],
            "digest": index_artifact["digest"],
        }
    weights_fingerprint = canonical_digest(
        {
            "source": source,
            "revision": revision,
            "artifacts": stable_artifacts,
            "index": stable_index,
        }
    )
    return {
        "snapshot": snapshot_label,
        "revision": revision,
        "weight_artifacts": artifacts,
        "index_artifact": index_artifact,
        "weights_fingerprint": weights_fingerprint,
    }


def torchvision_provenance(spec: ModelSpec) -> dict[str, object]:
    from torchvision import models

    constructor = getattr(models, str(spec.constructor))
    weights = models.get_model_weights(constructor).DEFAULT
    checkpoint = torchvision_checkpoint(spec)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Torchvision checkpoint is missing: {checkpoint}.")
    return build_weight_provenance(
        source=spec.source,
        snapshot_path=checkpoint.parent,
        snapshot_label=str(weights.url),
        revision=str(weights),
        files=(checkpoint,),
    )


def load_torchvision(spec: ModelSpec, allow_download: bool) -> MappingModel:
    from torchvision import models

    constructor = getattr(models, str(spec.constructor))
    weights = models.get_model_weights(constructor).DEFAULT
    checkpoint = torchvision_checkpoint(spec)
    if not allow_download and not checkpoint.is_file():
        raise FileNotFoundError(
            f"Pretrained torchvision weights are not cached at {checkpoint}. "
            "Use --allow-download to fetch them."
        )
    module = constructor(weights=weights).eval().to(dtype=torch.bfloat16)
    tensors = floating_state_dict(module)
    provenance = torchvision_provenance(spec)
    provenance["floating_numel"] = sum(tensor.numel() for tensor in tensors.values())
    del module
    return MappingModel(spec.source, tensors, provenance)


def torchvision_checkpoint(spec: ModelSpec) -> Path:
    from torchvision import models

    constructor = getattr(models, str(spec.constructor))
    weights = models.get_model_weights(constructor).DEFAULT
    return Path(torch.hub.get_dir()) / "checkpoints" / Path(str(weights.url)).name


def snapshot_weight_files(
    spec: ModelSpec,
    *,
    cache_dir: Path,
    allow_download: bool,
    revision: str | None = None,
) -> CheckpointSelection:
    from huggingface_hub import snapshot_download

    prefix = f"{spec.subfolder}/" if spec.subfolder else ""
    if spec.weight_file:
        patterns = (
            spec.weight_file,
            "config.json",
            f"{prefix}config.json",
        )
    else:
        patterns = (
            f"{prefix}*.safetensors",
            f"{prefix}*.safetensors.index.json",
            f"{prefix}*.bin",
            f"{prefix}*.bin.index.json",
            f"{prefix}*.pt",
            "config.json",
            f"{prefix}config.json",
        )
    snapshot = Path(
        snapshot_download(
            repo_id=spec.source,
            revision=revision,
            cache_dir=cache_dir,
            allow_patterns=patterns,
            local_files_only=not allow_download,
        )
    )
    root = snapshot / spec.subfolder if spec.subfolder else snapshot
    if spec.weight_file:
        selected = snapshot / spec.weight_file
        if not selected.is_file():
            raise FileNotFoundError(
                f"Requested checkpoint weight file is missing: {selected}."
            )
        return CheckpointSelection(
            files=(selected,),
            snapshot=snapshot,
            revision=snapshot.name,
        )

    # An index is authoritative.  It prevents similarly named alternate or
    # quantized checkpoint files from being counted as additional tensors.
    index_paths = sorted(root.glob("*.safetensors.index.json"))
    if not index_paths:
        index_paths = sorted(root.glob("*.bin.index.json"))
    if index_paths:
        canonical_indexes = [
            path
            for path in index_paths
            if path.name
            in {"model.safetensors.index.json", "pytorch_model.bin.index.json"}
        ]
        if len(canonical_indexes) == 1:
            preferred_index = canonical_indexes[0]
        elif len(index_paths) == 1:
            preferred_index = index_paths[0]
        else:
            raise ValueError(
                "Checkpoint contains ambiguous shard indexes: "
                + ", ".join(path.name for path in index_paths)
            )
        index = json.loads(preferred_index.read_text())
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Checkpoint index has no weight map: {preferred_index}.")
        shard_names = sorted(set(str(name) for name in weight_map.values()))
        files = [preferred_index.parent / name for name in shard_names]
        missing = [path for path in files if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Checkpoint index references missing shards: "
                + ", ".join(str(path) for path in missing)
            )
        return CheckpointSelection(
            files=tuple(files),
            snapshot=snapshot,
            revision=snapshot.name,
            index_path=preferred_index,
            expected_keys=frozenset(str(key) for key in weight_map),
        )

    preferred_files = (
        root / "model.safetensors",
        root / "diffusion_pytorch_model.safetensors",
        root / "pytorch_model.bin",
    )
    files = [path for path in preferred_files if path.is_file()]
    if files:
        return CheckpointSelection(
            files=(files[0],),
            snapshot=snapshot,
            revision=snapshot.name,
        )

    files = [
        path
        for path in sorted(root.glob("*.safetensors"))
        if ".fp16." not in path.name and ".non_ema." not in path.name
    ]
    if not files:
        files = sorted(root.glob("*.bin")) + sorted(root.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No checkpoint weight files found under {root}.")
    if len(files) != 1:
        raise ValueError(
            "Checkpoint has multiple weight files but no authoritative shard "
            f"index under {root}: {', '.join(path.name for path in files)}."
        )
    if re.search(r"-\d+-of-\d+\.", files[0].name):
        raise ValueError(
            f"Refusing orphan checkpoint shard {files[0].name} without an index."
        )
    return CheckpointSelection(
        files=(files[0],),
        snapshot=snapshot,
        revision=snapshot.name,
    )


def hf_provenance(spec: ModelSpec, selection: CheckpointSelection) -> dict[str, object]:
    return build_weight_provenance(
        source=spec.source,
        snapshot_path=selection.snapshot,
        snapshot_label=str(selection.snapshot),
        revision=selection.revision,
        files=selection.files,
        index_path=selection.index_path,
    )


def load_hf_weights(
    spec: ModelSpec,
    *,
    cache_dir: Path,
    allow_download: bool,
    revision: str | None = None,
) -> MappingModel:
    selection = snapshot_weight_files(
        spec,
        cache_dir=cache_dir,
        allow_download=allow_download,
        revision=revision,
    )
    tensors: dict[str, torch.Tensor] = {}
    loaded_keys: set[str] = set()
    for path in selection.files:
        if path.suffix == ".safetensors":
            from safetensors.torch import load_file

            state = load_file(str(path), device="cpu")
        else:
            state = torch.load(path, map_location="cpu", weights_only=True)
        for name, tensor in flatten_state(state):
            if name in loaded_keys:
                raise ValueError(f"Duplicate checkpoint tensor key {name!r} in {path}.")
            loaded_keys.add(name)
            if not tensor.is_floating_point():
                continue
            value = tensor.detach().cpu().contiguous()
            if value.dtype != torch.bfloat16:
                value = value.to(dtype=torch.bfloat16)
            tensors[name] = value
        del state
        gc.collect()
    if not tensors:
        raise ValueError(f"{spec.source} contains no floating-point tensors.")
    if selection.expected_keys is not None:
        missing_keys = sorted(selection.expected_keys - loaded_keys)
        unexpected_keys = sorted(loaded_keys - selection.expected_keys)
        if missing_keys:
            preview = ", ".join(missing_keys[:5])
            raise ValueError(
                f"Checkpoint {spec.source} is missing {len(missing_keys)} indexed "
                f"tensor(s), including: {preview}."
            )
        if unexpected_keys:
            preview = ", ".join(unexpected_keys[:5])
            raise ValueError(
                f"Checkpoint {spec.source} contains {len(unexpected_keys)} tensor(s) "
                f"not declared by its index, including: {preview}."
            )
    floating_numel = sum(tensor.numel() for tensor in tensors.values())
    if spec.parameters_b is not None:
        minimum_numel = int(spec.parameters_b * 1e9 * MIN_PARAMETER_COVERAGE)
        if floating_numel < minimum_numel:
            raise ValueError(
                f"Checkpoint {spec.source} contains only {floating_numel:,} floating "
                f"values; expected at least {minimum_numel:,} from the declared "
                f"{spec.parameters_b:g}B parameters. Refusing a likely partial snapshot."
            )
    provenance = hf_provenance(spec, selection)
    provenance["floating_numel"] = floating_numel
    provenance["declared_parameters_b"] = spec.parameters_b
    provenance["minimum_parameter_coverage"] = MIN_PARAMETER_COVERAGE
    return MappingModel(spec.source, tensors, provenance)


def load_model(
    spec: ModelSpec,
    *,
    cache_dir: Path,
    allow_download: bool,
    revision: str | None = None,
) -> MappingModel:
    if spec.backend == "torchvision":
        return load_torchvision(spec, allow_download)
    if spec.backend == "hf":
        return load_hf_weights(
            spec,
            cache_dir=cache_dir,
            allow_download=allow_download,
            revision=revision,
        )
    raise ValueError(f"Unsupported backend {spec.backend!r}.")


def tensor_content_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous().view(torch.uint8)
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def make_smoke_model() -> MappingModel:
    generator = torch.Generator().manual_seed(20260717)
    tensor = torch.randn(
        512,
        512,
        dtype=torch.bfloat16,
        generator=generator,
    )
    artifact = {
        "repository_path": "synthetic/weight",
        "cache_path": None,
        "resolved_path": None,
        "size_bytes": int(tensor.numel() * tensor.element_size()),
        "digest_algorithm": "sha256",
        "digest": tensor_content_digest(tensor),
        "digest_source": "computed",
    }
    revision = f"seed-20260717-torch-{torch.__version__}"
    provenance = {
        "snapshot": "synthetic://bf16-512x512",
        "revision": revision,
        "weight_artifacts": [artifact],
        "index_artifact": None,
        "weights_fingerprint": canonical_digest(
            {
                "source": "synthetic-smoke",
                "revision": revision,
                "artifacts": [
                    {
                        key: artifact[key]
                        for key in (
                            "repository_path",
                            "size_bytes",
                            "digest_algorithm",
                            "digest",
                        )
                    }
                ],
            }
        ),
        "floating_numel": int(tensor.numel()),
    }
    return MappingModel("synthetic-smoke", {"weight": tensor}, provenance)


def resolve_weight_provenance(
    spec: ModelSpec,
    *,
    cache_dir: Path,
    smoke: bool,
    revision: str | None = None,
) -> dict[str, object]:
    if smoke:
        return make_smoke_model().provenance
    if spec.backend == "torchvision":
        return torchvision_provenance(spec)
    if spec.backend == "hf":
        selection = snapshot_weight_files(
            spec,
            cache_dir=cache_dir,
            allow_download=False,
            revision=revision,
        )
        return hf_provenance(spec, selection)
    raise ValueError(f"Unsupported backend {spec.backend!r}.")


def parse_zipnn_bf16_exponent(payload: bytes, parser) -> tuple[int, int, int]:
    """Return exponent payload, exponent-group metadata, and chunk count."""

    header_bytes = int(parser._retrieve_header(payload))
    original_bytes = int(parser.original_len)
    chunk_bytes = int(parser.compression_chunk)
    if original_bytes % 2:
        raise ValueError("BF16 ZipNN payload has an odd original byte length.")
    chunk_count = math.ceil(original_bytes / chunk_bytes)
    entry_count = 2 * chunk_count
    offsets_start = header_bytes + entry_count
    payload_start = offsets_start + 8 * entry_count
    if payload_start > len(payload):
        raise ValueError("ZipNN payload is shorter than its chunk metadata.")
    offsets = struct.unpack_from(
        f"<{entry_count}Q",
        payload,
        offsets_start,
    )
    low_bytes = int(offsets[chunk_count - 1]) if chunk_count else 0
    exponent_bytes = int(offsets[-1]) if chunk_count else 0
    data_bytes = len(payload) - payload_start
    if low_bytes + exponent_bytes != data_bytes:
        raise ValueError(
            "Unsupported ZipNN archive layout: byte-stream sizes do not match."
        )
    # Count only the exponent byte group's chunk-type and cumulative-size
    # entries.  The common tensor header and the low-byte group's entries are
    # outside the exponent-only scope, matching NeuralZip group accounting.
    metadata_bytes = chunk_count * (1 + struct.calcsize("P"))
    return exponent_bytes, metadata_bytes, chunk_count


def zipnn_src_pass(
    model: MappingModel,
    *,
    threads: int,
    verify: bool,
) -> dict[str, object]:
    from zipnn import ZipNN

    total_encode = 0.0
    total_codec = 0.0
    total_copy = 0.0
    exponent_raw = 0
    exponent_data = 0
    exponent_metadata = 0
    tensors = 0
    chunks = 0
    roundtrip_ok = True
    parser = ZipNN(
        method="HUFFMAN",
        input_format="torch",
        threads=threads,
        compression_threshold=DEFAULT_COMPRESSION_THRESHOLD,
        compression_chunk=DEFAULT_CHUNK_SIZE,
    )
    decompressor = None
    if verify:
        decompressor = ZipNN(
            method="HUFFMAN",
            input_format="torch",
            threads=threads,
            compression_threshold=DEFAULT_COMPRESSION_THRESHOLD,
            compression_chunk=DEFAULT_CHUNK_SIZE,
        )

    for name, tensor in iter_float_tensors(model):
        original = tensor.detach().cpu().contiguous()
        if original.dtype != torch.bfloat16:
            raise TypeError(f"{name} is {original.dtype}; expected BF16.")
        compressor = ZipNN(
            method="HUFFMAN",
            input_format="torch",
            threads=threads,
            compression_threshold=DEFAULT_COMPRESSION_THRESHOLD,
            compression_chunk=DEFAULT_CHUNK_SIZE,
        )

        copy_start = time.perf_counter()
        compressor_input = original.clone()
        copy_seconds = time.perf_counter() - copy_start
        codec_start = time.perf_counter()
        payload = compressor.compress(compressor_input)
        codec_seconds = time.perf_counter() - codec_start
        total_copy += copy_seconds
        total_codec += codec_seconds
        total_encode += copy_seconds + codec_seconds

        stream_bytes, metadata_bytes, tensor_chunks = parse_zipnn_bf16_exponent(
            payload, parser
        )
        exponent_raw += int(original.numel())
        exponent_data += stream_bytes
        exponent_metadata += metadata_bytes
        chunks += tensor_chunks
        tensors += 1

        if verify:
            restored = decompressor.decompress(payload)
            same = torch.equal(
                restored.detach().cpu().contiguous().view(torch.uint8),
                original.view(torch.uint8),
            )
            roundtrip_ok = bool(roundtrip_ok and same)
            if not same:
                raise RuntimeError(f"ZipNN src round-trip failed for tensor {name}.")
        del compressor_input, payload

    exponent_compressed = exponent_data + exponent_metadata
    return {
        "encode_seconds": total_encode,
        "codec_encode_seconds": total_codec,
        "input_copy_seconds": total_copy,
        "setup_seconds": 0.0,
        "exponent_raw_bytes": exponent_raw,
        "exponent_data_bytes": exponent_data,
        "exponent_metadata_bytes": exponent_metadata,
        "exponent_compressed_bytes": exponent_compressed,
        "exponent_ratio_pct": (
            100.0 * exponent_compressed / exponent_raw if exponent_raw else None
        ),
        "tensor_count": tensors,
        "chunk_count": chunks,
        "roundtrip_ok": bool(roundtrip_ok),
    }


def neuralzip_pass(
    model: MappingModel,
    *,
    threads: int,
    verify: bool,
) -> dict[str, object]:
    result = Compressor(
        "neuralzip",
        verify=verify,
        keep_payloads=False,
        threads=threads,
        chunk_size=DEFAULT_CHUNK_SIZE,
        compression_threshold=DEFAULT_COMPRESSION_THRESHOLD,
    ).compress(model)
    if verify and result.get("roundtrip_ok") is not True:
        raise RuntimeError("NeuralZip round-trip verification failed.")
    streams = result.get("stream_results") or ()
    exponent = next(
        (row for row in streams if int(row.get("stream_id", -1)) == 1),
        None,
    )
    if exponent is None:
        raise ValueError("NeuralZip did not report its BF16 exponent stream.")
    codebook_bytes = sum(
        int(codebook.metadata_bytes)
        for codebook in (result.get("codebooks") or ())
        if str(codebook.dtype) == "bfloat16" and int(codebook.stream_id) == 1
    )
    exponent_raw = int(exponent["raw_bytes"])
    exponent_payload = int(exponent["payload_bytes"])
    exponent_group_metadata = int(exponent["metadata_bytes"])
    exponent_metadata = exponent_group_metadata + codebook_bytes
    exponent_compressed = exponent_payload + exponent_metadata
    return {
        "encode_seconds": float(result["encode_seconds"]),
        "codec_encode_seconds": float(result["encode_seconds"]),
        "input_copy_seconds": 0.0,
        "setup_seconds": float(
            result.get("precompute_total_seconds")
            or result.get("precompute_seconds")
            or 0.0
        ),
        "exponent_raw_bytes": exponent_raw,
        "exponent_data_bytes": exponent_payload,
        "exponent_metadata_bytes": exponent_metadata,
        "exponent_group_metadata_bytes": exponent_group_metadata,
        "exponent_codebook_metadata_bytes": codebook_bytes,
        "exponent_compressed_bytes": exponent_compressed,
        "exponent_ratio_pct": (
            100.0 * exponent_compressed / exponent_raw if exponent_raw else None
        ),
        "tensor_count": int(result["num_tensors"]),
        "chunk_count": int(exponent.get("local_huff0_chunks", 0))
        + int(exponent.get("shared_huff0_chunks", 0))
        + int(exponent.get("packed_ngram_chunks", 0))
        + int(exponent.get("raw_chunks", 0))
        + int(exponent.get("zero_chunks", 0)),
        "roundtrip_ok": bool(result.get("roundtrip_ok")) if verify else True,
        "clusters": int(result.get("clusters") or 0),
        "codebooks": len(result.get("codebooks") or ()),
    }


def run_pass(
    method: str,
    model: MappingModel,
    *,
    threads: int,
    verify: bool,
) -> dict[str, object]:
    if method == "zipnn-src":
        return zipnn_src_pass(model, threads=threads, verify=verify)
    if method == "neuralzip":
        return neuralzip_pass(model, threads=threads, verify=verify)
    raise ValueError(method)


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str))
    temporary.replace(path)


def worker(
    *,
    spec: ModelSpec,
    method: str,
    output: Path,
    cache_dir: Path,
    threads: int,
    cpu_set: str,
    allow_download: bool,
    smoke: bool,
    revision: str | None,
) -> int:
    cpus = parse_cpu_set(cpu_set)
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, cpus)
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    started = utc_now()
    host = platform.node()
    configuration = configuration_payload(
        spec,
        method,
        threads=threads,
        cpu_set=cpu_set,
        smoke=smoke,
    )
    code = code_provenance()
    provenance: dict[str, object] | None = None
    load_start = time.perf_counter()
    try:
        if (
            not smoke
            and platform.node().startswith("relela")
            and spec.preferred_host == "godel"
        ):
            raise RuntimeError(
                "This 70B checkpoint requires whole-model state in the current "
                "NeuralZip path and is marked for Godel, not relela-06."
            )
        if smoke:
            model = make_smoke_model()
        else:
            model = load_model(
                spec,
                cache_dir=cache_dir,
                allow_download=allow_download,
                revision=revision,
            )
        provenance = model.provenance
        load_seconds = time.perf_counter() - load_start

        warmup = run_pass(method, model, threads=threads, verify=True)
        warmup_roundtrip_ok = warmup.get("roundtrip_ok") is True
        if not warmup_roundtrip_ok:
            raise RuntimeError("The verified warmup did not round-trip exactly.")
        del warmup
        gc.collect()

        measured = run_pass(method, model, threads=threads, verify=False)
        identity = job_identity(
            host=host,
            configuration=configuration,
            weights_fingerprint=str(provenance["weights_fingerprint"]),
        )
        row = {
            "status": "ok",
            "host": host,
            "model_key": spec.key,
            "model": spec.label,
            "family": spec.family,
            "suites": list(spec.suites),
            "source": spec.source,
            "parameters_b": spec.parameters_b,
            "compressor": COMPRESSOR_LABELS[method],
            "snapshot": provenance["snapshot"],
            "revision": provenance["revision"],
            "weight_artifacts": provenance["weight_artifacts"],
            "index_artifact": provenance.get("index_artifact"),
            "weights_fingerprint": provenance["weights_fingerprint"],
            "floating_numel": provenance.get("floating_numel"),
            "configuration": configuration,
            **identity,
            "dtype": "bfloat16",
            "threads": threads,
            "cpu_set": sorted(cpus),
            "chunk_size": DEFAULT_CHUNK_SIZE,
            "compression_threshold": DEFAULT_COMPRESSION_THRESHOLD,
            "warmup_runs": 1,
            "measured_runs": 1,
            "verified_warmup": True,
            "load_seconds": load_seconds,
            "started_at": started,
            "finished_at": utc_now(),
            **measured,
            "roundtrip_ok": warmup_roundtrip_ok,
        }
    except Exception as error:
        if provenance is None:
            try:
                provenance = resolve_weight_provenance(
                    spec,
                    cache_dir=cache_dir,
                    smoke=smoke,
                    revision=revision,
                )
            except Exception:
                provenance = None
        identity = (
            job_identity(
                host=host,
                configuration=configuration,
                weights_fingerprint=str(provenance["weights_fingerprint"]),
            )
            if provenance is not None
            else {
                "configuration_fingerprint": canonical_digest(configuration),
                "code_digest": str(code["digest"]),
                "host_fingerprint": str(host_provenance()["fingerprint"]),
                "job_fingerprint": None,
            }
        )
        row = {
            "status": "failed",
            "host": host,
            "model_key": spec.key,
            "model": spec.label,
            "family": spec.family,
            "suites": list(spec.suites),
            "source": spec.source,
            "parameters_b": spec.parameters_b,
            "compressor": COMPRESSOR_LABELS[method],
            "snapshot": provenance.get("snapshot") if provenance else None,
            "revision": provenance.get("revision") if provenance else None,
            "weight_artifacts": (
                provenance.get("weight_artifacts") if provenance else None
            ),
            "index_artifact": provenance.get("index_artifact") if provenance else None,
            "weights_fingerprint": (
                provenance.get("weights_fingerprint") if provenance else None
            ),
            "floating_numel": provenance.get("floating_numel") if provenance else None,
            "configuration": configuration,
            **identity,
            "dtype": "bfloat16",
            "threads": threads,
            "cpu_set": sorted(cpus),
            "chunk_size": DEFAULT_CHUNK_SIZE,
            "compression_threshold": DEFAULT_COMPRESSION_THRESHOLD,
            "warmup_runs": 1,
            "measured_runs": 1,
            "started_at": started,
            "finished_at": utc_now(),
            "error": f"{type(error).__name__}: {error}",
        }
    atomic_write_json(output, row)
    print(json.dumps(row, default=str), flush=True)
    return 0 if row["status"] == "ok" else 1


def read_job_rows(job_dir: Path) -> list[dict[str, object]]:
    rows = []
    for path in sorted(job_dir.glob("*.json")):
        try:
            rows.append(json.loads(path.read_text()))
        except (OSError, json.JSONDecodeError):
            continue
    return rows


def write_summary(output_dir: Path, *, expected_host: str | None = None) -> None:
    rows = read_job_rows(output_dir / "jobs")
    hosts = {str(row.get("host")) for row in rows if row.get("host")}
    if len(hosts) > 1:
        raise RuntimeError(
            f"Refusing to combine result rows from multiple hosts: {sorted(hosts)}."
        )
    if expected_host is not None and hosts and hosts != {expected_host}:
        raise RuntimeError(
            f"Result directory belongs to host {next(iter(hosts))}, not {expected_host}."
        )
    json_path = output_dir / "results.json"
    csv_path = output_dir / "results.csv"
    atomic_write_json(json_path, rows)
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            serializable = {
                key: json.dumps(value) if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            }
            writer.writerow(serializable)
    temporary.replace(csv_path)


def completed_job_provenance(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    keys = (
        "status",
        "host",
        "model_key",
        "model",
        "compressor",
        "snapshot",
        "revision",
        "weight_artifacts",
        "index_artifact",
        "weights_fingerprint",
        "floating_numel",
        "configuration_fingerprint",
        "code_digest",
        "host_fingerprint",
        "job_fingerprint",
    )
    return [{key: row.get(key) for key in keys} for row in rows]


def methods_for_model(index: int) -> tuple[str, ...]:
    return METHODS if index % 2 == 0 else tuple(reversed(METHODS))


def manifest_payload(
    args,
    specs: list[ModelSpec],
    skipped_for_host: list[ModelSpec],
    *,
    host: str,
    created_at: str,
    rows: list[dict[str, object]],
    complete: bool,
) -> dict[str, object]:
    code = code_provenance()
    manifest_configuration = {
        "host": host,
        "model_keys": [spec.key for spec in specs],
        "compressors": [COMPRESSOR_LABELS[method] for method in METHODS],
        "dtype": "bfloat16",
        "threads": args.threads,
        "cpu_set": sorted(parse_cpu_set(args.cpu_set)),
        "warmup_runs": 1,
        "measured_runs": 1,
        "compression_threshold": DEFAULT_COMPRESSION_THRESHOLD,
        "chunk_size": DEFAULT_CHUNK_SIZE,
        "smoke": bool(args.smoke),
        "execution_order": {
            spec.key: [
                COMPRESSOR_LABELS[method]
                for method in methods_for_model(index)
            ]
            for index, spec in enumerate(
                [item for item in specs if item not in skipped_for_host]
            )
        },
    }
    return {
        "created_at": created_at,
        "updated_at": utc_now(),
        **manifest_configuration,
        "manifest_configuration_fingerprint": canonical_digest(
            manifest_configuration
        ),
        "code_digest": code["digest"],
        "code_provenance": code,
        "host_fingerprint": host_provenance()["fingerprint"],
        "host_provenance": host_provenance(),
        "complete": bool(complete),
        "ratio_scope": "BF16 exponent stream including coding metadata",
        "zipnn_time_scope": (
            "defensive input clone plus official ZipNN codec; both components "
            "are also reported separately"
        ),
        "models": [asdict(spec) for spec in specs],
        "skipped_for_host": [asdict(spec) for spec in skipped_for_host],
        "completed_jobs": completed_job_provenance(rows),
    }


def expected_job_identity(
    spec: ModelSpec,
    method: str,
    args,
    host: str,
    *,
    revision: str | None,
) -> dict[str, str]:
    provenance = resolve_weight_provenance(
        spec,
        cache_dir=args.cache_dir,
        smoke=args.smoke,
        revision=revision,
    )
    configuration = configuration_payload(
        spec,
        method,
        threads=args.threads,
        cpu_set=args.cpu_set,
        smoke=args.smoke,
    )
    return job_identity(
        host=host,
        configuration=configuration,
        weights_fingerprint=str(provenance["weights_fingerprint"]),
    )


def prior_matches_expected(
    prior: dict[str, object],
    expected: dict[str, str],
    *,
    host: str,
) -> bool:
    return bool(
        prior.get("host") == host
        and prior.get("configuration_fingerprint")
        == expected["configuration_fingerprint"]
        and prior.get("code_digest") == expected["code_digest"]
        and prior.get("host_fingerprint") == expected["host_fingerprint"]
        and prior.get("job_fingerprint") == expected["job_fingerprint"]
    )


def validate_completed_pairs(
    rows: list[dict[str, object]],
    specs: list[ModelSpec],
    args,
    *,
    host: str,
    pinned_revisions: dict[str, str | None],
) -> None:
    by_key: dict[tuple[str, str], dict[str, object]] = {}
    for row in rows:
        key = (str(row.get("model_key")), str(row.get("compressor")))
        if key in by_key:
            raise RuntimeError(f"Duplicate result row for {key}.")
        by_key[key] = row

    failures = []
    for spec in specs:
        for method in METHODS:
            key = (spec.key, COMPRESSOR_LABELS[method])
            row = by_key.get(key)
            if row is None:
                failures.append(f"missing {key[0]}:{key[1]}")
                continue
            expected = expected_job_identity(
                spec,
                method,
                args,
                host,
                revision=pinned_revisions.get(spec.key),
            )
            if row.get("status") != "ok":
                failures.append(f"failed {key[0]}:{key[1]}")
            elif not prior_matches_expected(row, expected, host=host):
                failures.append(f"stale {key[0]}:{key[1]}")
    if failures:
        raise RuntimeError(
            "Final summary requires a complete, current compressor pair for every "
            "model: " + "; ".join(failures)
        )


def select_specs(model_keys: str | None, suites: str) -> list[ModelSpec]:
    requested_suites = {
        value.strip() for value in suites.split(",") if value.strip()
    }
    if model_keys:
        keys = [value.strip() for value in model_keys.split(",") if value.strip()]
        unknown = [key for key in keys if key not in SPECS_BY_KEY]
        if unknown:
            raise ValueError(f"Unknown model keys: {', '.join(unknown)}")
        return [SPECS_BY_KEY[key] for key in keys]
    if "all" in requested_suites:
        return list(MODEL_SPECS)
    selected = [
        spec
        for spec in MODEL_SPECS
        if requested_suites.intersection(spec.suites)
    ]
    if "model-storing" in requested_suites:
        selected.sort(
            key=lambda spec: (
                MODEL_STORAGE_FAMILY_ORDER.get(spec.family, 100),
                spec.parameters_b if spec.parameters_b is not None else float("inf"),
            )
        )
    return selected


def cached_hf_repos(cache_dir: Path) -> set[str]:
    try:
        from huggingface_hub import scan_cache_dir

        return {repo.repo_id for repo in scan_cache_dir(cache_dir).repos}
    except Exception:
        return set()


def audit(specs: list[ModelSpec], cache_dir: Path) -> dict[str, object]:
    cached = cached_hf_repos(cache_dir)
    available_gib = None
    try:
        import psutil

        available_gib = psutil.virtual_memory().available / (1 << 30)
    except ImportError:
        pass
    rows = []
    for spec in specs:
        weight_gib = (
            spec.parameters_b * 2e9 / (1 << 30)
            if spec.parameters_b is not None
            else None
        )
        estimated_peak_gib = 1.7 * weight_gib if weight_gib is not None else None
        rows.append(
            {
                **asdict(spec),
                "cached": (
                    torchvision_checkpoint(spec).is_file()
                    if spec.backend == "torchvision"
                    else spec.source in cached
                ),
                "estimated_bf16_weight_gib": weight_gib,
                "estimated_neuralzip_peak_gib": estimated_peak_gib,
                "fits_available_memory_estimate": (
                    estimated_peak_gib <= available_gib
                    if estimated_peak_gib is not None and available_gib is not None
                    else None
                ),
            }
        )
    return {
        "created_at": utc_now(),
        "host": platform.node(),
        "cpu_count": os.cpu_count(),
        "available_memory_gib": available_gib,
        "cache_dir": str(cache_dir),
        "disk_free_gib": shutil.disk_usage(cache_dir.parent).free / (1 << 30)
        if cache_dir.parent.exists()
        else None,
        "packages": {
            name: package_version(name)
            for name in (
                "torch",
                "torchvision",
                "neuralzip",
                "zipnn",
                "huggingface-hub",
                "safetensors",
                "transformers",
                "diffusers",
            )
        },
        "models": rows,
    }


def orchestrate(args, specs: list[ModelSpec]) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    job_dir = args.output_dir / "jobs"
    job_dir.mkdir(parents=True, exist_ok=True)
    host = platform.node()
    skipped_for_host = [
        spec
        for spec in specs
        if host.startswith("relela") and spec.preferred_host == "godel"
    ]
    runnable_specs = [spec for spec in specs if spec not in skipped_for_host]
    rows = read_job_rows(job_dir)
    allowed_keys = {spec.key for spec in runnable_specs}
    allowed_compressors = set(COMPRESSOR_LABELS.values())
    current_host_fingerprint = str(host_provenance()["fingerprint"])
    for row in rows:
        if (
            row.get("host") != host
            or row.get("host_fingerprint") != current_host_fingerprint
        ):
            raise RuntimeError(
                "Refusing an output directory containing legacy or foreign-host "
                f"rows: expected {host!r}/{current_host_fingerprint}, found "
                f"{row.get('host')!r}/{row.get('host_fingerprint')!r}."
            )
        if row.get("model_key") not in allowed_keys:
            raise RuntimeError(
                f"Output directory contains out-of-scope model {row.get('model_key')!r}."
            )
        if row.get("compressor") not in allowed_compressors:
            raise RuntimeError(
                "Output directory contains an unknown compressor row: "
                f"{row.get('compressor')!r}."
            )

    manifest_path = args.output_dir / "manifest.json"
    created_at = utc_now()
    if manifest_path.is_file():
        try:
            previous_manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError as error:
            raise RuntimeError(f"Existing manifest is corrupt: {manifest_path}.") from error
        if rows and previous_manifest.get("host") != host:
            raise RuntimeError(
                f"Output directory manifest belongs to {previous_manifest.get('host')!r}, "
                f"not {host!r}."
            )
        created_at = str(previous_manifest.get("created_at") or created_at)

    pinned_revisions: dict[str, str | None] = {}
    for spec in runnable_specs:
        revisions = {
            str(row["revision"])
            for row in rows
            if row.get("model_key") == spec.key and row.get("revision")
        }
        if len(revisions) > 1:
            raise RuntimeError(
                f"Model {spec.key} already mixes checkpoint revisions: {sorted(revisions)}."
            )
        pinned_revisions[spec.key] = next(iter(revisions), None)

    atomic_write_json(
        manifest_path,
        manifest_payload(
            args,
            specs,
            skipped_for_host,
            host=host,
            created_at=created_at,
            rows=rows,
            complete=False,
        ),
    )
    if not args.run:
        print(
            f"planned_jobs={len(runnable_specs) * len(METHODS)} "
            f"host_skips={len(skipped_for_host)} "
            f"output={args.output_dir}",
            flush=True,
        )
        return 0

    # Job JSON files are the incremental source of truth.  A previous aggregate
    # must not remain visible while only one side of a refreshed pair exists.
    for aggregate in (args.output_dir / "results.json", args.output_dir / "results.csv"):
        if aggregate.exists():
            aggregate.unlink()

    failures = 0
    for spec in skipped_for_host:
        print(f"skip-host={spec.key}:requires-{spec.preferred_host}", flush=True)
    for spec_index, spec in enumerate(runnable_specs):
        pinned_revision = pinned_revisions.get(spec.key)
        for method in methods_for_model(spec_index):
            output = job_dir / f"{spec.key}__{method}.json"
            if output.is_file() and not args.force:
                try:
                    prior = json.loads(output.read_text())
                except json.JSONDecodeError:
                    prior = {}
                try:
                    expected = expected_job_identity(
                        spec,
                        method,
                        args,
                        host,
                        revision=pinned_revision,
                    )
                except Exception:
                    expected = None
                if (
                    prior.get("status") == "ok"
                    and expected is not None
                    and prior_matches_expected(prior, expected, host=host)
                ):
                    print(f"skip={spec.key}:{method}", flush=True)
                    continue
                print(f"rerun-stale={spec.key}:{method}", flush=True)
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker-model",
                spec.key,
                "--worker-method",
                method,
                "--worker-output",
                str(output),
                "--cache-dir",
                str(args.cache_dir),
                "--threads",
                str(args.threads),
                "--cpu-set",
                args.cpu_set,
            ]
            if pinned_revision:
                command.extend(("--worker-revision", pinned_revision))
            if args.allow_download:
                command.append("--allow-download")
            if args.smoke:
                command.append("--smoke")
            environment = os.environ.copy()
            environment.update(
                {
                    "OMP_NUM_THREADS": str(args.threads),
                    "MKL_NUM_THREADS": str(args.threads),
                    "OPENBLAS_NUM_THREADS": str(args.threads),
                    "TOKENIZERS_PARALLELISM": "false",
                }
            )
            print(f"start={spec.key}:{method}", flush=True)
            completed = subprocess.run(command, env=environment, check=False)
            failures += int(completed.returncode != 0)
            try:
                current = json.loads(output.read_text())
            except (OSError, json.JSONDecodeError):
                current = {}
            current_revision = current.get("revision")
            if current_revision:
                if pinned_revision and str(current_revision) != pinned_revision:
                    raise RuntimeError(
                        f"Worker changed pinned revision for {spec.key}: "
                        f"{pinned_revision} -> {current_revision}."
                    )
                pinned_revision = str(current_revision)
                pinned_revisions[spec.key] = pinned_revision
            rows = read_job_rows(job_dir)
            atomic_write_json(
                args.output_dir / "progress.json",
                completed_job_provenance(rows),
            )
            atomic_write_json(
                manifest_path,
                manifest_payload(
                    args,
                    specs,
                    skipped_for_host,
                    host=host,
                    created_at=created_at,
                    rows=rows,
                    complete=False,
                ),
            )
    if failures:
        return 1
    rows = read_job_rows(job_dir)
    validate_completed_pairs(
        rows,
        runnable_specs,
        args,
        host=host,
        pinned_revisions=pinned_revisions,
    )
    write_summary(args.output_dir, expected_host=host)
    atomic_write_json(
        manifest_path,
        manifest_payload(
            args,
            specs,
            skipped_for_host,
            host=host,
            created_at=created_at,
            rows=rows,
            complete=True,
        ),
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="all")
    parser.add_argument("--models", help="Comma-separated model keys.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/general-rerun"))
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=default_hf_cache(),
    )
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--cpu-set", default="0-11")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--worker-model", choices=tuple(SPECS_BY_KEY), help=argparse.SUPPRESS)
    parser.add_argument("--worker-method", choices=METHODS, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-revision", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.threads < 1:
        raise SystemExit("--threads must be positive.")
    if args.worker_model:
        if not args.worker_method or not args.worker_output:
            raise SystemExit("Worker mode requires method and output path.")
        return worker(
            spec=SPECS_BY_KEY[args.worker_model],
            method=args.worker_method,
            output=args.worker_output,
            cache_dir=args.cache_dir,
            threads=args.threads,
            cpu_set=args.cpu_set,
            allow_download=args.allow_download,
            smoke=args.smoke,
            revision=args.worker_revision,
        )

    specs = select_specs(args.models, args.suite)
    if args.list:
        for spec in specs:
            print(
                f"{spec.key}\t{spec.label}\t{spec.family}\t"
                f"{','.join(spec.suites)}\t{spec.source}"
            )
        return 0
    if args.audit:
        payload = audit(specs, args.cache_dir)
        atomic_write_json(args.output_dir / "audit.json", payload)
        print(json.dumps(payload, indent=2, default=str))
        return 0
    return orchestrate(args, specs)


if __name__ == "__main__":
    raise SystemExit(main())
