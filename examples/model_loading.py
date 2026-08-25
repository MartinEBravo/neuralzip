"""Pinned checkpoint loading and provenance for thesis experiments."""

from __future__ import annotations

import gc
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

from neuralzip.checkpoint_loading import select_indexed_weights
from neuralzip.hashing import canonical_digest, sha256_file
from neuralzip.system import host_facts, module_path, package_versions

try:
    from .model_catalog import ModelSpec
except ImportError:  # Direct execution from the examples directory.
    from model_catalog import ModelSpec


DEFAULT_CHUNK_SIZE = 256 << 10
DEFAULT_COMPRESSION_THRESHOLD = 0.95
MIN_PARAMETER_COVERAGE = 0.70
CODE_SUFFIXES = {".py", ".so", ".pyd", ".dylib", ".dll"}


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
    fields = host_facts(include_machine_id=True)
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

    roots: dict[str, Path] = {"examples": Path(__file__).resolve().parent}
    for module_name in ("neuralzip", "zipnn", "zipnn_core"):
        path = module_path(module_name)
        roots[module_name] = path.parent if path.name.startswith("__init__.") else path

    artifacts = []
    for label, root in roots.items():
        paths = (
            [root]
            if root.is_file()
            else sorted(
                path
                for path in root.rglob("*")
                if path.is_file() and path.suffix.lower() in CODE_SUFFIXES
            )
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
    versions = package_versions("torch", "torchvision", "neuralzip", "zipnn")
    _CODE_PROVENANCE = {
        "digest": canonical_digest({"artifacts": artifacts, "versions": versions}),
        "artifacts": artifacts,
        "versions": versions,
    }
    return _CODE_PROVENANCE


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


def _cached_hf_snapshot(
    repo_id: str,
    *,
    cache_dir: Path,
    revision: str | None,
) -> Path:
    """Resolve one Hub snapshot without consulting or mutating the network cache.

    ``snapshot_download(local_files_only=True)`` still applies ``allow_patterns``
    to the complete cached Hub manifest.  For repositories that publish both an
    indexed checkpoint and an alternate monolithic checkpoint, that makes an
    otherwise complete indexed snapshot look incomplete when the alternate was
    deliberately not cached.  Resolve the revision directly from the standard
    Hub cache layout instead; checkpoint completeness is validated afterwards
    against the selected index.
    """

    from huggingface_hub.constants import DEFAULT_REVISION
    from huggingface_hub.file_download import repo_folder_name

    requested_revision = str(revision or DEFAULT_REVISION)
    revision_path = Path(requested_revision)
    if (
        not requested_revision
        or revision_path.is_absolute()
        or ".." in revision_path.parts
        or "\\" in requested_revision
    ):
        raise ValueError(f"Invalid Hugging Face revision {requested_revision!r}.")

    repo_cache = cache_dir / repo_folder_name(repo_id=repo_id, repo_type="model")
    snapshots_dir = repo_cache / "snapshots"
    refs_dir = repo_cache / "refs"
    if not snapshots_dir.is_dir():
        raise FileNotFoundError(
            f"No cached Hugging Face snapshots found for {repo_id!r} under "
            f"{repo_cache}."
        )

    ref_path = refs_dir.joinpath(*revision_path.parts)
    direct_snapshot = (
        snapshots_dir / requested_revision if len(revision_path.parts) == 1 else None
    )
    ref_snapshot: Path | None = None
    if ref_path.is_file():
        ref_lines = [line.strip() for line in ref_path.read_text().splitlines()]
        ref_lines = [line for line in ref_lines if line]
        if len(ref_lines) != 1:
            raise ValueError(
                f"Cached Hugging Face ref {ref_path} must contain exactly one "
                "snapshot revision."
            )
        resolved_revision = ref_lines[0]
        if (
            Path(resolved_revision).name != resolved_revision
            or resolved_revision in {".", ".."}
            or "\\" in resolved_revision
        ):
            raise ValueError(
                f"Cached Hugging Face ref {ref_path} contains an invalid snapshot "
                f"revision {resolved_revision!r}."
            )
        ref_snapshot = snapshots_dir / resolved_revision

    if (
        ref_snapshot is not None
        and direct_snapshot is not None
        and direct_snapshot.is_dir()
        and direct_snapshot != ref_snapshot
    ):
        raise ValueError(
            f"Cached Hugging Face revision {requested_revision!r} for {repo_id!r} "
            f"is ambiguous: the ref selects {ref_snapshot.name!r}, while a direct "
            f"snapshot named {direct_snapshot.name!r} also exists."
        )

    if ref_snapshot is not None:
        snapshot = ref_snapshot
    elif direct_snapshot is not None and direct_snapshot.is_dir():
        # A resolved commit SHA passed by a previous job is pinned directly to
        # its snapshot directory and must not silently follow a moving ref.
        snapshot = direct_snapshot
    else:
        available = sorted(
            path.name for path in snapshots_dir.iterdir() if path.is_dir()
        )
        if len(available) > 1:
            raise ValueError(
                f"Cached Hugging Face revision {requested_revision!r} for {repo_id!r} "
                "cannot be resolved unambiguously because its ref is missing; "
                f"available snapshots: {', '.join(available)}."
            )
        detail = f" Available snapshot: {available[0]}." if available else ""
        raise FileNotFoundError(
            f"Cached Hugging Face revision {requested_revision!r} for {repo_id!r} "
            f"has no ref or matching snapshot under {repo_cache}.{detail}"
        )

    if not snapshot.is_dir():
        raise FileNotFoundError(
            f"Cached Hugging Face revision {requested_revision!r} for {repo_id!r} "
            f"points to missing snapshot {snapshot}."
        )
    return snapshot


def snapshot_weight_files(
    spec: ModelSpec,
    *,
    cache_dir: Path,
    allow_download: bool,
    revision: str | None = None,
) -> CheckpointSelection:
    prefix = f"{spec.subfolder}/" if spec.subfolder else ""
    if spec.weight_file:
        patterns = tuple(
            dict.fromkeys(
                (
                    spec.weight_file,
                    "config.json",
                    f"{prefix}config.json",
                )
            )
        )
    else:
        patterns = tuple(
            dict.fromkeys(
                (
                    f"{prefix}*.safetensors",
                    f"{prefix}*.safetensors.index.json",
                    f"{prefix}*.bin",
                    f"{prefix}*.bin.index.json",
                    f"{prefix}*.pt",
                    "config.json",
                    f"{prefix}config.json",
                )
            )
        )
    if allow_download:
        from huggingface_hub import snapshot_download

        snapshot = Path(
            snapshot_download(
                repo_id=spec.source,
                revision=revision,
                cache_dir=cache_dir,
                allow_patterns=patterns,
                local_files_only=False,
            )
        )
    else:
        snapshot = _cached_hf_snapshot(
            spec.source,
            cache_dir=cache_dir,
            revision=revision,
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

    indexed = select_indexed_weights(root, allow_bin=True)
    if indexed is not None:
        return CheckpointSelection(
            files=indexed.files,
            snapshot=snapshot,
            revision=snapshot.name,
            index_path=indexed.index_path,
            expected_keys=indexed.expected_keys,
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
