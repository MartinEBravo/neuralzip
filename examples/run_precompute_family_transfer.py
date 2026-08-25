from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neuralzip.experiment_io import atomic_write_json
from neuralzip.hashing import (
    canonical_digest as _shared_canonical_digest,
    sha256_file as _shared_sha256_file,
)
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    resnet18,
    resnet34,
)
from transformers import (
    Gemma3ForConditionalGeneration,
    Gemma3TextConfig,
    Gemma3TextModel,
    Qwen3_5Config,
    Qwen3_5TextModel,
    SiglipConfig,
    SiglipVisionModel,
    WhisperModel,
)

from neuralzip import Compressor
from neuralzip.codec_protocol import (
    decode_tensor_batch,
    encode_tensor_batch,
    measure_exponent_streams,
    prepare_neuralzip_codec,
)
from neuralzip.tensors import bitwise_equal
from neuralzip.zipnn_codec import ZipNNCodec
from neuralzip.checkpoint_loading import (
    SnapshotSelection,
    download_safetensors_snapshot,
    select_safetensors_weights as _select_safetensors_weights,
)


@dataclass(frozen=True)
class PairSpec:
    key: str
    family: str
    kind: str
    source_id: str
    source_dir: str | None
    target_id: str
    target_dir: str | None
    source_revision: str | None = None
    target_revision: str | None = None


PAIR_SPECS = {
    spec.key: spec
    for spec in (
        PairSpec(
            key="resnet",
            family="ResNet",
            kind="resnet",
            source_id="torchvision/resnet34-imagenet1k-v1",
            source_dir=None,
            target_id="torchvision/resnet18-imagenet1k-v1",
            target_dir=None,
        ),
        PairSpec(
            key="qwen",
            family="Qwen 3.5",
            kind="qwen_text",
            source_id="Qwen/Qwen3.5-2B-Base",
            source_dir="qwen35-2b",
            target_id="Qwen/Qwen3.5-0.8B-Base",
            target_dir="qwen35-08b",
            source_revision="b1485b2fa6dfa1287294f269f5fb618e03d52d7c",
            target_revision="dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68",
        ),
        PairSpec(
            key="gemma",
            family="Gemma 3",
            kind="gemma_text",
            source_id="google/gemma-3-4b-pt",
            source_dir="gemma3-4b-pt",
            target_id="google/gemma-3-1b-pt",
            target_dir="gemma3-1b-pt",
            source_revision="cc012e0a6d0787b4adcc0fa2c4da74402494554d",
            target_revision="fcf18a2a879aab110ca39f8bffbccd5d49d8eb29",
        ),
        PairSpec(
            key="siglip",
            family="SigLIP 2",
            kind="siglip_vision",
            source_id="google/siglip2-giant-opt-patch16-384",
            source_dir="siglip2-giant-opt-384",
            target_id="google/siglip2-so400m-patch16-384",
            target_dir="siglip2-so400m-384",
            source_revision="a713301b217d38485fb2204c808367d10bc3cc40",
            target_revision="dd658faac399427308559e2c3ac1e99cbe43845d",
        ),
        PairSpec(
            key="whisper",
            family="Whisper",
            kind="whisper",
            source_id="openai/whisper-large-v3",
            source_dir="whisper-large-v3",
            target_id="openai/whisper-large-v3-turbo",
            target_dir="whisper-large-v3-turbo",
            source_revision="06f233fe06e710322aca913c1bc4249a0d71fce1",
            target_revision="41f01f3fe87f28c78e2fbf8b568835947dd65ed9",
        ),
    )
}

SHARED_MODES = {"shared_huffman", "shared_huff0", "packed_huff0"}
PACKED_MODES = {"packed_huff0"}
LOCAL_MODES = {"huffman", "local_huff0"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", choices=PAIR_SPECS, required=True)
    parser.add_argument(
        "--transfer-mode",
        choices=("direct", "js-probe"),
        default="direct",
    )
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument(
        "--source-revision",
        help="Optional source commit override; defaults to the pinned pair commit.",
    )
    parser.add_argument(
        "--target-revision",
        help="Optional target commit override; defaults to the pinned pair commit.",
    )
    return parser.parse_args()


def _sha256_file(path: Path, *, block_bytes: int = 8 << 20) -> str:
    return _shared_sha256_file(path, block_bytes=block_bytes)


def _canonical_digest(value: object) -> str:
    return _shared_canonical_digest(value, stringify_unknown=False)


def select_snapshot_weights(path: Path) -> SnapshotSelection:
    """Compatibility entry point with the family suite's config requirement."""

    return _select_safetensors_weights(path, require_config=True)


def _download_metadata(path: Path, relative_path: str) -> dict | None:
    metadata_path = (
        path
        / ".cache"
        / "huggingface"
        / "download"
        / f"{relative_path}.metadata"
    )
    if not metadata_path.is_file():
        return None
    lines = metadata_path.read_text().splitlines()
    if len(lines) < 3:
        return None
    try:
        timestamp = float(lines[2])
    except ValueError:
        return None
    artifact = path / relative_path
    if not artifact.is_file() or artifact.stat().st_mtime - 1 > timestamp:
        return None
    return {"commit": lines[0], "etag": lines[1]}


def snapshot_provenance(
    repo_id: str,
    path: Path,
    selection: SnapshotSelection,
    *,
    expected_revision: str,
) -> dict:
    artifacts = []
    commits = set()
    every_weight_has_metadata = True
    for file in selection.files:
        relative_path = str(file.relative_to(path))
        metadata = _download_metadata(path, relative_path)
        if metadata is not None:
            commits.add(str(metadata["commit"]))
            digest = str(metadata["etag"])
            digest_source = "huggingface-local-dir-metadata"
        else:
            every_weight_has_metadata = False
            digest = _sha256_file(file)
            digest_source = "computed-sha256"
        artifacts.append(
            {
                "repository_path": relative_path,
                "size_bytes": int(file.stat().st_size),
                "digest": digest,
                "digest_source": digest_source,
            }
        )
    if len(commits) > 1:
        raise ValueError(f"Snapshot mixes multiple revisions under {path}: {commits}")
    resolved_revision = (
        next(iter(commits))
        if every_weight_has_metadata and len(commits) == 1
        else None
    )
    if resolved_revision is not None and resolved_revision != expected_revision:
        raise ValueError(
            f"Snapshot revision mismatch for {repo_id}: "
            f"{resolved_revision} != {expected_revision}"
        )
    payload = {
        "repo_id": repo_id,
        "requested_revision": expected_revision,
        "resolved_revision": resolved_revision or expected_revision,
        "revision_verified_by_metadata": resolved_revision is not None,
        "weight_files": artifacts,
        "weight_index": (
            str(selection.index_path.relative_to(path))
            if selection.index_path is not None
            else None
        ),
        "indexed_tensor_keys": (
            len(selection.expected_keys)
            if selection.expected_keys is not None
            else None
        ),
    }
    payload["weight_fingerprint"] = _canonical_digest(artifacts)
    return payload


def ensure_snapshot(repo_id: str, path: Path, revision: str) -> dict:
    if not re.fullmatch(r"[0-9a-f]{40}", str(revision)):
        raise ValueError(f"Family snapshots require a pinned commit: {revision!r}")
    try:
        selection = select_snapshot_weights(path)
        metadata = [
            _download_metadata(path, str(file.relative_to(path)))
            for file in selection.files
        ]
        commits = {
            str(row["commit"])
            for row in metadata
            if row is not None
        }
        if len(commits) > 1:
            raise ValueError(
                f"Snapshot mixes multiple revisions under {path}: {commits}"
            )
        metadata_revision = (
            next(iter(commits))
            if metadata and all(row is not None for row in metadata)
            else None
        )
        if metadata_revision == revision:
            return snapshot_provenance(
                repo_id,
                path,
                selection,
                expected_revision=revision,
            )
    except (FileNotFoundError, ValueError):
        # Reconcile an incomplete, stale, or unverified local directory against
        # the immutable requested commit before accepting it as experiment input.
        pass
    path.mkdir(parents=True, exist_ok=True)
    _, selection = download_safetensors_snapshot(
        repo_id=repo_id,
        revision=revision,
        local_dir=path,
        local_files_only=False,
        include_config=True,
    )
    provenance = snapshot_provenance(
        repo_id,
        path,
        selection,
        expected_revision=revision,
    )
    if not provenance["revision_verified_by_metadata"]:
        raise RuntimeError(
            f"Downloaded snapshot has no verifiable revision metadata: {path}"
        )
    return provenance


def load_hf_model(kind: str, path: Path):
    common = {
        "dtype": torch.bfloat16,
        "local_files_only": True,
    }
    if kind == "qwen_text":
        config = Qwen3_5Config.from_pretrained(
            path, local_files_only=True
        ).text_config
        model = Qwen3_5TextModel.from_pretrained(path, config=config, **common)
    elif kind == "gemma_text":
        config_data = json.loads((path / "config.json").read_text())
        if "vision_config" in config_data:
            full_model = Gemma3ForConditionalGeneration.from_pretrained(
                path, **common
            )
            model = full_model.model.language_model
            full_model.model.language_model = None
            del full_model
            gc.collect()
        else:
            config = Gemma3TextConfig.from_pretrained(
                path, local_files_only=True
            )
            model = Gemma3TextModel.from_pretrained(
                path, config=config, **common
            )
    elif kind == "siglip_vision":
        config = SiglipConfig.from_pretrained(
            path, local_files_only=True
        ).vision_config
        model = SiglipVisionModel.from_pretrained(path, config=config, **common)
    elif kind == "whisper":
        model = WhisperModel.from_pretrained(path, **common)
    else:
        raise ValueError(f"Unsupported model kind: {kind}")
    model.eval()
    model.requires_grad_(False)
    dtypes = {parameter.dtype for parameter in model.parameters()}
    if dtypes != {torch.bfloat16}:
        raise RuntimeError(f"Model is not uniformly BF16: {dtypes}")
    return model


def _torchvision_weight_provenance(model_id: str, weights) -> dict:
    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / Path(weights.url).name
    if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
        raise FileNotFoundError(f"Torchvision checkpoint is missing: {checkpoint}")
    artifacts = [
        {
            "repository_path": checkpoint.name,
            "size_bytes": int(checkpoint.stat().st_size),
            "digest": _sha256_file(checkpoint),
            "digest_source": "computed-sha256",
        }
    ]
    return {
        "repo_id": model_id,
        "requested_revision": str(weights),
        "resolved_revision": str(weights),
        "revision_verified_by_metadata": True,
        "weight_files": artifacts,
        "weight_index": None,
        "indexed_tensor_keys": None,
        "weight_fingerprint": _canonical_digest(artifacts),
    }


def load_pair(
    spec: PairSpec,
    model_root: Path,
    *,
    source_revision: str | None = None,
    target_revision: str | None = None,
):
    if spec.kind == "resnet":
        source = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        target = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        return (
            source.eval().to(torch.bfloat16),
            target.eval().to(torch.bfloat16),
            _torchvision_weight_provenance(
                spec.source_id, ResNet34_Weights.IMAGENET1K_V1
            ),
            _torchvision_weight_provenance(
                spec.target_id, ResNet18_Weights.IMAGENET1K_V1
            ),
        )

    source_path = model_root / str(spec.source_dir)
    target_path = model_root / str(spec.target_dir)
    source_revision = source_revision or spec.source_revision
    target_revision = target_revision or spec.target_revision
    if source_revision is None or target_revision is None:
        raise ValueError(f"Pair {spec.key} has no pinned source/target revision.")
    source_provenance = ensure_snapshot(
        spec.source_id, source_path, source_revision
    )
    target_provenance = ensure_snapshot(
        spec.target_id, target_path, target_revision
    )
    return (
        load_hf_model(spec.kind, source_path),
        load_hf_model(spec.kind, target_path),
        source_provenance,
        target_provenance,
    )


def floating_parameters(model):
    """Return the complete floating model state exactly once per state key.

    ``named_parameters()`` omits persistent floating buffers such as
    BatchNorm running means and variances.  Family transfer is a checkpoint
    experiment, so its input must match the complete floating ``state_dict``
    used by the general storage runner.  Reading a single ``state_dict`` also
    avoids the duplicate entries that would result from concatenating
    parameters and buffers manually.
    """

    named_tensors = []
    seen_names = set()
    for raw_name, tensor in model.state_dict(keep_vars=True).items():
        name = str(raw_name)
        if name in seen_names:
            raise RuntimeError(f"Duplicate state key: {name}")
        seen_names.add(name)
        if (
            torch.is_tensor(tensor)
            and tensor.is_floating_point()
            and tensor.numel() > 0
        ):
            named_tensors.append((name, tensor))
    return named_tensors


def exponent_metrics(payloads, codebooks=()):
    payloads = tuple(payloads)
    codebooks = tuple(codebooks)
    sizes = measure_exponent_streams(payloads, codebooks)
    mode_symbols = {}
    mode_chunks = {}
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
        "exponent_pct": sizes.ratio_pct,
        "shared_exponent_pct": symbol_pct(SHARED_MODES),
        "packed_exponent_pct": symbol_pct(PACKED_MODES),
        "local_exponent_pct": symbol_pct(LOCAL_MODES),
        "raw_exponent_pct": symbol_pct({"raw"}),
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
        compression_threshold=0.95,
        threads=threads,
        # Mode scoring and runtime use the same uncached direct-encoding
        # boundary. No target transformation is moved into setup.
        enable_prepacked_cache=False,
    ).codec
    report = prepare_neuralzip_codec(codec, named_tensors)
    return codec, {
        "clustering_seconds": report.clustering_seconds,
        "codebook_seconds": report.codebook_setup_seconds,
        "total_seconds": report.total_setup_seconds,
        "clusters": int(codec.cluster_count),
        "codebooks": len(codec.export_codebooks()),
        "selected_modes": dict(codec.selected_modes),
    }


def run_pair(
    spec: PairSpec,
    model_root: Path,
    threads: int,
    transfer_mode: str,
    *,
    source_revision: str | None = None,
    target_revision: str | None = None,
):
    if threads <= 0:
        raise ValueError("threads must be positive.")
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    print(f"[{spec.key}] loading source and target", flush=True)
    source, target, source_provenance, target_provenance = load_pair(
        spec,
        model_root,
        source_revision=source_revision,
        target_revision=target_revision,
    )
    source_named = floating_parameters(source)
    target_named = floating_parameters(target)
    print(
        f"[{spec.key}] source={sum(t.numel() for _, t in source_named):,} "
        f"target={sum(t.numel() for _, t in target_named):,}",
        flush=True,
    )

    print(f"[{spec.key}] source setup", flush=True)
    source_codec, source_setup = setup_neuralzip(source_named, threads=threads)

    if transfer_mode == "direct":
        print(f"[{spec.key}] direct structural target mapping", flush=True)
        started = time.perf_counter()
        transfer_stats = source_codec.prepare_direct_transfer_target(target_named)
        transfer_seconds = time.perf_counter() - started
    else:
        print(f"[{spec.key}] JS transfer preparation", flush=True)
        started = time.perf_counter()
        transfer_stats = source_codec.prepare_transfer_target(target_named)
        transfer_seconds = time.perf_counter() - started

    # The production transfer path reuses only the frozen family plan.  Exact
    # target materialization is deliberately disabled after the plan has been
    # selected so the measured encode includes every target-dependent step.
    source_codec.clear_prepacked_cache()
    source_codec.enable_prepacked_cache = False
    if (
        source_codec.prepacked_cache_bytes != 0
        or source_codec.prepacked_cache_stats["entries"] != 0
    ):
        raise RuntimeError("Direct transfer unexpectedly retained an exact cache.")

    print(
        f"[{spec.key}] direct-transfer uncached warmup and measurement",
        flush=True,
    )
    direct_verification = compress_with_codec(
        source_codec,
        target_named,
        verify=True,
    )
    transferred_direct = compress_with_codec(source_codec, target_named)
    if (
        source_codec.prepacked_cache_bytes != 0
        or source_codec.prepacked_cache_stats["entries"] != 0
    ):
        raise RuntimeError("Direct transfer populated an exact target cache.")

    print(f"[{spec.key}] target-specific setup", flush=True)
    native_codec, native_setup = setup_neuralzip(target_named, threads=threads)
    native_codec.clear_prepacked_cache()
    native_codec.enable_prepacked_cache = False
    native_setup["target_materialization_performed"] = False
    native_setup["target_materialization_bytes"] = 0
    native_verification = compress_with_codec(native_codec, target_named, verify=True)
    native = compress_with_codec(native_codec, target_named)
    if (
        native_codec.prepacked_cache_bytes != 0
        or native_codec.prepacked_cache_stats["entries"] != 0
    ):
        raise RuntimeError("Target-specific direct encoding populated an exact cache.")

    print(f"[{spec.key}] ZipNN warmup and measurement", flush=True)
    zipnn_codec = ZipNNCodec(compression_threshold=0.95, threads=threads)
    zipnn_verification = compress_with_codec(
        zipnn_codec,
        target_named,
        verify=True,
    )
    zipnn = compress_with_codec(zipnn_codec, target_named)
    verification = {
        "zipnn": zipnn_verification["roundtrip_ok"],
        "neuralzip_target_specific_direct": native_verification["roundtrip_ok"],
        "neuralzip_transferred_direct": direct_verification["roundtrip_ok"],
    }
    failed_roundtrips = [name for name, ok in verification.items() if ok is not True]
    if failed_roundtrips:
        raise RuntimeError(
            "Bitwise round-trip failed for: " + ", ".join(failed_roundtrips)
        )

    output = {
        "pair": spec.key,
        "family": spec.family,
        "source_model": spec.source_id,
        "target_model": spec.target_id,
        "source_weights": source_provenance,
        "target_weights": target_provenance,
        "dtype": "bfloat16",
        "threads": threads,
        "transfer_mode": transfer_mode,
        "measured_repetitions": 1,
        "runner_sha256": _sha256_file(Path(__file__).resolve()),
        "warmup": (
            "one uncached direct pass per measured encoding path; "
            "no exact target materialization"
        ),
        "measurement_regime": "warmed direct encode without exact target cache",
        "direct_encode_includes": [
            "target byte reorder and stream extraction",
            "packed-exponent tokenization",
            "local Huff0 tree construction and fallback",
            "entropy encoding",
            "payload metadata assembly",
        ],
        "direct_encode_excludes": [
            "family setup",
            "target mapping",
            "verification and decoding",
            "metric accounting",
        ],
        "source_parameters": sum(int(tensor.numel()) for _, tensor in source_named),
        "target_parameters": sum(int(tensor.numel()) for _, tensor in target_named),
        "source_setup": source_setup,
        "target_native_setup": native_setup,
        "transfer_setup": {
            "seconds": transfer_seconds,
            **transfer_stats,
        },
        "target_materialization": {
            "performed": False,
            "seconds": None,
            "resident_bytes": 0,
        },
        "zipnn": {
            **zipnn,
            "roundtrip_ok": zipnn_verification["roundtrip_ok"],
        },
        "neuralzip_target_specific_direct": {
            **native,
            "roundtrip_ok": native_verification["roundtrip_ok"],
        },
        # Compatibility alias for existing consumers. This is now an uncached
        # direct encode, not an exact-target hot-cache measurement.
        "neuralzip_native": {
            **native,
            "roundtrip_ok": native_verification["roundtrip_ok"],
        },
        "neuralzip_transferred_direct": {
            **transferred_direct,
            "roundtrip_ok": direct_verification["roundtrip_ok"],
        },
        # Historical alias. New consumers should use the explicit direct key.
        "neuralzip_transferred": {
            **transferred_direct,
            "roundtrip_ok": direct_verification["roundtrip_ok"],
        },
        "comparison": {
            "transfer_setup_reduction_vs_native_pct": (
                100.0 * (1.0 - transfer_seconds / native_setup["total_seconds"])
            ),
            "direct_transfer_speedup_vs_zipnn": (
                zipnn["encode_seconds"] / transferred_direct["encode_seconds"]
            ),
            "direct_transfer_speedup_vs_target_specific": (
                native["encode_seconds"] / transferred_direct["encode_seconds"]
            ),
            "direct_transfer_ratio_delta_vs_zipnn_pp": (
                transferred_direct["exponent_pct"] - zipnn["exponent_pct"]
            ),
            "direct_transfer_ratio_delta_vs_target_specific_pp": (
                transferred_direct["exponent_pct"] - native["exponent_pct"]
            ),
            # Historical comparison aliases now refer to the uncached direct
            # regime. Explicit ``direct_*`` keys above are preferred.
            "transfer_speedup_vs_zipnn": (
                zipnn["encode_seconds"] / transferred_direct["encode_seconds"]
            ),
            "transfer_speedup_vs_native": (
                native["encode_seconds"] / transferred_direct["encode_seconds"]
            ),
            "transfer_ratio_delta_vs_zipnn_pp": (
                transferred_direct["exponent_pct"] - zipnn["exponent_pct"]
            ),
            "transfer_ratio_delta_vs_native_pp": (
                transferred_direct["exponent_pct"] - native["exponent_pct"]
            ),
            "incremental_target_seconds": (
                transfer_seconds + transferred_direct["encode_seconds"]
            ),
            "first_family_member_total_seconds": (
                source_setup["total_seconds"]
                + transfer_seconds
                + transferred_direct["encode_seconds"]
            ),
        },
    }

    del source, target, source_codec, native_codec, source_named, target_named
    gc.collect()
    return output


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = run_pair(
        PAIR_SPECS[args.pair],
        args.model_root,
        args.threads,
        args.transfer_mode,
        source_revision=args.source_revision,
        target_revision=args.target_revision,
    )
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
