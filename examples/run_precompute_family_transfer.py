from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
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
from neuralzip.archive import serialized_group_size
from neuralzip.zipnn_codec import ZipNNCodec


@dataclass(frozen=True)
class PairSpec:
    key: str
    family: str
    kind: str
    source_id: str
    source_dir: str | None
    target_id: str
    target_dir: str | None


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
        ),
        PairSpec(
            key="gemma",
            family="Gemma 3",
            kind="gemma_text",
            source_id="google/gemma-3-4b-pt",
            source_dir="gemma3-4b-pt",
            target_id="google/gemma-3-1b-pt",
            target_dir="gemma3-1b-pt",
        ),
        PairSpec(
            key="siglip",
            family="SigLIP 2",
            kind="siglip_vision",
            source_id="google/siglip2-giant-opt-patch16-384",
            source_dir="siglip2-giant-opt-384",
            target_id="google/siglip2-so400m-patch16-384",
            target_dir="siglip2-so400m-384",
        ),
        PairSpec(
            key="whisper",
            family="Whisper",
            kind="whisper",
            source_id="openai/whisper-large-v3",
            source_dir="whisper-large-v3",
            target_id="openai/whisper-large-v3-turbo",
            target_dir="whisper-large-v3-turbo",
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
    return parser.parse_args()


def ensure_snapshot(repo_id: str, path: Path):
    if (path / "config.json").is_file():
        return
    path.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=path,
        allow_patterns=("*.json", "*.model", "*.safetensors", "*.txt"),
        ignore_patterns=("*.bin", "*.gguf", "*.h5", "*.msgpack", "*.onnx"),
    )


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


def load_pair(spec: PairSpec, model_root: Path):
    if spec.kind == "resnet":
        source = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        target = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        return (
            source.eval().to(torch.bfloat16),
            target.eval().to(torch.bfloat16),
        )

    source_path = model_root / str(spec.source_dir)
    target_path = model_root / str(spec.target_dir)
    ensure_snapshot(spec.source_id, source_path)
    ensure_snapshot(spec.target_id, target_path)
    return (
        load_hf_model(spec.kind, source_path),
        load_hf_model(spec.kind, target_path),
    )


def floating_parameters(model):
    return [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.is_floating_point() and parameter.numel() > 0
    ]


def exponent_metrics(payloads):
    raw_symbols = 0
    compressed_bytes = 0
    mode_symbols = {}
    mode_chunks = {}
    referenced_codebooks = set()
    for payload in payloads:
        group = payload.groups[-1]
        raw_symbols += int(group.symbol_count)
        compressed_bytes += int(serialized_group_size(group))
        for chunk in group.chunks:
            mode = str(chunk.mode)
            mode_symbols[mode] = mode_symbols.get(mode, 0) + int(chunk.symbol_count)
            mode_chunks[mode] = mode_chunks.get(mode, 0) + 1
            if chunk.codebook_id is not None:
                referenced_codebooks.add(int(chunk.codebook_id))

    def symbol_pct(modes):
        symbols = sum(mode_symbols.get(mode, 0) for mode in modes)
        return 100.0 * symbols / raw_symbols if raw_symbols else 0.0

    return {
        "raw_exponents": raw_symbols,
        "compressed_exponent_bytes": compressed_bytes,
        "exponent_pct": (
            100.0 * compressed_bytes / raw_symbols if raw_symbols else 0.0
        ),
        "shared_exponent_pct": symbol_pct(SHARED_MODES),
        "packed_exponent_pct": symbol_pct(PACKED_MODES),
        "local_exponent_pct": symbol_pct(LOCAL_MODES),
        "raw_exponent_pct": symbol_pct({"raw"}),
        "mode_symbols": mode_symbols,
        "mode_chunks": mode_chunks,
        "referenced_codebooks": len(referenced_codebooks),
    }


def compress_with_codec(codec, named_tensors, *, verify=False):
    started = time.perf_counter()
    payloads = [codec.compress_tensor(name, tensor) for name, tensor in named_tensors]
    encode_seconds = time.perf_counter() - started
    roundtrip_ok = None
    if verify:
        roundtrip_ok = all(
            torch.equal(
                codec.decompress_tensor(payload).contiguous().view(torch.uint8),
                tensor.detach().cpu().contiguous().view(torch.uint8),
            )
            for (_, tensor), payload in zip(named_tensors, payloads, strict=True)
        )
    return {
        "encode_seconds": encode_seconds,
        "roundtrip_ok": roundtrip_ok,
        **exponent_metrics(payloads),
    }


def setup_neuralzip(named_tensors):
    codec = Compressor(
        "neuralzip",
        verify=False,
        compression_threshold=0.95,
        threads=32,
    ).codec
    started = time.perf_counter()
    codec.prepare_clusters(named_tensors)
    clustering_seconds = time.perf_counter() - started
    started = time.perf_counter()
    codec.build_codebooks_and_select_modes(named_tensors)
    codebook_seconds = time.perf_counter() - started
    return codec, {
        "clustering_seconds": clustering_seconds,
        "codebook_seconds": codebook_seconds,
        "total_seconds": clustering_seconds + codebook_seconds,
        "clusters": int(codec.cluster_count),
        "codebooks": len(codec.export_codebooks()),
        "selected_modes": dict(codec.selected_modes),
    }


def direct_transfer_stats(codec, target_named):
    source_names = set(codec.cluster_map) | set(codec._computed_cluster_map)
    matched_tensors = 0
    matched_exponents = 0
    target_exponents = 0
    for name, tensor in target_named:
        symbols = int(tensor.numel())
        target_exponents += symbols
        if name in source_names:
            matched_tensors += 1
            matched_exponents += symbols
    return {
        "mapping": "exact source tensor name",
        "target_tensors": len(target_named),
        "matched_tensors": matched_tensors,
        "matched_tensor_pct": 100.0 * matched_tensors / len(target_named),
        "target_exponents": target_exponents,
        "matched_exponents": matched_exponents,
        "matched_exponent_pct": 100.0 * matched_exponents / target_exponents,
        "target_histograms_computed": False,
        "target_js_matching": False,
        "target_mode_probing": False,
    }


def run_pair(spec: PairSpec, model_root: Path, threads: int, transfer_mode: str):
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    print(f"[{spec.key}] loading source and target", flush=True)
    source, target = load_pair(spec, model_root)
    source_named = floating_parameters(source)
    target_named = floating_parameters(target)
    print(
        f"[{spec.key}] source={sum(t.numel() for _, t in source_named):,} "
        f"target={sum(t.numel() for _, t in target_named):,}",
        flush=True,
    )

    print(f"[{spec.key}] source setup", flush=True)
    source_codec, source_setup = setup_neuralzip(source_named)

    if transfer_mode == "direct":
        print(f"[{spec.key}] direct target encoding", flush=True)
        transfer_seconds = 0.0
        transfer_stats = direct_transfer_stats(source_codec, target_named)
        verification = compress_with_codec(source_codec, target_named, verify=True)
        transferred = compress_with_codec(source_codec, target_named)
    else:
        print(f"[{spec.key}] JS transfer preparation", flush=True)
        started = time.perf_counter()
        transfer_stats = source_codec.prepare_transfer_target(target_named)
        transfer_seconds = time.perf_counter() - started
        verification = compress_with_codec(source_codec, target_named, verify=True)
        transferred = compress_with_codec(source_codec, target_named)

    print(f"[{spec.key}] target-specific setup", flush=True)
    native_codec, native_setup = setup_neuralzip(target_named)
    compress_with_codec(native_codec, target_named)
    native = compress_with_codec(native_codec, target_named)

    print(f"[{spec.key}] ZipNN warmup and measurement", flush=True)
    zipnn_codec = ZipNNCodec(compression_threshold=0.95, threads=threads)
    compress_with_codec(zipnn_codec, target_named)
    zipnn = compress_with_codec(zipnn_codec, target_named)

    output = {
        "pair": spec.key,
        "family": spec.family,
        "source_model": spec.source_id,
        "target_model": spec.target_id,
        "dtype": "bfloat16",
        "threads": threads,
        "transfer_mode": transfer_mode,
        "measured_repetitions": 1,
        "warmup": "one pass per measured encoding path",
        "source_parameters": sum(int(tensor.numel()) for _, tensor in source_named),
        "target_parameters": sum(int(tensor.numel()) for _, tensor in target_named),
        "source_setup": source_setup,
        "target_native_setup": native_setup,
        "transfer_setup": {
            "seconds": transfer_seconds,
            **transfer_stats,
        },
        "zipnn": zipnn,
        "neuralzip_native": native,
        "neuralzip_transferred": {
            **transferred,
            "roundtrip_ok": verification["roundtrip_ok"],
        },
        "comparison": {
            "transfer_setup_reduction_vs_native_pct": (
                100.0
                if transfer_mode == "direct"
                else 100.0
                * (1.0 - transfer_seconds / native_setup["total_seconds"])
            ),
            "transfer_speedup_vs_zipnn": (
                zipnn["encode_seconds"] / transferred["encode_seconds"]
            ),
            "transfer_speedup_vs_native": (
                native["encode_seconds"] / transferred["encode_seconds"]
            ),
            "transfer_ratio_delta_vs_zipnn_pp": (
                transferred["exponent_pct"] - zipnn["exponent_pct"]
            ),
            "transfer_ratio_delta_vs_native_pp": (
                transferred["exponent_pct"] - native["exponent_pct"]
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
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
