"""Compatibility implementation for historical per-checkpoint experiments.

The final thesis runners use the trajectory-wide protocol in
examples.checkpoint_compression. This module keeps the older public helper
isolated from model-loading and federated-training code.
"""

from __future__ import annotations

import gc
from pathlib import Path

import pandas as pd
import torch

from .dtypes import dtype_name
from .codec_protocol import (
    CODEC_PROTOCOL_VERSION,
    assert_tensor_roundtrip,
    decode_tensor_batch,
    encode_tensor_batch,
    is_current_codec_protocol,
    payload_exponent_sizes,
    prepare_neuralzip_codec,
)
from .neuralzip_codec import NeuralZipCodec
from .zipnn_codec import ZipNNCodec


def _ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _canonical_cifar_architecture(architecture):
    key = str(architecture).lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "resnet_18": "resnet18",
        "resnet_34": "resnet34",
        "vgg16": "vgg16_bn",
        "vgg_16": "vgg16_bn",
        "vgg_16_bn": "vgg16_bn",
        "vit": "vit_b_16",
        "vit_base": "vit_b_16",
        "vit_b16": "vit_b_16",
    }
    key = aliases.get(key, key)
    supported = {"resnet18", "resnet34", "vgg16_bn", "vit_b_16"}
    if key not in supported:
        raise ValueError(
            f"Unsupported CIFAR architecture {architecture!r}: "
            + ", ".join(sorted(supported))
            + "."
        )
    return key


def _floating_tensors_from_state(state, *, dtype):
    tensors = {}
    for name, tensor in state.items():
        if not torch.is_tensor(tensor) or not tensor.is_floating_point():
            continue
        value = tensor.detach().cpu().contiguous()
        if dtype is not None:
            value = value.to(dtype=dtype)
        tensors[name] = value
    return tensors

def compress_training_checkpoints(
    checkpoint_dir,
    output_dir,
    *,
    architecture,
    dtype=None,
    chunk_size=256 << 10,
    compression_threshold=0.95,
    clustering="js-distance",
    cluster_js_threshold=0.05,
    max_codebook_sample_bytes=8 << 20,
    ngram_values=(2, 3, 4),
    top_k=254,
    packed_source_budget_ratio=0.30,
    verify=True,
    resume=True,
):
    """Compress each checkpoint with a NeuralZip setup fitted to that checkpoint."""

    architecture = _canonical_cifar_architecture(architecture)
    checkpoint_dir = Path(checkpoint_dir)
    output_dir = _ensure_dir(output_dir)
    result_path = output_dir / "checkpoint_compression_results.csv"
    summary_path = output_dir / "checkpoint_compression_summary.csv"
    checkpoint_paths = sorted(
        checkpoint_dir.glob(f"{architecture}_cifar100_epoch_*.pt")
    )
    if not checkpoint_paths:
        raise ValueError(
            f"No {architecture} CIFAR-100 checkpoints found in {checkpoint_dir}."
        )

    zipnn_codec = ZipNNCodec(
        chunk_size=chunk_size,
        compression_threshold=compression_threshold,
    )

    rows = (
        pd.read_csv(result_path).to_dict("records")
        if resume and result_path.exists()
        else []
    )
    if rows and any(row.get("setup_scope") != "per-checkpoint" for row in rows):
        rows = []
    expected_encoder_paths = {
        "ZipNN": "compress_tensor",
        "NeuralZip": "compress_tensors",
    }
    rows = [
        row
        for row in rows
        if is_current_codec_protocol(row.get("codec_protocol_version"))
        and row.get("encoder_path")
        == expected_encoder_paths.get(str(row.get("method")))
    ]
    completed = {
        (str(row["method"]), int(row["epoch"]))
        for row in rows
        if str(row.get("status", "ok")) == "ok"
    }

    for checkpoint_path in checkpoint_paths:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        epoch = int(checkpoint["epoch"])
        tensors = _floating_tensors_from_state(checkpoint["model_state"], dtype=dtype)
        protocols = [
            (
                "ZipNN",
                zipnn_codec,
                {
                    "setup_seconds": 0.0,
                    "cluster_setup_seconds": 0.0,
                    "codebook_setup_seconds": 0.0,
                    "setup_bytes": 0,
                    "exponent_codebook_bytes": 0,
                    "clusters": 0,
                    "codebooks": 0,
                },
            )
        ]
        if ("NeuralZip", epoch) not in completed:
            neuralzip_codec = NeuralZipCodec(
                chunk_size=chunk_size,
                compression_threshold=compression_threshold,
                clustering=clustering,
                cluster_js_threshold=cluster_js_threshold,
                max_codebook_sample_bytes=max_codebook_sample_bytes,
                ngram_values=ngram_values,
                top_k=top_k,
                packed_source_budget_ratio=packed_source_budget_ratio,
            )
            neuralzip_setup = prepare_neuralzip_codec(
                neuralzip_codec,
                tensors.items(),
            )
            codebooks = neuralzip_codec.export_codebooks()
            element_sizes = {"float16": 2, "bfloat16": 2, "float32": 4}
            exponent_codebook_bytes = sum(
                int(codebook.metadata_bytes)
                for codebook in codebooks
                if int(codebook.stream_id)
                == element_sizes[str(codebook.dtype)] - 1
            )
            protocols.append(
                (
                    "NeuralZip",
                    neuralzip_codec,
                    {
                        "setup_seconds": neuralzip_setup.total_setup_seconds,
                        "cluster_setup_seconds": neuralzip_setup.precompute_seconds,
                        "codebook_setup_seconds": (
                            neuralzip_setup.codebook_setup_seconds
                        ),
                        "setup_bytes": int(
                            neuralzip_codec.codebook_metadata_bytes()
                        ),
                        "exponent_codebook_bytes": int(
                            exponent_codebook_bytes
                        ),
                        "clusters": int(neuralzip_codec.cluster_count),
                        "codebooks": int(len(codebooks)),
                    },
                )
            )
        for method, codec, setup in protocols:
            if (method, epoch) in completed:
                continue
            encoded = encode_tensor_batch(codec, tensors)
            exponent_raw_bytes = 0
            exponent_compressed_bytes = 0
            for payload in encoded.payloads:
                raw_size, compressed_size = payload_exponent_sizes(payload)
                exponent_raw_bytes += raw_size
                exponent_compressed_bytes += compressed_size
            exponent_compressed_bytes += int(setup["exponent_codebook_bytes"])
            stored_bytes = int(encoded.compressed_bytes) + int(
                setup["setup_bytes"]
            )

            decoded, decode_seconds = decode_tensor_batch(
                codec,
                encoded.payloads,
            )
            if verify:
                assert_tensor_roundtrip(
                    tensors,
                    decoded,
                    method=method,
                    checkpoint=checkpoint_path.name,
                )
            row = {
                "experiment": "training-checkpoints",
                "architecture": architecture,
                "checkpoint": checkpoint_path.name,
                "epoch": epoch,
                "method": method,
                "codec_protocol_version": CODEC_PROTOCOL_VERSION,
                "encoder_path": encoded.encoder_path,
                "setup_scope": "per-checkpoint",
                "dtype": (
                    dtype_name(dtype)
                    if dtype is not None
                    else "checkpoint"
                ),
                "tensor_count": len(tensors),
                "raw_bytes": int(encoded.raw_bytes),
                "payload_compressed_bytes": int(encoded.compressed_bytes),
                "stored_bytes": stored_bytes,
                "compression_ratio_pct": (
                    100 * stored_bytes / encoded.raw_bytes
                    if encoded.raw_bytes
                    else 0.0
                ),
                "exponent_raw_bytes": int(exponent_raw_bytes),
                "exponent_compressed_bytes": int(exponent_compressed_bytes),
                "exponent_compression_ratio_pct": (
                    100 * exponent_compressed_bytes / exponent_raw_bytes
                    if exponent_raw_bytes
                    else 0.0
                ),
                "encode_seconds": float(encoded.encode_seconds),
                "total_compression_seconds": float(encoded.encode_seconds)
                + float(setup["setup_seconds"]),
                "decode_seconds": float(decode_seconds),
                "huffman_chunks": int(encoded.huffman_chunks),
                "raw_chunks": int(encoded.raw_chunks),
                "shared_codebook_refs": int(encoded.shared_codebook_refs),
                "roundtrip_exact": True,
                "status": "ok",
                **setup,
            }
            rows = [
                existing
                for existing in rows
                if not (
                    str(existing["method"]) == method
                    and int(existing["epoch"]) == epoch
                )
            ]
            rows.append(row)
            rows.sort(key=lambda item: (int(item["epoch"]), str(item["method"])))
            pd.DataFrame(rows).to_csv(result_path, index=False)
            print(
                f"{architecture} checkpoint={epoch} method={method} "
                f"ratio={row['compression_ratio_pct']:.3f}% "
                f"exponents={row['exponent_compression_ratio_pct']:.3f}% "
                f"setup={row['setup_seconds']:.3f}s "
                f"encode={row['encode_seconds']:.3f}s",
                flush=True,
            )
            del decoded, encoded
        del checkpoint, tensors
        gc.collect()

    result_df = pd.DataFrame(rows)
    if any(
        group[["codec_protocol_version", "encoder_path"]].nunique().max() != 1
        for _, group in result_df.groupby("method")
    ):
        raise RuntimeError("Checkpoint summary mixes incompatible codec protocols.")
    summary = (
        result_df.groupby("method", as_index=False)
        .agg(
            architecture=("architecture", "first"),
            codec_protocol_version=("codec_protocol_version", "first"),
            encoder_path=("encoder_path", "first"),
            checkpoints=("epoch", "count"),
            setup_seconds=("setup_seconds", "sum"),
            setup_bytes=("setup_bytes", "sum"),
            clusters_mean=("clusters", "mean"),
            codebooks_mean=("codebooks", "mean"),
            raw_bytes=("raw_bytes", "sum"),
            payload_compressed_bytes=("payload_compressed_bytes", "sum"),
            stored_bytes=("stored_bytes", "sum"),
            exponent_raw_bytes=("exponent_raw_bytes", "sum"),
            exponent_compressed_bytes=("exponent_compressed_bytes", "sum"),
            encode_seconds=("encode_seconds", "sum"),
            total_compression_seconds=("total_compression_seconds", "sum"),
            decode_seconds=("decode_seconds", "sum"),
        )
    )
    summary["compression_ratio_pct"] = (
        100 * summary["stored_bytes"] / summary["raw_bytes"]
    )
    summary["exponent_compression_ratio_pct"] = (
        100
        * summary["exponent_compressed_bytes"]
        / summary["exponent_raw_bytes"]
    )
    summary.to_csv(summary_path, index=False)
    return result_df, summary


