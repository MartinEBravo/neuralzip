import json
import statistics
import time
from pathlib import Path

import timm
import torch

from neuralzip import Compressor
from neuralzip.archive import serialized_group_size
from neuralzip.zipnn_codec import (
    ZipNNCodec,
    _zipnn_dtype_config,
    _zipnn_reordered_bytes,
)


MODEL_ID = "timm/resnet18.tv_in1k"
REPEATS = 20


def load_model():
    return timm.create_model(
        f"hf_hub:{MODEL_ID}",
        pretrained=True,
    ).eval().to(torch.bfloat16)


def exponent_streams(named_tensors):
    streams = []
    for name, tensor in named_tensors:
        original = tensor.detach().cpu().contiguous()
        config = _zipnn_dtype_config(original.dtype)
        stream_id = int(config["element_size"]) - 1
        reordered = _zipnn_reordered_bytes(original, config)
        stream = reordered[stream_id :: int(config["element_size"])]
        streams.append((name, original, config, stream_id, stream))
    return streams


def encode_zipnn(codec, streams):
    groups = []
    started = time.perf_counter()
    for _, tensor, config, _, stream in streams:
        groups.append(
            codec._compress_stream(
                stream,
                "exponent",
                chunk_size=codec._stream_chunk_size(config["element_size"]),
            )
        )
    return time.perf_counter() - started, groups


def encode_neuralzip(codec, streams):
    groups = []
    codebook_count = len(codec.export_codebooks())
    started = time.perf_counter()
    for name, tensor, _, stream_id, stream in streams:
        groups.append(
            codec._compress_cluster_stream(
                name,
                tensor,
                stream,
                "exponent",
                stream_id,
                is_exponent=True,
            )
        )
    elapsed = time.perf_counter() - started
    if len(codec.export_codebooks()) != codebook_count:
        raise RuntimeError("Hot exponent encoding created a new codebook.")
    return elapsed, groups


def pipeline_zipnn(codec, named_tensors):
    groups = []
    started = time.perf_counter()
    for _, tensor in named_tensors:
        original = tensor.detach().cpu().contiguous()
        config = _zipnn_dtype_config(original.dtype)
        stream_id = int(config["element_size"]) - 1
        reordered = _zipnn_reordered_bytes(original, config)
        stream = reordered[stream_id :: int(config["element_size"])]
        groups.append(
            codec._compress_stream(
                stream,
                "exponent",
                chunk_size=codec._stream_chunk_size(config["element_size"]),
            )
        )
    return time.perf_counter() - started, groups


def pipeline_neuralzip(codec, named_tensors):
    groups = []
    codebook_count = len(codec.export_codebooks())
    started = time.perf_counter()
    for name, tensor in named_tensors:
        original = tensor.detach().cpu().contiguous()
        config = _zipnn_dtype_config(original.dtype)
        stream_id = int(config["element_size"]) - 1
        reordered = _zipnn_reordered_bytes(original, config)
        stream = reordered[stream_id :: int(config["element_size"])]
        groups.append(
            codec._compress_cluster_stream(
                name,
                original,
                stream,
                "exponent",
                stream_id,
                is_exponent=True,
            )
        )
    elapsed = time.perf_counter() - started
    if len(codec.export_codebooks()) != codebook_count:
        raise RuntimeError("Hot exponent pipeline created a new codebook.")
    return elapsed, groups


def verify_groups(zipnn_codec, neuralzip_codec, streams, zipnn_groups, neuralzip_groups):
    for (_, _, _, _, expected), zipnn_group, neuralzip_group in zip(
        streams,
        zipnn_groups,
        neuralzip_groups,
        strict=True,
    ):
        if zipnn_codec._decompress_stream(zipnn_group) != expected:
            raise RuntimeError("ZipNN exponent roundtrip failed.")
        if neuralzip_codec._decompress_cluster_stream(neuralzip_group) != expected:
            raise RuntimeError("NeuralZip exponent roundtrip failed.")


def group_stats(groups, raw_bytes):
    compressed_bytes = sum(serialized_group_size(group) for group in groups)
    return {
        "raw_bytes": raw_bytes,
        "compressed_bytes_without_shared_codebooks": compressed_bytes,
        "compression_pct_without_shared_codebooks": 100.0 * compressed_bytes / raw_bytes,
        "local_huffman_chunks": sum(group.local_huffman_chunks for group in groups),
        "shared_huffman_chunks": sum(
            group.shared_codebook_refs - group.packed_ngram_chunks for group in groups
        ),
        "packed_ngram_chunks": sum(group.packed_ngram_chunks for group in groups),
        "raw_chunks": sum(group.raw_chunks for group in groups),
    }


def timing_summary(values):
    return {
        "median_seconds": statistics.median(values),
        "min_seconds": min(values),
        "max_seconds": max(values),
        "repeats": len(values),
    }


def main():
    torch.set_num_threads(32)
    model = load_model()
    named_tensors = list(model.named_parameters())
    streams = exponent_streams(named_tensors)
    raw_bytes = sum(len(stream) for *_, stream in streams)

    zipnn_codec = ZipNNCodec(compression_threshold=0.95)
    neuralzip_compressor = Compressor(
        "neuralzip",
        verify=False,
        compression_threshold=0.95,
    )
    setup_result = neuralzip_compressor.compress(model)
    neuralzip_codec = neuralzip_compressor.codec
    shared_compressor = Compressor(
        "neuralzip",
        verify=False,
        compression_threshold=0.95,
        ngram_values=(),
    )
    shared_setup_result = shared_compressor.compress(model)
    shared_codec = shared_compressor.codec

    _, zipnn_groups = encode_zipnn(zipnn_codec, streams)
    _, neuralzip_groups = encode_neuralzip(neuralzip_codec, streams)
    _, shared_groups = encode_neuralzip(shared_codec, streams)
    verify_groups(
        zipnn_codec,
        neuralzip_codec,
        streams,
        zipnn_groups,
        neuralzip_groups,
    )
    verify_groups(
        zipnn_codec,
        shared_codec,
        streams,
        zipnn_groups,
        shared_groups,
    )

    timings = {
        "zipnn_encode_only": [],
        "shared_only_encode_only": [],
        "neuralzip_encode_only": [],
        "zipnn_extraction_and_encode": [],
        "shared_only_extraction_and_encode": [],
        "neuralzip_extraction_and_encode": [],
    }
    for repeat in range(REPEATS):
        if repeat % 2 == 0:
            elapsed, _ = encode_zipnn(zipnn_codec, streams)
            timings["zipnn_encode_only"].append(elapsed)
            elapsed, _ = encode_neuralzip(shared_codec, streams)
            timings["shared_only_encode_only"].append(elapsed)
            elapsed, _ = encode_neuralzip(neuralzip_codec, streams)
            timings["neuralzip_encode_only"].append(elapsed)
            elapsed, _ = pipeline_zipnn(zipnn_codec, named_tensors)
            timings["zipnn_extraction_and_encode"].append(elapsed)
            elapsed, _ = pipeline_neuralzip(shared_codec, named_tensors)
            timings["shared_only_extraction_and_encode"].append(elapsed)
            elapsed, _ = pipeline_neuralzip(neuralzip_codec, named_tensors)
            timings["neuralzip_extraction_and_encode"].append(elapsed)
        else:
            elapsed, _ = pipeline_neuralzip(neuralzip_codec, named_tensors)
            timings["neuralzip_extraction_and_encode"].append(elapsed)
            elapsed, _ = pipeline_neuralzip(shared_codec, named_tensors)
            timings["shared_only_extraction_and_encode"].append(elapsed)
            elapsed, _ = pipeline_zipnn(zipnn_codec, named_tensors)
            timings["zipnn_extraction_and_encode"].append(elapsed)
            elapsed, _ = encode_neuralzip(neuralzip_codec, streams)
            timings["neuralzip_encode_only"].append(elapsed)
            elapsed, _ = encode_neuralzip(shared_codec, streams)
            timings["shared_only_encode_only"].append(elapsed)
            elapsed, _ = encode_zipnn(zipnn_codec, streams)
            timings["zipnn_encode_only"].append(elapsed)
        print(f"repeat {repeat + 1}/{REPEATS}", flush=True)

    summaries = {name: timing_summary(values) for name, values in timings.items()}
    output = {
        "model_id": MODEL_ID,
        "dataset": "ImageNet-1K",
        "recipe": "original torchvision weights",
        "dtype": "bfloat16",
        "threads": torch.get_num_threads(),
        "neuralzip_setup": {
            "clustering_seconds": setup_result["precompute_seconds"],
            "codebook_setup_seconds": setup_result["codebook_setup_seconds"],
            "total_seconds": (
                setup_result["precompute_seconds"]
                + setup_result["codebook_setup_seconds"]
            ),
            "clusters": setup_result["clusters"],
            "codebooks": len(setup_result["codebooks"]),
        },
        "shared_only_setup": {
            "clustering_seconds": shared_setup_result["precompute_seconds"],
            "codebook_setup_seconds": shared_setup_result["codebook_setup_seconds"],
            "total_seconds": (
                shared_setup_result["precompute_seconds"]
                + shared_setup_result["codebook_setup_seconds"]
            ),
            "clusters": shared_setup_result["clusters"],
            "codebooks": len(shared_setup_result["codebooks"]),
        },
        "timings": summaries,
        "comparison": {
            "encode_only_speedup": (
                summaries["zipnn_encode_only"]["median_seconds"]
                / summaries["neuralzip_encode_only"]["median_seconds"]
            ),
            "shared_only_encode_speedup": (
                summaries["zipnn_encode_only"]["median_seconds"]
                / summaries["shared_only_encode_only"]["median_seconds"]
            ),
            "extraction_and_encode_speedup": (
                summaries["zipnn_extraction_and_encode"]["median_seconds"]
                / summaries["neuralzip_extraction_and_encode"]["median_seconds"]
            ),
            "shared_only_extraction_and_encode_speedup": (
                summaries["zipnn_extraction_and_encode"]["median_seconds"]
                / summaries["shared_only_extraction_and_encode"]["median_seconds"]
            ),
        },
        "zipnn": group_stats(zipnn_groups, raw_bytes),
        "shared_only": group_stats(shared_groups, raw_bytes),
        "neuralzip": group_stats(neuralzip_groups, raw_bytes),
        "roundtrip_ok": True,
    }
    destination = Path(__file__).with_name(
        "resnet18_exponent_time_20260713.json"
    )
    destination.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
