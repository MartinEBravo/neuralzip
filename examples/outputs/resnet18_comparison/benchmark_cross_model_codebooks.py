import gc
import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import timm
import torch

from neuralzip import Compressor
from neuralzip.archive import serialized_group_size, serialized_size
from neuralzip.zipnn_codec import _dtype_name, _zipnn_dtype_config


MODEL_IDS = (
    "timm/resnet18.a1_in1k",
    "timm/resnet34.a1_in1k",
    "timm/resnet50.a1_in1k",
)


def architecture_role(name):
    parts = str(name).split(".")
    if len(parts) > 2 and parts[0].startswith("layer") and parts[1].isdigit():
        parts[1] = "#"
    return ".".join(parts)


def load_model(model_id):
    model_name = model_id.split("/", 1)[1]
    return timm.create_model(
        f"hf_hub:{model_id}",
        pretrained=True,
    ).eval().to(torch.bfloat16)


def source_cluster_index(codec, named_tensors):
    exact = {}
    candidates = defaultdict(Counter)
    for name, tensor in named_tensors:
        cluster_id = codec.cluster_id(name)
        exact[name] = cluster_id
        candidates[architecture_role(name)][cluster_id] += int(tensor.numel())
    by_role = {
        role: counts.most_common(1)[0][0]
        for role, counts in candidates.items()
    }
    return exact, by_role


def install_transfer_map(codec, named_tensors, exact, by_role):
    mapped_bytes = 0
    total_bytes = 0
    mapped_tensors = 0
    for name, tensor in named_tensors:
        raw_bytes = int(tensor.numel() * tensor.element_size())
        total_bytes += raw_bytes
        role = architecture_role(name)
        cluster_id = exact.get(name)
        if cluster_id is None:
            cluster_id = by_role.get(role)
        if cluster_id is not None:
            mapped_bytes += raw_bytes
            mapped_tensors += 1
        else:
            cluster_id = f"transfer-raw:{role}"
        codec.cluster_map[name] = cluster_id

        config = _zipnn_dtype_config(tensor.dtype)
        for stream_id in range(int(config["num_buffers"])):
            key = (cluster_id, _dtype_name(tensor.dtype), stream_id)
            if key not in codec._mode_by_key:
                codec._mode_by_key[key] = {"kind": "raw", "fixed_cost": 0}
    return {
        "mapped_tensors": mapped_tensors,
        "total_tensors": len(named_tensors),
        "mapped_bytes_pct": 100.0 * mapped_bytes / total_bytes,
    }


def exponent_ratio(payloads):
    raw_bytes = 0
    compressed_bytes = 0
    for payload in payloads:
        group = payload.groups[-1]
        raw_bytes += int(group.symbol_count)
        compressed_bytes += serialized_group_size(group)
    return 100.0 * compressed_bytes / raw_bytes


def transfer_compress(codec, named_tensors, *, verify=False):
    codebooks_before = len(codec.export_codebooks())
    started = time.perf_counter()
    payloads = [
        codec.compress_tensor(name, tensor)
        for name, tensor in named_tensors
    ]
    compression_seconds = time.perf_counter() - started
    if len(codec.export_codebooks()) != codebooks_before:
        raise RuntimeError("The target model created new codebooks.")

    roundtrip_ok = None
    if verify:
        roundtrip_ok = all(
            torch.equal(
                codec.decompress_tensor(payload).view(torch.uint8),
                tensor.contiguous().view(torch.uint8),
            )
            for (_, tensor), payload in zip(named_tensors, payloads, strict=True)
        )

    raw_bytes = sum(
        int(tensor.numel() * tensor.element_size())
        for _, tensor in named_tensors
    )
    compressed_bytes = serialized_size(
        "neuralzip",
        codec.export_codebooks(),
        payloads,
    )
    return {
        "compression_seconds": compression_seconds,
        "raw_bytes": raw_bytes,
        "compressed_bytes": compressed_bytes,
        "compression_pct": 100.0 * compressed_bytes / raw_bytes,
        "exponent_pct": exponent_ratio(payloads),
        "roundtrip_ok": roundtrip_ok,
    }


def baseline_compress(model, method):
    options = {"verify": False, "compression_threshold": 0.95}
    if method == "neuralzip":
        options.update(ngram_values=(), packed_time_budget_ratio=0.0)
    result = Compressor(method, **options).compress(model)
    exponent = result["stream_results"][-1]
    return {
        "compression_seconds": result["compression_seconds"],
        "compression_pct": 100.0 * result["compression_ratio"],
        "exponent_pct": 100.0 * exponent["compression_ratio"],
    }


def median_result(rows):
    first = rows[0]
    return {
        "compression_seconds": statistics.median(
            row["compression_seconds"] for row in rows
        ),
        "compression_pct": first["compression_pct"],
        "exponent_pct": first["exponent_pct"],
    }


def benchmark_target(model_id, model, source_codec, exact, by_role, repeats=5):
    named_tensors = list(model.named_parameters())
    coverage = install_transfer_map(source_codec, named_tensors, exact, by_role)
    verification = transfer_compress(
        source_codec,
        named_tensors,
        verify=True,
    )["roundtrip_ok"]

    methods = {"zipnn": [], "neuralzip_target": [], "neuralzip_transfer": []}
    baseline_compress(model, "zipnn")
    baseline_compress(model, "neuralzip")
    transfer_compress(source_codec, named_tensors)
    for repeat in range(1, repeats + 1):
        methods["zipnn"].append(baseline_compress(model, "zipnn"))
        methods["neuralzip_target"].append(
            baseline_compress(model, "neuralzip")
        )
        methods["neuralzip_transfer"].append(
            transfer_compress(source_codec, named_tensors)
        )
        print(model_id, repeat, flush=True)

    summary = {name: median_result(rows) for name, rows in methods.items()}
    summary["transfer_speedup_vs_zipnn"] = (
        summary["zipnn"]["compression_seconds"]
        / summary["neuralzip_transfer"]["compression_seconds"]
    )
    summary["transfer_speedup_vs_target_neuralzip"] = (
        summary["neuralzip_target"]["compression_seconds"]
        / summary["neuralzip_transfer"]["compression_seconds"]
    )
    summary["transfer_ratio_penalty_vs_zipnn_pp"] = (
        summary["neuralzip_transfer"]["compression_pct"]
        - summary["zipnn"]["compression_pct"]
    )
    summary["transfer_exponent_penalty_vs_zipnn_pp"] = (
        summary["neuralzip_transfer"]["exponent_pct"]
        - summary["zipnn"]["exponent_pct"]
    )
    return {
        "coverage": coverage,
        "roundtrip_ok": verification,
        "summary": summary,
    }


def main():
    torch.set_num_threads(32)
    source = load_model(MODEL_IDS[0])
    source_compressor = Compressor(
        "neuralzip",
        verify=False,
        ngram_values=(),
        packed_time_budget_ratio=0.0,
    )
    source_result = source_compressor.compress(source)
    source_codec = source_compressor.codec
    source_named_tensors = list(source.named_parameters())
    exact, by_role = source_cluster_index(source_codec, source_named_tensors)

    output = {
        "dataset": "ImageNet-1K",
        "recipe": "timm ResNet Strikes Back A1",
        "source": {
            "model_id": MODEL_IDS[0],
            "clustering_seconds": source_result["precompute_seconds"],
            "compression_seconds": source_result["compression_seconds"],
            "clusters": source_result["clusters"],
            "codebooks": len(source_result["codebooks"]),
        },
        "targets": {},
    }
    for model_id in MODEL_IDS:
        model = source if model_id == MODEL_IDS[0] else load_model(model_id)
        output["targets"][model_id] = benchmark_target(
            model_id,
            model,
            source_codec,
            exact,
            by_role,
        )
        if model is not source:
            del model
        gc.collect()

    destination = Path(__file__).with_name("cross_model_codebooks_20260712.json")
    destination.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
