import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import AutoModel

from neuralzip import Compressor
from neuralzip.archive import serialized_group_size, serialized_size
from neuralzip.zipnn_codec import _dtype_name, _zipnn_dtype_config


MODEL_IDS = (
    "google-bert/bert-base-uncased",
    "google-bert/bert-large-uncased",
)


def architecture_role(name):
    parts = str(name).split(".")
    for index in range(len(parts) - 2):
        if (
            parts[index] == "encoder"
            and parts[index + 1] == "layer"
            and parts[index + 2].isdigit()
        ):
            parts[index + 2] = "#"
            break
    return ".".join(parts)


def load_model(model_id):
    return AutoModel.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
    ).eval()


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
    for name, tensor in named_tensors:
        raw_bytes = int(tensor.numel() * tensor.element_size())
        total_bytes += raw_bytes
        cluster_id = exact.get(name) or by_role.get(architecture_role(name))
        if cluster_id is None:
            cluster_id = f"transfer-raw:{architecture_role(name)}"
        else:
            mapped_bytes += raw_bytes
        codec.cluster_map[name] = cluster_id

        config = _zipnn_dtype_config(tensor.dtype)
        for stream_id in range(int(config["num_buffers"])):
            key = (cluster_id, _dtype_name(tensor.dtype), stream_id)
            if key not in codec._mode_by_key:
                codec._mode_by_key[key] = {"kind": "raw", "fixed_cost": 0}
    return 100.0 * mapped_bytes / total_bytes


def payload_metrics(payloads):
    exponent_raw = 0
    exponent_compressed = 0
    packed_chunks = 0
    hits = 0
    escapes = 0
    for payload in payloads:
        exponent = payload.groups[-1]
        exponent_raw += int(exponent.symbol_count)
        exponent_compressed += serialized_group_size(exponent)
        packed_chunks += payload.packed_ngram_chunks
        hits += payload.ngram_hits
        escapes += payload.ngram_escapes
    return {
        "exponent_pct": 100.0 * exponent_compressed / exponent_raw,
        "packed_chunks": packed_chunks,
        "ngram_hits": hits,
        "ngram_escapes": escapes,
        "ngram_hit_rate": hits / (hits + escapes) if hits + escapes else 0.0,
    }


def transfer_compress(codec, named_tensors, *, verify=False):
    codebook_count = len(codec.export_codebooks())
    started = time.perf_counter()
    payloads = [
        codec.compress_tensor(name, tensor)
        for name, tensor in named_tensors
    ]
    compression_seconds = time.perf_counter() - started
    if len(codec.export_codebooks()) != codebook_count:
        raise RuntimeError("The target model created a new codebook.")

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
        "setup_seconds": 0.0,
        "compression_seconds": compression_seconds,
        "one_shot_seconds": compression_seconds,
        "compression_pct": 100.0 * compressed_bytes / raw_bytes,
        "roundtrip_ok": roundtrip_ok,
        **payload_metrics(payloads),
    }


def baseline_compress(model, method):
    result = Compressor(
        method,
        verify=False,
        compression_threshold=0.95,
    ).compress(model)
    exponent = result["stream_results"][-1]
    return {
        "setup_seconds": (
            result["precompute_seconds"] + result["codebook_setup_seconds"]
        ),
        "compression_seconds": result["encode_seconds"],
        "one_shot_seconds": result["total_compression_seconds"],
        "compression_pct": 100.0 * result["compression_ratio"],
        "exponent_pct": 100.0 * exponent["compression_ratio"],
        "packed_chunks": result["packed_ngram_chunks"],
        "ngram_hit_rate": result["ngram_hit_rate"],
    }


def median_result(rows):
    first = rows[0]
    output = {
        field: statistics.median(row[field] for row in rows)
        for field in ("setup_seconds", "compression_seconds", "one_shot_seconds")
    }
    for field in ("compression_pct", "exponent_pct", "packed_chunks", "ngram_hit_rate"):
        output[field] = first[field]
    return output


def run_direction(source_id, source, target_id, target):
    source_compressor = Compressor(
        "neuralzip",
        verify=False,
        compression_threshold=0.95,
    )
    source_result = source_compressor.compress(source)
    source_codec = source_compressor.codec
    exact, by_role = source_cluster_index(
        source_codec,
        list(source.named_parameters()),
    )

    target_named = list(target.named_parameters())
    coverage = install_transfer_map(source_codec, target_named, exact, by_role)
    verification = transfer_compress(
        source_codec,
        target_named,
        verify=True,
    )["roundtrip_ok"]

    rows = {"zipnn": [], "target_specific": [], "transferred": []}
    baseline_compress(target, "zipnn")
    baseline_compress(target, "neuralzip")
    transfer_compress(source_codec, target_named)
    for repeat in range(1, 6):
        rows["zipnn"].append(baseline_compress(target, "zipnn"))
        rows["target_specific"].append(baseline_compress(target, "neuralzip"))
        rows["transferred"].append(transfer_compress(source_codec, target_named))
        print(f"{source_id} -> {target_id}: {repeat}/5", flush=True)

    summary = {name: median_result(values) for name, values in rows.items()}
    zipnn = summary["zipnn"]
    target_specific = summary["target_specific"]
    transferred = summary["transferred"]
    return {
        "source": {
            "model_id": source_id,
            "clustering_seconds": source_result["precompute_seconds"],
            "codebook_setup_seconds": source_result["codebook_setup_seconds"],
            "offline_setup_seconds": (
                source_result["precompute_seconds"]
                + source_result["codebook_setup_seconds"]
            ),
            "clusters": source_result["clusters"],
            "codebooks": len(source_result["codebooks"]),
            "selected_modes": source_result["selected_modes"],
        },
        "target": {
            "model_id": target_id,
            "architecture_coverage_pct": coverage,
            "roundtrip_ok": verification,
        },
        "results": summary,
        "comparison": {
            "transfer_speedup_vs_zipnn": (
                zipnn["compression_seconds"] / transferred["compression_seconds"]
            ),
            "transfer_speedup_vs_target_specific": (
                target_specific["compression_seconds"]
                / transferred["compression_seconds"]
            ),
            "transfer_ratio_penalty_vs_zipnn_pp": (
                transferred["compression_pct"] - zipnn["compression_pct"]
            ),
            "transfer_exponent_penalty_vs_zipnn_pp": (
                transferred["exponent_pct"] - zipnn["exponent_pct"]
            ),
        },
    }


def main():
    torch.set_num_threads(32)
    models = {model_id: load_model(model_id) for model_id in MODEL_IDS}
    output = {
        "dataset": ["BookCorpus", "English Wikipedia"],
        "dtype": "bfloat16",
        "directions": {},
    }
    for source_id, target_id in (MODEL_IDS, tuple(reversed(MODEL_IDS))):
        output["directions"][f"{source_id}->{target_id}"] = run_direction(
            source_id,
            models[source_id],
            target_id,
            models[target_id],
        )

    destination = Path(__file__).with_name(
        "bert_packed_bidirectional_20260712.json"
    )
    destination.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
