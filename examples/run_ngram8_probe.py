from __future__ import annotations

import argparse
import json
from pathlib import Path

import timm
import torch

from neuralzip import Compressor


MODEL_ID = "timm/resnet18.tv_in1k"


def load_model():
    return timm.create_model(
        f"hf_hub:{MODEL_ID}",
        pretrained=True,
    ).eval().to(torch.bfloat16)


def packed_stats(result):
    chunks_by_n = {}
    hits_by_n = {}
    escapes_by_n = {}
    for tensor_result in result["tensor_results"]:
        payload = tensor_result["payload"]
        for group in payload.groups:
            for chunk in group.chunks:
                if chunk.mode != "packed_huff0":
                    continue
                n = int(chunk.ngram_n)
                chunks_by_n[n] = chunks_by_n.get(n, 0) + 1
                hits_by_n[n] = hits_by_n.get(n, 0) + int(chunk.ngram_hits)
                escapes_by_n[n] = escapes_by_n.get(n, 0) + int(
                    chunk.ngram_escapes
                )
    return {
        "chunks_by_n": chunks_by_n,
        "hits_by_n": hits_by_n,
        "escapes_by_n": escapes_by_n,
        "hit_rate_by_n": {
            n: hits_by_n[n] / (hits_by_n[n] + escapes_by_n[n])
            for n in chunks_by_n
            if hits_by_n[n] + escapes_by_n[n]
        },
    }


def run(model, ngram_values):
    result = Compressor(
        "neuralzip",
        verify=True,
        keep_payloads=True,
        compression_threshold=0.95,
        ngram_values=ngram_values,
        packed_source_budget_ratio=0.30,
    ).compress(model)
    exponent_stream = result["stream_results"][-1]
    return {
        "ngram_values": list(ngram_values),
        "raw_bytes": int(result["raw_bytes"]),
        "compressed_bytes": int(result["compressed_bytes"]),
        "compression_ratio": float(result["compression_ratio"]),
        "compression_pct": 100.0 * float(result["compression_ratio"]),
        "exponent_compression_pct": 100.0
        * float(exponent_stream["compression_ratio"]),
        "precompute_seconds": float(result["precompute_seconds"]),
        "codebook_setup_seconds": float(result["codebook_setup_seconds"]),
        "encode_seconds": float(result["encode_seconds"]),
        "total_compression_seconds": float(result["total_compression_seconds"]),
        "selected_modes": result["selected_modes"],
        "selection_stats": result["selection_stats"],
        "packed": packed_stats(result),
        "roundtrip_ok": bool(result["roundtrip_ok"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    torch.set_num_threads(32)
    model = load_model()
    baseline = run(model, (2, 3, 4))
    experimental = run(model, (2, 3, 4, 8))
    output = {
        "model_id": MODEL_ID,
        "dtype": "bfloat16",
        "repetitions": 1,
        "baseline": baseline,
        "ngram8_candidate": experimental,
        "delta": {
            "compressed_bytes": experimental["compressed_bytes"]
            - baseline["compressed_bytes"],
            "compression_pp": experimental["compression_pct"]
            - baseline["compression_pct"],
            "exponent_compression_pp": experimental["exponent_compression_pct"]
            - baseline["exponent_compression_pct"],
            "codebook_setup_seconds": experimental["codebook_setup_seconds"]
            - baseline["codebook_setup_seconds"],
            "encode_seconds": experimental["encode_seconds"]
            - baseline["encode_seconds"],
        },
    }
    encoded = json.dumps(output, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    print(encoded, flush=True)


if __name__ == "__main__":
    main()
