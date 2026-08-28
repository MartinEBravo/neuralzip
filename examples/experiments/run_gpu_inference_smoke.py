"""Validate post-setup NeuralZip inference in a Hugging Face smoke experiment."""

import argparse
import json
import time

import torch
from transformers import AutoModelForCausalLM

from neuralzip import NeuralZipPlan


def _timed_forward(model, inputs, *, warmup, repetitions):
    with torch.inference_mode():
        output = None
        for _ in range(warmup):
            output = model(**inputs, use_cache=False).logits
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(repetitions):
            output = model(**inputs, use_cache=False).logits
        torch.cuda.synchronize()
    return output.detach().clone(), (time.perf_counter() - started) / repetitions


def _timed_decoder(runtime, *, warmup, repetitions):
    with torch.inference_mode():
        for _ in range(warmup):
            for block_name in runtime.export.block_order:
                runtime._decode_block(block_name)
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(repetitions):
            for block_name in runtime.export.block_order:
                runtime._decode_block(block_name)
        torch.cuda.synchronize()
    return (time.perf_counter() - started) / repetitions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()

    if args.batch_size <= 0 or args.sequence_length <= 0:
        parser.error("--batch-size and --sequence-length must be positive")
    if args.warmup < 0 or args.repetitions <= 0:
        parser.error("--warmup must be non-negative and --repetitions positive")

    device = torch.device(args.device)
    if device.type != "cuda":
        parser.error("the smoke test requires a CUDA device")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
    ).eval()
    vocab_size = int(model.config.vocab_size)
    input_ids = (
        torch.arange(1, args.sequence_length + 1, dtype=torch.long)
        .remainder(vocab_size)
        .unsqueeze(0)
        .repeat(args.batch_size, 1)
    )
    attention_mask = torch.ones_like(input_ids)

    setup_started = time.perf_counter()
    plan = NeuralZipPlan.fit(
        dict(model.named_parameters()),
        threads=args.threads,
        ngram_values=(),
        packed_source_budget_ratio=0.0,
        packed_time_budget_ratio=0.0,
    )
    setup_seconds = time.perf_counter() - setup_started

    model = model.to(device)
    inputs = {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
    }
    baseline_logits, baseline_seconds = _timed_forward(
        model,
        inputs,
        warmup=args.warmup,
        repetitions=args.repetitions,
    )

    attach_started = time.perf_counter()
    model = plan.prepare_inference(model)
    attach_seconds = time.perf_counter() - attach_started
    compressed_logits, compressed_seconds = _timed_forward(
        model,
        inputs,
        warmup=args.warmup,
        repetitions=args.repetitions,
    )

    exact = torch.equal(
        baseline_logits.view(torch.uint8),
        compressed_logits.view(torch.uint8),
    )
    runtime = model.neuralzip_runtime
    decode_seconds = _timed_decoder(
        runtime,
        warmup=args.warmup,
        repetitions=args.repetitions,
    )
    groups_per_block = [
        sum(group.block_name == block for group in runtime.export.groups)
        for block in runtime.export.block_order
    ]
    result = {
        "model": args.model,
        "batch_size": args.batch_size,
        "device": torch.cuda.get_device_name(device),
        "bitwise_exact_logits": exact,
        "setup_seconds": setup_seconds,
        "attach_seconds": attach_seconds,
        "baseline_forward_ms": baseline_seconds * 1000.0,
        "neuralzip_forward_ms": compressed_seconds * 1000.0,
        "latency_ratio": compressed_seconds / baseline_seconds,
        "decode_all_blocks_ms": decode_seconds * 1000.0,
        "compressed_weights": runtime.export.compressed_weight_count,
        "compressed_groups": len(runtime.export.groups),
        "compressed_blocks": len(runtime.export.block_order),
        "max_groups_per_block": max(groups_per_block, default=0),
        "resident_candidate_weights": runtime.export.resident_weight_count,
        "compressed_bytes": runtime.export.compressed_bytes,
        "gpu_launch_descriptor_bytes": runtime.export.launch_descriptor_bytes,
        "runtime_candidate_bytes": runtime.export.runtime_candidate_bytes,
        "raw_compressed_weight_bytes": runtime.export.raw_bytes,
        "payload_ratio": (
            runtime.export.compressed_bytes / runtime.export.raw_bytes
            if runtime.export.raw_bytes
            else 0.0
        ),
        "workspace_bytes": runtime.export.workspace_bytes,
        "selected_modes": plan.codec.selected_modes,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if not exact:
        raise SystemExit("NeuralZip GPU logits differ from the BF16 baseline")


if __name__ == "__main__":
    main()
