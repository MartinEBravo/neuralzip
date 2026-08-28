"""Validate persistent NeuralZip inference in a GPU smoke experiment."""

from __future__ import annotations

import argparse
import gc
import json
import shutil
from pathlib import Path

import torch
from accelerate import cpu_offload, dispatch_model
from transformers import LlamaConfig, LlamaForCausalLM

from neuralzip import Compressor, NeuralZipModel


def _config() -> LlamaConfig:
    return LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        bos_token_id=1,
        eos_token_id=127,
        pad_token_id=0,
        tie_word_embeddings=False,
    )


def _ordinary(config, state, *, device=None, device_map=None, main_device=None):
    model = LlamaForCausalLM(config).to(dtype=torch.bfloat16).eval()
    model.load_state_dict(state)
    if device_map is None:
        return model.to(device)
    if all(value == "cpu" for value in device_map.values()):
        return cpu_offload(
            model,
            execution_device=main_device,
            state_dict=model.state_dict(),
        )
    return dispatch_model(
        model,
        device_map=device_map,
        main_device=main_device,
        force_hooks=True,
    )


def _logits(model, input_ids):
    with torch.inference_mode():
        return model(input_ids, use_cache=False).logits.detach().cpu()


def _release(*models):
    for model in models:
        del model
    gc.collect()
    torch.cuda.empty_cache()


def _comparison(expected, actual):
    return {
        "allclose": torch.allclose(expected, actual, atol=3e-3, rtol=0.0),
        "bitwise_exact": torch.equal(
            expected.view(torch.uint8), actual.view(torch.uint8)
        ),
        "different_words": int(
            torch.count_nonzero(
                expected.view(torch.uint16) != actual.view(torch.uint16)
            ).item()
        ),
        "max_abs_error": float((expected.float() - actual.float()).abs().max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument(
        "--modes",
        default="single,auto,multi,offload",
        help="Comma-separated subset of single,auto,multi,offload.",
    )
    args = parser.parse_args()
    modes = tuple(value.strip() for value in args.modes.split(",") if value.strip())
    unknown = set(modes) - {"single", "auto", "multi", "offload"}
    if unknown:
        parser.error(f"unknown modes: {sorted(unknown)}")
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    if {"auto", "multi"}.intersection(modes) and torch.cuda.device_count() < 2:
        parser.error("auto/multi modes require two visible CUDA devices")

    torch.manual_seed(2026)
    config = _config()
    source = LlamaForCausalLM(config).to(dtype=torch.bfloat16).eval()
    state = {
        name: value.detach().cpu().clone()
        for name, value in source.state_dict().items()
    }
    compressor = Compressor(
        "neuralzip",
        threads=2,
        ngram_values=(),
        packed_source_budget_ratio=0.0,
        packed_time_budget_ratio=0.0,
        max_codebook_sample_bytes=1 << 20,
    )
    compressor.fit(source)
    shutil.rmtree(args.directory, ignore_errors=True)
    NeuralZipModel.save_pretrained(
        source,
        compressor,
        args.directory,
        max_shard_size="8KB",
    )
    del source

    input_ids = torch.tensor([[1, 7, 11, 19]], dtype=torch.long)
    results = {}
    if "single" in modes:
        baseline = _ordinary(config, state, device="cuda:0")
        compressed = NeuralZipModel.from_pretrained(args.directory, device="cuda:0")
        expected = _logits(baseline, input_ids.to("cuda:0"))
        actual = _logits(compressed, input_ids.to("cuda:0"))
        results["single"] = {
            **_comparison(expected, actual),
            "runtime_devices": [
                str(value) for value in compressed.neuralzip_runtime.devices
            ],
        }
        _release(baseline, compressed)

    if "auto" in modes:
        baseline = _ordinary(config, state, device="cuda:0")
        compressed = NeuralZipModel.from_pretrained(
            args.directory,
            device_map="auto",
            max_memory={0: "240KiB", 1: "240KiB", "cpu": "1GiB"},
        )
        expected = _logits(baseline, input_ids.to("cuda:0"))
        actual = _logits(compressed, input_ids.to("cuda:0"))
        results["auto"] = {
            **_comparison(expected, actual),
            "device_map": {
                name: str(value)
                for name, value in getattr(compressed, "hf_device_map", {}).items()
            },
            "runtime_devices": [
                str(value) for value in compressed.neuralzip_runtime.devices
            ],
        }
        _release(baseline, compressed)

    if "multi" in modes:
        device_map = {
            "model.embed_tokens": 0,
            "model.layers.0": 0,
            "model.layers.1": 1,
            "model.norm": 1,
            "model.rotary_emb": 0,
            "lm_head": 1,
        }
        baseline = _ordinary(
            config,
            state,
            device_map=device_map,
            main_device=torch.device("cuda:0"),
        )
        compressed = NeuralZipModel.from_pretrained(
            args.directory,
            device_map=device_map,
        )
        materialized = compressed.neuralzip_runtime.materialize()
        weight_mismatches = [
            name
            for name, value in materialized.items()
            if not torch.equal(
                value.detach().cpu().view(torch.uint8),
                state[name].view(torch.uint8),
            )
        ]
        resident_mismatches = [
            name
            for name, value in compressed.named_parameters()
            if name in state
            and not torch.equal(
                value.detach().cpu().view(torch.uint8),
                state[name].view(torch.uint8),
            )
        ]
        expected = _logits(baseline, input_ids.to("cuda:0"))
        actual = _logits(compressed, input_ids.to("cuda:0"))
        results["multi"] = {
            **_comparison(expected, actual),
            "runtime_devices": [
                str(value) for value in compressed.neuralzip_runtime.devices
            ],
            "resident_mismatches": resident_mismatches,
            "weight_mismatches": weight_mismatches,
        }
        _release(baseline, compressed)

    if "offload" in modes:
        device_map = {"": "cpu"}
        execution = torch.device("cuda:0")
        baseline = _ordinary(
            config,
            state,
            device_map=device_map,
            main_device=execution,
        )
        compressed = NeuralZipModel.from_pretrained(
            args.directory,
            device_map=device_map,
            execution_device=execution,
            cpu_offload=True,
        )
        expected = _logits(baseline, input_ids.to(execution))
        actual = _logits(compressed, input_ids.to(execution))
        results["offload"] = {
            **_comparison(expected, actual),
            "offloaded_blocks": sorted(compressed.neuralzip_runtime._offloaded_blocks),
            "runtime_devices": [
                str(value) for value in compressed.neuralzip_runtime.devices
            ],
        }
        _release(baseline, compressed)

    print(json.dumps(results, indent=2, sort_keys=True))
    if not all(
        result["allclose"] and not result.get("weight_mismatches")
        for result in results.values()
    ):
        raise SystemExit("A persistent GPU inference mode failed validation.")


if __name__ == "__main__":
    main()
