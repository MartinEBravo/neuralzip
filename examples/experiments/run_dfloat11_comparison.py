"""Measure lossless GPU compression and inference across BF16 LLM families.

``suite`` launches BF16, NeuralZip, and DFloat11 in separate processes so
CUDA allocator state cannot leak between methods.  Every method is required
to cover the same seven linear weights per decoder block and to reproduce the
BF16 logits bit for bit.  The older single-method commands remain available
for debugging and for reproducing earlier measurements.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM

SEVEN_LINEAR_MODULES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


@dataclass(frozen=True)
class BlockLayout:
    """A decoder-block path and the linears compressed inside each block."""

    name: str
    block_pattern: str
    relative_modules: tuple[str, ...] = SEVEN_LINEAR_MODULES

    def __post_init__(self) -> None:
        if not self.name or not self.block_pattern:
            raise ValueError("Block-layout names and patterns must be non-empty.")
        if "index" not in re.compile(self.block_pattern).groupindex:
            raise ValueError("A block pattern must provide a named 'index' group.")
        if len(self.relative_modules) != 7:
            raise ValueError("The GPU comparison requires exactly seven linears.")
        if len(set(self.relative_modules)) != len(self.relative_modules):
            raise ValueError("Relative linear names must be unique.")

    @property
    def dfloat_pattern(self) -> dict[str, list[str]]:
        return {self.block_pattern: list(self.relative_modules)}


STANDARD_TEXT_LAYOUT = BlockLayout(
    name="decoder-model-layers",
    block_pattern=r"model\.layers\.(?P<index>\d+)",
)
NESTED_TEXT_LAYOUT = BlockLayout(
    name="nested-language-model-layers",
    block_pattern=r"model\.language_model\.layers\.(?P<index>\d+)",
)
SUPPORTED_BLOCK_LAYOUTS = (STANDARD_TEXT_LAYOUT, NESTED_TEXT_LAYOUT)

# Compatibility name used by older commands and notes.  New code obtains the
# selected pattern from the discovered TargetScope rather than assuming Qwen.
QWEN_BLOCK_PATTERN = STANDARD_TEXT_LAYOUT.dfloat_pattern


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    family: str
    model_id: str
    revision: str | None = None
    layouts: tuple[BlockLayout, ...] = (STANDARD_TEXT_LAYOUT,)


MODEL_CATALOG = {
    spec.key: spec
    for spec in (
        ModelSpec(
            "qwen25-05b",
            "Qwen2.5 0.5B",
            "Qwen2.5",
            "Qwen/Qwen2.5-0.5B",
            "060db6499f32faf8b98477b0a26969ef7d8b9987",
        ),
        ModelSpec(
            "qwen25-15b",
            "Qwen2.5 1.5B",
            "Qwen2.5",
            "Qwen/Qwen2.5-1.5B",
            "8faed761d45a263340a0528343f099c05c9a4323",
        ),
        ModelSpec(
            "qwen3-06b",
            "Qwen3 0.6B",
            "Qwen3",
            "Qwen/Qwen3-0.6B",
            "c1899de289a04d12100db370d81485cdf75e47ca",
        ),
        ModelSpec(
            "qwen3-17b",
            "Qwen3 1.7B",
            "Qwen3",
            "Qwen/Qwen3-1.7B",
            "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        ),
        ModelSpec(
            "qwen3-4b",
            "Qwen3 4B",
            "Qwen3",
            "Qwen/Qwen3-4B",
            "1cfa9a7208912126459214e8b04321603b3df60c",
        ),
        ModelSpec(
            "llama32-1b",
            "Llama 3.2 1B",
            "Llama 3.2",
            "meta-llama/Llama-3.2-1B",
            "4e20de362430cd3b72f300e6b0f18e50e7166e08",
        ),
        ModelSpec(
            "llama32-3b",
            "Llama 3.2 3B",
            "Llama 3.2",
            "meta-llama/Llama-3.2-3B",
            "13afe5124825b4f3751f836b40dafda64c1ed062",
        ),
        ModelSpec(
            "gemma3-1b",
            "Gemma 3 1B",
            "Gemma 3",
            "google/gemma-3-1b-it",
            "dcc83ea841ab6100d6b47a070329e1ba4cf78752",
        ),
        ModelSpec(
            "smollm2-17b",
            "SmolLM2 1.7B",
            "SmolLM2",
            "HuggingFaceTB/SmolLM2-1.7B",
            "effd688a12921b4cc83e3312b6feb579f70f9c71",
        ),
        ModelSpec(
            "tinyllama-11b",
            "TinyLlama 1.1B",
            "TinyLlama",
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "fe8a4ea1ffedaf415f4da2f062534de366a451e6",
        ),
        ModelSpec(
            "granite33-2b",
            "Granite 3.3 2B",
            "Granite 3.3",
            "ibm-granite/granite-3.3-2b-base",
            "c15dc681f4b2aeb4e576cdaae5434edfc17ed14d",
        ),
        ModelSpec(
            "falcon3-3b",
            "Falcon3 3B",
            "Falcon3",
            "tiiuae/Falcon3-3B-Base",
            "092c29e3114ff551bd06b93b85424cdb525d227c",
        ),
    )
}

DEFAULT_SUITE_MODEL_KEYS = (
    "qwen25-05b",
    "qwen25-15b",
    "qwen3-06b",
    "qwen3-17b",
    "qwen3-4b",
    "smollm2-17b",
    "tinyllama-11b",
    "granite33-2b",
    "falcon3-3b",
)


@dataclass(frozen=True)
class ResolvedModel:
    key: str
    label: str
    family: str
    model_id: str
    revision: str | None
    layouts: tuple[BlockLayout, ...]


def _resolve_model(model: str, revision: str | None = None) -> ResolvedModel:
    spec = MODEL_CATALOG.get(model)
    if spec is None:
        return ResolvedModel(
            key=model.replace("/", "--"),
            label=model.rsplit("/", 1)[-1],
            family="custom",
            model_id=model,
            revision=revision,
            layouts=SUPPORTED_BLOCK_LAYOUTS,
        )
    return ResolvedModel(
        key=spec.key,
        label=spec.label,
        family=spec.family,
        model_id=spec.model_id,
        revision=revision if revision is not None else spec.revision,
        layouts=spec.layouts,
    )


@dataclass(frozen=True)
class TargetWeight:
    block_name: str
    module_name: str
    tensor_name: str
    numel: int
    raw_bytes: int


@dataclass(frozen=True)
class TargetScope:
    layout: BlockLayout
    block_names: tuple[str, ...]
    weights: tuple[TargetWeight, ...]

    @property
    def weight_names(self) -> tuple[str, ...]:
        return tuple(weight.tensor_name for weight in self.weights)

    @property
    def raw_bytes(self) -> int:
        return sum(weight.raw_bytes for weight in self.weights)

    @property
    def raw_exponent_bytes(self) -> int:
        # BF16 has one eight-bit exponent symbol per element.
        return sum(weight.numel for weight in self.weights)

    @property
    def dfloat_pattern(self) -> dict[str, list[str]]:
        return self.layout.dfloat_pattern

    def weights_for_block(self, block_name: str) -> tuple[TargetWeight, ...]:
        return tuple(
            weight for weight in self.weights if weight.block_name == block_name
        )


@dataclass(frozen=True)
class DFloatScope:
    raw_bytes: int
    exponent_stream_bytes: int
    sign_mantissa_bytes: int
    exponent_metadata_bytes: int
    exponent_codebook_bytes: int
    split_positions_host_bytes: int
    blocks: int

    @property
    def raw_exponent_bytes(self) -> int:
        return self.raw_bytes // 2

    @property
    def exponent_compressed_bytes(self) -> int:
        return (
            self.exponent_stream_bytes
            + self.exponent_metadata_bytes
            + self.exponent_codebook_bytes
        )

    @property
    def exponent_ratio(self) -> float:
        return self.exponent_compressed_bytes / self.raw_exponent_bytes

    @property
    def gpu_payload_bytes(self) -> int:
        return self.exponent_stream_bytes + self.sign_mantissa_bytes

    @property
    def gpu_metadata_bytes(self) -> int:
        return self.exponent_metadata_bytes

    @property
    def gpu_codebook_bytes(self) -> int:
        return self.exponent_codebook_bytes

    @property
    def compressed_bytes(self) -> int:
        return (
            self.gpu_payload_bytes
            + self.exponent_metadata_bytes
            + self.exponent_codebook_bytes
        )


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _version_fields() -> dict[str, str | None]:
    return {
        "dfloat11_version": _package_version("dfloat11"),
        "torch_version": str(torch.__version__),
        "transformers_version": str(transformers.__version__),
    }


def _resolved_from_args(args) -> ResolvedModel:
    resolved = getattr(args, "resolved_model", None)
    if isinstance(resolved, ResolvedModel):
        return resolved
    return _resolve_model(args.model, getattr(args, "revision", None))


def _model_fields(resolved: ResolvedModel) -> dict[str, str | None]:
    return {
        "model_key": resolved.key,
        "model_label": resolved.label,
        "model_family": resolved.family,
        "model": resolved.model_id,
        "revision": resolved.revision,
    }


def _discover_layout(
    model,
    layouts: tuple[BlockLayout, ...] = SUPPORTED_BLOCK_LAYOUTS,
) -> tuple[BlockLayout, tuple[str, ...]]:
    """Find one unambiguous, contiguous decoder stack."""

    named_modules = tuple(model.named_modules())
    matches = []
    for layout in layouts:
        indexed_names = []
        for block_name, _block in named_modules:
            match = re.fullmatch(layout.block_pattern, block_name)
            if match is not None:
                indexed_names.append((int(match.group("index")), block_name))
        if not indexed_names:
            continue
        indexed_names.sort()
        indices = tuple(index for index, _name in indexed_names)
        if indices != tuple(range(len(indexed_names))):
            raise RuntimeError(
                f"Decoder blocks for {layout.name!r} are not contiguous from zero: "
                f"{indices}."
            )
        matches.append((layout, tuple(name for _index, name in indexed_names)))

    if not matches:
        supported = ", ".join(layout.block_pattern for layout in layouts)
        raise RuntimeError(
            f"The model has no supported decoder block stack. Tried: {supported}."
        )
    if len(matches) != 1:
        names = ", ".join(layout.name for layout, _blocks in matches)
        raise RuntimeError(f"Decoder block discovery is ambiguous: {names}.")
    return matches[0]


def _target_scope(
    model,
    layouts: tuple[BlockLayout, ...] = SUPPORTED_BLOCK_LAYOUTS,
) -> TargetScope:
    layout, blocks = _discover_layout(model, layouts)
    weights = []
    seen_names = set()
    for block_name in blocks:
        for relative_name in layout.relative_modules:
            module_name = f"{block_name}.{relative_name}"
            try:
                module = model.get_submodule(module_name)
            except AttributeError as error:
                raise RuntimeError(
                    f"Decoder block {block_name} is missing target {relative_name}."
                ) from error
            parameter = module._parameters.get("weight")
            if not isinstance(module, torch.nn.Linear) or not isinstance(
                parameter, torch.nn.Parameter
            ):
                raise TypeError(
                    f"Target {module_name} must be an nn.Linear with a weight."
                )
            if parameter.dtype != torch.bfloat16:
                raise RuntimeError(
                    f"Target {module_name}.weight must use torch.bfloat16."
                )
            tensor_name = f"{module_name}.weight"
            if tensor_name in seen_names:
                raise RuntimeError(f"Duplicate target weight {tensor_name}.")
            seen_names.add(tensor_name)
            weights.append(
                TargetWeight(
                    block_name=block_name,
                    module_name=module_name,
                    tensor_name=tensor_name,
                    numel=parameter.numel(),
                    raw_bytes=parameter.numel() * parameter.element_size(),
                )
            )
    expected_weights = len(blocks) * len(layout.relative_modules)
    if len(weights) != expected_weights:
        raise RuntimeError(
            f"Target scope found {len(weights)} weights; expected {expected_weights}."
        )
    return TargetScope(layout=layout, block_names=blocks, weights=tuple(weights))


def _target_fields(scope: TargetScope) -> dict[str, object]:
    return {
        "target_layout": scope.layout.name,
        "target_block_pattern": scope.layout.block_pattern,
        "target_block_count": len(scope.block_names),
        "target_block_names": scope.block_names,
        "target_weight_count": len(scope.weights),
        "target_weight_names": scope.weight_names,
        "target_raw_bytes": scope.raw_bytes,
        "target_raw_exponent_bytes": scope.raw_exponent_bytes,
    }


def _require_exact_coverage(method: str, actual_names, target: TargetScope) -> None:
    actual = set(actual_names)
    expected = set(target.weight_names)
    if actual == expected:
        return
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    raise RuntimeError(
        f"{method} target coverage mismatch: "
        f"missing={missing[:8]} ({len(missing)} total), "
        f"extra={extra[:8]} ({len(extra)} total)."
    )


def _load_original(
    model_id: str,
    revision: str | None,
    *,
    cache_dir: str | None = None,
):
    return AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        dtype=torch.bfloat16,
        cache_dir=cache_dir,
    ).eval()


def _load_resolved(args, resolved: ResolvedModel):
    return _load_original(
        resolved.model_id,
        resolved.revision,
        cache_dir=getattr(args, "cache_dir", None),
    )


def _inputs(model, *, batch_size: int, sequence_length: int, device):
    input_ids = (
        torch.arange(1, sequence_length + 1, dtype=torch.long)
        .remainder(int(model.config.vocab_size))
        .unsqueeze(0)
        .repeat(batch_size, 1)
    )
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": torch.ones_like(input_ids).to(device),
    }


def _logit_digest(logits: torch.Tensor) -> str:
    raw = logits.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _measure(model, inputs, *, warmup: int, repetitions: int, trials: int):
    with torch.inference_mode():
        logits = None
        torch.cuda.reset_peak_memory_stats()
        for _ in range(warmup):
            logits = model(**inputs, use_cache=False).logits
        torch.cuda.synchronize()

        samples = []
        for _ in range(trials):
            torch.cuda.synchronize()
            started = time.perf_counter()
            for _ in range(repetitions):
                logits = model(**inputs, use_cache=False).logits
            torch.cuda.synchronize()
            samples.append(1000.0 * (time.perf_counter() - started) / repetitions)

    assert logits is not None
    logit_sha256 = _logit_digest(logits)
    del logits
    gc.collect()
    torch.cuda.synchronize()
    steady_allocated_bytes = torch.cuda.memory_allocated()
    return {
        "forward_ms": statistics.median(samples),
        "inference_ms": statistics.median(samples),
        "forward_ms_trials": samples,
        "allocated_bytes": steady_allocated_bytes,
        "steady_gpu_allocated_bytes": steady_allocated_bytes,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "logit_sha256": logit_sha256,
    }


def _exactness_fields(args, logit_sha256: str) -> dict[str, object]:
    reference = getattr(args, "reference_logit_sha256", None)
    return {
        "reference_logit_sha256": reference,
        "bitwise_exact_logits": (
            None if reference is None else logit_sha256 == reference
        ),
    }


def _require_reference_match(result: dict[str, object]) -> None:
    if result.get("bitwise_exact_logits") is False:
        raise SystemExit(
            f"{result['method']} logits differ from the separate BF16 control"
        )


def _emit_result(args, result: dict[str, object]) -> None:
    destination = getattr(args, "result_file", None)
    if destination:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    if not getattr(args, "no_print_result", False):
        print(json.dumps(result, indent=2, sort_keys=True))


def _emit_verified_result(args, result: dict[str, object]) -> None:
    """Reject known-inexact output before it can be persisted as a result."""

    _require_reference_match(result)
    _emit_result(args, result)


def _prepare_dfloat(args) -> None:
    from dfloat11 import compress_model

    resolved = _resolved_from_args(args)
    destination = Path(args.dfloat_directory)
    if destination.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{destination} already exists; pass --overwrite to replace it"
            )
        shutil.rmtree(destination)

    started = time.perf_counter()
    model = _load_resolved(args, resolved)
    load_seconds = time.perf_counter() - started
    target = _target_scope(model, resolved.layouts)
    compress_started = time.perf_counter()
    compress_model(
        model,
        pattern_dict=target.dfloat_pattern,
        save_path=str(destination),
        save_single_file=False,
        check_correctness=False,
    )
    compression_seconds = time.perf_counter() - compress_started
    disk_bytes = sum(
        path.stat().st_size for path in destination.rglob("*") if path.is_file()
    )
    _emit_result(
        args,
        {
            "method": "dfloat11-prepare",
            **_model_fields(resolved),
            "load_seconds": load_seconds,
            "setup_seconds": compression_seconds,
            "encoding_seconds": compression_seconds,
            "encoding_time_scope": "official_compress_model_from_preloaded_bf16",
            "disk_bytes": disk_bytes,
            "directory": str(destination),
            **_target_fields(target),
            **_version_fields(),
        },
    )


def _run_bf16(args, device) -> None:
    resolved = _resolved_from_args(args)
    model = _load_resolved(args, resolved)
    target = _target_scope(model, resolved.layouts)
    model = model.to(device)
    inputs = _inputs(
        model,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        device=device,
    )
    measurement = _measure(
        model,
        inputs,
        warmup=args.warmup,
        repetitions=args.repetitions,
        trials=args.trials,
    )
    result = {
        "method": "bf16",
        **_model_fields(resolved),
        "device": torch.cuda.get_device_name(device),
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "use_cache": False,
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "trials": args.trials,
        "post_setup_encode_seconds": None,
        "raw_exponent_bytes": target.raw_exponent_bytes,
        "compressed_exponent_bytes": target.raw_exponent_bytes,
        "exponent_ratio": 1.0,
        "exponent_ratio_pct": 100.0,
        "steady_gpu_allocated_bytes": measurement["steady_gpu_allocated_bytes"],
        "inference_ms": measurement["inference_ms"],
        "logit_sha256": measurement["logit_sha256"],
        "measurement": measurement,
        **_target_fields(target),
        **_version_fields(),
    }
    _emit_result(args, result)


def _neuralzip_exponent_metrics(exported, target: TargetScope) -> dict[str, object]:
    if exported.resident_raw_bytes % 2:
        raise RuntimeError("A BF16 resident fallback has an odd raw byte count.")
    exponent_stream_bytes = sum(group.exponent_bytes for group in exported.groups)
    exponent_metadata_bytes = sum(group.metadata_bytes for group in exported.groups)
    exponent_codebook_bytes = exported.codebook_bytes
    raw_fallback_exponent_bytes = exported.resident_raw_bytes // 2
    compressed_exponent_bytes = (
        exponent_stream_bytes
        + exponent_metadata_bytes
        + exponent_codebook_bytes
        + raw_fallback_exponent_bytes
    )
    ratio = compressed_exponent_bytes / target.raw_exponent_bytes
    return {
        "raw_exponent_bytes": target.raw_exponent_bytes,
        "exponent_stream_bytes": exponent_stream_bytes,
        "exponent_metadata_bytes": exponent_metadata_bytes,
        "exponent_codebook_bytes": exponent_codebook_bytes,
        "raw_fallback_exponent_bytes": raw_fallback_exponent_bytes,
        "compressed_exponent_bytes": compressed_exponent_bytes,
        "exponent_ratio": ratio,
        "exponent_ratio_pct": 100.0 * ratio,
    }


def _run_neuralzip(args, device) -> None:
    from neuralzip import NeuralZipPlan
    from neuralzip.gpu_inference import NeuralZipRuntime, export_model_for_gpu

    resolved = _resolved_from_args(args)
    model = _load_resolved(args, resolved)
    target = _target_scope(model, resolved.layouts)
    target_parameters = {
        name: model.get_parameter(name) for name in target.weight_names
    }
    setup_started = time.perf_counter()
    plan = NeuralZipPlan.fit(
        target_parameters,
        threads=args.threads,
        exponent_prior_count=args.exponent_prior_count,
        max_codebook_sample_bytes=args.max_codebook_sample_bytes,
        ngram_values=(),
        packed_source_budget_ratio=0.0,
        packed_time_budget_ratio=0.0,
    )
    setup_seconds = time.perf_counter() - setup_started

    encode_started = time.perf_counter()
    exported = export_model_for_gpu(model, plan)
    post_setup_encode_seconds = time.perf_counter() - encode_started
    covered_names = (
        tuple(binding.tensor_name for binding in exported.bindings)
        + exported.resident_weight_names
    )
    _require_exact_coverage("NeuralZip", covered_names, target)
    if exported.original_candidate_bytes != target.raw_bytes:
        raise RuntimeError(
            "NeuralZip target bytes differ from the fixed seven-linear scope: "
            f"{exported.original_candidate_bytes} != {target.raw_bytes}."
        )
    exponent_metrics = _neuralzip_exponent_metrics(exported, target)
    payload_bytes = exported.compressed_bytes + exported.resident_raw_bytes
    payload_ratio = payload_bytes / target.raw_bytes
    runtime_resident_bytes = exported.runtime_candidate_bytes
    runtime_resident_ratio = runtime_resident_bytes / target.raw_bytes

    # The setup mapping points at the original Parameters.  Drop this extra
    # owner before moving the model to CUDA; otherwise removed BF16 weights
    # would remain allocated and inflate NeuralZip's steady-memory result.
    del target_parameters
    model = model.to(device)
    inputs = _inputs(
        model,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        device=device,
    )
    runtime_attach_started = time.perf_counter()
    NeuralZipRuntime.attach(model, exported, device=device)
    runtime_attach_seconds = time.perf_counter() - runtime_attach_started
    attach_seconds = post_setup_encode_seconds + runtime_attach_seconds
    compressed = _measure(
        model,
        inputs,
        warmup=args.warmup,
        repetitions=args.repetitions,
        trials=args.trials,
    )
    runtime = model.neuralzip_runtime
    result = {
        "method": "neuralzip",
        **_model_fields(resolved),
        "device": torch.cuda.get_device_name(device),
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "use_cache": False,
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "trials": args.trials,
        "setup_seconds": setup_seconds,
        "post_setup_encode_seconds": post_setup_encode_seconds,
        "runtime_attach_seconds": runtime_attach_seconds,
        "attach_seconds": attach_seconds,
        "attach_seconds_scope": "post_setup_encode_plus_runtime_attach",
        "total_preparation_seconds": setup_seconds + attach_seconds,
        "compressed": compressed,
        "steady_gpu_allocated_bytes": compressed["steady_gpu_allocated_bytes"],
        "inference_ms": compressed["inference_ms"],
        "logit_sha256": compressed["logit_sha256"],
        **_exactness_fields(args, compressed["logit_sha256"]),
        **exponent_metrics,
        "eligible_raw_bytes": target.raw_bytes,
        "eligible_compressed_bytes": payload_bytes,
        "gpu_payload_bytes": runtime.export.payload_bytes,
        "gpu_metadata_bytes": runtime.export.metadata_bytes,
        "gpu_codebook_bytes": runtime.export.codebook_bytes,
        "gpu_launch_descriptor_bytes": runtime.export.launch_descriptor_bytes,
        "resident_raw_bytes": runtime.export.resident_raw_bytes,
        "payload_bytes": payload_bytes,
        "payload_ratio": payload_ratio,
        "eligible_payload_ratio": payload_ratio,
        "runtime_candidate_bytes": runtime_resident_bytes,
        "runtime_resident_bytes": runtime_resident_bytes,
        "runtime_resident_ratio": runtime_resident_ratio,
        "workspace_bytes": runtime.export.workspace_bytes,
        "compressed_blocks": len(runtime.export.block_order),
        "compressed_groups": len(runtime.export.groups),
        "unique_codebooks": len({group.codebook_id for group in runtime.export.groups}),
        "maximum_code_bits": max(
            (group.codebook.max_bits for group in runtime.export.groups),
            default=0,
        ),
        "exponent_prior_count": args.exponent_prior_count,
        "max_codebook_sample_bytes": args.max_codebook_sample_bytes,
        **_target_fields(target),
        **_version_fields(),
    }
    _emit_verified_result(args, result)


def _dfloat_scope(model, target: TargetScope) -> DFloatScope:
    exponent_stream_bytes = 0
    sign_mantissa_bytes = 0
    exponent_metadata_bytes = 0
    exponent_codebook_bytes = 0
    raw_bytes = 0
    split_positions_host_bytes = 0
    compressed_blocks = {}
    for module_name, module in model.named_modules():
        encoded = getattr(module, "encoded_exponent", None)
        sign_mantissa = getattr(module, "sign_mantissa", None)
        if encoded is None or sign_mantissa is None:
            continue
        compressed_blocks[module_name] = module

    actual_blocks = set(compressed_blocks)
    expected_blocks = set(target.block_names)
    if actual_blocks != expected_blocks:
        missing = sorted(expected_blocks - actual_blocks)
        extra = sorted(actual_blocks - expected_blocks)
        raise RuntimeError(
            "DFloat11 compressed-block coverage mismatch: "
            f"missing={missing}, extra={extra}."
        )

    covered_weights = []
    for block_name in target.block_names:
        module = compressed_blocks[block_name]
        block_weights = target.weights_for_block(block_name)
        expected_numel = sum(weight.numel for weight in block_weights)
        if module.sign_mantissa.numel() != expected_numel:
            raise RuntimeError(
                f"DFloat11 block {block_name} represents "
                f"{module.sign_mantissa.numel()} elements; expected {expected_numel}."
            )

        injection_modules = tuple(getattr(module, "weight_injection_modules", ()))
        expected_modules = tuple(
            model.get_submodule(weight.module_name) for weight in block_weights
        )
        if len(injection_modules) != len(expected_modules) or any(
            actual is not expected
            for actual, expected in zip(
                injection_modules,
                expected_modules,
                strict=True,
            )
        ):
            raise RuntimeError(
                f"DFloat11 block {block_name} does not inject the seven target weights."
            )

        expected_splits = []
        offset = 0
        for weight in block_weights[:-1]:
            offset += weight.numel
            expected_splits.append(offset)
        split_positions = getattr(module, "split_positions", None)
        if torch.is_tensor(split_positions):
            actual_splits = tuple(int(value) for value in split_positions.tolist())
            split_bytes = split_positions.numel() * split_positions.element_size()
        elif isinstance(split_positions, (list, tuple)):
            actual_splits = tuple(int(value) for value in split_positions)
            # DFloat11 serializes split positions as int64, then restores them
            # as a host list. Keep this logical host/storage metadata separate
            # from GPU-resident payload bytes.
            split_bytes = 8 * len(actual_splits)
        else:
            raise RuntimeError(f"DFloat11 block {block_name} has no split_positions.")
        if actual_splits != tuple(expected_splits):
            raise RuntimeError(
                f"DFloat11 block {block_name} has incorrect split_positions."
            )

        raw_bytes += module.sign_mantissa.numel() * 2
        encoded_exponent = module.encoded_exponent
        if not torch.is_tensor(encoded_exponent):
            raise RuntimeError(
                f"DFloat11 {block_name}.encoded_exponent is not a tensor."
            )
        exponent_stream_bytes += (
            encoded_exponent.numel() * encoded_exponent.element_size()
        )
        sign_mantissa = module.sign_mantissa
        if not torch.is_tensor(sign_mantissa):
            raise RuntimeError(f"DFloat11 {block_name}.sign_mantissa is not a tensor.")
        sign_mantissa_bytes += sign_mantissa.numel() * sign_mantissa.element_size()
        for name in ("output_positions", "gaps"):
            value = getattr(module, name)
            if not torch.is_tensor(value):
                raise RuntimeError(f"DFloat11 {block_name}.{name} is not a tensor.")
            exponent_metadata_bytes += value.numel() * value.element_size()
        luts = module.luts
        if not torch.is_tensor(luts):
            raise RuntimeError(f"DFloat11 {block_name}.luts is not a tensor.")
        exponent_codebook_bytes += luts.numel() * luts.element_size()
        split_positions_host_bytes += split_bytes
        covered_weights.extend(weight.tensor_name for weight in block_weights)

    _require_exact_coverage("DFloat11", covered_weights, target)
    if raw_bytes != target.raw_bytes:
        raise RuntimeError(
            "DFloat11 target bytes differ from the fixed Qwen/Llama scope: "
            f"{raw_bytes} != {target.raw_bytes}."
        )
    return DFloatScope(
        raw_bytes=raw_bytes,
        exponent_stream_bytes=exponent_stream_bytes,
        sign_mantissa_bytes=sign_mantissa_bytes,
        exponent_metadata_bytes=exponent_metadata_bytes,
        exponent_codebook_bytes=exponent_codebook_bytes,
        split_positions_host_bytes=split_positions_host_bytes,
        blocks=len(compressed_blocks),
    )


def _dfloat_in_memory_api():
    try:
        from examples.experiments.dfloat11_in_memory import (
            encode_dfloat11_in_memory,
            install_dfloat11_runtime,
        )
    except ModuleNotFoundError as error:
        if error.name != "examples":
            raise
        # Direct script execution places the experiments directory, rather
        # than the repository root, on sys.path.
        from dfloat11_in_memory import (  # type: ignore[no-redef]
            encode_dfloat11_in_memory,
            install_dfloat11_runtime,
        )
    return encode_dfloat11_in_memory, install_dfloat11_runtime


def _dfloat_workspace_bytes(device: torch.device) -> int:
    from dfloat11.dfloat11 import TensorManager

    workspace = TensorManager._tensors.get(device)
    if workspace is None:
        workspace = TensorManager._tensors.get(str(device))
    return 0 if workspace is None else workspace.numel() * workspace.element_size()


def _dfloat_result(
    args,
    device: torch.device,
    *,
    resolved: ResolvedModel,
    target: TargetScope,
    scope: DFloatScope,
    compressed: dict[str, object],
    source_model_load_seconds: float,
    encoding_seconds: float | None,
    runtime_attach_seconds: float,
    encoding_time_scope: str,
    disk_bytes: int | None,
) -> dict[str, object]:
    workspace_bytes = _dfloat_workspace_bytes(device)
    payload_ratio = scope.compressed_bytes / target.raw_bytes
    runtime_resident_bytes = scope.compressed_bytes + workspace_bytes
    result = {
        "method": "dfloat11",
        **_model_fields(resolved),
        "device": torch.cuda.get_device_name(device),
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "use_cache": False,
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "trials": args.trials,
        "source_model_load_seconds": source_model_load_seconds,
        "encoding_seconds": encoding_seconds,
        "post_setup_encode_seconds": encoding_seconds,
        "encoding_time_scope": encoding_time_scope,
        "runtime_attach_seconds": runtime_attach_seconds,
        "compressed": compressed,
        "steady_gpu_allocated_bytes": compressed["steady_gpu_allocated_bytes"],
        "inference_ms": compressed["inference_ms"],
        "logit_sha256": compressed["logit_sha256"],
        **_exactness_fields(args, str(compressed["logit_sha256"])),
        "raw_exponent_bytes": scope.raw_exponent_bytes,
        "exponent_stream_bytes": scope.exponent_stream_bytes,
        "exponent_metadata_bytes": scope.exponent_metadata_bytes,
        "exponent_codebook_bytes": scope.exponent_codebook_bytes,
        "raw_fallback_exponent_bytes": 0,
        "compressed_exponent_bytes": scope.exponent_compressed_bytes,
        "exponent_ratio": scope.exponent_ratio,
        "exponent_ratio_pct": 100.0 * scope.exponent_ratio,
        "eligible_raw_bytes": target.raw_bytes,
        "eligible_compressed_bytes": scope.compressed_bytes,
        "resident_raw_bytes": 0,
        "gpu_payload_bytes": scope.gpu_payload_bytes,
        "gpu_metadata_bytes": scope.gpu_metadata_bytes,
        "gpu_codebook_bytes": scope.gpu_codebook_bytes,
        "gpu_launch_descriptor_bytes": 0,
        "payload_bytes": scope.compressed_bytes,
        "payload_ratio": payload_ratio,
        "eligible_payload_ratio": payload_ratio,
        "runtime_candidate_bytes": runtime_resident_bytes,
        "runtime_resident_bytes": runtime_resident_bytes,
        "runtime_resident_ratio": runtime_resident_bytes / target.raw_bytes,
        "workspace_bytes": workspace_bytes,
        "split_positions_host_metadata_bytes": scope.split_positions_host_bytes,
        "compressed_blocks": scope.blocks,
        "disk_bytes": disk_bytes,
        **_target_fields(target),
        **_version_fields(),
    }
    return result


def _run_dfloat_in_memory(args, device) -> None:
    encode_dfloat11_in_memory, install_dfloat11_runtime = _dfloat_in_memory_api()
    resolved = _resolved_from_args(args)
    source_load_started = time.perf_counter()
    model = _load_resolved(args, resolved)
    source_model_load_seconds = time.perf_counter() - source_load_started
    target = _target_scope(model, resolved.layouts)

    encoding = encode_dfloat11_in_memory(model, target.dfloat_pattern)
    _require_exact_coverage("DFloat11 in-memory", encoding.weight_names, target)
    if encoding.raw_exponent_bytes != target.raw_exponent_bytes:
        raise RuntimeError(
            "DFloat11 encoded exponent count differs from the target scope: "
            f"{encoding.raw_exponent_bytes} != {target.raw_exponent_bytes}."
        )

    attach_started = time.perf_counter()
    install_dfloat11_runtime(model, encoding)
    model = model.to(device).eval()
    runtime_attach_seconds = time.perf_counter() - attach_started
    scope = _dfloat_scope(model, target)
    expected_metrics = (
        ("stream", encoding.exponent_stream_bytes, scope.exponent_stream_bytes),
        (
            "metadata",
            encoding.exponent_metadata_bytes,
            scope.exponent_metadata_bytes,
        ),
        (
            "codebooks",
            encoding.exponent_codebook_bytes,
            scope.exponent_codebook_bytes,
        ),
    )
    for name, encoded_bytes, runtime_bytes in expected_metrics:
        if encoded_bytes != runtime_bytes:
            raise RuntimeError(
                f"DFloat11 {name} bytes changed during runtime installation: "
                f"{encoded_bytes} != {runtime_bytes}."
            )
    inputs = _inputs(
        model,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        device=device,
    )
    compressed = _measure(
        model,
        inputs,
        warmup=args.warmup,
        repetitions=args.repetitions,
        trials=args.trials,
    )
    result = _dfloat_result(
        args,
        device,
        resolved=resolved,
        target=target,
        scope=scope,
        compressed=compressed,
        source_model_load_seconds=source_model_load_seconds,
        encoding_seconds=float(encoding.encode_seconds),
        runtime_attach_seconds=runtime_attach_seconds,
        encoding_time_scope="in_memory_official_dfloat11_primitives",
        disk_bytes=None,
    )
    _emit_verified_result(args, result)


def _run_dfloat_from_directory(args, device) -> None:
    from dfloat11 import DFloat11Model

    resolved = _resolved_from_args(args)
    source_load_started = time.perf_counter()
    bfloat16_model = _load_resolved(args, resolved)
    source_model_load_seconds = time.perf_counter() - source_load_started
    target = _target_scope(bfloat16_model, resolved.layouts)
    attach_started = time.perf_counter()
    model = DFloat11Model.from_pretrained(
        args.dfloat_directory,
        device=str(device),
        bfloat16_model=bfloat16_model,
    ).eval()
    runtime_attach_seconds = time.perf_counter() - attach_started
    scope = _dfloat_scope(model, target)
    inputs = _inputs(
        model,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        device=device,
    )
    compressed = _measure(
        model,
        inputs,
        warmup=args.warmup,
        repetitions=args.repetitions,
        trials=args.trials,
    )
    directory = Path(args.dfloat_directory)
    disk_bytes = sum(
        path.stat().st_size for path in directory.rglob("*") if path.is_file()
    )
    result = _dfloat_result(
        args,
        device,
        resolved=resolved,
        target=target,
        scope=scope,
        compressed=compressed,
        source_model_load_seconds=source_model_load_seconds,
        encoding_seconds=None,
        runtime_attach_seconds=runtime_attach_seconds,
        encoding_time_scope="not_measured_precompressed_directory",
        disk_bytes=disk_bytes,
    )
    _emit_verified_result(args, result)


def _run_dfloat(args, device) -> None:
    if args.dfloat_directory:
        _run_dfloat_from_directory(args, device)
    else:
        _run_dfloat_in_memory(args, device)


def _suite_configuration(args) -> dict[str, object]:
    return {
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "use_cache": False,
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "trials": args.trials,
        "device": args.device,
        "threads": args.threads,
        "exponent_prior_count": args.exponent_prior_count,
        "max_codebook_sample_bytes": args.max_codebook_sample_bytes,
        "target_linears_per_block": len(SEVEN_LINEAR_MODULES),
    }


def _suite_fingerprint(args, configuration: dict[str, object]) -> dict[str, object]:
    repository = Path(__file__).resolve().parents[2]
    files = [
        Path(__file__).resolve(),
        repository / "examples" / "experiments" / "dfloat11_in_memory.py",
    ]
    sources = (repository / "neuralzip").rglob("*")
    files += [
        path
        for path in sources
        if path.is_file()
        and path.suffix in {".py", ".cpp", ".cu", ".h", ".so", ".dylib"}
    ]
    code = hashlib.sha256()
    for path in sorted(files):
        code.update(path.relative_to(repository).as_posix().encode())
        code.update(path.read_bytes())
    runtime = {
        "python": sys.version,
        **_version_fields(),
        "cuda": torch.version.cuda,
        "device": args.device,
        "cupy": _package_version("cupy-cuda12x"),
    }
    if torch.cuda.is_available():
        runtime["gpu"] = torch.cuda.get_device_name(torch.device(args.device))
    components = {
        "code": code.hexdigest(),
        "runtime": runtime,
        "config": configuration,
    }
    digest = hashlib.sha256(json.dumps(components, sort_keys=True).encode()).hexdigest()
    return {"digest": digest, **components}


def _child_command(
    args,
    method: str,
    resolved: ResolvedModel,
    result_file: Path,
    *,
    reference_logit_sha256: str | None = None,
) -> list[str]:
    model_argument = (
        resolved.key if resolved.key in MODEL_CATALOG else resolved.model_id
    )
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        method,
        "--model",
        model_argument,
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
        "--sequence-length",
        str(args.sequence_length),
        "--warmup",
        str(args.warmup),
        "--repetitions",
        str(args.repetitions),
        "--trials",
        str(args.trials),
        "--threads",
        str(args.threads),
        "--exponent-prior-count",
        str(args.exponent_prior_count),
        "--max-codebook-sample-bytes",
        str(args.max_codebook_sample_bytes),
        "--result-file",
        str(result_file),
        "--no-print-result",
    ]
    if resolved.revision is not None:
        command.extend(("--revision", resolved.revision))
    if args.cache_dir:
        command.extend(("--cache-dir", args.cache_dir))
    if reference_logit_sha256 is not None:
        command.extend(("--reference-logit-sha256", reference_logit_sha256))
    return command


def _run_child_method(
    args,
    method: str,
    resolved: ResolvedModel,
    *,
    reference_logit_sha256: str | None = None,
) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="neuralzip-gpu-suite-") as temporary:
        result_file = Path(temporary) / "result.json"
        command = _child_command(
            args,
            method,
            resolved,
            result_file,
            reference_logit_sha256=reference_logit_sha256,
        )
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        subprocess.run(
            command,
            check=True,
            cwd=Path(__file__).resolve().parents[2],
            env=environment,
        )
        if not result_file.is_file():
            raise RuntimeError(f"{method} did not write its result file.")
        result = json.loads(result_file.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise TypeError(f"{method} result must be a JSON object.")
    return result


def _method_metrics(result: dict[str, object]) -> dict[str, object]:
    method = str(result["method"])
    if method == "bf16":
        encoding_seconds = None
    elif method == "neuralzip":
        encoding_seconds = result["post_setup_encode_seconds"]
    elif method == "dfloat11":
        encoding_seconds = result["encoding_seconds"]
    else:
        raise ValueError(f"Unexpected suite method {method!r}.")
    return {
        "encoding_seconds": encoding_seconds,
        "exponent_ratio": result["exponent_ratio"],
        "exponent_ratio_pct": result["exponent_ratio_pct"],
        "steady_gpu_allocated_bytes": result["steady_gpu_allocated_bytes"],
        "inference_ms": result["inference_ms"],
        "logit_sha256": result["logit_sha256"],
    }


def _result_method_name(child_method: str) -> str:
    return "dfloat11" if child_method == "dfloat" else child_method


def _checkpoint_result_valid(
    resolved: ResolvedModel,
    configuration: dict[str, object],
    child_method: str,
    result: dict[str, object],
    *,
    reference_logit_sha256: str | None = None,
) -> bool:
    expected = _result_method_name(child_method)
    fixed = {
        "method": expected,
        "model": resolved.model_id,
        "revision": resolved.revision,
        **{
            key: configuration[key]
            for key in (
                "batch_size",
                "sequence_length",
                "use_cache",
                "warmup",
                "repetitions",
                "trials",
            )
        },
    }
    if any(result.get(key) != value for key, value in fixed.items()):
        return False
    required = (
        "target_block_names",
        "target_weight_names",
        "target_raw_bytes",
        "target_raw_exponent_bytes",
        "logit_sha256",
        "exponent_ratio",
        "steady_gpu_allocated_bytes",
        "inference_ms",
    )
    if any(result.get(field) is None for field in required):
        return False
    timing = {
        "neuralzip": "post_setup_encode_seconds",
        "dfloat11": "encoding_seconds",
    }.get(expected)
    if timing and result.get(timing) is None:
        return False
    return reference_logit_sha256 is None or (
        result.get("reference_logit_sha256") == reference_logit_sha256
        and result.get("logit_sha256") == reference_logit_sha256
        and result.get("bitwise_exact_logits") is True
    )


def _suite_record(
    kind: str,
    resolved: ResolvedModel,
    configuration: dict[str, object],
    fingerprint: dict[str, object],
    **fields,
) -> dict[str, object]:
    return {
        "schema_version": 2,
        "experiment": "gpu-compression-and-inference",
        "record_type": kind,
        **_model_fields(resolved),
        "configuration": configuration,
        "resume_fingerprint": fingerprint["digest"],
        "fingerprint": fingerprint,
        **fields,
    }


def _failure_record(resolved, configuration, fingerprint, stage, error):
    fields = {
        "status": "failed",
        "stage": stage,
        "error_type": type(error).__name__,
        "error": str(error),
    }
    if isinstance(error, subprocess.CalledProcessError):
        fields["returncode"] = error.returncode
    return _suite_record("failure", resolved, configuration, fingerprint, **fields)


def _validate_suite_results(
    resolved: ResolvedModel,
    configuration: dict[str, object],
    bf16: dict[str, object],
    neuralzip: dict[str, object],
    dfloat11: dict[str, object],
    *,
    fingerprint: dict[str, object],
) -> dict[str, object]:
    results = (bf16, neuralzip, dfloat11)
    if tuple(result.get("method") for result in results) != (
        "bf16",
        "neuralzip",
        "dfloat11",
    ):
        raise RuntimeError("Suite subprocesses returned unexpected method labels.")

    scope_fields = (
        "model",
        "revision",
        "target_layout",
        "target_block_pattern",
        "target_block_names",
        "target_weight_names",
        "target_raw_bytes",
        "target_raw_exponent_bytes",
    )
    for field in scope_fields:
        values = tuple(result.get(field) for result in results)
        if values[1:] != values[:-1]:
            raise RuntimeError(
                f"Methods disagree on {field} for {resolved.label}: {values}."
            )

    digest = str(bf16["logit_sha256"])
    if any(str(result["logit_sha256"]) != digest for result in results[1:]):
        raise RuntimeError(
            f"Compressed logits are not bitwise exact for {resolved.label}."
        )
    if any(result.get("bitwise_exact_logits") is not True for result in results[1:]):
        raise RuntimeError(
            f"A compressed subprocess did not validate BF16 exactness for {resolved.label}."
        )

    metrics = {
        "bf16": _method_metrics(bf16),
        "neuralzip": _method_metrics(neuralzip),
        "dfloat11": _method_metrics(dfloat11),
    }
    for method, values in metrics.items():
        for field in (
            "exponent_ratio",
            "steady_gpu_allocated_bytes",
            "inference_ms",
        ):
            value = values[field]
            if not isinstance(value, (int, float)) or value <= 0:
                raise RuntimeError(
                    f"{resolved.label} {method} returned invalid {field}: {value}."
                )
        encoding_seconds = values["encoding_seconds"]
        if method != "bf16" and (
            not isinstance(encoding_seconds, (int, float)) or encoding_seconds <= 0
        ):
            raise RuntimeError(f"{resolved.label} {method} has no valid encoding time.")

    return {
        **_suite_record("complete", resolved, configuration, fingerprint),
        "status": "complete",
        "bitwise_exact_logits": True,
        "target_block_count": bf16["target_block_count"],
        "target_weight_count": bf16["target_weight_count"],
        "target_raw_bytes": bf16["target_raw_bytes"],
        "target_raw_exponent_bytes": bf16["target_raw_exponent_bytes"],
        "metrics": metrics,
        "raw_results": {
            "bf16": bf16,
            "neuralzip": neuralzip,
            "dfloat11": dfloat11,
        },
    }


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSONL row {line_number} in {path}.") from error
        if not isinstance(row, dict):
            raise TypeError(f"JSONL row {line_number} in {path} is not an object.")
        rows.append(row)
    return rows


def _record_type(row: dict[str, object]) -> str | None:
    return row.get("record_type") or ("complete" if "metrics" in row else None)


def _run_record_matches(
    row: dict[str, object],
    resolved: ResolvedModel,
    configuration: dict[str, object],
    resume_fingerprint: str,
) -> bool:
    return (
        row.get("experiment") == "gpu-compression-and-inference"
        and row.get("model") == resolved.model_id
        and row.get("revision") == resolved.revision
        and row.get("configuration") == configuration
        and row.get("resume_fingerprint") == resume_fingerprint
    )


def _record_key(row: dict[str, object]) -> tuple[object, ...]:
    kind = _record_type(row)
    slot = row.get("method") if kind == "method_result" else row.get("stage")
    return (
        kind,
        slot,
        row.get("model"),
        row.get("revision"),
        json.dumps(row.get("configuration"), sort_keys=True),
        row.get("resume_fingerprint"),
    )


def _write_jsonl_atomic(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _upsert_jsonl(
    path: Path,
    rows: list[dict[str, object]],
    record: dict[str, object],
) -> list[dict[str, object]]:
    key = _record_key(record)
    updated = [row for row in rows if _record_key(row) != key] + [record]
    _write_jsonl_atomic(path, updated)
    return updated


def _discard_records(
    path: Path,
    rows: list[dict[str, object]],
    predicate,
) -> list[dict[str, object]]:
    updated = [row for row in rows if not predicate(row)]
    if len(updated) != len(rows):
        _write_jsonl_atomic(path, updated)
    return updated


def _method_result_for(
    rows: list[dict[str, object]],
    resolved: ResolvedModel,
    configuration: dict[str, object],
    resume_fingerprint: str,
    child_method: str,
    *,
    reference_logit_sha256: str | None = None,
) -> dict[str, object] | None:
    expected_method = _result_method_name(child_method)
    for row in reversed(rows):
        if (
            _record_type(row) != "method_result"
            or row.get("status") != "complete"
            or row.get("method") != expected_method
            or not _run_record_matches(
                row,
                resolved,
                configuration,
                resume_fingerprint,
            )
        ):
            continue
        result = row.get("result")
        if isinstance(result, dict) and _checkpoint_result_valid(
            resolved,
            configuration,
            child_method,
            result,
            reference_logit_sha256=reference_logit_sha256,
        ):
            return result
    return None


def _complete_row_for(
    rows: list[dict[str, object]],
    resolved: ResolvedModel,
    configuration: dict[str, object],
    resume_fingerprint: str,
) -> dict[str, object] | None:
    return next(
        (
            row
            for row in reversed(rows)
            if (
                _record_type(row) == "complete"
                and row.get("status") == "complete"
                and _run_record_matches(
                    row,
                    resolved,
                    configuration,
                    resume_fingerprint,
                )
                and row.get("bitwise_exact_logits") is True
            )
        ),
        None,
    )


_CSV_FIELDS = (
    "model_key",
    "model_label",
    "model_family",
    "model",
    "revision",
    "bf16_exponent_ratio_pct",
    "bf16_gpu_memory_gib",
    "bf16_inference_ms",
    "neuralzip_encoding_seconds",
    "neuralzip_exponent_ratio_pct",
    "neuralzip_gpu_memory_gib",
    "neuralzip_inference_ms",
    "dfloat11_encoding_seconds",
    "dfloat11_exponent_ratio_pct",
    "dfloat11_gpu_memory_gib",
    "dfloat11_inference_ms",
    "bitwise_exact_logits",
)


def _csv_row(row: dict[str, object]) -> dict[str, object]:
    metrics = row["metrics"]
    if not isinstance(metrics, dict):
        raise TypeError("A completed suite row has invalid metrics.")

    def method(name: str) -> dict[str, object]:
        value = metrics[name]
        if not isinstance(value, dict):
            raise TypeError(f"A completed suite row has invalid {name} metrics.")
        return value

    bf16 = method("bf16")
    neuralzip = method("neuralzip")
    dfloat11 = method("dfloat11")
    gib = float(1 << 30)
    return {
        "model_key": row["model_key"],
        "model_label": row["model_label"],
        "model_family": row["model_family"],
        "model": row["model"],
        "revision": row["revision"],
        "bf16_exponent_ratio_pct": bf16["exponent_ratio_pct"],
        "bf16_gpu_memory_gib": float(bf16["steady_gpu_allocated_bytes"]) / gib,
        "bf16_inference_ms": bf16["inference_ms"],
        "neuralzip_encoding_seconds": neuralzip["encoding_seconds"],
        "neuralzip_exponent_ratio_pct": neuralzip["exponent_ratio_pct"],
        "neuralzip_gpu_memory_gib": (
            float(neuralzip["steady_gpu_allocated_bytes"]) / gib
        ),
        "neuralzip_inference_ms": neuralzip["inference_ms"],
        "dfloat11_encoding_seconds": dfloat11["encoding_seconds"],
        "dfloat11_exponent_ratio_pct": dfloat11["exponent_ratio_pct"],
        "dfloat11_gpu_memory_gib": (
            float(dfloat11["steady_gpu_allocated_bytes"]) / gib
        ),
        "dfloat11_inference_ms": dfloat11["inference_ms"],
        "bitwise_exact_logits": row["bitwise_exact_logits"],
    }


def _write_csv(path: Path, rows: list[dict[str, object]], fingerprint=None) -> None:
    complete_rows = [
        row
        for row in rows
        if (
            _record_type(row) == "complete"
            and row.get("status") == "complete"
            and (fingerprint is None or row.get("resume_fingerprint") == fingerprint)
        )
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(_csv_row(row) for row in complete_rows)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _obtain_suite_method(
    args,
    output_jsonl: Path,
    rows: list[dict[str, object]],
    resolved: ResolvedModel,
    configuration: dict[str, object],
    fingerprint: dict[str, object],
    child_method: str,
    *,
    reference_logit_sha256: str | None = None,
) -> tuple[dict[str, object], list[dict[str, object]], bool]:
    cached = _method_result_for(
        rows,
        resolved,
        configuration,
        str(fingerprint["digest"]),
        child_method,
        reference_logit_sha256=reference_logit_sha256,
    )
    if cached is not None:
        return cached, rows, True

    result = _run_child_method(
        args,
        child_method,
        resolved,
        reference_logit_sha256=reference_logit_sha256,
    )
    if not _checkpoint_result_valid(
        resolved,
        configuration,
        child_method,
        result,
        reference_logit_sha256=reference_logit_sha256,
    ):
        raise RuntimeError(f"Invalid {child_method} result for {resolved.label}.")
    record = _suite_record(
        "method_result",
        resolved,
        configuration,
        fingerprint,
        status="complete",
        method=_result_method_name(child_method),
        result=result,
    )
    return result, _upsert_jsonl(output_jsonl, rows, record), False


def _run_suite(args) -> None:
    model_arguments = tuple(args.suite_model or DEFAULT_SUITE_MODEL_KEYS)
    resolved_models = tuple(
        _resolve_model(model, args.revision if len(model_arguments) == 1 else None)
        for model in model_arguments
    )
    output_jsonl = Path(args.output_jsonl)
    output_csv = (
        Path(args.output_csv) if args.output_csv else output_jsonl.with_suffix(".csv")
    )
    configuration = _suite_configuration(args)
    fingerprint = _suite_fingerprint(args, configuration)
    resume_fingerprint = str(fingerprint["digest"])
    rows = _read_jsonl(output_jsonl)
    failures = []

    for index, resolved in enumerate(resolved_models, start=1):

        def matches_run(record, resolved=resolved):
            return _run_record_matches(
                record, resolved, configuration, resume_fingerprint
            )

        if args.force_suite:
            rows = _discard_records(output_jsonl, rows, matches_run)
        previous = _complete_row_for(rows, resolved, configuration, resume_fingerprint)
        if previous is not None:
            print(
                f"[{index}/{len(resolved_models)}] {resolved.label}: complete, skipping",
                flush=True,
            )
            continue

        results = {}
        for child_method, label in (
            ("bf16", "BF16"),
            ("neuralzip", "NeuralZip"),
            ("dfloat", "DFloat11"),
        ):
            if child_method != "bf16" and "bf16" not in results:
                break
            reference = (
                None if child_method == "bf16" else str(results["bf16"]["logit_sha256"])
            )
            try:
                result, rows, reused = _obtain_suite_method(
                    args,
                    output_jsonl,
                    rows,
                    resolved,
                    configuration,
                    fingerprint,
                    child_method,
                    reference_logit_sha256=reference,
                )
                results[_result_method_name(child_method)] = result
                state = "resumed" if reused else "saved"
                print(
                    f"[{index}/{len(resolved_models)}] {resolved.label}: {label} {state}",
                    flush=True,
                )
            except Exception as error:  # noqa: BLE001 - continue the overnight suite.
                failure = _failure_record(
                    resolved,
                    configuration,
                    fingerprint,
                    _result_method_name(child_method),
                    error,
                )
                rows = _upsert_jsonl(output_jsonl, rows, failure)
                failures.append(failure)
                print(
                    f"[{index}/{len(resolved_models)}] {resolved.label}: "
                    f"{label} failed: {error}",
                    file=sys.stderr,
                    flush=True,
                )

        if set(results) != {"bf16", "neuralzip", "dfloat11"}:
            continue
        try:
            row = _validate_suite_results(
                resolved,
                configuration,
                results["bf16"],
                results["neuralzip"],
                results["dfloat11"],
                fingerprint=fingerprint,
            )
        except Exception as error:  # noqa: BLE001 - record invalid cross-method rows.
            failure = _failure_record(
                resolved, configuration, fingerprint, "validation", error
            )
            rows = _upsert_jsonl(output_jsonl, rows, failure)
            failures.append(failure)
            print(
                f"[{index}/{len(resolved_models)}] {resolved.label}: "
                f"validation failed: {error}",
                file=sys.stderr,
                flush=True,
            )
            continue

        rows = _discard_records(
            output_jsonl,
            rows,
            lambda record, matches=matches_run: (
                _record_type(record) == "failure" and matches(record)
            ),
        )
        rows = _upsert_jsonl(output_jsonl, rows, row)
        _write_csv(output_csv, rows, resume_fingerprint)
        print(
            f"[{index}/{len(resolved_models)}] {resolved.label}: complete",
            flush=True,
        )

    _write_csv(output_csv, rows, resume_fingerprint)
    print(
        json.dumps(
            {
                "experiment": "gpu-compression-and-inference",
                "output_jsonl": str(output_jsonl),
                "output_csv": str(output_csv),
                "selected_models": [model.key for model in resolved_models],
                "resume_fingerprint": resume_fingerprint,
                "completed_models": sum(
                    _complete_row_for(rows, model, configuration, resume_fingerprint)
                    is not None
                    for model in resolved_models
                ),
                "failed_stages": len(failures),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if failures:
        raise SystemExit(1)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "method",
        choices=(
            "catalog",
            "suite",
            "prepare-dfloat",
            "bf16",
            "neuralzip",
            "dfloat",
        ),
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--revision")
    parser.add_argument("--cache-dir")
    parser.add_argument("--dfloat-directory")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--exponent-prior-count", type=int, default=1)
    parser.add_argument(
        "--max-codebook-sample-bytes",
        type=int,
        default=8 << 20,
    )
    parser.add_argument("--reference-logit-sha256")
    parser.add_argument("--result-file")
    parser.add_argument("--no-print-result", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--suite-model",
        action="append",
        help=(
            "Catalog key or Hugging Face model ID; repeat to override the "
            "nine-model public default."
        ),
    )
    parser.add_argument("--output-jsonl")
    parser.add_argument("--output-csv")
    parser.add_argument("--force-suite", action="store_true")
    return parser


def _print_catalog() -> None:
    default_keys = set(DEFAULT_SUITE_MODEL_KEYS)
    print(
        json.dumps(
            [
                {
                    "key": spec.key,
                    "label": spec.label,
                    "family": spec.family,
                    "model": spec.model_id,
                    "revision": spec.revision,
                    "default_suite": spec.key in default_keys,
                    "block_patterns": [layout.block_pattern for layout in spec.layouts],
                    "linears_per_block": len(spec.layouts[0].relative_modules),
                }
                for spec in MODEL_CATALOG.values()
            ],
            indent=2,
            sort_keys=True,
        )
    )


def _validate_args(parser: argparse.ArgumentParser, args) -> torch.device:
    if args.batch_size <= 0 or args.sequence_length <= 0:
        parser.error("batch size and sequence length must be positive")
    if args.warmup < 0 or args.repetitions <= 0 or args.trials <= 0:
        parser.error("warmup must be non-negative; repetitions and trials positive")
    if (
        args.threads < 0
        or args.exponent_prior_count <= 0
        or args.max_codebook_sample_bytes <= 0
    ):
        parser.error("threads, codebook prior, and sample size are invalid")
    try:
        device = torch.device(args.device)
    except RuntimeError as error:
        parser.error(str(error))
    if device.type != "cuda":
        parser.error("the comparison requires a CUDA device")
    if torch.cuda.is_available() and (
        device.index is not None and device.index >= torch.cuda.device_count()
    ):
        parser.error(f"CUDA device index {device.index} is unavailable")
    if args.method == "suite":
        models = args.suite_model or DEFAULT_SUITE_MODEL_KEYS
        if not args.output_jsonl or len(models) != len(set(models)):
            parser.error("suite requires an output and unique models")
        if args.revision is not None and len(models) != 1:
            parser.error("--revision requires a one-model suite")
        if args.output_csv and Path(args.output_csv) == Path(args.output_jsonl):
            parser.error("JSONL and CSV outputs must be different files")
    return device


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.method == "catalog":
        _print_catalog()
        return
    if args.method == "prepare-dfloat":
        args.resolved_model = _resolve_model(args.model, args.revision)
        if not args.dfloat_directory:
            parser.error("prepare-dfloat requires --dfloat-directory")
        _prepare_dfloat(args)
        return
    device = _validate_args(parser, args)
    if args.method == "suite":
        _run_suite(args)
        return

    args.resolved_model = _resolve_model(args.model, args.revision)
    torch.cuda.set_device(device)
    if args.method == "bf16":
        _run_bf16(args, device)
    elif args.method == "neuralzip":
        _run_neuralzip(args, device)
    else:
        _run_dfloat(args, device)


if __name__ == "__main__":
    main()
