"""Trusted, directly loadable NeuralZip inference checkpoints."""

import hashlib
import importlib
import json
import math
import os
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from .inference import (
    BF16WeightGroup,
    GPUCodebook,
    GPUEncodedBF16,
    GPUEncodedExponents,
    LinearWeightBinding,
    NeuralZipModelExport,
    NeuralZipRuntime,
    export_model_for_gpu,
)

FORMAT_NAME = "neuralzip-gpu-pretrained"
FORMAT_VERSION = 3
MANIFEST_FILENAME = "neuralzip_config.json"
EXPORT_FILENAME = "neuralzip-export.pt"
RESIDENT_FILENAME = "neuralzip-resident.safetensors"

def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

def _checkpoint_source(
    value,
    *,
    revision="main",
    token=None,
    cache_dir=None,
    local_files_only=False,
):
    source = os.fspath(value)
    local = Path(source).expanduser()
    if local.is_dir():
        return local
    if local.exists() or local.is_absolute() or source.startswith((".", "~")):
        raise FileNotFoundError(local)

    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=source,
            revision=revision,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
    )

def _transformers_model(directory, module_name, class_name):
    if not module_name.startswith("transformers."):
        raise ValueError("The checkpoint does not contain a Transformers model.")
    model_class = getattr(importlib.import_module(module_name), class_name)
    if not isinstance(model_class, type) or not issubclass(model_class, nn.Module):
        raise TypeError("The saved Transformers model class is unavailable.")

    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(directory, local_files_only=True)
    return model_class(config)

def _resident_state(runtime):
    return {
        name: tensor.detach().to(device="cpu").contiguous() for name, tensor in runtime.resident_state_dict().items()
    }

def _export_state(exported):
    codebooks = {}
    groups = []
    for group in exported.groups:
        codebooks.setdefault(
            group.codebook_id,
            {
                "lengths": group.codebook.code_lengths,
                "values": group.codebook.code_values,
            },
        )
        groups.append(
            {
                "block": group.block_name,
                "codebook": group.codebook_id,
                "encoded": {
                    "bitstream": group.encoded.exponents.bitstream,
                    "bit_length": group.encoded.exponents.bit_length,
                    "symbol_count": group.encoded.exponents.symbol_count,
                    "packed_gaps": group.encoded.exponents.packed_gaps,
                    "output_positions": group.encoded.exponents.output_positions,
                    "sign_mantissa": group.encoded.sign_mantissa,
                },
                "bindings": [
                    {
                        "module": binding.module_name,
                        "tensor": binding.tensor_name,
                        "shape": binding.shape,
                        "requires_grad": binding.requires_grad,
                        "kind": binding.module_kind,
                    }
                    for binding in group.bindings
                ],
            }
        )
    return {
        "codebooks": codebooks,
        "groups": groups,
    }

def _load_export(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    codebooks = {
        int(codebook_id): GPUCodebook(value["lengths"], value["values"])
        for codebook_id, value in state["codebooks"].items()
    }
    groups = []
    workspace_offsets = {}
    for value in state["groups"]:
        block = value["block"]
        data = value["encoded"]
        encoded = GPUEncodedBF16(
            sign_mantissa=data["sign_mantissa"],
            exponents=GPUEncodedExponents(
                bitstream=data["bitstream"],
                bit_length=data["bit_length"],
                symbol_count=data["symbol_count"],
                packed_gaps=data["packed_gaps"],
                output_positions=data["output_positions"],
            ),
        )
        workspace_offset = workspace_offsets.get(block, 0)
        group_offset = 0
        bindings = []
        for binding in value["bindings"]:
            shape = tuple(binding["shape"])
            numel = math.prod(shape)
            bindings.append(
                LinearWeightBinding(
                    module_name=binding["module"],
                    tensor_name=binding["tensor"],
                    shape=shape,
                    numel=numel,
                    workspace_offset=workspace_offset + group_offset,
                    requires_grad=binding["requires_grad"],
                    module_kind=binding["kind"],
                )
            )
            group_offset += numel
        if group_offset != encoded.numel:
            raise ValueError("Compressed weights do not match their tensor shapes.")
        groups.append(
            BF16WeightGroup(
                block_name=block,
                codebook_id=value["codebook"],
                codebook=codebooks[value["codebook"]],
                encoded=encoded,
                bindings=tuple(bindings),
                workspace_offset=workspace_offset,
            )
        )
        workspace_offsets[block] = workspace_offset + encoded.numel
    return NeuralZipModelExport(tuple(groups))

def save_gpu_pretrained(model, directory, *, tokenizer=None):
    """Save one trusted export and one resident Safetensors file."""

    runtime = getattr(model, "neuralzip_runtime", None)
    if not isinstance(runtime, NeuralZipRuntime):
        raise TypeError("model must have an attached NeuralZip runtime.")

    target = Path(directory).expanduser()
    target.mkdir(parents=True, exist_ok=True)
    torch.save(_export_state(runtime.export), target / EXPORT_FILENAME)
    save_file(_resident_state(runtime), target / RESIDENT_FILENAME)

    config = getattr(model, "config", None)
    if config is None or not callable(getattr(config, "save_pretrained", None)):
        raise TypeError("Persistent GPU checkpoints require a Transformers model.")
    config.save_pretrained(target)
    generation = getattr(model, "generation_config", None)
    if generation is not None:
        generation.save_pretrained(target)
    elif (target / "generation_config.json").is_file():
        (target / "generation_config.json").unlink()
    if tokenizer is not None:
        tokenizer.save_pretrained(target)

    manifest = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "model_module": type(model).__module__,
        "model_class": type(model).__name__,
        "sha256": {
            filename: _sha256(target / filename)
            for filename in (
                EXPORT_FILENAME,
                RESIDENT_FILENAME,
                "config.json",
                "generation_config.json",
            )
            if (target / filename).is_file()
        },
    }
    (target / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return target

def attach_gpu_model(
    model,
    setup,
    *,
    threads=0,
    device=None,
    include_embeddings=False,
):
    """Compress eligible weights and attach the single-GPU runtime."""

    exported = export_model_for_gpu(
        model,
        setup,
        threads=threads,
        include_embeddings=include_embeddings,
    )
    NeuralZipRuntime.attach(model, exported, device=device)
    return model

def load_gpu_pretrained(
    pretrained_model_name_or_path,
    *,
    device=None,
    revision="main",
    token=None,
    cache_dir=None,
    local_files_only=False,
):
    """Load a trusted local or Hugging Face NeuralZip checkpoint."""

    target = _checkpoint_source(
        pretrained_model_name_or_path,
        revision=revision,
        token=token,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    manifest = json.loads((target / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT_NAME or manifest.get("version") != FORMAT_VERSION:
        raise ValueError("Unsupported NeuralZip GPU checkpoint.")
    hashes = manifest["sha256"]
    if not {EXPORT_FILENAME, RESIDENT_FILENAME, "config.json"} <= hashes.keys():
        raise ValueError("NeuralZip checkpoint hashes are incomplete.")
    for filename, expected in hashes.items():
        path = target / filename
        if path.parent != target or not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"Corrupted NeuralZip checkpoint file: {filename}.")

    exported = _load_export(target / EXPORT_FILENAME)
    model = _transformers_model(
        target,
        manifest["model_module"],
        manifest["model_class"],
    )
    resident = load_file(target / RESIDENT_FILENAME, device="cpu")
    missing, unexpected = model.load_state_dict(resident, strict=False, assign=True)
    compressed = {binding.tensor_name for binding in exported.bindings}
    if set(missing) != compressed or unexpected:
        raise ValueError("The resident state does not match the model architecture.")
    tie_weights = getattr(model, "tie_weights", None)
    if callable(tie_weights):
        tie_weights()

    generation_path = target / "generation_config.json"
    if generation_path.is_file():
        from transformers import GenerationConfig

        model.generation_config = GenerationConfig.from_pretrained(
            target,
            local_files_only=True,
        )
    NeuralZipRuntime.attach(model, exported, device=device)
    return model

__all__ = [
    "FORMAT_NAME",
    "FORMAT_VERSION",
    "MANIFEST_FILENAME",
    "attach_gpu_model",
    "load_gpu_pretrained",
    "save_gpu_pretrained",
]
