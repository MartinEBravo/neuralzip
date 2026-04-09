import gc
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import torch

from constants import HF_ALLOW_PATTERNS

SafetensorEntries = list[tuple[Path, list[str]]]


def resolve_safetensor_source(
    model_name: str,
    hf_token: str | None,
    local_files_only: bool,
) -> Path:
    local_path = Path(model_name)
    if local_path.exists():
        return local_path

    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=model_name,
            token=hf_token,
            local_files_only=local_files_only,
            allow_patterns=HF_ALLOW_PATTERNS,
        )
    )


def discover_safetensor_entries(source_path: Path) -> SafetensorEntries:
    if source_path.is_file():
        if source_path.suffix != ".safetensors":
            raise RuntimeError(f"{source_path} is not a .safetensors file.")
        return [(source_path, read_safetensor_keys(source_path))]

    index_candidates = sorted(source_path.rglob("*.safetensors.index.json"))
    if index_candidates:
        index_data = json.loads(index_candidates[0].read_text())
        weight_map = index_data.get("weight_map", {})
        if not weight_map:
            raise RuntimeError(f"{index_candidates[0]} does not contain a weight_map.")
        files_to_keys: dict[str, list[str]] = defaultdict(list)
        for key, filename in weight_map.items():
            files_to_keys[filename].append(key)
        return [(source_path / filename, keys) for filename, keys in files_to_keys.items()]

    shard_paths = sorted(source_path.rglob("*.safetensors"))
    if not shard_paths:
        raise RuntimeError(
            f"No .safetensors shards were found for {source_path}. "
            "Use --loader transformers if you need the legacy loader."
        )
    return [(shard_path, read_safetensor_keys(shard_path)) for shard_path in shard_paths]


def read_safetensor_keys(shard_path: Path) -> list[str]:
    from safetensors.torch import safe_open

    with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
        return list(handle.keys())


def iter_tensors_from_safetensors(
    entries: SafetensorEntries,
) -> Iterator[tuple[str, torch.Tensor]]:
    from safetensors.torch import safe_open

    for shard_path, keys in entries:
        with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
            for key in keys:
                yield key, handle.get_tensor(key)


def _hf_kwargs(
    hf_token: str | None,
    trust_remote_code: bool,
    **extra: object,
) -> dict[str, object]:
    kwargs: dict[str, object] = dict(extra)
    if hf_token:
        kwargs["token"] = hf_token
    if trust_remote_code:
        kwargs["trust_remote_code"] = True
    return kwargs


def iter_tensors_from_transformers(
    model_name: str,
    hf_token: str | None,
    trust_remote_code: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.from_pretrained(
        model_name, **_hf_kwargs(hf_token, trust_remote_code)
    )
    if hasattr(config, "add_pooling_layer"):
        config.add_pooling_layer = False

    model = AutoModel.from_pretrained(
        model_name,
        **_hf_kwargs(hf_token, trust_remote_code, config=config, low_cpu_mem_usage=True),
    )
    try:
        for name, param in model.named_parameters():
            yield name, param.detach().cpu()
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
