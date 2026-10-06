"""Pinned paper inputs and model adapters. No work is performed on import."""

import hashlib
import json
import re
import struct
import tarfile
import urllib.request
import zipfile
from contextlib import nullcontext
from pathlib import Path

import torch
from huggingface_hub import HfApi, hf_hub_download
from safetensors import safe_open
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForMaskedLM,
    AutoModelForPreTraining,
    AutoModelForSpeechSeq2Seq,
    ParakeetForTDT,
)
from transformers.conversion_mapping import get_model_conversion_mapping
from transformers.core_model_loading import (
    WeightConverter,
    WeightRenaming,
    convert_and_load_state_dict_in_model,
    rename_source_key,
    revert_weight_conversion,
)
from transformers.modeling_utils import LoadStateDictConfig

try:
    import torchvision.models as vision_models
except ImportError:
    vision_models = None
try:
    from diffusers import UNet2DConditionModel
except ImportError:
    UNet2DConditionModel = None
try:
    from accelerate import init_empty_weights
except ImportError:
    init_empty_weights = None

RESULTS = Path(__file__).resolve().parents[1] / "results"
NEMO_MAPPING = {
    r"encoder\.pre_encode\.conv\.": r"encoder.subsampling.layers.",
    r"encoder\.pre_encode\.out\.": r"encoder.subsampling.linear.",
    r"encoder\.pos_enc\.": r"encoder.encode_positions.",
    r"encoder\.layers\.(\d+)\.conv\.batch_norm\.": r"encoder.layers.\1.conv.norm.",
    r"linear_([kv])": r"\1_proj",
    r"linear_out": r"o_proj",
    r"linear_q": r"q_proj",
    r"pos_bias_([uv])": r"bias_\1",
    r"linear_pos": r"relative_k_proj",
    r"decoder\.prediction\.embed\.": r"decoder.embedding.",
    r"decoder\.prediction\.dec_rnn\.lstm\.": r"decoder.lstm.",
    r"joint\.enc\.": r"encoder_projector.",
    r"joint\.pred\.": r"decoder.decoder_projector.",
    r"joint\.joint_net\.2\.": r"joint.head.",
}


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(8 << 20), b""):
            digest.update(part)
    return digest.hexdigest()


def fingerprint(tensor):
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
    ).hexdigest()


def exact(left, right):
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and torch.equal(
            left.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
            right.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
        )
    )


def tensor_manifest(named):
    return [
        dict(name=n, shape=list(t.shape), dtype=str(t.dtype), sha256=fingerprint(t))
        for n, t in sorted(named)
    ]


def tensor_digest(named):
    digest = hashlib.sha256()
    for name, tensor in sorted(named):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(struct.pack("<Q", value.ndim))
        for dimension in value.shape:
            digest.update(struct.pack("<Q", int(dimension)))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def catalogue():
    return json.loads((RESULTS / "paper/catalogue.json").read_text())


def evidence_bytes(name):
    with zipfile.ZipFile(RESULTS / "paper/evidence.zip") as archive:
        data = archive.read(name)
    manifest = json.loads((RESULTS / "paper/evidence-members.json").read_text())
    assert hashlib.sha256(data).hexdigest() == manifest[name]["sha256"], name
    return data


def evidence(name):
    return json.loads(evidence_bytes(name))


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def prepare(key, cache, inputs):
    """Download only an exact pinned checkpoint, or validate a provided .pt input."""
    spec = catalogue()["inputs"][key]
    source = spec["source"]
    inputs, cache = Path(inputs), Path(cache)
    destination = inputs / (key + ".pt")
    if destination.exists():
        named = tuple(
            sorted(
                torch.load(destination, map_location="cpu", weights_only=True).items()
            )
        )
    else:
        cache.mkdir(parents=True, exist_ok=True)
        tensors = {}
        files = source.get("files", {})
        if source.get("kind") == "torchvision":
            (filename,) = files
            path = cache / filename
            if not path.exists():
                urllib.request.urlretrieve(source["url"], path)
            assert file_sha(path) == files[filename], filename
            state = torch.load(path, map_location="cpu", weights_only=True)
            tensors = {
                n: t.to(torch.bfloat16).contiguous()
                for n, t in state.items()
                if t.is_floating_point() and t.numel()
            }
        else:
            repo, revision = source["id"], source["revision"]
            assert len(revision) == 40, "A commit, not a mutable model tag, is required"
            folder = source.get("subfolder") or ""
            prefix = folder.rstrip("/") + "/" if folder else ""
            if not files:
                names = HfApi().list_repo_files(repo, revision=revision)
                index = next(
                    (
                        prefix + n
                        for n in (
                            "model.safetensors.index.json",
                            "pytorch_model.bin.index.json",
                        )
                        if prefix + n in names
                    ),
                    None,
                )
                if index:
                    path = hf_hub_download(
                        repo, index, revision=revision, cache_dir=cache
                    )
                    files = {
                        n: None
                        for n in sorted(
                            set(
                                json.loads(Path(path).read_text())[
                                    "weight_map"
                                ].values()
                            )
                        )
                    }
                elif (
                    prefix + "pytorch_model.bin" in names
                    and prefix + "model.safetensors" not in names
                ):
                    files = {"pytorch_model.bin": None}
                else:
                    files = {
                        n.removeprefix(prefix): None
                        for n in names
                        if n.startswith(prefix)
                        and "/" not in n.removeprefix(prefix)
                        and n.endswith((".safetensors", ".nemo"))
                    }
                assert files, "No pinned Safetensors/NeMo checkpoint found"
            for filename, digest in files.items():
                if not filename.endswith((".safetensors", ".nemo", ".bin", ".pth")):
                    continue
                path = hf_hub_download(
                    repo, prefix + filename, revision=revision, cache_dir=cache
                )
                if digest:
                    assert file_sha(path) == digest, filename
                if filename.endswith(".safetensors"):
                    with safe_open(path, framework="pt", device="cpu") as shard:
                        state = {n: shard.get_tensor(n) for n in shard.keys()}
                elif filename.endswith(".nemo"):
                    with tarfile.open(path) as archive:
                        (member,) = [
                            m
                            for m in archive.getmembers()
                            if Path(m.name).name == "model_weights.ckpt"
                        ]
                        with archive.extractfile(member) as stream:
                            state = torch.load(
                                stream, map_location="cpu", weights_only=True
                            )
                else:
                    state = torch.load(path, map_location="cpu", weights_only=True)
                for name, tensor in state.items():
                    if name.endswith(("featurizer.window", "featurizer.fb")) or (
                        key.startswith("gpt2")
                        and name.endswith((".attn.bias", ".attn.masked_bias"))
                    ):
                        continue
                    if filename.endswith(".nemo"):
                        for pattern, replacement in NEMO_MAPPING.items():
                            name = re.sub(pattern, replacement, name)
                    if tensor.is_floating_point() and tensor.numel():
                        assert name not in tensors, name
                        tensors[name] = tensor.to(torch.bfloat16).contiguous()
                del state
        named = tuple(sorted(tensors.items()))
    actual = tensor_manifest(named)
    assert (
        hashlib.sha256(json.dumps(actual, sort_keys=True).encode()).hexdigest()
        == spec["bf16_manifest_sha256"]
    ), key
    assert (
        len(named) == spec["tensors"]
        and sum(t.numel() * 2 for _, t in named) == spec["raw_bytes"]
    )
    assert all(t.dtype == torch.bfloat16 for _, t in named)
    checkpoint = catalogue()["checkpoints"].get(key)
    if checkpoint:
        assert tensor_digest(named) == checkpoint["input_digest"], key
    if not destination.exists():
        inputs.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".pt.tmp")
        assert not temporary.exists(), (
            "Preserve incomplete preparation; use a new inputs directory"
        )
        torch.save(dict(named), temporary)
        temporary.replace(destination)
    return named


def load_model(spec, named, cache):
    """Published construction, official renamings/conversions and bitwise loading."""
    control = (
        spec.get("key")
        if spec.get("key") in catalogue()["six"]
        else spec.get("control_key")
    )
    config = None
    if spec.get("kind") == "torchvision":
        assert vision_models is not None, "Install the experiment dependencies"
        model = (
            getattr(vision_models, spec["constructor"])(weights=None).eval().bfloat16()
        )
    elif spec.get("subfolder"):
        assert UNet2DConditionModel is not None, (
            "Install diffusers for the UNet experiment"
        )
        config = UNet2DConditionModel.load_config(
            spec["id"],
            subfolder=spec["subfolder"],
            revision=spec["revision"],
            cache_dir=cache,
        )
        model = UNet2DConditionModel.from_config(config).eval().bfloat16()
    else:
        config = AutoConfig.from_pretrained(
            spec["id"], revision=spec["revision"], cache_dir=cache
        )
        if control == "parakeet":
            model = (
                ParakeetForTDT.from_pretrained(
                    spec["id"],
                    revision=spec["revision"],
                    cache_dir=cache,
                    dtype=torch.bfloat16,
                    attn_implementation="eager",
                )
                .eval()
                .bfloat16()
            )
            assert all(
                n in model.state_dict() and exact(t, model.state_dict()[n])
                for n, t in named.items()
            )
            return model, ()
        kind = config.model_type
        factory = (
            AutoModelForImageTextToText
            if "qwen3_5" in kind
            or ("gemma" in kind and hasattr(config, "vision_config"))
            else AutoModelForSpeechSeq2Seq
            if kind == "whisper"
            else (
                AutoModelForPreTraining
                if any(n.startswith("cls.seq_relationship.") for n in named)
                else AutoModelForMaskedLM
            )
            if kind == "bert"
            else AutoModel
            if control in ("dinov3-convnext", "dinov3-vit", "siglip2")
            or "siglip" in kind
            else AutoModelForCausalLM
        )
        # Meta is scoped only to construction, never assignment (including ARM hosts).
        with (
            init_empty_weights(include_buffers=False)
            if init_empty_weights
            else nullcontext()
        ):
            model = (
                factory.from_config(
                    config, dtype=torch.bfloat16, attn_implementation="eager"
                )
                .eval()
                .bfloat16()
            )
    transforms = (
        get_model_conversion_mapping(model)
        if config is not None and not spec.get("subfolder")
        else ()
    )
    renamings = [t for t in transforms if isinstance(t, WeightRenaming)]
    converters = [t for t in transforms if isinstance(t, WeightConverter)]
    patterns = getattr(model, "_keys_to_ignore_on_load_unexpected", None) or ()
    ignored = tuple(n for n in named if any(re.match(p, n) for p in patterns))
    checkpoint = {n: t for n, t in named.items() if n not in ignored}
    state_names = set(model.state_dict())
    if any(
        n not in state_names
        and rename_source_key(n, renamings, converters)[1] is not None
        for n in checkpoint
    ):
        settings = LoadStateDictConfig(
            dtype=torch.bfloat16, device_map={"": "cpu"}, weight_mapping=transforms
        )
        status, _ = convert_and_load_state_dict_in_model(
            model, checkpoint, settings, tp_plan=None
        )
        assert not (
            status.unexpected_keys
            or status.mismatched_keys
            or status.conversion_errors
            or status.error_msgs
        ), status
        model.tie_weights()
        loaded = {
            id(p)
            for n, p in model.named_parameters(remove_duplicate=False)
            if n not in status.missing_keys
        }
        assert all(
            n not in status.missing_keys or id(p) in loaded
            for n, p in model.named_parameters(remove_duplicate=False)
        ), status
        restored = revert_weight_conversion(model, model.state_dict())
        assert all(
            n in restored and exact(restored[n], t) for n, t in checkpoint.items()
        ), "Expert conversion changed BF16 bits"
    else:
        loaded = {}
        for name, tensor in checkpoint.items():
            name = name.replace(".LayerNorm.gamma", ".LayerNorm.weight").replace(
                ".LayerNorm.beta", ".LayerNorm.bias"
            )
            if name not in state_names:
                for transform in renamings:
                    name, _ = transform.rename_source_key(name)
            if name not in state_names:
                name = (
                    (getattr(model, "base_model_prefix", "model") or "model")
                    + "."
                    + name
                )
            assert name not in loaded, name
            loaded[name] = tensor
        status = model.load_state_dict(loaded, assign=True, strict=False)
        if hasattr(model, "tie_weights"):
            model.tie_weights()
        assert not status.unexpected_keys, status
        ids = {
            id(p)
            for n, p in model.named_parameters(remove_duplicate=False)
            if n in loaded
        }
        assert all(
            n in loaded or id(p) in ids
            for n, p in model.named_parameters(remove_duplicate=False)
        ), status
        state = model.state_dict()
        assert all(n in state and exact(state[n], t) for n, t in loaded.items())
    return model, ignored


def inputs_for(key, model):
    source = catalogue()["inputs"][key]["source"]
    control = key if key in catalogue()["six"] else source.get("control_key")
    generator = torch.Generator().manual_seed(7)
    if source.get("kind") == "torchvision":
        return {
            "__pixels__": torch.randn(
                1, 3, 224, 224, generator=generator, dtype=torch.bfloat16
            ).cuda()
        }
    if source.get("subfolder"):
        return dict(
            sample=torch.randn(
                1, 4, 64, 64, generator=generator, dtype=torch.bfloat16
            ).cuda(),
            timestep=torch.tensor(500, device="cuda"),
            encoder_hidden_states=torch.randn(
                1,
                77,
                model.config.cross_attention_dim,
                generator=generator,
                dtype=torch.bfloat16,
            ).cuda(),
        )
    if control == "parakeet":
        return dict(
            input_features=torch.randn(
                1, 512, 128, generator=generator, dtype=torch.bfloat16
            ).cuda(),
            attention_mask=torch.ones(1, 512, dtype=torch.long, device="cuda"),
            decoder_input_ids=torch.ones(1, 1, dtype=torch.long, device="cuda"),
            use_decoder_cache=False,
        )
    if control in ("dinov3-convnext", "dinov3-vit", "siglip2"):
        values = {
            "pixel_values": torch.randn(
                1, 3, 224, 224, generator=generator, dtype=torch.bfloat16
            ).cuda()
        }
        if control == "siglip2":
            values["input_ids"] = torch.randint(
                model.config.text_config.vocab_size, (1, 64), generator=generator
            ).cuda()
        return values
    config = getattr(model.config, "text_config", model.config)
    if model.config.model_type == "whisper":
        return dict(
            input_features=torch.randn(
                1,
                model.config.num_mel_bins,
                3000,
                generator=generator,
                dtype=torch.bfloat16,
            ).cuda(),
            decoder_input_ids=torch.ones(1, 1, dtype=torch.long, device="cuda"),
            use_cache=False,
        )
    values = dict(
        input_ids=torch.randint(config.vocab_size, (1, 128), generator=generator).cuda()
    )
    if model.config.model_type != "bert":
        values["use_cache"] = False
    if "siglip" in model.config.model_type:
        values.pop("use_cache", None)
        values["input_ids"] = values["input_ids"][:, :64]
        size = model.config.vision_config.image_size
        values["pixel_values"] = torch.randn(
            1, 3, size, size, generator=generator, dtype=torch.bfloat16
        ).cuda()
    return values


def outputs_for(output):
    if isinstance(output, torch.Tensor):
        return {"output": output.detach().cpu()}
    return {
        k: v.detach().cpu() for k, v in output.items() if isinstance(v, torch.Tensor)
    }
