"""Measured optimizer experiment: MLP, CIFAR ResNet18, real ViT-B/16 and GPT-2."""

import csv
import gzip
import hashlib
import io
import json
import math
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import CIFAR10, CIFAR100, MNIST
from torchvision.models import resnet18, vit_b_16
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast

from _experiment_inputs import evidence, evidence_bytes

MODELS = ("mlp", "resnet18", "vit", "gpt2")
OPTIMIZERS = ("SGD", "Adam", "AdamW", "Muon")
LR = {
    "mlp": dict(SGD=0.05, Adam=0.001, AdamW=0.001, Muon=0.001),
    "resnet18": dict(SGD=0.05, Adam=0.001, AdamW=0.001, Muon=0.001),
    "vit": dict(SGD=0.05, Adam=0.0003, AdamW=0.0003, Muon=0.0003),
    "gpt2": dict(SGD=0.1, Adam=0.0003, AdamW=0.0003, Muon=0.0003),
}
TOKENIZER_REPO = "openai-community/gpt2"
TOKENIZER_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
MICROBATCH = {"mlp": 128, "resnet18": 128, "vit": 8, "gpt2": 4}
TEXT_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
TEXT_SHA = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def entropy(counts):
    counts = np.asarray(counts, dtype=np.int64)
    probabilities = counts[counts > 0].astype(np.float64) / counts.sum()
    return float(-np.sum(probabilities * np.log2(probabilities)))


@torch.no_grad()
def exponent_counts(values):
    # A new BF16 tensor: measurement does not quantize the trainable FP32 weights.
    bits = (
        values.detach()
        .contiguous()
        .to(torch.bfloat16)
        .view(torch.int16)
        .cpu()
        .numpy()
        .view(np.uint16)
    )
    return np.bincount(((bits >> 7) & 255).reshape(-1), minlength=256)


@torch.no_grad()
def measure(model):
    # Exact pooled histogram without allocating a full FP32 weight concatenation.
    counts = np.zeros(256, dtype=np.int64)
    elements = 0
    for parameter in model.parameters():
        flat = parameter.detach().reshape(-1)
        for part in flat.split(1 << 20):
            counts += exponent_counts(part)
        elements += parameter.numel()
    assert counts.sum() == elements and counts[255] == 0
    return counts


def state_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(json.dumps([name, list(value.shape), str(value.dtype)]).encode())
        digest.update(
            value.detach()
            .cpu()
            .contiguous()
            .reshape(-1)
            .view(torch.uint8)
            .numpy()
            .tobytes()
        )
    return digest.hexdigest()


def make_model(key, seed, device):
    torch.manual_seed(seed)
    if key == "mlp":
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )
    elif key == "resnet18":
        model = resnet18(weights=None, num_classes=10)
        model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    elif key == "vit":
        model = vit_b_16(weights=None, num_classes=100)
        assert model.image_size == 224 and model.patch_size == 16
        assert len(model.encoder.layers) == 12 and model.hidden_dim == 768
        assert model.mlp_dim == 3072 and model.encoder.layers[0].num_heads == 12
        assert sum(p.numel() for p in model.parameters()) == 85875556
    else:
        config = GPT2Config(use_cache=False)
        config._attn_implementation = "eager"
        model = GPT2LMHeadModel(config)
        assert (
            config.vocab_size,
            config.n_positions,
            config.n_embd,
            config.n_layer,
            config.n_head,
        ) == (50257, 1024, 768, 12, 12)
        assert model.transformer.wte.weight is model.lm_head.weight
        assert sum(p.numel() for p in model.parameters()) == 124439808
    return model.to(device=device, dtype=torch.float32)


def muon_hidden(key, name, parameter):
    if key == "mlp":
        return parameter.ndim == 2 and name in ("1.weight", "3.weight")
    if key == "resnet18":
        return parameter.ndim == 4 and name.startswith("layer")
    if key == "vit":
        return parameter.ndim == 2 and name.startswith("encoder.layers.")
    return parameter.ndim == 2 and name.startswith("transformer.h.")


def make_optimizers(model, key, kind):
    named = list(model.named_parameters())
    regularized = [p for _, p in named if p.ndim >= 2]
    other = [p for _, p in named if p.ndim < 2]
    groups = [
        dict(params=regularized, weight_decay=0.01),
        dict(params=other, weight_decay=0.0),
    ]
    lr = LR[key][kind]
    aliases, muon_names, auxiliary_names = [], [], []
    if kind == "SGD":
        optimizers = [torch.optim.SGD(groups, lr=lr, momentum=0.9)]
    elif kind == "Adam":
        optimizers = [torch.optim.Adam(groups, lr=lr, betas=(0.9, 0.999))]
    elif kind == "AdamW":
        optimizers = [torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.999))]
    else:
        matrices, auxiliary_decay, auxiliary_other = [], [], []
        for name, p in named:
            if muon_hidden(key, name, p):
                if p.ndim == 4:
                    # Standard convolution filter matrix: [out_channels, in*kh*kw].
                    # This is a storage-sharing view, not a changed architecture.
                    matrix = nn.Parameter(p.detach().flatten(1))
                    assert matrix.data_ptr() == p.data_ptr()
                    aliases.append((p, matrix))
                else:
                    matrix = p
                matrices.append(matrix)
                muon_names.append(name)
            else:
                (auxiliary_decay if p.ndim >= 2 else auxiliary_other).append(p)
                auxiliary_names.append(name)
        assert matrices and len(muon_names) + len(auxiliary_names) == len(named)
        optimizers = [
            torch.optim.Muon(
                matrices,
                lr=lr,
                weight_decay=0.01,
                momentum=0.95,
                ns_steps=5,
                adjust_lr_fn="match_rms_adamw",
            ),
            torch.optim.AdamW(
                [
                    dict(params=auxiliary_decay, weight_decay=0.01),
                    dict(params=auxiliary_other, weight_decay=0.0),
                ],
                lr=lr,
                betas=(0.9, 0.999),
            ),
        ]
    return (
        optimizers,
        aliases,
        dict(
            base_lr=lr,
            muon_parameters=muon_names,
            auxiliary_adamw_parameters=auxiliary_names,
        ),
    )


def load_small_data(key, cache, device):
    cache.mkdir(parents=True, exist_ok=True)
    dataset = MNIST if key == "mlp" else CIFAR10
    train = dataset(str(cache), train=True, download=True)
    heldout = dataset(str(cache), train=False, download=True)
    x = torch.as_tensor(np.asarray(train.data)).to(device=device, dtype=torch.float32)
    vx = torch.as_tensor(np.asarray(heldout.data)[:1024]).to(
        device=device, dtype=torch.float32
    )
    if key == "mlp":
        x, vx = x.unsqueeze(1), vx.unsqueeze(1)
        mean, std = 0.1307, 0.3081
    else:
        x, vx = x.permute(0, 3, 1, 2).contiguous(), vx.permute(0, 3, 1, 2).contiguous()
        mean = torch.tensor([0.4914, 0.4822, 0.4465], device=device)[
            None, :, None, None
        ]
        std = torch.tensor([0.247, 0.2435, 0.2616], device=device)[None, :, None, None]
    x, vx = (x / 255.0 - mean) / std, (vx / 255.0 - mean) / std
    y = torch.tensor(train.targets, device=device, dtype=torch.long)
    vy = torch.tensor(heldout.targets[:1024], device=device, dtype=torch.long)
    files = {
        str(p.relative_to(cache)): sha(p)
        for p in cache.rglob("*")
        if p.is_file() and ("MNIST" in str(p) if key == "mlp" else "cifar" in str(p))
    }
    return (
        x,
        y,
        vx,
        vy,
        dict(
            dataset=dataset.__name__,
            input_sha256=files,
            train_examples=len(x),
            evaluation_examples=len(vx),
            evaluation="first 1024 official test examples; no tuning on test",
            augmentation="none",
            preprocessing="fixed channel standardization",
        ),
    )


def load_large_data(key, cache, device):
    cache.mkdir(parents=True, exist_ok=True)
    if key == "gpt2":
        path = cache / "tinyshakespeare.txt"
        if not path.exists():
            with urllib.request.urlopen(TEXT_URL, timeout=60) as response:
                data = response.read(2_000_000)
            assert 1_000_000 < len(data) < 2_000_000
            path.write_bytes(data)
        assert sha(path) == TEXT_SHA, "The text input differs from the published study"
        tokenizer_folder = cache / "gpt2-tokenizer"
        snapshot_download(
            repo_id=TOKENIZER_REPO,
            revision=TOKENIZER_REVISION,
            allow_patterns=[
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.json",
                "merges.txt",
                "config.json",
            ],
            local_dir=tokenizer_folder,
            token=False,
        )
        tokenizer = GPT2TokenizerFast.from_pretrained(
            str(tokenizer_folder), local_files_only=True
        )
        assert len(tokenizer) == 50257
        data = np.asarray(
            tokenizer.encode(path.read_text(), add_special_tokens=False), dtype=np.int64
        )
        split = int(0.9 * len(data))
        train = torch.from_numpy(data[:split]).to(device=device, dtype=torch.long)
        heldout = torch.from_numpy(data[split:]).to(device=device, dtype=torch.long)
        generator = torch.Generator().manual_seed(991)
        starts = torch.randint(len(heldout) - 129, (256,), generator=generator).to(
            device
        )
        offsets = torch.arange(129, device=device)
        evaluation = heldout[starts[:, None] + offsets[None, :]]
        return (
            train,
            None,
            evaluation[:, :-1],
            evaluation[:, 1:],
            dict(
                dataset="Tiny Shakespeare, GPT-2 BPE next-token prediction",
                tokenizer_repo=TOKENIZER_REPO,
                tokenizer_revision=TOKENIZER_REVISION,
                tokenizer_files_sha256={
                    p.name: sha(p) for p in tokenizer_folder.iterdir() if p.is_file()
                },
                url=TEXT_URL,
                sha256=sha(path),
                split="contiguous 90% train / 10% held-out",
                train_tokens=len(train),
                heldout_tokens=len(heldout),
                evaluation_sequences=256,
            ),
        )
    dataset = CIFAR100
    train = dataset(str(cache), train=True, download=True)
    heldout = dataset(str(cache), train=False, download=True)
    x = torch.as_tensor(np.asarray(train.data)).to(device=device, dtype=torch.float32)
    vx = torch.as_tensor(np.asarray(heldout.data)[:1024]).to(
        device=device, dtype=torch.float32
    )
    x, vx = x.permute(0, 3, 1, 2).contiguous(), vx.permute(0, 3, 1, 2).contiguous()
    mean = torch.tensor([0.5, 0.5, 0.5], device=device)[None, :, None, None]
    std = torch.tensor([0.5, 0.5, 0.5], device=device)[None, :, None, None]
    x, vx = (x / 255.0 - mean) / std, (vx / 255.0 - mean) / std
    y = torch.tensor(train.targets, device=device, dtype=torch.long)
    vy = torch.tensor(heldout.targets[:1024], device=device, dtype=torch.long)
    files = {
        str(p.relative_to(cache)): sha(p)
        for p in (cache / "cifar-100-python").iterdir()
        if p.is_file()
    }
    return (
        x,
        y,
        vx,
        vy,
        dict(
            dataset=dataset.__name__,
            input_sha256=files,
            train_examples=len(x),
            evaluation_examples=len(vx),
            evaluation="first 1024 official test examples; no tuning on test",
            augmentation="none",
            preprocessing="x/255 then (x-0.5)/0.5 per channel; bilinear resize to 224x224, align_corners=False",
        ),
    )


def load_data(key, cache, device):
    return (load_small_data if key in ("mlp", "resnet18") else load_large_data)(
        key, cache, device
    )


def batch_indices(size, steps, batch_size, seed, language=False):
    generator = torch.Generator().manual_seed(seed + 10000)
    order, position = torch.empty(0, dtype=torch.long), 0
    for _ in range(steps):
        if language:
            yield torch.randint(size - 129, (batch_size,), generator=generator)
        else:
            if position + batch_size > len(order):
                order, position = torch.randperm(size, generator=generator), 0
            result = order[position : position + batch_size]
            position += batch_size
            yield result


def loss_for(model, key, x, y):
    if key == "vit":
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    logits = model(x).logits if key == "gpt2" else model(x)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1)), logits


@torch.no_grad()
def evaluate(model, key, x, y):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for start in range(
        0, len(x), (64 if key in ("mlp", "resnet18") else MICROBATCH[key])
    ):
        end = start + (64 if key in ("mlp", "resnet18") else MICROBATCH[key])
        loss, logits = loss_for(model, key, x[start:end], y[start:end])
        count = y[start:end].numel()
        total_loss += loss.item() * count
        correct += (logits.argmax(-1) == y[start:end]).sum().item()
        total += count
    model.train()
    return dict(heldout_loss=total_loss / total, heldout_accuracy=correct / total)


def run_one(root, key, kind, seed, steps, data):
    folder = root / f"{key}-{kind}-seed{seed}"
    if (folder / "summary.json").exists():
        value = json.loads((folder / "summary.json").read_text())
        assert value["steps"] == steps and value["runner_sha256"] == sha(__file__)
        print("preserved completed run", folder.name, flush=True)
        return
    folder.mkdir(parents=True, exist_ok=True)
    assert not (folder / "trace.jsonl.gz").exists(), (
        "Preserve partial runs; use a new output directory"
    )
    x, y, vx, vy, data_identity = data
    torch.cuda.reset_peak_memory_stats()
    model = make_model(key, seed, x.device)
    initial_digest = state_digest(model)
    optimizers, aliases, opt_info = make_optimizers(model, key, kind)
    manifest = [
        dict(name=n, shape=list(p.shape), dtype=str(p.dtype), elements=p.numel())
        for n, p in model.named_parameters()
    ]
    write_json(
        folder / "protocol.json",
        dict(
            model=key,
            optimizer=kind,
            seed=seed,
            steps=steps,
            batch_size=32 if key == "gpt2" else 128,
            microbatch_size=MICROBATCH[key],
            gradient_accumulation=key in ("vit", "gpt2"),
            architecture=dict(
                mlp="784-256-256-10 MLP",
                resnet18="CIFAR ResNet18, 3x3 stem and no max pool",
                vit="torchvision vit_b_16, 100-class head",
                gpt2="standard GPT-2 small",
            )[key],
            initial_state_sha256=initial_digest,
            parameters=manifest,
            data=data_identity,
            optimizer_configuration=opt_info,
            runner_sha256=sha(__file__),
        ),
    )
    order_hash, started = hashlib.sha256(), time.monotonic()
    batch_size = 32 if key == "gpt2" else 128
    rows = batch_indices(len(x), steps, batch_size, seed, language=key == "gpt2")
    fields = [
        "step",
        "entropy_bits",
        "train_loss",
        "heldout_loss",
        "heldout_accuracy",
        "gradient_norm_before_clipping",
        "elapsed_seconds",
    ]
    final_evaluation = None
    with (
        gzip.open(folder / "trace.jsonl.gz", "xt", compresslevel=5) as raw,
        (folder / "metrics.csv").open("x", newline="") as table,
    ):
        writer = csv.DictWriter(table, fieldnames=fields)
        writer.writeheader()
        for step in range(steps + 1):
            loss_value, gradient_norm = None, None
            if step:
                ids = next(rows)
                order_hash.update(ids.numpy().tobytes())
                ids = ids.to(x.device)
                if key == "gpt2":
                    sequence = x[
                        ids[:, None] + torch.arange(129, device=x.device)[None, :]
                    ]
                    bx, by = sequence[:, :-1], sequence[:, 1:]
                else:
                    bx, by = x[ids], y[ids]
                model.zero_grad(set_to_none=True)
                if key in ("mlp", "resnet18"):
                    loss, _ = loss_for(model, key, bx, by)
                    assert torch.isfinite(loss)
                    loss.backward()
                    loss_sum = loss.item()
                else:
                    loss_sum = 0.0
                    for start in range(0, len(bx), MICROBATCH[key]):
                        end = min(start + MICROBATCH[key], len(bx))
                        loss, logits = loss_for(
                            model, key, bx[start:end], by[start:end]
                        )
                        assert torch.isfinite(loss)
                        (loss * ((end - start) / len(bx))).backward()
                        loss_sum += loss.item() * ((end - start) / len(bx))
                        del loss, logits

                gradient_norm = float(
                    nn.utils.clip_grad_norm_(
                        model.parameters(), 1.0, error_if_nonfinite=True
                    )
                )
                for original, matrix in aliases:
                    matrix.grad = (
                        original.grad.flatten(1) if original.grad is not None else None
                    )
                scale = min(step / 50.0, 1.0)
                for optimizer in optimizers:
                    for group in optimizer.param_groups:
                        group["lr"] = opt_info["base_lr"] * scale
                    optimizer.step()
                loss_value = loss_sum
            counts = measure(model)
            value = entropy(counts)
            evaluation = {}
            if step % 100 == 0 or step == steps:
                evaluation = evaluate(model, key, vx, vy)
                final_evaluation = evaluation
            record = dict(
                step=step,
                entropy_bits=value,
                train_loss=loss_value,
                gradient_norm_before_clipping=gradient_norm,
                elapsed_seconds=time.monotonic() - started,
                **evaluation,
            )
            writer.writerow(record)
            raw.write(
                json.dumps(
                    dict(**record, exponent_counts=counts.tolist()),
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )
            if step % 100 == 0 or step == steps:
                table.flush()
                raw.flush()
                print(
                    key,
                    kind,
                    seed,
                    step,
                    "H",
                    round(value, 5),
                    "heldout",
                    evaluation,
                    "seconds",
                    round(record["elapsed_seconds"], 1),
                    flush=True,
                )
    summary = dict(
        model=key,
        optimizer=kind,
        seed=seed,
        steps=steps,
        initial_state_sha256=initial_digest,
        final_state_sha256=state_digest(model),
        minibatch_order_sha256=order_hash.hexdigest(),
        runner_sha256=sha(__file__),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        parameters=sum(p.numel() for p in model.parameters()),
        final_entropy_bits=value,
        final_evaluation=final_evaluation,
        elapsed_seconds=time.monotonic() - started,
        trace_sha256=sha(folder / "trace.jsonl.gz"),
        metrics_sha256=sha(folder / "metrics.csv"),
    )
    write_json(folder / "summary.json", summary)
    del model, optimizers, aliases
    torch.cuda.empty_cache()


def saved_trace(key, kind, seed):
    raw = evidence_bytes(f"optimizers/{key}-{kind}-seed{seed}/trace.jsonl.gz")
    return [json.loads(line) for line in gzip.decompress(raw).decode().splitlines()]


def check_saved(models=MODELS, optimizers=OPTIMIZERS, seeds=(17, 29, 43)):
    total = 0
    for key in models:
        for seed in seeds:
            initial, order, step_zero = set(), set(), []
            for kind in optimizers:
                base = f"optimizers/{key}-{kind}-seed{seed}/"
                summary = evidence(base + "summary.json")
                protocol = evidence(base + "protocol.json")
                trace = saved_trace(key, kind, seed)
                metrics = list(
                    csv.DictReader(
                        io.StringIO(evidence_bytes(base + "metrics.csv").decode())
                    )
                )
                assert len(trace) == len(metrics) == summary["steps"] + 1 == 1001
                assert (
                    summary["initial_state_sha256"] == protocol["initial_state_sha256"]
                )
                assert (
                    hashlib.sha256(evidence_bytes(base + "trace.jsonl.gz")).hexdigest()
                    == summary["trace_sha256"]
                )
                assert (
                    hashlib.sha256(evidence_bytes(base + "metrics.csv")).hexdigest()
                    == summary["metrics_sha256"]
                )
                assert summary["parameters"] == sum(
                    p["elements"] for p in protocol["parameters"]
                )
                for step, (raw, row) in enumerate(zip(trace, metrics, strict=True)):
                    counts = np.asarray(raw["exponent_counts"], dtype=np.int64)
                    assert raw["step"] == int(row["step"]) == step
                    assert (
                        counts.shape == (256,)
                        and np.all(counts >= 0)
                        and counts[255] == 0
                    )
                    assert counts.sum() == summary["parameters"]
                    assert abs(entropy(counts) - raw["entropy_bits"]) < 1e-12
                    assert float(row["entropy_bits"]) == raw["entropy_bits"]
                    if step:
                        assert math.isfinite(raw["train_loss"]) and math.isfinite(
                            raw["gradient_norm_before_clipping"]
                        )
                assert trace[-1]["entropy_bits"] == summary["final_entropy_bits"]
                initial.add(summary["initial_state_sha256"])
                order.add(summary["minibatch_order_sha256"])
                step_zero.append(trace[0]["exponent_counts"])
                total += len(trace)
            assert len(initial) == len(order) == 1
            assert all(counts == step_zero[0] for counts in step_zero)
    print(
        f"PASS: {total} exact optimizer histograms, CSVs, initial states and minibatch orders"
    )


def verify_prefix(root, key, kind, seed):
    folder = root / f"{key}-{kind}-seed{seed}"
    summary = json.loads((folder / "summary.json").read_text())
    old = evidence(f"optimizers/{folder.name}/summary.json")
    assert summary["initial_state_sha256"] == old["initial_state_sha256"]
    with gzip.open(folder / "trace.jsonl.gz", "rt") as stream:
        actual = [json.loads(line) for line in stream]
    expected = saved_trace(key, kind, seed)
    for row, ref in zip(actual, expected, strict=False):
        assert (
            row["step"] == ref["step"]
            and row["exponent_counts"] == ref["exponent_counts"]
        ), row["step"]
        assert row["entropy_bits"] == ref["entropy_bits"], row["step"]
    assert len(actual) == summary["steps"] + 1 <= len(expected)
    return dict(
        status="PASS",
        steps=summary["steps"],
        histograms=len(actual),
        initial_state_exact=True,
        exponent_histograms_exact=True,
        timing_compared=False,
    )


def export_curves(models, optimizers, seeds, root, saved=False):
    root.mkdir(parents=True, exist_ok=True)
    for key in models:
        curves = {}
        for kind in optimizers:
            runs = []
            for seed in seeds:
                if saved:
                    rows = saved_trace(key, kind, seed)
                else:
                    with gzip.open(
                        root / f"{key}-{kind}-seed{seed}" / "trace.jsonl.gz", "rt"
                    ) as stream:
                        rows = [json.loads(line) for line in stream]
                runs.append([r["entropy_bits"] for r in rows])
            values = np.asarray(runs)
            curves[kind] = (
                values.mean(0),
                values.std(0, ddof=1) if len(seeds) > 1 else np.zeros(values.shape[1]),
            )
        names = ["step"] + [
            f"{kind}_{stat}"
            for kind in optimizers
            for stat in ("mean", "sd", "low", "high")
        ]
        with (root / f"optimizer-{key}.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=names)
            writer.writeheader()
            for step in range(len(next(iter(curves.values()))[0])):
                row = dict(step=step)
                for kind, (mean, sd) in curves.items():
                    row.update(
                        {
                            f"{kind}_mean": mean[step],
                            f"{kind}_sd": sd[step],
                            f"{kind}_low": mean[step] - sd[step],
                            f"{kind}_high": mean[step] + sd[step],
                        }
                    )
                writer.writerow(row)
