"""GPT-2/BERT checkpoint trajectories for WikiText-2.

Run independent ``--train-only`` jobs in separate output directories when
using multiple GPUs, then gather both model directories under one root and run
one joint ``--compression-only`` process.  Compression writes a shared CSV and
must not be launched concurrently against the same output directory.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import time
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from examples.checkpoint_common import (
    add_checkpoint_run_arguments,
    atomic_torch_save,
    checkpoint_files,
    initialize_checkpoint_run,
    limited,
    load_or_fit_frozen_codec,
    missing_checkpoint_slugs,
    restore_rng_state,
    rng_state,
    run_checkpoint_models,
    run_checkpoint_trajectory,
    set_seed,
    upsert_model_epoch_row,
    validate_checkpoint_results,
)
from examples.checkpoint_compression import (
    BASE_RESULT_FIELDS,
    CODEC_PROTOCOL_VERSION,
    benchmark_checkpoint_codecs,
    completed_checkpoint_methods,
    floating_state,
    is_current_codec_result,
    nonfloating_state,
    tensor_digest,
)
from neuralzip.checkpoint_loading import load_torch_checkpoint
from neuralzip.experiment_io import atomic_write_json

DEFAULT_MODELS = ("gpt2", "bert-base")
RESULT_FIELDS = BASE_RESULT_FIELDS + (
    "objective",
    "sequence_length",
    "state_tensor_count",
    "storage_alias_count",
    "storage_alias_metadata_bytes",
    "checkpoint_tensor_digest",
    "epoch0_setup_digest",
    "compression_protocol_digest",
)
TRAINING_FIELDS = (
    "model",
    "model_label",
    "epoch",
    "objective",
    "train_loss",
    "train_accuracy",
    "test_loss",
    "test_accuracy",
    "test_perplexity",
    "learning_rate",
    "train_tokens",
    "test_tokens",
    "train_seconds",
    "eval_seconds",
    "checkpoint_path",
)


@dataclass(frozen=True)
class LanguageModelSpec:
    slug: str
    label: str
    tokenizer_id: str
    objective: str
    sequence_length: int
    batch_size: int
    learning_rate: float
    weight_decay: float


MODEL_SPECS = {
    spec.slug: spec
    for spec in (
        LanguageModelSpec(
            "gpt2",
            "GPT-2",
            "openai-community/gpt2",
            "causal-lm",
            256,
            8,
            5e-4,
            0.01,
        ),
        LanguageModelSpec(
            "bert-base",
            "BERT Base",
            "google-bert/bert-base-uncased",
            "masked-lm-15pct",
            128,
            16,
            5e-4,
            0.01,
        ),
    )
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train GPT-2 Small and BERT Base from random initialization on real "
            "WikiText-2, save exact FP32 epoch trajectories, and benchmark ZipNN "
            "src against a NeuralZip setup built at epoch 0."
        )
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODEL_SPECS),
        default=list(DEFAULT_MODELS),
    )
    add_checkpoint_run_arguments(
        parser,
        epochs=20,
        batch_size=None,
        batch_size_help="Override the per-model defaults (GPT-2=8, BERT Base=16).",
        data_dir=ROOT / "results" / "datasets" / "wikitext",
        output_dir=ROOT / "results" / "checkpoints-wikitext2",
    )
    parser.add_argument(
        "--download",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download Salesforce/wikitext parquet shards when absent.",
    )
    parser.add_argument("--train-block-limit", type=int)
    parser.add_argument("--validation-block-limit", type=int)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Use tiny randomly initialized models and synthetic token blocks. "
            "This only validates the code path and is never thesis data."
        ),
    )
    return parser.parse_args()


def upsert_result(path: Path, row: Mapping) -> None:
    upsert_model_epoch_row(
        path,
        row,
        fields=RESULT_FIELDS,
        model_order=DEFAULT_MODELS,
        include_compressor=True,
        lineterminator="\n",
    )


def upsert_training(path: Path, row: Mapping) -> None:
    upsert_model_epoch_row(
        path,
        row,
        fields=TRAINING_FIELDS,
        model_order=DEFAULT_MODELS,
        include_compressor=False,
        lineterminator="\n",
    )


def update_progress(path: Path, **updates) -> None:
    progress = {}
    if path.is_file():
        progress = json.loads(path.read_text())
    progress.update(updates)
    progress["updated_unix_seconds"] = time.time()
    atomic_write_json(path, progress)


def _exact_tensor_storage_key(tensor: torch.Tensor) -> tuple:
    """Identify an exact tensor view without confusing overlapping views.

    Tied Transformer weights have the same storage, offset, shape and stride.
    Views that only overlap are deliberately kept as independent tensors.
    """

    if tensor.numel() == 0:
        return ("empty", id(tensor), str(tensor.dtype), tuple(tensor.shape))
    storage = tensor.untyped_storage()
    return (
        tensor.device.type,
        tensor.device.index,
        int(storage.data_ptr()),
        int(storage.nbytes()),
        int(tensor.storage_offset()),
        tuple(int(value) for value in tensor.shape),
        tuple(int(value) for value in tensor.stride()),
        str(tensor.dtype),
    )


def unique_cpu_model_state(
    model: nn.Module,
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Return one FP32 CPU tensor per exact storage plus its alias manifest."""

    canonical: dict[str, torch.Tensor] = {}
    aliases: dict[str, str] = {}
    seen: dict[tuple, str] = {}
    for name, value in model.state_dict(keep_vars=True).items():
        tensor = value.detach()
        key = _exact_tensor_storage_key(tensor)
        if key in seen:
            aliases[str(name)] = seen[key]
            continue
        canonical_name = str(name)
        seen[key] = canonical_name
        canonical[canonical_name] = tensor.cpu().contiguous().clone()

    invalid = {
        name: tensor.dtype
        for name, tensor in canonical.items()
        if tensor.is_floating_point() and tensor.dtype != torch.float32
    }
    if invalid:
        preview = ", ".join(
            f"{name}={dtype}" for name, dtype in list(invalid.items())[:8]
        )
        raise TypeError(
            f"Language checkpoints require exact FP32 model states; got {preview}."
        )
    return canonical, aliases


def expand_storage_aliases(
    canonical: Mapping[str, torch.Tensor],
    aliases: Mapping[str, str],
) -> dict[str, torch.Tensor]:
    expanded = dict(canonical)
    for alias, source in aliases.items():
        if alias in expanded:
            raise ValueError(
                f"Storage alias {alias!r} duplicates a canonical state key."
            )
        if source not in canonical:
            raise KeyError(
                f"Storage alias {alias!r} refers to missing tensor {source!r}."
            )
        expanded[str(alias)] = canonical[str(source)]
    return expanded


def checkpoint_unique_state(
    checkpoint: Mapping,
    *,
    require_alias_manifest: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    if require_alias_manifest and checkpoint.get("unique_storage_version") != 1:
        raise ValueError(
            "Language checkpoint lacks the versioned unique-storage manifest; "
            "refusing to double-count tied embedding/output weights."
        )
    state = checkpoint.get("model_state", checkpoint)
    if not isinstance(state, Mapping):
        raise TypeError("Checkpoint model_state is not a mapping.")
    canonical = {
        str(name): tensor.detach().cpu().contiguous()
        for name, tensor in state.items()
        if torch.is_tensor(tensor)
    }
    aliases = {
        str(alias): str(source)
        for alias, source in checkpoint.get("storage_aliases", {}).items()
    }
    expand_storage_aliases(canonical, aliases)
    return canonical, aliases


def alias_digest(aliases: Mapping[str, str]) -> str:
    encoded = json.dumps(dict(sorted(aliases.items())), separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def alias_metadata_bytes(aliases: Mapping[str, str]) -> int:
    """Serialized bytes needed to restore omitted state-dict aliases."""

    return len(
        json.dumps(
            dict(sorted(aliases.items())),
            separators=(",", ":"),
        ).encode("utf-8")
    )


def stable_digest(value: Mapping) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def training_protocol(
    spec: LanguageModelSpec,
    args: argparse.Namespace,
    dataset_name: str,
    train_data: Dataset,
    validation_data: Dataset,
) -> dict:
    protocol = {
        "version": 1,
        "model": spec.slug,
        "model_label": spec.label,
        "tokenizer": None if args.smoke else spec.tokenizer_id,
        "objective": spec.objective,
        "sequence_length": 16 if args.smoke else spec.sequence_length,
        "batch_size": int(args.batch_size or spec.batch_size),
        "learning_rate": spec.learning_rate,
        "weight_decay": spec.weight_decay,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "epochs": int(args.epochs),
        "seed": int(args.seed),
        "dataset": dataset_name,
        "train_blocks": len(train_data),
        "validation_blocks": len(validation_data),
        "train_block_limit": args.train_block_limit,
        "validation_block_limit": args.validation_block_limit,
        "amp": not args.no_amp,
        "smoke": bool(args.smoke),
    }
    return {**protocol, "digest": stable_digest(protocol)}


def compression_protocol(args: argparse.Namespace) -> dict:
    protocol = {
        "version": 2,
        "codec_protocol_version": CODEC_PROTOCOL_VERSION,
        "threads": int(args.threads),
        "warmup_runs": 1,
        "measured_repetitions": 1,
        "zipnn": {
            "implementation": "src",
            "method": "HUFFMAN",
            "chunk_size": 256 << 10,
            "compression_threshold": 0.95,
        },
        "neuralzip": {
            "setup": "epoch-0-reused",
            "compression_threshold": 0.95,
        },
        "ratio_scope": "exponent symbols + raw fallback + metadata",
        "unique_storage": True,
        "alias_manifest_counted": True,
    }
    return {**protocol, "digest": stable_digest(protocol)}


def load_model_checkpoint(model: nn.Module, checkpoint: Mapping) -> None:
    canonical, aliases = checkpoint_unique_state(
        checkpoint,
        require_alias_manifest=True,
    )
    model.load_state_dict(expand_storage_aliases(canonical, aliases), strict=True)


class TokenBlockDataset(Dataset):
    def __init__(
        self,
        input_ids: torch.Tensor,
        special_tokens_mask: torch.Tensor | None = None,
    ) -> None:
        if input_ids.ndim != 2 or input_ids.dtype != torch.long:
            raise ValueError("input_ids must be a [blocks, sequence] int64 tensor.")
        if (
            special_tokens_mask is not None
            and special_tokens_mask.shape != input_ids.shape
        ):
            raise ValueError("special_tokens_mask must match input_ids.")
        self.input_ids = input_ids.contiguous()
        self.special_tokens_mask = (
            special_tokens_mask.to(torch.bool).contiguous()
            if special_tokens_mask is not None
            else torch.zeros_like(input_ids, dtype=torch.bool)
        )

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[int(index)],
            "special_tokens_mask": self.special_tokens_mask[int(index)],
        }


def find_wikitext2_parquet(data_dir: Path) -> tuple[Path, Path] | None:
    candidates = (
        data_dir / "wikitext-2-raw-v1",
        data_dir,
    )
    for root in candidates:
        train_path = root / "train-00000-of-00001.parquet"
        validation_path = root / "validation-00000-of-00001.parquet"
        if train_path.is_file() and validation_path.is_file():
            return train_path, validation_path
    return None


def ensure_wikitext2_parquet(data_dir: Path, *, download: bool) -> tuple[Path, Path]:
    found = find_wikitext2_parquet(data_dir)
    if found is not None:
        return found
    if not download:
        raise FileNotFoundError(
            f"WikiText-2 parquet shards were not found under {data_dir}. "
            "Pass --download or populate wikitext-2-raw-v1/."
        )
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError(
            "Downloading WikiText-2 requires the experiments dependency "
            "huggingface-hub."
        ) from error

    snapshot_download(
        repo_id="Salesforce/wikitext",
        repo_type="dataset",
        allow_patterns=["wikitext-2-raw-v1/*.parquet"],
        local_dir=data_dir,
    )
    found = find_wikitext2_parquet(data_dir)
    if found is None:
        raise FileNotFoundError(
            "Salesforce/wikitext was downloaded, but the expected "
            "wikitext-2-raw-v1 parquet shards are absent."
        )
    return found


def parquet_texts(path: Path) -> list[str]:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError(
            "Reading real WikiText-2 requires pyarrow; install the project's "
            "experiments dependencies."
        ) from error
    table = pq.read_table(path, columns=["text"])
    return [str(value) for value in table.column("text").to_pylist() if value]


def token_stream(texts: Iterable[str], tokenizer, *, append_eos: bool) -> list[int]:
    stream: list[int] = []
    batch: list[str] = []

    def flush() -> None:
        if not batch:
            return
        encoded = tokenizer(
            batch,
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )["input_ids"]
        for tokens in encoded:
            stream.extend(int(token) for token in tokens)
            if append_eos:
                stream.append(int(tokenizer.eos_token_id))
        batch.clear()

    for text in texts:
        if not text.strip():
            continue
        batch.append(text)
        if len(batch) >= 512:
            flush()
    flush()
    return stream


def build_real_token_blocks(
    spec: LanguageModelSpec,
    tokenizer,
    texts: Iterable[str],
) -> TokenBlockDataset:
    sequence_length = int(spec.sequence_length)
    if spec.objective == "causal-lm":
        stream = token_stream(texts, tokenizer, append_eos=True)
        block_count = len(stream) // sequence_length
        if block_count == 0:
            raise ValueError(
                "WikiText split is too small to form a causal token block."
            )
        values = torch.tensor(
            stream[: block_count * sequence_length], dtype=torch.long
        ).reshape(block_count, sequence_length)
        return TokenBlockDataset(values)

    payload_length = sequence_length - 2
    stream = token_stream(texts, tokenizer, append_eos=False)
    block_count = len(stream) // payload_length
    if block_count == 0:
        raise ValueError("WikiText split is too small to form a masked-LM token block.")
    payloads = torch.tensor(
        stream[: block_count * payload_length], dtype=torch.long
    ).reshape(block_count, payload_length)
    cls = torch.full((block_count, 1), int(tokenizer.cls_token_id), dtype=torch.long)
    sep = torch.full((block_count, 1), int(tokenizer.sep_token_id), dtype=torch.long)
    values = torch.cat((cls, payloads, sep), dim=1)
    special = torch.zeros_like(values, dtype=torch.bool)
    special[:, 0] = True
    special[:, -1] = True
    return TokenBlockDataset(values, special)


def smoke_token_blocks(
    spec: LanguageModelSpec,
    *,
    seed: int,
    train: bool,
) -> TokenBlockDataset:
    generator = torch.Generator().manual_seed(seed + (0 if train else 1))
    blocks = 8 if train else 4
    sequence_length = 16
    values = torch.randint(
        5,
        128,
        (blocks, sequence_length),
        generator=generator,
        dtype=torch.long,
    )
    if spec.objective == "causal-lm":
        return TokenBlockDataset(values)
    values[:, 0] = 1
    values[:, -1] = 2
    special = torch.zeros_like(values, dtype=torch.bool)
    special[:, 0] = True
    special[:, -1] = True
    return TokenBlockDataset(values, special)


def load_tokenizer(spec: LanguageModelSpec):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(spec.tokenizer_id, use_fast=True)
    if spec.objective == "causal-lm" and tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_model(spec: LanguageModelSpec, *, smoke: bool) -> nn.Module:
    from transformers import (
        BertConfig,
        BertForMaskedLM,
        GPT2Config,
        GPT2LMHeadModel,
    )

    if spec.slug == "gpt2":
        config = (
            GPT2Config(
                vocab_size=128,
                n_positions=32,
                n_ctx=32,
                n_embd=32,
                n_layer=1,
                n_head=2,
                bos_token_id=0,
                eos_token_id=0,
            )
            if smoke
            else GPT2Config(
                vocab_size=50257,
                n_positions=1024,
                n_ctx=1024,
                n_embd=768,
                n_layer=12,
                n_head=12,
                bos_token_id=50256,
                eos_token_id=50256,
            )
        )
        config.use_cache = False
        return GPT2LMHeadModel(config)
    if spec.slug == "bert-base":
        config = (
            BertConfig(
                vocab_size=128,
                hidden_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                intermediate_size=64,
                max_position_embeddings=32,
                pad_token_id=0,
            )
            if smoke
            else BertConfig(
                vocab_size=30522,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                max_position_embeddings=512,
                type_vocab_size=2,
                pad_token_id=0,
            )
        )
        config.mask_token_id = 4 if smoke else 103
        return BertForMaskedLM(config)
    raise ValueError(f"Unsupported language model: {spec.slug}")


def apply_mlm_mask(
    input_ids: torch.Tensor,
    special_tokens_mask: torch.Tensor,
    *,
    mask_token_id: int,
    vocab_size: int,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply BERT's 15% MLM selection with its 80/10/10 replacement rule."""

    masked_inputs = input_ids.clone()
    labels = input_ids.clone()
    probabilities = torch.full(input_ids.shape, 0.15, dtype=torch.float32)
    probabilities.masked_fill_(special_tokens_mask.to(torch.bool), 0.0)
    selected = torch.bernoulli(probabilities, generator=generator).to(torch.bool)
    labels[~selected] = -100

    replaced = (
        torch.bernoulli(
            torch.full(input_ids.shape, 0.8, dtype=torch.float32),
            generator=generator,
        ).to(torch.bool)
        & selected
    )
    masked_inputs[replaced] = int(mask_token_id)
    random_replacement = (
        torch.bernoulli(
            torch.full(input_ids.shape, 0.5, dtype=torch.float32),
            generator=generator,
        ).to(torch.bool)
        & selected
        & ~replaced
    )
    random_tokens = torch.randint(
        int(vocab_size),
        input_ids.shape,
        dtype=torch.long,
        generator=generator,
    )
    masked_inputs[random_replacement] = random_tokens[random_replacement]
    return masked_inputs, labels


def prepare_batch(
    batch: Mapping[str, torch.Tensor],
    spec: LanguageModelSpec,
    model: nn.Module,
    device: torch.device,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = batch["input_ids"]
    if spec.objective == "causal-lm":
        labels = input_ids.clone()
    else:
        input_ids, labels = apply_mlm_mask(
            input_ids,
            batch["special_tokens_mask"],
            mask_token_id=int(model.config.mask_token_id or 103),
            vocab_size=int(model.config.vocab_size),
            generator=generator,
        )
    return (
        input_ids.to(device, non_blocking=True),
        labels.to(device, non_blocking=True),
    )


def correct_token_counts(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    causal: bool,
) -> tuple[int, int]:
    if causal:
        predictions = logits[:, :-1].argmax(dim=-1)
        targets = labels[:, 1:]
    else:
        predictions = logits.argmax(dim=-1)
        targets = labels
    selected = targets.ne(-100)
    return (
        int(predictions[selected].eq(targets[selected]).sum().item()),
        int(selected.sum().item()),
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    spec: LanguageModelSpec,
    device: torch.device,
    *,
    optimizer=None,
    scaler=None,
    amp_enabled: bool,
    evaluation_seed: int | None = None,
) -> tuple[float, float, int]:
    training = optimizer is not None
    model.train(training)
    loss_total = 0.0
    correct = 0
    tokens = 0
    evaluation_generator = (
        torch.Generator().manual_seed(int(evaluation_seed))
        if evaluation_seed is not None
        else None
    )
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            input_ids, labels = prepare_batch(
                batch,
                spec,
                model,
                device,
                generator=evaluation_generator,
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                output = model(input_ids=input_ids, labels=labels)
                loss = output.loss
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            batch_correct, batch_tokens = correct_token_counts(
                output.logits.detach(),
                labels,
                causal=spec.objective == "causal-lm",
            )
            loss_total += float(loss.item()) * batch_tokens
            correct += batch_correct
            tokens += batch_tokens
    return loss_total / max(tokens, 1), correct / max(tokens, 1), tokens


def model_checkpoint_path(checkpoint_dir: Path, slug: str, epoch: int) -> Path:
    return checkpoint_dir / f"{slug}_wikitext2_epoch_{int(epoch):03d}.pt"


def make_optimizer(spec: LanguageModelSpec, model: nn.Module):
    return torch.optim.AdamW(
        model.parameters(),
        lr=spec.learning_rate,
        weight_decay=spec.weight_decay,
    )


def train_model(
    spec: LanguageModelSpec,
    train_data: Dataset,
    validation_data: Dataset,
    dataset_name: str,
    args: argparse.Namespace,
    model_dir: Path,
) -> Path:
    checkpoint_dir = model_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest_state_path = model_dir / "latest_training_state.pt"
    training_path = model_dir / "training_metrics.csv"
    progress_path = model_dir / "progress.json"
    device = torch.device(args.device)
    set_seed(args.seed)
    model = build_model(spec, smoke=args.smoke).to(device)
    optimizer = make_optimizer(spec, model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(args.epochs)),
    )
    amp_enabled = bool(device.type == "cuda" and not args.no_amp)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    loader_generator = torch.Generator().manual_seed(args.seed)
    loader_kwargs = {
        "num_workers": int(args.num_workers),
        "pin_memory": device.type == "cuda",
        "persistent_workers": int(args.num_workers) > 0,
    }
    batch_size = int(args.batch_size or spec.batch_size)
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        generator=loader_generator,
        **loader_kwargs,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    protocol = training_protocol(
        spec,
        args,
        dataset_name,
        train_data,
        validation_data,
    )

    start_epoch = 0
    if latest_state_path.is_file():
        latest = torch.load(latest_state_path, map_location=device, weights_only=False)
        if latest.get("training_protocol_digest") != protocol["digest"]:
            raise RuntimeError(
                f"[{spec.slug}] existing training state uses a different protocol; "
                "resume with the original arguments or use a fresh output directory."
            )
        if not model_checkpoint_path(checkpoint_dir, spec.slug, 0).is_file():
            raise FileNotFoundError(
                f"[{spec.slug}] latest training state exists but epoch 0 is missing."
            )
        start_epoch = int(latest["epoch"])
        load_model_checkpoint(model, latest)
        optimizer.load_state_dict(latest["optimizer_state"])
        scheduler.load_state_dict(latest["scheduler_state"])
        scaler.load_state_dict(latest.get("scaler_state", {}))
        restore_rng_state(latest["rng_state"], loader_generator)
        print(f"[{spec.slug}] resume training from epoch {start_epoch}", flush=True)
    elif any(epoch > 0 for epoch in checkpoint_files(checkpoint_dir, args.epochs)):
        raise FileNotFoundError(
            f"[{spec.slug}] partial checkpoints exist without latest_training_state.pt; "
            "use a fresh output directory rather than mixing trajectories."
        )

    initial_path = model_checkpoint_path(checkpoint_dir, spec.slug, 0)
    if not initial_path.is_file():
        canonical, aliases = unique_cpu_model_state(model)
        eval_started = time.perf_counter()
        test_loss, test_accuracy, test_tokens = run_epoch(
            model,
            validation_loader,
            spec,
            device,
            amp_enabled=amp_enabled,
            evaluation_seed=args.seed + 10_000,
        )
        eval_seconds = time.perf_counter() - eval_started
        atomic_torch_save(
            initial_path,
            {
                "epoch": 0,
                "architecture": spec.slug,
                "objective": spec.objective,
                "dataset": dataset_name,
                "seed": args.seed,
                "model_state": canonical,
                "storage_aliases": aliases,
                "unique_storage_version": 1,
                "checkpoint_dtype": "float32",
                "training_protocol_digest": protocol["digest"],
                "training_protocol": protocol,
            },
        )
        upsert_training(
            training_path,
            {
                "model": spec.slug,
                "model_label": spec.label,
                "epoch": 0,
                "objective": spec.objective,
                "train_loss": "",
                "train_accuracy": "",
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "test_perplexity": math.exp(min(test_loss, 20.0)),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train_tokens": 0,
                "test_tokens": test_tokens,
                "train_seconds": 0.0,
                "eval_seconds": eval_seconds,
                "checkpoint_path": str(initial_path),
            },
        )
        update_progress(
            progress_path,
            phase="training",
            status="running",
            completed_epoch=0,
            requested_epochs=args.epochs,
            storage_aliases=len(aliases),
            training_protocol_digest=protocol["digest"],
        )
        del canonical, aliases

    if start_epoch >= int(args.epochs):
        update_progress(
            progress_path,
            phase="training",
            status="complete",
            completed_epoch=start_epoch,
            requested_epochs=args.epochs,
            training_protocol_digest=protocol["digest"],
        )
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return checkpoint_dir

    for epoch in range(start_epoch + 1, int(args.epochs) + 1):
        started = time.perf_counter()
        train_loss, train_accuracy, train_tokens = run_epoch(
            model,
            train_loader,
            spec,
            device,
            optimizer=optimizer,
            scaler=scaler,
            amp_enabled=amp_enabled,
        )
        train_seconds = time.perf_counter() - started
        scheduler.step()
        eval_started = time.perf_counter()
        test_loss, test_accuracy, test_tokens = run_epoch(
            model,
            validation_loader,
            spec,
            device,
            amp_enabled=amp_enabled,
            evaluation_seed=args.seed + 10_000,
        )
        eval_seconds = time.perf_counter() - eval_started
        canonical, aliases = unique_cpu_model_state(model)
        checkpoint_path = model_checkpoint_path(checkpoint_dir, spec.slug, epoch)
        checkpoint = {
            "epoch": epoch,
            "architecture": spec.slug,
            "objective": spec.objective,
            "dataset": dataset_name,
            "seed": args.seed,
            "model_state": canonical,
            "storage_aliases": aliases,
            "unique_storage_version": 1,
            "checkpoint_dtype": "float32",
            "training_protocol_digest": protocol["digest"],
            "training_protocol": protocol,
        }
        atomic_torch_save(checkpoint_path, checkpoint)
        atomic_torch_save(
            latest_state_path,
            {
                **checkpoint,
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict(),
                "rng_state": rng_state(loader_generator),
            },
        )
        upsert_training(
            training_path,
            {
                "model": spec.slug,
                "model_label": spec.label,
                "epoch": epoch,
                "objective": spec.objective,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "test_perplexity": math.exp(min(test_loss, 20.0)),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train_tokens": train_tokens,
                "test_tokens": test_tokens,
                "train_seconds": train_seconds,
                "eval_seconds": eval_seconds,
                "checkpoint_path": str(checkpoint_path),
            },
        )
        update_progress(
            progress_path,
            phase="training",
            status="complete" if epoch == int(args.epochs) else "running",
            completed_epoch=epoch,
            requested_epochs=args.epochs,
            storage_aliases=len(aliases),
            training_protocol_digest=protocol["digest"],
        )
        print(
            f"[{spec.slug}] epoch={epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.4f} test_loss={test_loss:.4f} "
            f"seconds={train_seconds + eval_seconds:.1f}",
            flush=True,
        )
        del canonical, aliases, checkpoint

    del model, optimizer, scheduler, scaler, train_loader, validation_loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return checkpoint_dir


def add_alias_manifest_to_metrics(metrics: Mapping, aliases: Mapping[str, str]) -> dict:
    """Count the common alias manifest needed by either lossless codec."""

    output = dict(metrics)
    metadata_bytes = alias_metadata_bytes(aliases)
    output["compressed_bytes"] = int(output["compressed_bytes"]) + metadata_bytes
    output["exponent_compressed_bytes"] = (
        int(output["exponent_compressed_bytes"]) + metadata_bytes
    )
    output["full_ratio_pct"] = (
        100.0 * output["compressed_bytes"] / int(output["raw_bytes"])
        if int(output["raw_bytes"])
        else 0.0
    )
    output["exponent_ratio_pct"] = (
        100.0 * output["exponent_compressed_bytes"] / int(output["exponent_raw_bytes"])
        if int(output["exponent_raw_bytes"])
        else 0.0
    )
    return output


def benchmark_checkpoint(
    *,
    spec: LanguageModelSpec,
    epoch: int,
    path: Path,
    checkpoint_source: str,
    plan,
    setup: Mapping,
    args: argparse.Namespace,
    result_path: Path,
) -> None:
    checkpoint = load_torch_checkpoint(path)
    canonical, aliases = checkpoint_unique_state(
        checkpoint,
        require_alias_manifest=True,
    )
    floating = floating_state(canonical, error_prefix="Language checkpoints")
    nonfloating = nonfloating_state(canonical)
    named_tensors = list(floating.items())
    checkpoint_tensor_digest = tensor_digest(named_tensors)
    compression = compression_protocol(args)
    done = completed_checkpoint_methods(
        result_path,
        spec.slug,
        epoch,
        identities={
            "checkpoint_tensor_digest": checkpoint_tensor_digest,
            "epoch0_setup_digest": setup["codebook_digest"],
            "compression_protocol_digest": compression["digest"],
        },
    )
    if done == {"NeuralZip", "ZipNN"}:
        print(
            f"[{spec.slug}] epoch={epoch:03d} compression already complete", flush=True
        )
        return
    print(
        f"[{spec.slug}] epoch={epoch:03d} unique_tensors={len(canonical)} "
        f"aliases={len(aliases)} parameters={sum(t.numel() for t in floating.values()):,}",
        flush=True,
    )

    common = {
        "model": spec.slug,
        "model_label": spec.label,
        "epoch": epoch,
        "checkpoint_path": str(path),
        "checkpoint_source": checkpoint_source,
        "dtype": "float32",
        "tensor_count": len(floating),
        "nonfloating_tensor_count": len(nonfloating),
        "threads": args.threads,
        "warmup_runs": 1,
        "measured_repetitions": 1,
        "ratio_scope": "exponent symbols + raw fallback + metadata",
        "status": "ok",
        "objective": spec.objective,
        "sequence_length": 16 if args.smoke else spec.sequence_length,
        "state_tensor_count": len(canonical) + len(aliases),
        "storage_alias_count": len(aliases),
        "storage_alias_metadata_bytes": alias_metadata_bytes(aliases),
        "checkpoint_tensor_digest": checkpoint_tensor_digest,
        "epoch0_setup_digest": setup["codebook_digest"],
        "compression_protocol_digest": compression["digest"],
    }
    benchmark_checkpoint_codecs(
        slug=spec.slug,
        epoch=epoch,
        plan=plan,
        setup=setup,
        tensors=floating,
        common_row=common,
        completed=done,
        threads=args.threads,
        upsert_result=lambda row: upsert_result(result_path, row),
        transform_metrics=lambda metrics: add_alias_manifest_to_metrics(
            metrics,
            aliases,
        ),
    )


def frozen_codec_for_trajectory(
    spec: LanguageModelSpec,
    checkpoint_dir: Path,
    model_dir: Path,
    args: argparse.Namespace,
):
    def prepare(checkpoint):
        canonical, aliases = checkpoint_unique_state(
            checkpoint,
            require_alias_manifest=True,
        )
        floating = floating_state(canonical, error_prefix="Language checkpoints")
        return (
            list(floating.items()),
            {
                "epoch0_tensor_digest": tensor_digest(floating.items()),
                "epoch0_alias_digest": alias_digest(aliases),
            },
            {
                "storage_aliases": aliases,
                "unique_storage_compression": True,
            },
        )

    return load_or_fit_frozen_codec(
        slug=spec.slug,
        checkpoint_dir=checkpoint_dir,
        model_dir=model_dir,
        epochs=args.epochs,
        threads=args.threads,
        prepare_epoch_zero=prepare,
    )


def process_trajectory(
    spec: LanguageModelSpec,
    checkpoint_dir: Path,
    checkpoint_source: str,
    args: argparse.Namespace,
    result_path: Path,
) -> None:
    progress_path = args.output_dir / spec.slug / "progress.json"

    def update_compression_progress(epoch: int) -> None:
        update_progress(
            progress_path,
            phase="compression",
            status="complete" if epoch == int(args.epochs) else "running",
            completed_compression_epoch=epoch,
            requested_epochs=args.epochs,
            compression_protocol_digest=compression_protocol(args)["digest"],
        )

    run_checkpoint_trajectory(
        spec=spec,
        checkpoint_dir=checkpoint_dir,
        checkpoint_source=checkpoint_source,
        args=args,
        result_path=result_path,
        frozen_codec_for_trajectory=frozen_codec_for_trajectory,
        benchmark_checkpoint=benchmark_checkpoint,
        after_epoch=update_compression_progress,
    )


def validate_complete_results(
    result_path: Path,
    *,
    models: Iterable[str],
    epochs: int,
) -> None:
    validate_checkpoint_results(
        result_path,
        models=models,
        epochs=epochs,
        is_current_result=is_current_codec_result,
        error_label="Language checkpoint results",
    )


def write_metadata(args: argparse.Namespace, dataset_name: str) -> None:
    atomic_write_json(
        args.output_dir / "run_metadata.json",
        {
            "experiment": "language-training-checkpoint-suite",
            "models": [asdict(MODEL_SPECS[slug]) for slug in args.models],
            "dataset": dataset_name,
            "dataset_is_real": not args.smoke,
            "epochs": args.epochs,
            "seed": args.seed,
            "train_block_limit": args.train_block_limit,
            "validation_block_limit": args.validation_block_limit,
            "checkpoint_dtype": "exact FP32 model state",
            "checkpoint_schedule": "epoch 0 and the end of every epoch",
            "objectives": {
                "gpt2": "causal next-token prediction",
                "bert-base": "15% masked language modeling with 80/10/10 replacement",
            },
            "unique_storage": (
                "exact tied-weight views are stored and compressed once; aliases "
                "are recorded as metadata and reconstructed on load"
            ),
            "compression_schedule": "post-hoc isolated CPU benchmark of every saved epoch",
            "compressors": ["ZipNN 0.5.4", "NeuralZip epoch-0 setup"],
            "ratio_scope": "exponent symbols + raw fallback + metadata",
            "storage_alias_metadata": "counted once in both compressors",
            "compression_protocol": compression_protocol(args),
            "timing": {
                "threads": args.threads,
                "warmup_runs": 1,
                "measured_repetitions": 1,
                "neuralzip_setup": "once at epoch 0, outside encode time",
                "neuralzip_target_mapping": "outside encode time and reported separately",
                "zipnn_src_input_clone": (
                    "inside encode time; required because ZipNN 0.5.4 mutates its input"
                ),
            },
            "verification": (
                "bitwise round-trip for every unique FP32 floating tensor, input "
                "immutability, and explicit tied-weight alias reconstruction"
            ),
            "smoke_only": args.smoke,
            "concurrency": (
                "shared checkpoint_results.csv is single-writer; use parallel "
                "train-only directories followed by one joint compression process"
            ),
        },
    )


def datasets_for_model(
    spec: LanguageModelSpec,
    args: argparse.Namespace,
    parquet_paths: tuple[Path, Path] | None,
) -> tuple[Dataset, Dataset, str]:
    if args.smoke:
        return (
            smoke_token_blocks(spec, seed=args.seed, train=True),
            smoke_token_blocks(spec, seed=args.seed, train=False),
            "synthetic-token-blocks-smoke-only",
        )
    if parquet_paths is None:
        raise RuntimeError("Real WikiText-2 parquet paths were not initialized.")
    tokenizer = load_tokenizer(spec)
    train_path, validation_path = parquet_paths
    train = build_real_token_blocks(spec, tokenizer, parquet_texts(train_path))
    validation = build_real_token_blocks(
        spec,
        tokenizer,
        parquet_texts(validation_path),
    )
    return train, validation, "Salesforce/wikitext:wikitext-2-raw-v1"


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.epochs = min(int(args.epochs), 1)
        args.batch_size = min(int(args.batch_size or 2), 2)
        args.num_workers = 0
        args.device = "cpu"
        args.output_dir = args.output_dir / "smoke"
    result_path = initialize_checkpoint_run(args)
    parquet_paths = None
    if (
        not args.compression_only
        and missing_checkpoint_slugs(
            slugs=args.models,
            external_root=args.existing_checkpoint_root,
            output_dir=args.output_dir,
            epochs=args.epochs,
        )
        and not args.smoke
    ):
        if not args.train_missing:
            raise FileNotFoundError(
                "Requested language checkpoints are missing and --no-train-missing was set."
            )
        parquet_paths = ensure_wikitext2_parquet(
            args.data_dir,
            download=args.download,
        )
    dataset_name = (
        "synthetic-token-blocks-smoke-only"
        if args.smoke
        else "Salesforce/wikitext:wikitext-2-raw-v1"
    )
    write_metadata(args, dataset_name)

    def train_trajectory(spec: LanguageModelSpec) -> Path:
        train_data, validation_data, model_dataset_name = datasets_for_model(
            spec,
            args,
            parquet_paths,
        )
        train_data = limited(train_data, args.train_block_limit)
        validation_data = limited(validation_data, args.validation_block_limit)
        checkpoint_dir = train_model(
            spec,
            train_data,
            validation_data,
            model_dataset_name,
            args,
            args.output_dir / spec.slug,
        )
        del train_data, validation_data
        gc.collect()
        return checkpoint_dir

    run_checkpoint_models(
        args=args,
        specs=MODEL_SPECS,
        result_path=result_path,
        local_source="trained-by-language-runner",
        train_trajectory=train_trajectory,
        process_trajectory=process_trajectory,
        validate_results=validate_complete_results,
    )


if __name__ == "__main__":
    main()
