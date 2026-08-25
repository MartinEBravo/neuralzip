from __future__ import annotations

import argparse
import gc
import io
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from examples.checkpoint_common import (  # noqa: E402
    atomic_torch_save,
    checkpoint_files,
    find_complete_local_checkpoint_dir,
    find_external_checkpoint_dir,
    limited,
    read_csv,
    restore_rng_state,
    rng_state,
    set_seed,
    upsert_model_epoch_row,
)
from neuralzip.experiment_io import atomic_write_json  # noqa: E402
from examples.checkpoint_compression import (  # noqa: E402
    BASE_RESULT_FIELDS,
    CODEC_PROTOCOL_VERSION,
    ENCODER_PATHS,
    benchmark_neuralzip,
    benchmark_zipnn_src,
    is_current_codec_result,
    load_codec_plan,
    nonfloating_state,
    save_codec_plan,
    setup_neuralzip,
    tensor_digest,
)


DEFAULT_MODELS = (
    "resnet18",
    "resnet34",
    "convnext_tiny",
    "efficientnet_b0",
)
RESULT_FIELDS = BASE_RESULT_FIELDS
TRAINING_FIELDS = (
    "model",
    "model_label",
    "epoch",
    "train_loss",
    "train_accuracy",
    "test_loss",
    "test_accuracy",
    "learning_rate",
    "train_seconds",
    "eval_seconds",
    "checkpoint_path",
)


@dataclass(frozen=True)
class ModelSpec:
    slug: str
    label: str
    optimizer: str
    learning_rate: float
    weight_decay: float


MODEL_SPECS = {
    spec.slug: spec
    for spec in (
        ModelSpec("resnet18", "ResNet-18", "sgd", 0.1, 5e-4),
        ModelSpec("resnet34", "ResNet-34", "sgd", 0.1, 5e-4),
        ModelSpec("convnext_tiny", "ConvNeXt-Tiny", "adamw", 4e-3, 0.05),
        ModelSpec("efficientnet_b0", "EfficientNet-B0", "sgd", 0.1, 5e-4),
    )
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train CIFAR-100 models from random initialization and benchmark "
            "every exact FP32 checkpoint with ZipNN src and a NeuralZip plan "
            "frozen at epoch 0."
        )
    )
    parser.add_argument("--models", nargs="+", choices=tuple(MODEL_SPECS), default=list(DEFAULT_MODELS))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "examples" / "outputs" / "datasets" / "cifar100_hf")
    parser.add_argument(
        "--dataset-source",
        choices=("auto", "hf-parquet", "torchvision"),
        default="auto",
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--test-limit", type=int)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "final-checkpoints-cifar100",
    )
    parser.add_argument(
        "--existing-checkpoint-root",
        type=Path,
        help=(
            "Optional root containing <model>/checkpoints. Complete external "
            "trajectories are recompressed without retraining."
        ),
    )
    parser.add_argument(
        "--train-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train models whose requested checkpoints are not available externally.",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Train and save exact checkpoints without running either compressor.",
    )
    parser.add_argument("--compression-only", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Tiny synthetic code-path check; never thesis data.")
    return parser.parse_args()


def build_model(slug: str, *, num_classes: int = 100) -> nn.Module:
    from torchvision import models

    if slug in {"resnet18", "resnet34"}:
        model = getattr(models, slug)(weights=None, num_classes=num_classes)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        return model
    if slug == "convnext_tiny":
        model = models.convnext_tiny(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    if slug == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    raise ValueError(f"Unsupported model: {slug}")


def cifar100_transforms(*, train: bool):
    from torchvision import transforms

    steps = []
    if train:
        steps.extend(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        )
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408),
                (0.2675, 0.2565, 0.2761),
            ),
        ]
    )
    return transforms.Compose(steps)


class HFCIFAR100Parquet(Dataset):
    def __init__(self, path: Path, *, transform=None) -> None:
        from PIL import Image
        import pyarrow.parquet as pq

        schema_names = set(pq.read_schema(path).names)
        label_column = "fine_label" if "fine_label" in schema_names else "label"
        if label_column not in schema_names:
            raise ValueError(
                f"No CIFAR-100 label column in {path}; columns={sorted(schema_names)}"
            )
        rows = pq.read_table(path, columns=["img", label_column]).to_pylist()
        self.images = [row["img"]["bytes"] for row in rows]
        self.labels = [int(row[label_column]) for row in rows]
        self.transform = transform
        self.image_cls = Image

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        image = self.image_cls.open(io.BytesIO(self.images[int(index)])).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[int(index)]


def find_hf_parquet(data_dir: Path) -> tuple[Path, Path] | None:
    candidates = (
        data_dir / "cifar100",
        data_dir,
    )
    for root in candidates:
        train_path = root / "train-00000-of-00001.parquet"
        test_path = root / "test-00000-of-00001.parquet"
        if train_path.is_file() and test_path.is_file():
            return train_path, test_path
    return None


def smoke_datasets(seed: int) -> tuple[Dataset, Dataset, str]:
    generator = torch.Generator().manual_seed(seed)
    train_images = torch.randn(12, 3, 32, 32, generator=generator)
    train_labels = torch.arange(12, dtype=torch.long).remainder(100)
    test_images = torch.randn(8, 3, 32, 32, generator=generator)
    test_labels = torch.arange(8, dtype=torch.long).remainder(100)
    return (
        TensorDataset(train_images, train_labels),
        TensorDataset(test_images, test_labels),
        "synthetic-smoke-only",
    )


def load_cifar100(args: argparse.Namespace) -> tuple[Dataset, Dataset, str]:
    if args.smoke:
        return smoke_datasets(args.seed)

    parquet = find_hf_parquet(args.data_dir)
    if args.dataset_source in {"auto", "hf-parquet"} and parquet is not None:
        train_path, test_path = parquet
        return (
            HFCIFAR100Parquet(train_path, transform=cifar100_transforms(train=True)),
            HFCIFAR100Parquet(test_path, transform=cifar100_transforms(train=False)),
            "uoft-cs/cifar100-parquet",
        )
    if args.dataset_source == "hf-parquet":
        raise FileNotFoundError(
            f"CIFAR-100 parquet files not found under {args.data_dir}."
        )

    from torchvision.datasets import CIFAR100

    return (
        CIFAR100(
            root=args.data_dir,
            train=True,
            download=args.download,
            transform=cifar100_transforms(train=True),
        ),
        CIFAR100(
            root=args.data_dir,
            train=False,
            download=args.download,
            transform=cifar100_transforms(train=False),
        ),
        "torchvision-cifar100",
    )


def upsert_result(path: Path, row: Mapping) -> None:
    upsert_model_epoch_row(
        path,
        row,
        fields=RESULT_FIELDS,
        model_order=DEFAULT_MODELS,
        include_compressor=True,
    )


def upsert_training(path: Path, row: Mapping) -> None:
    upsert_model_epoch_row(
        path,
        row,
        fields=TRAINING_FIELDS,
        model_order=DEFAULT_MODELS,
        include_compressor=False,
    )


def checkpoint_state(checkpoint: Mapping) -> Mapping[str, torch.Tensor]:
    state = checkpoint.get("model_state", checkpoint)
    if not isinstance(state, Mapping):
        raise TypeError("Checkpoint model_state is not a mapping.")
    return state


def floating_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    tensors = {
        str(name): tensor.detach().cpu().contiguous()
        for name, tensor in state.items()
        if torch.is_tensor(tensor) and tensor.is_floating_point() and tensor.numel() > 0
    }
    invalid = {name: tensor.dtype for name, tensor in tensors.items() if tensor.dtype != torch.float32}
    if invalid:
        preview = ", ".join(f"{name}={dtype}" for name, dtype in list(invalid.items())[:8])
        raise TypeError(f"Final checkpoint suite requires exact FP32 tensors; got {preview}.")
    return tensors


def model_checkpoint_path(checkpoint_dir: Path, slug: str, epoch: int) -> Path:
    return checkpoint_dir / f"{slug}_cifar100_epoch_{int(epoch):03d}.pt"


def cpu_model_state(model: nn.Module) -> dict[str, torch.Tensor]:
    state = {
        name: tensor.detach().cpu().contiguous().clone()
        for name, tensor in model.state_dict().items()
    }
    floating_state(state)
    return state


def completed_methods(result_path: Path, slug: str, epoch: int) -> set[str]:
    return {
        str(row["compressor"])
        for row in read_csv(result_path)
        if row.get("model") == slug
        and int(row.get("epoch", -1)) == int(epoch)
        and row.get("status") == "ok"
        and row.get("roundtrip_exact", "").lower() == "true"
        and row.get("input_unchanged", "").lower() == "true"
        and is_current_codec_result(row)
    }


def benchmark_checkpoint(
    *,
    spec: ModelSpec,
    epoch: int,
    path: Path,
    checkpoint_source: str,
    plan,
    setup: Mapping,
    args: argparse.Namespace,
    result_path: Path,
) -> None:
    done = completed_methods(result_path, spec.slug, epoch)
    if done == {"NeuralZip", "ZipNN src"}:
        print(f"[{spec.slug}] epoch={epoch:03d} compression already complete", flush=True)
        return

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint_state(checkpoint)
    tensors = floating_state(state)
    nonfloating = nonfloating_state(state)
    named_tensors = list(tensors.items())
    print(
        f"[{spec.slug}] epoch={epoch:03d} tensors={len(tensors)} "
        f"parameters={sum(t.numel() for t in tensors.values()):,}",
        flush=True,
    )

    common = {
        "model": spec.slug,
        "model_label": spec.label,
        "epoch": epoch,
        "checkpoint_path": str(path),
        "checkpoint_source": checkpoint_source,
        "dtype": "float32",
        "tensor_count": len(tensors),
        "nonfloating_tensor_count": len(nonfloating),
        "threads": args.threads,
        "warmup_runs": 1,
        "measured_repetitions": 1,
        "ratio_scope": "exponent symbols + raw fallback + metadata",
        "status": "ok",
    }
    if "NeuralZip" not in done:
        metrics = benchmark_neuralzip(
            plan,
            named_tensors,
            epoch=epoch,
            setup=setup,
        )
        row = {**common, "compressor": "NeuralZip", **metrics}
        upsert_result(result_path, row)
        print(
            f"[{spec.slug}] epoch={epoch:03d} NeuralZip "
            f"ratio={metrics['exponent_ratio_pct']:.3f}% "
            f"encode={metrics['encode_seconds']:.6f}s",
            flush=True,
        )
    if "ZipNN src" not in done:
        metrics = benchmark_zipnn_src(tensors, threads=args.threads)
        row = {**common, "compressor": "ZipNN src", **metrics}
        upsert_result(result_path, row)
        print(
            f"[{spec.slug}] epoch={epoch:03d} ZipNN src "
            f"ratio={metrics['exponent_ratio_pct']:.3f}% "
            f"encode={metrics['encode_seconds']:.6f}s",
            flush=True,
        )
    del checkpoint, state, tensors, nonfloating, named_tensors
    gc.collect()


def make_optimizer(spec: ModelSpec, model: nn.Module):
    if spec.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=spec.learning_rate,
            weight_decay=spec.weight_decay,
        )
    return torch.optim.SGD(
        model.parameters(),
        lr=spec.learning_rate,
        momentum=0.9,
        weight_decay=spec.weight_decay,
        nesterov=True,
    )


@torch.no_grad()
def evaluate(model, loader, criterion, device, *, amp_enabled: bool) -> tuple[float, float]:
    model.eval()
    loss_total = 0.0
    correct = 0
    examples = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            logits = model(images)
            loss = criterion(logits, labels)
        batch = int(labels.numel())
        loss_total += float(loss.item()) * batch
        correct += int(logits.argmax(dim=1).eq(labels).sum().item())
        examples += batch
    return loss_total / max(examples, 1), correct / max(examples, 1)


def train_one_epoch(model, loader, optimizer, scaler, criterion, device, *, amp_enabled: bool):
    model.train()
    loss_total = 0.0
    correct = 0
    examples = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch = int(labels.numel())
        loss_total += float(loss.item()) * batch
        correct += int(logits.argmax(dim=1).eq(labels).sum().item())
        examples += batch
    return loss_total / max(examples, 1), correct / max(examples, 1)


def train_model(
    spec: ModelSpec,
    train_data: Dataset,
    test_data: Dataset,
    dataset_name: str,
    args: argparse.Namespace,
    model_dir: Path,
    training_path: Path,
) -> Path:
    checkpoint_dir = model_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest_state_path = model_dir / "latest_training_state.pt"
    device = torch.device(args.device)
    set_seed(args.seed)
    model = build_model(spec.slug).to(device)
    criterion = nn.CrossEntropyLoss()
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
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        generator=loader_generator,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    start_epoch = 0
    if latest_state_path.is_file():
        latest = torch.load(latest_state_path, map_location=device, weights_only=False)
        start_epoch = int(latest["epoch"])
        model.load_state_dict(latest["model_state"], strict=True)
        optimizer.load_state_dict(latest["optimizer_state"])
        scheduler.load_state_dict(latest["scheduler_state"])
        scaler.load_state_dict(latest.get("scaler_state", {}))
        restore_rng_state(latest["rng_state"], loader_generator)
        print(f"[{spec.slug}] resume training from epoch {start_epoch}", flush=True)

    initial_path = model_checkpoint_path(checkpoint_dir, spec.slug, 0)
    if not initial_path.is_file():
        initial_state = cpu_model_state(model)
        eval_started = time.perf_counter()
        test_loss, test_accuracy = evaluate(
            model, test_loader, criterion, device, amp_enabled=amp_enabled
        )
        eval_seconds = time.perf_counter() - eval_started
        atomic_torch_save(
            initial_path,
            {
                "epoch": 0,
                "architecture": spec.slug,
                "dataset": dataset_name,
                "seed": args.seed,
                "model_state": initial_state,
            },
        )
        upsert_training(
            training_path,
            {
                "model": spec.slug,
                "model_label": spec.label,
                "epoch": 0,
                "train_loss": "",
                "train_accuracy": "",
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train_seconds": 0.0,
                "eval_seconds": eval_seconds,
                "checkpoint_path": str(initial_path),
            },
        )
        del initial_state

    if start_epoch >= int(args.epochs):
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return checkpoint_dir

    for epoch in range(start_epoch + 1, int(args.epochs) + 1):
        started = time.perf_counter()
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            criterion,
            device,
            amp_enabled=amp_enabled,
        )
        train_seconds = time.perf_counter() - started
        scheduler.step()
        eval_started = time.perf_counter()
        test_loss, test_accuracy = evaluate(
            model, test_loader, criterion, device, amp_enabled=amp_enabled
        )
        eval_seconds = time.perf_counter() - eval_started
        state = cpu_model_state(model)
        checkpoint_path = model_checkpoint_path(checkpoint_dir, spec.slug, epoch)
        atomic_torch_save(
            checkpoint_path,
            {
                "epoch": epoch,
                "architecture": spec.slug,
                "dataset": dataset_name,
                "seed": args.seed,
                "model_state": state,
            },
        )
        atomic_torch_save(
            latest_state_path,
            {
                "epoch": epoch,
                "model_state": state,
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
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train_seconds": train_seconds,
                "eval_seconds": eval_seconds,
                "checkpoint_path": str(checkpoint_path),
            },
        )
        print(
            f"[{spec.slug}] epoch={epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.4f} test_acc={100.0 * test_accuracy:.2f}% "
            f"seconds={train_seconds + eval_seconds:.1f}",
            flush=True,
        )
        del state

    del model, optimizer, scheduler, scaler, train_loader, test_loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return checkpoint_dir


def external_checkpoint_dir(root: Path | None, spec: ModelSpec, epochs: int) -> Path | None:
    return find_external_checkpoint_dir(root, spec.slug, epochs)


def complete_local_checkpoint_dir(output_dir: Path, spec: ModelSpec, epochs: int) -> Path | None:
    return find_complete_local_checkpoint_dir(output_dir, spec.slug, epochs)


def frozen_codec_for_trajectory(
    spec: ModelSpec,
    checkpoint_dir: Path,
    model_dir: Path,
    args: argparse.Namespace,
):
    files = checkpoint_files(checkpoint_dir, args.epochs)
    if 0 not in files:
        raise FileNotFoundError(f"Missing epoch-0 checkpoint in {checkpoint_dir}.")
    plan_path = model_dir / "neuralzip_epoch0_plan.pkl"
    setup_path = model_dir / "neuralzip_epoch0_setup.json"
    checkpoint = torch.load(files[0], map_location="cpu", weights_only=False)
    tensors = floating_state(checkpoint_state(checkpoint))
    epoch0_digest = tensor_digest(tensors.items())

    if plan_path.is_file() and setup_path.is_file():
        setup = json.loads(setup_path.read_text())
        if setup.get("epoch0_tensor_digest") != epoch0_digest:
            raise RuntimeError(
                f"Frozen plan in {plan_path} belongs to a different epoch-0 checkpoint."
            )
        if int(setup.get("threads", -1)) != int(args.threads):
            raise RuntimeError(
                f"Frozen plan in {plan_path} was built with "
                f"threads={setup.get('threads')}, not {args.threads}. Use a fresh "
                "output directory for a different timing configuration."
            )
        plan = load_codec_plan(plan_path)
        if plan.digest != setup["codebook_digest"]:
            raise RuntimeError(f"Frozen NeuralZip plan is corrupt: {plan_path}.")
        print(f"[{spec.slug}] loaded frozen epoch-0 NeuralZip plan", flush=True)
    else:
        plan, setup = setup_neuralzip(list(tensors.items()), threads=args.threads)
        setup = {
            **setup,
            "model": spec.slug,
            "dtype": "float32",
            "epoch0_checkpoint": str(files[0]),
            "epoch0_tensor_digest": epoch0_digest,
            "threads": args.threads,
            "enable_prepacked_cache": False,
        }
        save_codec_plan(plan_path, plan, list(tensors.items()))
        atomic_write_json(setup_path, setup)
    print(
        f"[{spec.slug}] epoch-0 NeuralZip setup={setup['total_seconds']:.3f}s",
        flush=True,
    )
    del checkpoint, tensors
    gc.collect()
    return plan, setup, files


def process_trajectory(
    spec: ModelSpec,
    checkpoint_dir: Path,
    checkpoint_source: str,
    args: argparse.Namespace,
    result_path: Path,
) -> None:
    model_dir = args.output_dir / spec.slug
    model_dir.mkdir(parents=True, exist_ok=True)
    plan, setup, files = frozen_codec_for_trajectory(spec, checkpoint_dir, model_dir, args)
    missing = [epoch for epoch in range(int(args.epochs) + 1) if epoch not in files]
    if missing:
        raise FileNotFoundError(
            f"Missing checkpoints for {spec.slug}: {missing[:8]}"
            + (f" (+{len(missing) - 8} more)" if len(missing) > 8 else "")
        )
    for epoch in range(int(args.epochs) + 1):
        benchmark_checkpoint(
            spec=spec,
            epoch=epoch,
            path=files[epoch],
            checkpoint_source=checkpoint_source,
            plan=plan,
            setup=setup,
            args=args,
            result_path=result_path,
        )
    del plan
    gc.collect()


def render_figure(result_path: Path, output_dir: Path) -> tuple[Path, Path] | None:
    if not result_path.is_file():
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    frame = pd.read_csv(result_path)
    frame = frame[
        frame["status"].eq("ok")
        & frame["roundtrip_exact"].eq(True)  # noqa: E712
        & frame["input_unchanged"].eq(True)  # noqa: E712
        & frame.apply(is_current_codec_result, axis=1)
    ].copy()
    if frame.empty:
        return None
    colors = {"ZipNN src": "#4C78A8", "NeuralZip": "#E45756"}
    markers = {"ZipNN src": "o", "NeuralZip": "s"}
    fig, axes = plt.subplots(2, 4, figsize=(14.4, 6.2), sharex="col")
    for column, slug in enumerate(DEFAULT_MODELS):
        spec = MODEL_SPECS[slug]
        subset = frame[frame["model"].eq(slug)].sort_values("epoch")
        ratio_axis = axes[0, column]
        time_axis = axes[1, column]
        ratio_axis.set_title(spec.label, fontsize=11)
        if subset.empty:
            for axis in (ratio_axis, time_axis):
                axis.text(0.5, 0.5, "Pending", ha="center", va="center", transform=axis.transAxes)
        for method in ("ZipNN src", "NeuralZip"):
            group = subset[subset["compressor"].eq(method)]
            if group.empty:
                continue
            style = {
                "color": colors[method],
                "marker": markers[method],
                "markersize": 2.8,
                "markevery": max(1, len(group) // 10),
                "linewidth": 1.5,
                "label": method,
            }
            ratio_axis.plot(group["epoch"], group["exponent_ratio_pct"], **style)
            time_axis.plot(group["epoch"], group["encode_seconds"], **style)
        for axis in (ratio_axis, time_axis):
            axis.grid(True, axis="y", color="#dddddd", linewidth=0.6)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
        time_axis.set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Exponent ratio (%)")
    axes[1, 0].set_ylabel("Encoding time (s)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if not handles:
        for axis in axes[0, 1:]:
            handles, labels = axis.get_legend_handles_labels()
            if handles:
                break
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = figure_dir / "checkpoint_training.pdf"
    png_path = figure_dir / "checkpoint_training.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def validate_complete_results(
    result_path: Path,
    *,
    models: Iterable[str],
    epochs: int,
) -> None:
    """Reject partial or duplicated trajectories before final rendering."""

    import pandas as pd

    frame = pd.read_csv(result_path)
    expected = {
        (slug, epoch, compressor)
        for slug in models
        for epoch in range(int(epochs) + 1)
        for compressor in ("ZipNN src", "NeuralZip")
    }
    observed_rows = frame[
        frame["model"].isin(tuple(models))
    ].copy()
    keys = list(
        zip(
            observed_rows["model"],
            observed_rows["epoch"].astype(int),
            observed_rows["compressor"],
        )
    )
    observed = set(keys)
    missing = sorted(expected - observed)
    duplicates = len(keys) - len(observed)
    invalid = observed_rows[
        ~observed_rows["status"].eq("ok")
        | ~observed_rows["roundtrip_exact"].eq(True)  # noqa: E712
        | ~observed_rows["input_unchanged"].eq(True)  # noqa: E712
        | ~observed_rows.apply(is_current_codec_result, axis=1)
    ]
    if missing or duplicates or not invalid.empty:
        raise RuntimeError(
            "Checkpoint results are incomplete or invalid: "
            f"missing={len(missing)}, duplicates={duplicates}, "
            f"invalid={len(invalid)}."
        )


def write_metadata(args: argparse.Namespace, dataset_name: str | None) -> None:
    atomic_write_json(
        args.output_dir / "run_metadata.json",
        {
            "experiment": "final-training-checkpoint-suite",
            "models": [asdict(MODEL_SPECS[slug]) for slug in args.models],
            "dataset": dataset_name,
            "dataset_is_real": not args.smoke,
            "epochs": args.epochs,
            "seed": args.seed,
            "checkpoint_dtype": "float32 exact model state",
            "checkpoint_schedule": "epoch 0 and the end of every epoch",
            "compression_schedule": "post-hoc isolated CPU benchmark of every saved epoch",
            "compressors": ["ZipNN src 0.5.4", "NeuralZip frozen epoch-0 plan"],
            "codec_protocol_version": CODEC_PROTOCOL_VERSION,
            "encoder_paths": ENCODER_PATHS,
            "ratio_scope": "exponent symbols + raw fallback + metadata",
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
                "bitwise round-trip for every FP32 floating tensor and explicit "
                "input immutability after each compressor"
            ),
            "existing_checkpoint_root": (
                str(args.existing_checkpoint_root)
                if args.existing_checkpoint_root is not None
                else None
            ),
            "smoke_only": args.smoke,
        },
    )


def main() -> None:
    args = parse_args()
    exclusive_modes = sum(bool(value) for value in (args.train_only, args.compression_only, args.render_only))
    if exclusive_modes > 1:
        raise ValueError("--train-only, --compression-only, and --render-only are mutually exclusive.")
    if args.smoke:
        args.models = [args.models[0]]
        args.epochs = 1
        args.batch_size = min(args.batch_size, 4)
        args.num_workers = 0
        args.device = "cpu"
        args.output_dir = args.output_dir / "smoke"
    args.output_dir = args.output_dir.resolve()
    args.data_dir = args.data_dir.resolve()
    if args.existing_checkpoint_root is not None:
        args.existing_checkpoint_root = args.existing_checkpoint_root.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, int(args.threads)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    if args.render_only:
        validate_complete_results(
            args.output_dir / "checkpoint_results.csv",
            models=args.models,
            epochs=args.epochs,
        )
        rendered = render_figure(args.output_dir / "checkpoint_results.csv", args.output_dir)
        print(f"rendered={rendered}", flush=True)
        return

    result_path = args.output_dir / "checkpoint_results.csv"
    train_data = test_data = None
    dataset_name = (
        "uoft-cs/cifar100-parquet"
        if not args.smoke and find_hf_parquet(args.data_dir) is not None
        else None
    )
    if not args.compression_only:
        need_training = any(
            external_checkpoint_dir(args.existing_checkpoint_root, MODEL_SPECS[slug], args.epochs)
            is None
            and complete_local_checkpoint_dir(args.output_dir, MODEL_SPECS[slug], args.epochs)
            is None
            for slug in args.models
        )
        if need_training:
            if not args.train_missing:
                raise FileNotFoundError("Requested checkpoints are missing and --no-train-missing was set.")
            train_data, test_data, dataset_name = load_cifar100(args)
            train_data = limited(train_data, args.train_limit)
            test_data = limited(test_data, args.test_limit)
    write_metadata(args, dataset_name or "CIFAR-100 checkpoint trajectories")

    for slug in args.models:
        spec = MODEL_SPECS[slug]
        external = external_checkpoint_dir(args.existing_checkpoint_root, spec, args.epochs)
        if external is not None:
            checkpoint_dir = external
            source = "external-existing"
            print(f"[{slug}] reuse checkpoints from {checkpoint_dir}", flush=True)
        else:
            local = complete_local_checkpoint_dir(args.output_dir, spec, args.epochs)
            if local is not None:
                checkpoint_dir = local
                source = "trained-by-final-runner"
                print(f"[{slug}] reuse locally trained checkpoints from {checkpoint_dir}", flush=True)
            else:
                if args.compression_only or not args.train_missing:
                    raise FileNotFoundError(
                        f"No complete epoch 0..{args.epochs} trajectory for {slug}."
                    )
                if train_data is None or test_data is None or dataset_name is None:
                    raise RuntimeError("Training data were not initialized.")
                checkpoint_dir = train_model(
                    spec,
                    train_data,
                    test_data,
                    dataset_name,
                    args,
                    args.output_dir / slug,
                    args.output_dir / slug / "training_metrics.csv",
                )
                source = "trained-by-final-runner"
        if args.train_only:
            print(f"[{slug}] train-only complete: {checkpoint_dir}", flush=True)
            continue
        process_trajectory(spec, checkpoint_dir, source, args, result_path)
        rendered = render_figure(result_path, args.output_dir)
        print(f"[{slug}] rendered={rendered}", flush=True)

    if not args.train_only:
        validate_complete_results(
            result_path,
            models=args.models,
            epochs=args.epochs,
        )
        rendered = render_figure(result_path, args.output_dir)
        print(f"final_rendered={rendered}", flush=True)

    print(f"results={result_path}", flush=True)
    print(f"figure={args.output_dir / 'figures' / 'checkpoint_training.pdf'}", flush=True)


if __name__ == "__main__":
    main()
