from __future__ import annotations

import argparse
import gc
import io
import os
import time
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from examples.checkpoint_common import (
    add_checkpoint_run_arguments,
    atomic_torch_save,
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
    ENCODER_PATHS,
    benchmark_checkpoint_codecs,
    completed_checkpoint_methods,
    floating_state,
    is_current_codec_result,
    nonfloating_state,
    tensor_digest,
)
from neuralzip.checkpoint_loading import load_torch_checkpoint
from neuralzip.experiment_io import atomic_write_json

DEFAULT_MODELS = (
    "resnet18",
    "resnet34",
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
    learning_rate: float
    weight_decay: float


MODEL_SPECS = {
    spec.slug: spec
    for spec in (
        ModelSpec("resnet18", "ResNet-18", 0.1, 5e-4),
        ModelSpec("resnet34", "ResNet-34", 0.1, 5e-4),
    )
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train CIFAR-100 models from random initialization and benchmark "
            "every exact FP32 checkpoint with ZipNN and a NeuralZip setup "
            "built at epoch 0."
        )
    )
    parser.add_argument(
        "--models", nargs="+", choices=tuple(MODEL_SPECS), default=list(DEFAULT_MODELS)
    )
    add_checkpoint_run_arguments(
        parser,
        epochs=50,
        batch_size=128,
        data_dir=ROOT / "results" / "datasets" / "cifar100_hf",
        output_dir=ROOT / "results" / "checkpoints-cifar100",
    )
    parser.add_argument(
        "--dataset-source",
        choices=("auto", "hf-parquet", "torchvision"),
        default="auto",
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--test-limit", type=int)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny synthetic code-path check; never thesis data.",
    )
    return parser.parse_args()


def build_model(slug: str, *, num_classes: int = 100) -> nn.Module:
    from torchvision import models

    if slug in {"resnet18", "resnet34"}:
        model = getattr(models, slug)(weights=None, num_classes=num_classes)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
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
        import pyarrow.parquet as pq
        from PIL import Image

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


def model_checkpoint_path(checkpoint_dir: Path, slug: str, epoch: int) -> Path:
    return checkpoint_dir / f"{slug}_cifar100_epoch_{int(epoch):03d}.pt"


def cpu_model_state(model: nn.Module) -> dict[str, torch.Tensor]:
    state = {
        name: tensor.detach().cpu().contiguous().clone()
        for name, tensor in model.state_dict().items()
    }
    floating_state(state, error_prefix="Checkpoint suite")
    return state


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
    done = completed_checkpoint_methods(result_path, spec.slug, epoch)
    if done == {"NeuralZip", "ZipNN"}:
        print(
            f"[{spec.slug}] epoch={epoch:03d} compression already complete", flush=True
        )
        return

    checkpoint = load_torch_checkpoint(path)
    state = checkpoint_state(checkpoint)
    tensors = floating_state(state, error_prefix="Checkpoint suite")
    nonfloating = nonfloating_state(state)
    print(
        f"[{spec.slug}] epoch={epoch:03d} tensors={len(tensors)} "
        f"floating_values={sum(t.numel() for t in tensors.values()):,}",
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
    benchmark_checkpoint_codecs(
        slug=spec.slug,
        epoch=epoch,
        plan=plan,
        setup=setup,
        tensors=tensors,
        common_row=common,
        completed=done,
        threads=args.threads,
        upsert_result=lambda row: upsert_result(result_path, row),
    )
    del checkpoint, state, tensors, nonfloating
    gc.collect()


def make_optimizer(spec: ModelSpec, model: nn.Module):
    return torch.optim.SGD(
        model.parameters(),
        lr=spec.learning_rate,
        momentum=0.9,
        weight_decay=spec.weight_decay,
        nesterov=True,
    )


@torch.no_grad()
def evaluate(
    model, loader, criterion, device, *, amp_enabled: bool
) -> tuple[float, float]:
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


def train_one_epoch(
    model, loader, optimizer, scaler, criterion, device, *, amp_enabled: bool
):
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


def frozen_codec_for_trajectory(
    spec: ModelSpec,
    checkpoint_dir: Path,
    model_dir: Path,
    args: argparse.Namespace,
):
    def prepare(checkpoint):
        tensors = floating_state(
            checkpoint_state(checkpoint),
            error_prefix="Checkpoint suite",
        )
        return (
            list(tensors.items()),
            {"epoch0_tensor_digest": tensor_digest(tensors.items())},
            {},
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
    spec: ModelSpec,
    checkpoint_dir: Path,
    checkpoint_source: str,
    args: argparse.Namespace,
    result_path: Path,
) -> None:
    run_checkpoint_trajectory(
        spec=spec,
        checkpoint_dir=checkpoint_dir,
        checkpoint_source=checkpoint_source,
        args=args,
        result_path=result_path,
        frozen_codec_for_trajectory=frozen_codec_for_trajectory,
        benchmark_checkpoint=benchmark_checkpoint,
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
        error_label="Checkpoint results",
    )


def write_metadata(args: argparse.Namespace, dataset_name: str | None) -> None:
    atomic_write_json(
        args.output_dir / "run_metadata.json",
        {
            "experiment": "training-checkpoint-suite",
            "models": [asdict(MODEL_SPECS[slug]) for slug in args.models],
            "dataset": dataset_name,
            "dataset_is_real": not args.smoke,
            "epochs": args.epochs,
            "seed": args.seed,
            "checkpoint_dtype": "float32 exact model state",
            "checkpoint_schedule": "epoch 0 and the end of every epoch",
            "compression_schedule": "post-hoc isolated CPU benchmark of every saved epoch",
            "compressors": ["ZipNN 0.5.4", "NeuralZip epoch-0 setup"],
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
    if args.smoke:
        args.models = [args.models[0]]
        args.epochs = 1
        args.batch_size = min(args.batch_size, 4)
        args.num_workers = 0
        args.device = "cpu"
        args.output_dir = args.output_dir / "smoke"
    result_path = initialize_checkpoint_run(args)
    train_data = test_data = None
    dataset_name = (
        "uoft-cs/cifar100-parquet"
        if not args.smoke and find_hf_parquet(args.data_dir) is not None
        else None
    )
    if not args.compression_only and missing_checkpoint_slugs(
        slugs=args.models,
        external_root=args.existing_checkpoint_root,
        output_dir=args.output_dir,
        epochs=args.epochs,
    ):
        if not args.train_missing:
            raise FileNotFoundError(
                "Requested checkpoints are missing and --no-train-missing was set."
            )
        train_data, test_data, dataset_name = load_cifar100(args)
        train_data = limited(train_data, args.train_limit)
        test_data = limited(test_data, args.test_limit)
    write_metadata(args, dataset_name or "CIFAR-100 checkpoint trajectories")

    def train_trajectory(spec: ModelSpec) -> Path:
        if train_data is None or test_data is None or dataset_name is None:
            raise RuntimeError("Training data were not initialized.")
        return train_model(
            spec,
            train_data,
            test_data,
            dataset_name,
            args,
            args.output_dir / spec.slug,
            args.output_dir / spec.slug / "training_metrics.csv",
        )

    run_checkpoint_models(
        args=args,
        specs=MODEL_SPECS,
        result_path=result_path,
        local_source="trained-by-checkpoint-runner",
        train_trajectory=train_trajectory,
        process_trajectory=process_trajectory,
        validate_results=validate_complete_results,
    )


if __name__ == "__main__":
    main()
