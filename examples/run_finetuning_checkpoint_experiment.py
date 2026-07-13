from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import random
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neuralzip.experiment_scenarios import _compress_mapping  # noqa: E402


CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_TGZ = "cifar-10-python.tar.gz"
CIFAR10_DIR = "cifar-10-batches-py"
CIFAR10_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
CIFAR10_STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

METHODS = {
    "zipnn-local": ("ZipNN local", "zipnn-local"),
    "neuralzip": ("NeuralZip", "neuralzip"),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def download_cifar10(data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    archive_path = data_dir / CIFAR10_TGZ
    dataset_path = data_dir / CIFAR10_DIR
    if dataset_path.exists():
        return dataset_path
    if not archive_path.exists():
        print(f"downloading CIFAR-10 to {archive_path}", flush=True)
        urllib.request.urlretrieve(CIFAR10_URL, archive_path)
    print(f"extracting {archive_path}", flush=True)
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(data_dir)
    return dataset_path


def _load_batch(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        batch = pickle.load(handle, encoding="latin1")
    images = batch["data"].reshape(-1, 3, 32, 32).astype(np.uint8)
    labels = np.asarray(batch["labels"], dtype=np.int64)
    return images, labels


def load_cifar10(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dataset_path = download_cifar10(data_dir)
    train_images = []
    train_labels = []
    for index in range(1, 6):
        images, labels = _load_batch(dataset_path / f"data_batch_{index}")
        train_images.append(images)
        train_labels.append(labels)
    test_images, test_labels = _load_batch(dataset_path / "test_batch")
    return (
        np.concatenate(train_images, axis=0),
        np.concatenate(train_labels, axis=0),
        test_images,
        test_labels,
    )


class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        *,
        train: bool,
        input_size: int = 32,
        mean: torch.Tensor = CIFAR10_MEAN,
        std: torch.Tensor = CIFAR10_STD,
    ) -> None:
        self.images = images
        self.labels = labels
        self.train = train
        self.input_size = int(input_size)
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.images[index]
        label = int(self.labels[index])
        x = torch.from_numpy(image).float().div_(255.0)
        if self.train:
            x = F.pad(x, (4, 4, 4, 4), mode="reflect")
            top = random.randint(0, 8)
            left = random.randint(0, 8)
            x = x[:, top : top + 32, left : left + 32]
            if random.random() < 0.5:
                x = torch.flip(x, dims=(2,))
        if self.input_size != 32:
            x = F.interpolate(
                x.unsqueeze(0),
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        x = (x - self.mean) / self.std
        return x, torch.tensor(label, dtype=torch.long)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


class ResNet(nn.Module):
    def __init__(self, block: type[BasicBlock], layers: list[int], num_classes: int) -> None:
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(
        self,
        block: type[BasicBlock],
        planes: int,
        num_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        modules = []
        for block_stride in strides:
            modules.append(block(self.in_planes, planes, block_stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = torch.flatten(out, 1)
        return self.fc(out)


def resnet18_cifar10() -> ResNet:
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)


def torchvision_resnet18_cifar10(*, pretrained: bool) -> nn.Module:
    from torchvision.models import ResNet18_Weights, resnet18

    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model


def input_config(base_source: str, normalization: str, input_size: int | None):
    if normalization == "auto":
        normalization = (
            "imagenet"
            if base_source == "torchvision-imagenet1k-v1"
            else "cifar10"
        )
    if normalization == "imagenet":
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    elif normalization == "cifar10":
        mean, std = CIFAR10_MEAN, CIFAR10_STD
    else:
        raise ValueError("normalization must be 'auto', 'imagenet', or 'cifar10'.")

    if input_size is None:
        input_size = 224 if base_source == "torchvision-imagenet1k-v1" else 32
    return int(input_size), normalization, mean, std


def accuracy_from_logits(logits: torch.Tensor, target: torch.Tensor) -> int:
    return int(logits.argmax(dim=1).eq(target).sum().item())


def cosine_lr(step: int, total_steps: int, base_lr: float, min_lr: float) -> float:
    if total_steps <= 1:
        return float(base_lr)
    progress = min(max(step / float(total_steps - 1), 0.0), 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    finetune_epoch: int,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    start = time.perf_counter()
    for step, (images, labels) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        batch_size = int(labels.size(0))
        running_loss += float(loss.item()) * batch_size
        running_correct += accuracy_from_logits(logits, labels)
        running_total += batch_size
        if step % 50 == 0:
            elapsed = time.perf_counter() - start
            print(
                f"finetune_epoch={finetune_epoch:03d} step={step:03d}/{len(loader)} "
                f"loss={running_loss / running_total:.4f} "
                f"acc={100.0 * running_correct / running_total:.2f} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
    return running_loss / running_total, 100.0 * running_correct / running_total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        batch_size = int(labels.size(0))
        running_loss += float(loss.item()) * batch_size
        running_correct += accuracy_from_logits(logits, labels)
        running_total += batch_size
    return running_loss / running_total, 100.0 * running_correct / running_total


def _model_for_checkpoint(checkpoint: dict) -> nn.Module:
    architecture = checkpoint.get("architecture", "manual-resnet18-cifar10")
    if architecture == "torchvision-resnet18-imagenet1k-v1-cifar10":
        return torchvision_resnet18_cifar10(pretrained=False)
    return resnet18_cifar10()


def load_model(
    base_checkpoint: Path,
    device: torch.device,
    *,
    base_source: str = "checkpoint",
) -> tuple[nn.Module, dict]:
    if base_source == "torchvision-imagenet1k-v1":
        model = torchvision_resnet18_cifar10(pretrained=True)
        metadata = {
            "epoch": 0,
            "pretrain_epoch": 0,
            "architecture": "torchvision-resnet18-imagenet1k-v1-cifar10",
            "source_weights": "ResNet18_Weights.IMAGENET1K_V1",
        }
        return model.to(device), metadata
    if base_source != "checkpoint":
        raise ValueError("base_source must be 'checkpoint' or 'torchvision-imagenet1k-v1'.")

    checkpoint = torch.load(base_checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state", checkpoint)
    model = _model_for_checkpoint(checkpoint if isinstance(checkpoint, dict) else {})
    model.load_state_dict(state)
    return model.to(device), checkpoint if isinstance(checkpoint, dict) else {}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def checkpoint_finetune_epoch(path: Path) -> int:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    try:
        return int(checkpoint.get("finetune_epoch", checkpoint.get("epoch", -1)))
    finally:
        del checkpoint


def save_checkpoint(
    path: Path,
    model: nn.Module,
    *,
    pretrain_epoch: int,
    finetune_epoch: int,
    architecture: str,
    metadata: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(pretrain_epoch + finetune_epoch),
            "pretrain_epoch": int(pretrain_epoch),
            "finetune_epoch": int(finetune_epoch),
            "architecture": architecture,
            "dataset": "cifar10",
            "model_state": {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            },
            **metadata,
        },
        path,
    )


def finetune(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    checkpoint_dir = output_dir / "checkpoints"
    metrics_path = output_dir / "finetuning_metrics.csv"
    metadata_path = output_dir / "metadata.json"

    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("--require-cuda expects a CUDA device.")
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    print(f"device={device}", flush=True)
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)

    train_images, train_labels, test_images, test_labels = load_cifar10(args.data_dir)
    input_size, normalization, mean, std = input_config(
        args.base_source,
        args.normalization,
        args.input_size,
    )
    train_set = CIFAR10Dataset(
        train_images,
        train_labels,
        train=True,
        input_size=input_size,
        mean=mean,
        std=std,
    )
    test_set = CIFAR10Dataset(
        test_images,
        test_labels,
        train=False,
        input_size=input_size,
        mean=mean,
        std=std,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    model, base_metadata = load_model(
        args.base_checkpoint,
        device,
        base_source=args.base_source,
    )
    pretrain_epoch = int(base_metadata.get("epoch", args.pretrain_epoch))
    architecture = str(base_metadata.get("architecture", "manual-resnet18-cifar10"))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )

    experiment_metadata = {
        "experiment": "resnet18_cifar10_finetuning_checkpoints",
        "architecture": architecture,
        "dataset": "cifar10",
        "train_examples": int(len(train_set)),
        "test_examples": int(len(test_set)),
        "base_checkpoint": str(args.base_checkpoint),
        "base_source": str(args.base_source),
        "source_weights": base_metadata.get("source_weights"),
        "base_epoch": int(pretrain_epoch),
        "finetune_epochs": int(args.finetune_epochs),
        "input_size": int(input_size),
        "normalization": normalization,
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "optimizer": "sgd_nesterov",
        "base_lr": float(args.lr),
        "min_lr": float(args.min_lr),
        "weight_decay": float(args.weight_decay),
        "seed": int(args.seed),
        "device": str(device),
        "dtype_for_compression": "bf16",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(experiment_metadata, indent=2))

    rows: list[dict] = read_csv_rows(metrics_path) if args.resume else []
    base_path = checkpoint_dir / "resnet18_cifar10_finetune_epoch_000.pt"
    if not base_path.exists():
        base_test_loss, base_test_acc = evaluate(model, test_loader, criterion, device)
        save_checkpoint(
            base_path,
            model,
            pretrain_epoch=pretrain_epoch,
            finetune_epoch=0,
            architecture=architecture,
            metadata={
                "test_loss": base_test_loss,
                "test_accuracy": base_test_acc,
                "source_weights": base_metadata.get("source_weights"),
            },
        )
        rows.append(
            {
                "finetune_epoch": 0,
                "total_epoch": pretrain_epoch,
                "lr": "",
                "train_loss": "",
                "train_accuracy": "",
                "test_loss": base_test_loss,
                "test_accuracy": base_test_acc,
                "epoch_seconds": "",
                "checkpoint_path": str(base_path),
            }
        )
        write_csv(metrics_path, rows)
        print(
            f"saved_base_checkpoint={base_path} "
            f"test_acc={base_test_acc:.2f}",
            flush=True,
        )

    start_epoch = 1
    if args.resume:
        existing = sorted(
            checkpoint_dir.glob("resnet18_cifar10_finetune_epoch_*.pt"),
            key=checkpoint_finetune_epoch,
        )
        completed = [
            path
            for path in existing
            if checkpoint_finetune_epoch(path) > 0
        ]
        if completed:
            latest = completed[-1]
            latest_epoch = checkpoint_finetune_epoch(latest)
            model, latest_metadata = load_model(
                latest,
                device,
                base_source="checkpoint",
            )
            if latest_metadata.get("optimizer_state"):
                optimizer.load_state_dict(latest_metadata["optimizer_state"])
            start_epoch = latest_epoch + 1
            print(
                f"resume_from_checkpoint={latest} "
                f"next_finetune_epoch={start_epoch}",
                flush=True,
            )

    for finetune_epoch in range(start_epoch, int(args.finetune_epochs) + 1):
        checkpoint_path = (
            checkpoint_dir / f"resnet18_cifar10_finetune_epoch_{finetune_epoch:03d}.pt"
        )
        if checkpoint_path.exists():
            raise FileExistsError(
                f"{checkpoint_path} already exists. Use --resume to continue "
                "from the latest completed checkpoint or --no-resume with a "
                "fresh output directory."
            )

        lr = cosine_lr(
            finetune_epoch - 1,
            int(args.finetune_epochs),
            args.lr,
            args.min_lr,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        epoch_start = time.perf_counter()
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            finetune_epoch,
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        epoch_seconds = time.perf_counter() - epoch_start
        save_checkpoint(
            checkpoint_path,
            model,
            pretrain_epoch=pretrain_epoch,
            finetune_epoch=finetune_epoch,
            architecture=architecture,
            metadata={
                "lr": lr,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "test_loss": test_loss,
                "test_accuracy": test_acc,
                "source_weights": base_metadata.get("source_weights"),
                "optimizer_state": optimizer.state_dict(),
            },
        )
        rows.append(
            {
                "finetune_epoch": finetune_epoch,
                "total_epoch": pretrain_epoch + finetune_epoch,
                "lr": lr,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "test_loss": test_loss,
                "test_accuracy": test_acc,
                "epoch_seconds": epoch_seconds,
                "checkpoint_path": str(checkpoint_path),
            }
        )
        write_csv(metrics_path, rows)
        print(
            f"finetune_epoch={finetune_epoch:03d} lr={lr:.6f} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.2f} "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.2f} "
            f"epoch_seconds={epoch_seconds:.1f}",
            flush=True,
        )


def _checkpoint_sort_key(path: Path) -> tuple[int, str]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    try:
        return int(checkpoint.get("finetune_epoch", checkpoint.get("epoch", -1))), path.name
    finally:
        del checkpoint


def selected_methods(names: list[str]) -> tuple[tuple[str, str], ...]:
    methods = []
    for name in names:
        key = name.lower().replace("_", "-")
        if key not in METHODS:
            choices = ", ".join(sorted(METHODS))
            raise ValueError(f"Unknown method {name!r}. Expected one of: {choices}.")
        methods.append(METHODS[key])
    return tuple(methods)


def compress_checkpoints(args: argparse.Namespace) -> pd.DataFrame:
    checkpoint_dir = args.output_dir / "checkpoints"
    compression_dir = args.output_dir / "compression"
    compression_dir.mkdir(parents=True, exist_ok=True)
    result_path = compression_dir / "finetuning_checkpoint_exponent_results.csv"
    json_path = compression_dir / "finetuning_checkpoint_exponent_results.json"

    rows = []
    if args.resume and result_path.exists():
        rows = pd.read_csv(result_path).to_dict(orient="records")
    completed = {
        (str(row.get("checkpoint")), str(row.get("compressor")))
        for row in rows
        if row.get("status") == "ok"
    }

    methods = selected_methods(args.methods)
    checkpoint_paths = sorted(checkpoint_dir.glob("*.pt"), key=_checkpoint_sort_key)
    if not checkpoint_paths:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}.")

    for path in checkpoint_paths:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = checkpoint.get("model_state", checkpoint)
        tensors = {
            name: tensor.detach().cpu().contiguous().to(dtype=torch.bfloat16)
            for name, tensor in state.items()
            if torch.is_tensor(tensor) and tensor.is_floating_point()
        }
        needed_methods = [
            method
            for method in methods
            if (path.name, method[0]) not in completed
        ]
        if not needed_methods:
            print(f"skip_existing_compression={path.name}", flush=True)
            continue
        print(
            f"compress_checkpoint={path.name} tensors={len(tensors)}",
            flush=True,
        )
        for row in _compress_mapping(path.stem, tensors, methods=tuple(needed_methods)):
            row["experiment"] = "finetuning-checkpoints"
            row["checkpoint"] = path.name
            row["pretrain_epoch"] = int(checkpoint.get("pretrain_epoch", -1))
            row["finetune_epoch"] = int(checkpoint.get("finetune_epoch", -1))
            row["total_epoch"] = int(checkpoint.get("epoch", -1))
            row["dataset"] = checkpoint.get("dataset", "cifar10")
            row["architecture"] = checkpoint.get(
                "architecture",
                "manual-resnet18-cifar10",
            )
            row["dtype"] = "bf16"
            row["tensor_count"] = len(tensors)
            rows.append(row)
        pd.DataFrame(rows).to_csv(result_path, index=False)
        json_path.write_text(json.dumps(rows, indent=2, default=str))
        del checkpoint, state, tensors

    df = pd.DataFrame(rows)
    df.to_csv(result_path, index=False)
    json_path.write_text(json.dumps(df.to_dict(orient="records"), indent=2, default=str))
    return df


def plot_results(output_dir: Path) -> Path | None:
    result_path = output_dir / "compression" / "finetuning_checkpoint_exponent_results.csv"
    if not result_path.exists():
        return None

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as error:
        print(f"plot_skipped={type(error).__name__}: {error}", flush=True)
        return None

    df = pd.read_csv(result_path)
    df = df[df["status"].eq("ok")].copy()
    if df.empty:
        return None
    df = df.sort_values(["compressor", "finetune_epoch"])

    fig, ax = plt.subplots(figsize=(6.5, 3.7))
    for compressor, group in df.groupby("compressor", sort=False):
        ax.plot(
            group["finetune_epoch"],
            group["exponent_compressed_pct"],
            marker="o",
            linewidth=1.8,
            markersize=3.8,
            label=str(compressor),
        )
    ax.set_xlabel("Finetuning epoch")
    ax.set_ylabel("Exponent-stream compressed size (%)")
    ax.grid(True, axis="y", color="#d7d7d7", linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = figures_dir / "finetuning_checkpoint_epoch_vs_exponent_ratio.pdf"
    png_path = figures_dir / "finetuning_checkpoint_epoch_vs_exponent_ratio.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=180)
    plt.close(fig)
    print(f"saved_plot={pdf_path}", flush=True)
    return pdf_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    default_output = (
        ROOT
        / "examples"
        / "outputs"
        / "checkpoints"
        / "resnet18_cifar10_finetune_epoch200_relela05"
    )
    default_base = (
        ROOT
        / "examples"
        / "outputs"
        / "checkpoints"
        / "resnet18_cifar10_pretrained_epoch200_relela01"
        / "checkpoint_epoch_200.pt"
    )
    parser.add_argument("--base-checkpoint", type=Path, default=default_base)
    parser.add_argument(
        "--base-source",
        choices=["checkpoint", "torchvision-imagenet1k-v1"],
        default="checkpoint",
    )
    parser.add_argument("--data-dir", type=Path, default=Path.home() / "data" / "cifar10")
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--finetune-epochs", type=int, default=20)
    parser.add_argument("--pretrain-epoch", type=int, default=200)
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument(
        "--normalization",
        choices=["auto", "cifar10", "imagenet"],
        default="auto",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--min-lr", type=float, default=0.0005)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-compression", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["zipnn-local", "neuralzip"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.base_checkpoint = args.base_checkpoint.resolve()
    args.data_dir = args.data_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    start = time.perf_counter()
    if not args.skip_training:
        finetune(args)
    if not args.skip_compression:
        df = compress_checkpoints(args)
        display_cols = [
            "finetune_epoch",
            "compressor",
            "scope",
            "status",
            "exponent_raw_bytes",
            "exponent_compressed_bytes",
            "exponent_compressed_pct",
            "compression_seconds",
            "clusters",
        ]
        print(df[display_cols].to_string(index=False), flush=True)
        plot_results(args.output_dir)
    print(f"total_seconds={time.perf_counter() - start:.1f}", flush=True)


if __name__ == "__main__":
    main()
