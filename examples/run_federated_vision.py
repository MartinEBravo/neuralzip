"""Federated compression experiment for the thesis.

The experiment trains CIFAR ResNet models with FedAvg and evaluates two
lossless FP32 communication codecs on exactly the same messages:

* the official ZipNN source package (``zipnn.ZipNN``), and
* NeuralZip with one initial clustering/codebook setup reused by every client
  and every communication round.

The communicated downlink is decoded before local training and decoded
uplinks are aggregated.  Since both codecs are bitwise lossless, their decoded
floating-point states must be identical; NeuralZip's decoded state is used as
the canonical trajectory after this equality check.  Non-floating buffers do
not have an exponent field and are carried unchanged.

The default is the 10-client, 200-round ResNet-18/34 suite. Override the
CLI arguments for smoke tests; for example::

    python examples/run_federated_vision.py \
        --architectures resnet18 --rounds 1 --clients 2 \
        --train-subset 20 --batch-size 2 --threads 2 \
        --device cpu --output-dir /tmp/neuralzip-federated-smoke
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import platform
import random
from collections.abc import Iterable, Mapping
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, models, transforms

try:
    from examples.federated_common import (
        COMPRESSOR_NEURALZIP,
        COMPRESSOR_SRC,
        DTYPE_NAME,
        FederatedCodecProtocol,
        _write_csv_atomic,
        _write_json_atomic,
        build_round_rows,
        build_summary_rows,
        new_cumulative_metrics,
    )
except ModuleNotFoundError:  # Direct ``python examples/<script>.py`` execution.
    from federated_common import (  # type: ignore
        COMPRESSOR_NEURALZIP,
        COMPRESSOR_SRC,
        DTYPE_NAME,
        FederatedCodecProtocol,
        _write_csv_atomic,
        _write_json_atomic,
        build_round_rows,
        build_summary_rows,
        new_cumulative_metrics,
    )


ROOT = Path(__file__).resolve().parents[1]
DATASET_NAME = "CIFAR-10"


class HFCIFAR10Parquet(Dataset):
    """Real CIFAR-10 backed by the cached Hugging Face Parquet mirror."""

    def __init__(self, parquet_path: Path, transform=None):
        try:
            import pyarrow.parquet as pq
        except ImportError as error:
            raise ImportError(
                "pyarrow is required for the real CIFAR-10 Parquet mirror."
            ) from error

        table = pq.read_table(parquet_path, columns=["img", "label"])
        rows = table.to_pylist()
        self._images = [row["img"]["bytes"] for row in rows]
        self._labels = [int(row["label"]) for row in rows]
        self._transform = transform

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, index: int):
        image = Image.open(io.BytesIO(self._images[int(index)])).convert("RGB")
        if self._transform is not None:
            image = self._transform(image)
        return image, self._labels[int(index)]


class ClientBatchStream:
    """Persistent, deterministic minibatch stream for one federated client."""

    def __init__(self, loader: DataLoader):
        self.loader = loader
        self.iterator = iter(loader)

    def take(self, count: int):
        batches = []
        while len(batches) < int(count):
            try:
                batch = next(self.iterator)
            except StopIteration:
                self.iterator = iter(self.loader)
                batch = next(self.iterator)
            batches.append(batch)
        return batches


def _build_model(architecture: str) -> torch.nn.Module:
    if architecture not in {"resnet18", "resnet34"}:
        raise ValueError(f"Unsupported federated model: {architecture}.")
    model = getattr(models, architecture)(weights=None, num_classes=10)
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = torch.nn.Identity()
    return model


def _floating_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().contiguous()
        for name, value in state.items()
        if torch.is_tensor(value) and value.is_floating_point()
    }


def _merge_decoded_state(
    base: Mapping[str, torch.Tensor],
    decoded_floating: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    merged = {}
    for name, value in base.items():
        if name in decoded_floating:
            merged[name] = decoded_floating[name].detach().cpu().contiguous()
        elif torch.is_tensor(value):
            merged[name] = value.detach().cpu().clone()
        else:
            merged[name] = value
    return merged


def _average_decoded_states(
    states: list[Mapping[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    if not states:
        raise ValueError("At least one decoded client state is required.")
    averaged = {}
    for name, first in states[0].items():
        if torch.is_tensor(first) and first.is_floating_point():
            accumulator = first.detach().cpu().to(torch.float32).clone()
            for state in states[1:]:
                accumulator.add_(state[name].detach().cpu().to(torch.float32))
            accumulator.div_(len(states))
            averaged[name] = accumulator.to(dtype=first.dtype)
        elif torch.is_tensor(first):
            averaged[name] = first.detach().cpu().clone()
        else:
            averaged[name] = first
    return averaged


def _train_one_client(
    architecture: str,
    initial_state: Mapping[str, torch.Tensor],
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
    learning_rate: float,
) -> tuple[dict[str, torch.Tensor], float]:
    model = _build_model(architecture).to(device=device, dtype=torch.float32)
    model.load_state_dict(initial_state, strict=True)
    model.train()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(learning_rate),
        momentum=0.9,
        weight_decay=5e-4,
    )
    criterion = torch.nn.CrossEntropyLoss()
    weighted_loss = 0.0
    examples = 0
    for images, labels in batches:
        images = images.to(device=device, dtype=torch.float32, non_blocking=True)
        labels = labels.to(device=device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        weighted_loss += float(loss.detach().cpu()) * int(labels.numel())
        examples += int(labels.numel())

    state = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return state, weighted_loss / max(examples, 1)


def _client_streams(
    dataset: Dataset,
    *,
    clients: int,
    batch_size: int,
    seed: int,
    train_subset: int | None,
    num_workers: int,
) -> list[ClientBatchStream]:
    generator = torch.Generator().manual_seed(int(seed))
    indices = torch.randperm(len(dataset), generator=generator).tolist()
    if train_subset is not None:
        indices = indices[: int(train_subset)]
    if len(indices) < int(clients):
        raise ValueError(
            "The real-data subset must contain at least one item per client."
        )

    partitions = [indices[index :: int(clients)] for index in range(int(clients))]
    streams = []
    for client_index, partition in enumerate(partitions):
        loader_generator = torch.Generator().manual_seed(
            int(seed) + 10_000 + client_index
        )
        loader = DataLoader(
            Subset(dataset, partition),
            batch_size=int(batch_size),
            shuffle=True,
            generator=loader_generator,
            num_workers=int(num_workers),
            pin_memory=torch.cuda.is_available(),
            persistent_workers=int(num_workers) > 0,
        )
        streams.append(ClientBatchStream(loader))
    return streams


def _training_dataset(data_dir: Path) -> tuple[Dataset, str]:
    parquet_path = data_dir / "plain_text" / "train-00000-of-00001.parquet"
    transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2470, 0.2435, 0.2616),
            ),
        ]
    )
    if parquet_path.exists():
        return (
            HFCIFAR10Parquet(parquet_path, transform=transform),
            "uoft-cs/cifar10 Parquet",
        )

    torchvision_folder = data_dir / "cifar-10-batches-py"
    if torchvision_folder.is_dir():
        return (
            datasets.CIFAR10(
                root=data_dir,
                train=True,
                download=False,
                transform=transform,
            ),
            "CIFAR-10 Python archive",
        )

    raise FileNotFoundError(
        "Real CIFAR-10 was not found. Expected either "
        f"{parquet_path} or {torchvision_folder}."
    )


def _append_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
            lineterminator="\n",
        )
        if not exists:
            writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def _run_architecture(
    args,
    dataset: Dataset,
    dataset_source: str,
    architecture: str,
) -> None:
    architecture_dir = Path(args.output_dir) / architecture
    round_path = architecture_dir / "federated_round_results.csv"
    summary_path = architecture_dir / "federated_summary.csv"
    metadata_path = architecture_dir / "run_metadata.json"
    if round_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{round_path} already exists. Use a fresh output directory or --overwrite."
        )
    if args.overwrite:
        for path in (round_path, summary_path, metadata_path):
            if path.exists():
                path.unlink()

    torch.manual_seed(int(args.seed))
    random.seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    initial_model = _build_model(architecture)
    server_state = {
        name: value.detach().cpu().clone()
        for name, value in initial_model.state_dict().items()
    }
    del initial_model
    initial_floating = _floating_state(server_state)

    protocol = FederatedCodecProtocol(
        threads=args.threads,
        chunk_size=args.chunk_size,
        threshold=args.compression_threshold,
    )
    protocol.prepare_once(initial_floating)
    codecs = protocol.codecs
    setup_seconds = protocol.setup_seconds
    setup_bytes = protocol.setup_bytes

    # Warm both complete encode/decode paths on the same initial model after
    # NeuralZip setup.  This is intentionally outside every reported round.
    protocol.warmup(initial_floating, context=architecture)

    streams = _client_streams(
        dataset,
        clients=args.clients,
        batch_size=args.batch_size,
        seed=args.seed,
        train_subset=args.train_subset,
        num_workers=args.num_workers,
    )
    metadata = {
        "architecture": architecture,
        "dataset": DATASET_NAME,
        "dataset_source": dataset_source,
        "dtype": DTYPE_NAME,
        "clients": int(args.clients),
        "rounds": int(args.rounds),
        "local_batches": int(args.local_batches),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "seed": int(args.seed),
        "train_subset": args.train_subset,
        "device": str(device),
        "threads": int(args.threads),
        "chunk_size": int(args.chunk_size),
        "compression_threshold": float(args.compression_threshold),
        "compressors": [COMPRESSOR_SRC, COMPRESSOR_NEURALZIP],
        "warmup_runs": 1,
        "trajectory": (
            "NeuralZip-decoded FP32 states after bitwise equality with "
            "ZipNN-src decoded states"
        ),
        "time_metric": (
            "one server-side downlink encode per round; client encoding, decode, "
            "training, and automatic cyclic GC excluded"
        ),
        "automatic_gc_during_encode": False,
        "ratio_metric": (
            "exact exponent-group payload plus its group metadata divided by "
            "raw exponent symbols; the final summary also reports the ratio "
            "after charging the one-time codebooks once per client"
        ),
        "archive_metric": (
            "diagnostic full per-tensor payload bytes only; the thesis comparison "
            "uses exponent ratios, not this implementation-specific field"
        ),
        "host": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "setup": {
            name: {"seconds": setup_seconds[name], "bytes": setup_bytes[name]}
            for name in codecs
        },
    }
    _write_json_atomic(metadata_path, metadata)

    cumulative = new_cumulative_metrics()

    for round_index in range(1, int(args.rounds) + 1):
        floating_server = _floating_state(server_state)
        order = protocol.encoding_order(round_index)
        downlinks, decoded_floating_server = protocol.encode_verified(
            floating_server,
            order=order,
            context=f"{architecture} round {round_index}",
            phase="downlink",
        )
        decoded_server_state = _merge_decoded_state(
            server_state, decoded_floating_server
        )

        uplinks = {name: [] for name in codecs}
        decoded_client_states = []
        client_losses = []
        for client_index, stream in enumerate(streams):
            client_state, client_loss = _train_one_client(
                architecture,
                decoded_server_state,
                stream.take(args.local_batches),
                device=device,
                learning_rate=args.learning_rate,
            )
            client_losses.append(client_loss)
            floating_client = _floating_state(client_state)
            client_measurements, decoded_floating_client = protocol.encode_verified(
                floating_client,
                order=protocol.encoding_order(
                    round_index,
                    client_index=client_index,
                ),
                context=(f"{architecture} round {round_index} client {client_index}"),
                phase="uplink",
            )
            for name, encoded in client_measurements.items():
                uplinks[name].append(encoded)
            decoded_client_states.append(
                _merge_decoded_state(client_state, decoded_floating_client)
            )
            del client_state, floating_client, decoded_floating_client

        server_state = _average_decoded_states(decoded_client_states)
        del decoded_client_states, decoded_floating_server

        round_rows = build_round_rows(
            prefix={
                "architecture": architecture,
                "dataset": DATASET_NAME,
                "round": round_index,
                "clients": int(args.clients),
                "local_batches": int(args.local_batches),
                "batch_size": int(args.batch_size),
                "dtype": DTYPE_NAME,
            },
            suffix={
                "train_loss_mean": sum(client_losses) / len(client_losses),
                "roundtrip_ok": True,
            },
            downlinks=downlinks,
            uplinks=uplinks,
            clients=args.clients,
            cumulative=cumulative,
            setup_seconds=setup_seconds,
            setup_bytes=setup_bytes,
        )
        _append_csv(round_path, round_rows)

        summary_rows = build_summary_rows(
            prefix={
                "architecture": architecture,
                "dataset": DATASET_NAME,
            },
            dimensions={
                "completed_rounds": round_index,
                "clients": int(args.clients),
                "local_batches": int(args.local_batches),
                "batch_size": int(args.batch_size),
                "dtype": DTYPE_NAME,
            },
            clients=args.clients,
            cumulative=cumulative,
            setup_seconds=setup_seconds,
            setup_bytes=setup_bytes,
        )
        _write_csv_atomic(summary_path, summary_rows)
        print(
            f"{architecture} round={round_index}/{args.rounds} "
            f"loss={sum(client_losses) / len(client_losses):.4f} "
            + " ".join(
                f"{row['compressor']}="
                f"{row['downlink_encode_seconds']:.3f}s/"
                f"{row['exponent_ratio_pct']:.3f}%"
                for row in round_rows
            ),
            flush=True,
        )


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--architectures",
        nargs="+",
        choices=("resnet18", "resnet34"),
        default=("resnet18", "resnet34"),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "results" / "datasets" / "cifar10_hf",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "federated-cifar10-200x10",
    )
    parser.add_argument("--clients", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--local-batches", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--train-subset", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--chunk-size", type=int, default=256 << 10)
    parser.add_argument("--compression-threshold", type=float, default=0.95)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for field in (
        "clients",
        "rounds",
        "local_batches",
        "batch_size",
        "threads",
        "chunk_size",
    ):
        if int(getattr(args, field)) <= 0:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    if args.train_subset is not None and args.train_subset <= 0:
        parser.error("--train-subset must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    if not 0 < args.compression_threshold <= 1:
        parser.error("--compression-threshold must be in (0, 1]")
    if args.chunk_size & (args.chunk_size - 1):
        parser.error("--chunk-size must be a power of two")
    return args


def main() -> None:
    args = _parse_args()
    torch.set_num_threads(max(1, int(args.threads)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    dataset, dataset_source = _training_dataset(Path(args.data_dir))
    for architecture in args.architectures:
        _run_architecture(args, dataset, dataset_source, architecture)


if __name__ == "__main__":
    main()
