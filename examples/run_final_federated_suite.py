"""Final federated compression experiment for the thesis.

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

The default is the final 10-client, 200-round ResNet-18/34 suite.  Override the
CLI arguments for smoke tests; for example::

    python examples/run_final_federated_suite.py \
        --architectures resnet18 --rounds 1 --clients 2 \
        --train-subset 20 --batch-size 2 --threads 2 \
        --device cpu --output-dir /tmp/neuralzip-federated-smoke
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import platform
import random
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, models, transforms

from neuralzip.archive import serialized_tensor_size
from neuralzip.neuralzip_codec import NeuralZipCodec


COMPRESSOR_SRC = "ZipNN src"
COMPRESSOR_NEURALZIP = "NeuralZip"
DATASET_NAME = "CIFAR-10"
DTYPE_NAME = "float32"


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


@dataclass
class EncodedState:
    payloads: Mapping[str, object]
    encode_seconds: float
    exponent_raw_bytes: int
    exponent_compressed_bytes: int
    archive_bytes: int
    raw_floating_bytes: int


class ZipNNSrcStateCodec:
    """Non-destructive adapter around the official ZipNN source package."""

    def __init__(self, *, threads: int, chunk_size: int, threshold: float):
        try:
            from zipnn import ZipNN
        except ImportError as error:
            raise ImportError(
                "The official zipnn package is required: pip install zipnn."
            ) from error

        self._ZipNN = ZipNN
        self._settings = {
            "method": "HUFFMAN",
            "input_format": "torch",
            "threads": int(threads),
            "compression_threshold": float(threshold),
            "compression_chunk": int(chunk_size),
        }
        self._encoder = ZipNN(**self._settings)
        self._decoder = ZipNN(**self._settings)

    @staticmethod
    def _extended_header_end(payload: bytes) -> int:
        if len(payload) < 33 or payload[:2] != b"ZN":
            raise ValueError("Invalid ZipNN source payload header.")
        dimensions = int(payload[32])
        offset = 33
        for _ in range(dimensions):
            if offset >= len(payload):
                raise ValueError("Truncated ZipNN tensor-shape header.")
            width = int(payload[offset])
            if width not in {1, 2, 4, 8}:
                raise ValueError(f"Invalid ZipNN shape width {width}.")
            offset += 1 + width
        return offset

    @classmethod
    def _fp32_exponent_sizes(cls, payload: bytes) -> tuple[int, int]:
        """Return exact raw/compressed bytes for ZipNN's FP32 exponent group.

        ZipNN source writes, after its tensor header, one compression-type byte
        and one native ``size_t`` cumulative length per chunk and byte group,
        followed by the group payloads.  FP32 bit reordering places the
        exponent/sign stream in byte group 3.  We charge that group's own
        metadata and payload, matching the per-stream accounting used by
        NeuralZip while excluding the tensor-global header.
        """

        byte_reorder = int(payload[5])
        bit_reorder = int(payload[6])
        if byte_reorder != 220 or bit_reorder != 1:
            raise ValueError(
                "Expected the official ZipNN FP32 byte/bit-reorder layout."
            )

        original_bytes = int.from_bytes(payload[16:24], byteorder="little")
        chunk_size = 1 << int(payload[14])
        if original_bytes % 4:
            raise ValueError("ZipNN FP32 payload has a non-FP32 byte length.")
        chunks = max(1, math.ceil(original_bytes / chunk_size))
        groups = 4
        exponent_group = 3
        size_width = struct.calcsize("P")
        header_end = cls._extended_header_end(payload)
        type_bytes = groups * chunks
        cumulative_start = header_end + type_bytes

        totals = []
        for group in range(groups):
            last_index = group * chunks + chunks - 1
            start = cumulative_start + last_index * size_width
            end = start + size_width
            if end > len(payload):
                raise ValueError("Truncated ZipNN cumulative-size metadata.")
            totals.append(int.from_bytes(payload[start:end], sys.byteorder))

        metadata_bytes = groups * chunks * size_width
        expected_size = header_end + type_bytes + metadata_bytes + sum(totals)
        if expected_size != len(payload):
            raise ValueError(
                "ZipNN source payload layout changed: "
                f"parsed {expected_size} bytes, observed {len(payload)}."
            )

        raw_exponent_bytes = original_bytes // 4
        compressed_exponent_bytes = totals[exponent_group] + chunks * (
            1 + size_width
        )
        return raw_exponent_bytes, compressed_exponent_bytes

    def encode(self, tensors: Mapping[str, torch.Tensor]) -> EncodedState:
        payloads = {}
        exponent_raw_bytes = 0
        exponent_compressed_bytes = 0
        archive_bytes = 0
        raw_floating_bytes = 0

        encode_seconds = 0.0
        for name, tensor in tensors.items():
            if tensor.dtype != torch.float32:
                raise TypeError(f"{name} is {tensor.dtype}; FP32 is required.")
            # ZipNN source reorders the input buffer in place.  The clone is a
            # required part of a non-destructive encode and is timed.
            started = time.perf_counter()
            compressor_input = tensor.detach().cpu().contiguous().clone()
            if tensor.numel() and compressor_input.data_ptr() == tensor.data_ptr():
                raise AssertionError("ZipNN source input clone aliases the model state.")
            payload = self._encoder.compress(compressor_input)
            encode_seconds += time.perf_counter() - started

            # Archive bookkeeping and exponent-ratio parsing are deliberately
            # outside the encoding timer, matching the NeuralZip adapter.
            payloads[name] = payload
            raw_bytes, compressed_bytes = self._fp32_exponent_sizes(payload)
            exponent_raw_bytes += raw_bytes
            exponent_compressed_bytes += compressed_bytes
            archive_bytes += len(payload)
            raw_floating_bytes += int(tensor.numel()) * tensor.element_size()

        return EncodedState(
            payloads=payloads,
            encode_seconds=encode_seconds,
            exponent_raw_bytes=exponent_raw_bytes,
            exponent_compressed_bytes=exponent_compressed_bytes,
            archive_bytes=archive_bytes,
            raw_floating_bytes=raw_floating_bytes,
        )

    def decode(self, encoded: EncodedState) -> dict[str, torch.Tensor]:
        return {
            name: self._decoder.decompress(payload).detach().cpu().contiguous()
            for name, payload in encoded.payloads.items()
        }


class NeuralZipStateCodec:
    """NeuralZip adapter with one frozen initial setup."""

    def __init__(self, *, threads: int, chunk_size: int, threshold: float):
        self.codec = NeuralZipCodec(
            threads=int(threads),
            chunk_size=int(chunk_size),
            compression_threshold=float(threshold),
        )
        self.setup_seconds = 0.0
        self.setup_bytes = 0

    def prepare_once(self, tensors: Mapping[str, torch.Tensor]) -> None:
        named_tensors = list(tensors.items())
        started = time.perf_counter()
        self.codec.prepare_clusters(named_tensors)
        self.codec.build_codebooks_and_select_modes(named_tensors)
        self.setup_seconds = time.perf_counter() - started
        self.setup_bytes = int(self.codec.codebook_metadata_bytes())

    def encode(self, tensors: Mapping[str, torch.Tensor]) -> EncodedState:
        for name, tensor in tensors.items():
            if tensor.dtype != torch.float32:
                raise TypeError(f"{name} is {tensor.dtype}; FP32 is required.")

        started = time.perf_counter()
        payload_list = self.codec.compress_tensors(list(tensors.items()))
        encode_seconds = time.perf_counter() - started
        payloads = {payload.name: payload for payload in payload_list}
        if payloads.keys() != tensors.keys():
            raise RuntimeError("NeuralZip changed the floating-state tensor names.")

        exponent_raw_bytes = 0
        exponent_compressed_bytes = 0
        archive_bytes = 0
        raw_floating_bytes = 0
        for payload in payload_list:
            exponent_group = payload.groups[int(payload.element_size) - 1]
            exponent_raw_bytes += int(exponent_group.symbol_count)
            exponent_compressed_bytes += int(exponent_group.compressed_bytes)
            archive_bytes += int(serialized_tensor_size(payload))
            raw_floating_bytes += int(payload.raw_bytes)

        return EncodedState(
            payloads=payloads,
            encode_seconds=encode_seconds,
            exponent_raw_bytes=exponent_raw_bytes,
            exponent_compressed_bytes=exponent_compressed_bytes,
            archive_bytes=archive_bytes,
            raw_floating_bytes=raw_floating_bytes,
        )

    def decode(self, encoded: EncodedState) -> dict[str, torch.Tensor]:
        return {
            name: self.codec.decompress_tensor(payload).detach().cpu().contiguous()
            for name, payload in encoded.payloads.items()
        }


def _build_model(architecture: str) -> torch.nn.Module:
    if architecture not in {"resnet18", "resnet34"}:
        raise ValueError(f"Unsupported final federated model: {architecture}.")
    model = getattr(models, architecture)(weights=None, num_classes=10)
    model.conv1 = torch.nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
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


def _assert_same_tensors(
    expected: Mapping[str, torch.Tensor],
    observed: Mapping[str, torch.Tensor],
    *,
    context: str,
) -> None:
    if expected.keys() != observed.keys():
        raise AssertionError(f"Tensor names changed during {context}.")
    for name, tensor in expected.items():
        candidate = observed[name]
        if tensor.shape != candidate.shape or tensor.dtype != candidate.dtype:
            raise AssertionError(
                f"Tensor metadata changed during {context}: {name} "
                f"expected shape={tuple(tensor.shape)} dtype={tensor.dtype}, "
                f"observed shape={tuple(candidate.shape)} dtype={candidate.dtype}."
            )
        expected_bytes = tensor.detach().cpu().contiguous().view(torch.uint8)
        observed_bytes = candidate.detach().cpu().contiguous().view(torch.uint8)
        if not torch.equal(expected_bytes, observed_bytes):
            raise AssertionError(f"Bitwise mismatch during {context}: {name}.")


def _clone_tensor_map(
    tensors: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Create an untimed immutable reference for codec input checks."""

    return {
        name: tensor.detach().cpu().contiguous().clone()
        for name, tensor in tensors.items()
    }


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
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
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
        raise ValueError("The real-data subset must contain at least one item per client.")

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


def _write_csv_atomic(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _encoded_metrics(
    downlink: EncodedState,
    uplinks: list[EncodedState],
    *,
    clients: int,
) -> dict[str, float | int]:
    exponent_raw = downlink.exponent_raw_bytes * int(clients) + sum(
        encoded.exponent_raw_bytes for encoded in uplinks
    )
    exponent_compressed = downlink.exponent_compressed_bytes * int(clients) + sum(
        encoded.exponent_compressed_bytes for encoded in uplinks
    )
    archive_bytes = downlink.archive_bytes * int(clients) + sum(
        encoded.archive_bytes for encoded in uplinks
    )
    raw_floating_bytes = downlink.raw_floating_bytes * int(clients) + sum(
        encoded.raw_floating_bytes for encoded in uplinks
    )
    critical_seconds = downlink.encode_seconds + max(
        (encoded.encode_seconds for encoded in uplinks), default=0.0
    )
    return {
        "encoding_critical_seconds": critical_seconds,
        "exponent_raw_bytes": exponent_raw,
        "exponent_compressed_bytes": exponent_compressed,
        "exponent_ratio_pct": (
            100.0 * exponent_compressed / exponent_raw if exponent_raw else 0.0
        ),
        "archive_bytes": archive_bytes,
        "raw_floating_bytes": raw_floating_bytes,
    }


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

    src_codec = ZipNNSrcStateCodec(
        threads=args.threads,
        chunk_size=args.chunk_size,
        threshold=args.compression_threshold,
    )
    neuralzip_codec = NeuralZipStateCodec(
        threads=args.threads,
        chunk_size=args.chunk_size,
        threshold=args.compression_threshold,
    )
    neuralzip_codec.prepare_once(initial_floating)
    codecs = {
        COMPRESSOR_SRC: src_codec,
        COMPRESSOR_NEURALZIP: neuralzip_codec,
    }
    setup_seconds = {
        COMPRESSOR_SRC: 0.0,
        COMPRESSOR_NEURALZIP: neuralzip_codec.setup_seconds,
    }
    setup_bytes = {
        COMPRESSOR_SRC: 0,
        COMPRESSOR_NEURALZIP: neuralzip_codec.setup_bytes,
    }

    # Warm both complete encode/decode paths on the same initial model after
    # NeuralZip setup.  This is intentionally outside every reported round.
    warmup_input_reference = _clone_tensor_map(initial_floating)
    warmup_decoded = {}
    for name in (COMPRESSOR_SRC, COMPRESSOR_NEURALZIP):
        warmup_encoded = codecs[name].encode(initial_floating)
        _assert_same_tensors(
            warmup_input_reference,
            initial_floating,
            context=f"{architecture} {name} warmup input immutability",
        )
        warmup_decoded[name] = codecs[name].decode(warmup_encoded)
        _assert_same_tensors(
            initial_floating,
            warmup_decoded[name],
            context=f"{architecture} {name} warmup",
        )
    _assert_same_tensors(
        warmup_decoded[COMPRESSOR_SRC],
        warmup_decoded[COMPRESSOR_NEURALZIP],
        context=f"{architecture} cross-codec warmup",
    )
    del warmup_decoded, warmup_encoded, warmup_input_reference

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
            "downlink encode plus maximum client uplink encode per round; "
            "decode and training excluded"
        ),
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

    cumulative = {
        name: {
            "encoding_seconds": 0.0,
            "exponent_raw_bytes": 0,
            "exponent_compressed_bytes": 0,
            "archive_bytes": 0,
            "raw_floating_bytes": 0,
        }
        for name in codecs
    }

    for round_index in range(1, int(args.rounds) + 1):
        floating_server = _floating_state(server_state)
        floating_server_reference = _clone_tensor_map(floating_server)
        downlinks = {}
        decoded_downlinks = {}
        order = (
            [COMPRESSOR_SRC, COMPRESSOR_NEURALZIP]
            if round_index % 2
            else [COMPRESSOR_NEURALZIP, COMPRESSOR_SRC]
        )
        for name in order:
            downlinks[name] = codecs[name].encode(floating_server)
            _assert_same_tensors(
                floating_server_reference,
                floating_server,
                context=(
                    f"{architecture} round {round_index} {name} "
                    "downlink input immutability"
                ),
            )
            decoded_downlinks[name] = codecs[name].decode(downlinks[name])
            _assert_same_tensors(
                floating_server,
                decoded_downlinks[name],
                context=f"{architecture} round {round_index} {name} downlink",
            )
        _assert_same_tensors(
            decoded_downlinks[COMPRESSOR_SRC],
            decoded_downlinks[COMPRESSOR_NEURALZIP],
            context=f"{architecture} round {round_index} cross-codec downlink",
        )
        decoded_server_state = _merge_decoded_state(
            server_state, decoded_downlinks[COMPRESSOR_NEURALZIP]
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
            floating_client_reference = _clone_tensor_map(floating_client)
            decoded_uplinks = {}
            client_order = (
                order if client_index % 2 == 0 else list(reversed(order))
            )
            for name in client_order:
                encoded = codecs[name].encode(floating_client)
                _assert_same_tensors(
                    floating_client_reference,
                    floating_client,
                    context=(
                        f"{architecture} round {round_index} client "
                        f"{client_index} {name} uplink input immutability"
                    ),
                )
                uplinks[name].append(encoded)
                decoded_uplinks[name] = codecs[name].decode(encoded)
                _assert_same_tensors(
                    floating_client,
                    decoded_uplinks[name],
                    context=(
                        f"{architecture} round {round_index} client "
                        f"{client_index} {name} uplink"
                    ),
                )
            _assert_same_tensors(
                decoded_uplinks[COMPRESSOR_SRC],
                decoded_uplinks[COMPRESSOR_NEURALZIP],
                context=(
                    f"{architecture} round {round_index} client "
                    f"{client_index} cross-codec uplink"
                ),
            )
            decoded_client_states.append(
                _merge_decoded_state(
                    client_state, decoded_uplinks[COMPRESSOR_NEURALZIP]
                )
            )
            del (
                client_state,
                floating_client,
                floating_client_reference,
                decoded_uplinks,
            )

        server_state = _average_decoded_states(decoded_client_states)
        del decoded_client_states, decoded_downlinks, floating_server_reference

        round_rows = []
        for name in (COMPRESSOR_SRC, COMPRESSOR_NEURALZIP):
            metrics = _encoded_metrics(
                downlinks[name], uplinks[name], clients=args.clients
            )
            totals = cumulative[name]
            totals["encoding_seconds"] += float(
                metrics["encoding_critical_seconds"]
            )
            for field in (
                "exponent_raw_bytes",
                "exponent_compressed_bytes",
                "archive_bytes",
                "raw_floating_bytes",
            ):
                totals[field] += int(metrics[field])
            round_rows.append(
                {
                    "architecture": architecture,
                    "dataset": DATASET_NAME,
                    "round": round_index,
                    "clients": int(args.clients),
                    "local_batches": int(args.local_batches),
                    "batch_size": int(args.batch_size),
                    "dtype": DTYPE_NAME,
                    "compressor": name,
                    "setup_seconds_once": setup_seconds[name],
                    "setup_bytes_once": setup_bytes[name],
                    "downlink_encode_seconds": downlinks[name].encode_seconds,
                    "uplink_encode_max_seconds": max(
                        encoded.encode_seconds for encoded in uplinks[name]
                    ),
                    "encoding_critical_seconds": metrics[
                        "encoding_critical_seconds"
                    ],
                    "encoding_critical_cumulative_seconds": totals[
                        "encoding_seconds"
                    ],
                    "exponent_raw_bytes": metrics["exponent_raw_bytes"],
                    "exponent_compressed_bytes": metrics[
                        "exponent_compressed_bytes"
                    ],
                    "exponent_ratio_pct": metrics["exponent_ratio_pct"],
                    "archive_bytes": metrics["archive_bytes"],
                    "raw_floating_bytes": metrics["raw_floating_bytes"],
                    "train_loss_mean": sum(client_losses) / len(client_losses),
                    "roundtrip_ok": True,
                }
            )
        _append_csv(round_path, round_rows)

        summary_rows = []
        for name in (COMPRESSOR_SRC, COMPRESSOR_NEURALZIP):
            totals = cumulative[name]
            raw_exponent = int(totals["exponent_raw_bytes"])
            compressed_exponent = int(totals["exponent_compressed_bytes"])
            setup_communication_bytes = int(setup_bytes[name]) * int(args.clients)
            compressed_with_setup = (
                compressed_exponent + setup_communication_bytes
            )
            summary_rows.append(
                {
                    "architecture": architecture,
                    "dataset": DATASET_NAME,
                    "compressor": name,
                    "completed_rounds": round_index,
                    "clients": int(args.clients),
                    "local_batches": int(args.local_batches),
                    "batch_size": int(args.batch_size),
                    "dtype": DTYPE_NAME,
                    "setup_seconds_once": setup_seconds[name],
                    "setup_bytes_once": setup_bytes[name],
                    "encoding_critical_seconds": totals["encoding_seconds"],
                    "encoding_plus_setup_seconds": (
                        totals["encoding_seconds"] + setup_seconds[name]
                    ),
                    "exponent_raw_bytes": raw_exponent,
                    "exponent_compressed_bytes": compressed_exponent,
                    "exponent_ratio_pct": (
                        100.0 * compressed_exponent / raw_exponent
                        if raw_exponent
                        else 0.0
                    ),
                    "setup_communication_bytes": setup_communication_bytes,
                    "exponent_compressed_plus_setup_bytes": compressed_with_setup,
                    "exponent_ratio_with_setup_pct": (
                        100.0 * compressed_with_setup / raw_exponent
                        if raw_exponent
                        else 0.0
                    ),
                    "archive_bytes": totals["archive_bytes"],
                    "raw_floating_bytes": totals["raw_floating_bytes"],
                    "roundtrip_ok": True,
                }
            )
        _write_csv_atomic(summary_path, summary_rows)
        print(
            f"{architecture} round={round_index}/{args.rounds} "
            f"loss={sum(client_losses) / len(client_losses):.4f} "
            + " ".join(
                f"{row['compressor']}="
                f"{row['encoding_critical_seconds']:.3f}s/"
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
        default=Path("examples/outputs/datasets/cifar10_hf"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples/outputs/final_federated_cifar10_200x10"),
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
