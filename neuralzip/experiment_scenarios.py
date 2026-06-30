from __future__ import annotations

import gc
import math
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

from .compressor import Compressor
from .neuralzip_codec import NeuralZipCodec
from .zipnn_codec import ZipNNCodec


DEFAULT_METHODS = (
    ("Global Huffman", "global-huffman"),
    ("ZipNN local", "zipnn-local"),
    ("NeuralZip-Fast", "neuralzip-fast"),
    ("NeuralZip-Ratio", "neuralzip-ratio"),
)


FAST_METHODS = (
    ("ZipNN local", "zipnn-local"),
    ("NeuralZip-Fast", "neuralzip-fast"),
)


@dataclass
class MappingModel:
    model_name: str
    model: dict[str, torch.Tensor]


def _ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _bit_entropy_from_counts(ones, total):
    if total <= 0:
        return 0.0
    p = ones / total
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p)))


def _byte_bitplane_counts(data):
    if not data:
        return [0] * 8, 0
    tensor = torch.frombuffer(bytearray(data), dtype=torch.uint8)
    counts = [int(((tensor >> bit) & 1).sum().item()) for bit in range(8)]
    return counts, int(tensor.numel())


def _update_bitplane_counts(counts, data):
    bit_counts, byte_count = _byte_bitplane_counts(data)
    for bit, value in enumerate(bit_counts):
        counts[bit] += value
    return byte_count


def _tensor_mapping_from_module(module, *, dtype=None, predicate=None):
    tensors = {}
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            if not parameter.is_floating_point():
                continue
            if predicate is not None and not predicate(name):
                continue
            tensor = parameter.detach().cpu().contiguous()
            if dtype is not None:
                tensor = tensor.to(dtype=dtype)
            tensors[name] = tensor
    return tensors


def _normalise_dtype(dtype):
    if dtype is None:
        return None
    if isinstance(dtype, torch.dtype):
        return dtype
    dtype_key = str(dtype).lower().replace("torch.", "").replace("-", "_")
    dtype_map = {
        "auto": "auto",
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    if dtype_key not in dtype_map:
        choices = ", ".join(sorted(dtype_map))
        raise ValueError(f"Unsupported dtype {dtype!r}. Expected one of: {choices}.")
    return dtype_map[dtype_key]


def _model_spec_value(spec, key, default=None):
    if isinstance(spec, dict):
        return spec.get(key, default)
    if key == "name":
        return spec[0]
    if key == "model_id":
        return spec[1]
    return default


def _load_transformers_module(
    model_id,
    *,
    dtype=torch.bfloat16,
    local_files_only=True,
    trust_remote_code=True,
    **kwargs,
):
    from transformers import AutoModel, AutoModelForCausalLM

    dtype = _normalise_dtype(dtype)
    load_kwargs = {
        "local_files_only": bool(local_files_only),
        "trust_remote_code": bool(trust_remote_code),
        "low_cpu_mem_usage": True,
    }
    load_kwargs.update(kwargs)
    if dtype is not None and "dtype" not in load_kwargs and "torch_dtype" not in load_kwargs:
        load_kwargs["dtype"] = dtype

    errors = []
    for loader in (AutoModelForCausalLM, AutoModel):
        try:
            return loader.from_pretrained(model_id, **load_kwargs)
        except Exception as error:
            errors.append(f"{loader.__name__}: {type(error).__name__}: {error}")
    raise ValueError(
        f"Could not load {model_id} with supported model loaders:\n"
        + "\n".join(errors)
    )


def _load_safetensors_mapping(
    model_id,
    *,
    dtype=torch.bfloat16,
    local_files_only=True,
):
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    snapshot_path = Path(
        snapshot_download(
            model_id,
            local_files_only=bool(local_files_only),
            allow_patterns=("*.safetensors", "*.safetensors.index.json"),
        )
    )
    files = sorted(snapshot_path.glob("*.safetensors"))
    if not files:
        raise ValueError(f"No safetensors files found for {model_id}.")

    target_dtype = _normalise_dtype(dtype)
    if target_dtype == "auto":
        target_dtype = None

    tensors = {}
    for file_index, file_path in enumerate(files):
        state = load_file(str(file_path), device="cpu")
        for name, tensor in state.items():
            if not torch.is_tensor(tensor) or not tensor.is_floating_point():
                continue
            value = tensor.detach().cpu().contiguous()
            if target_dtype is not None:
                value = value.to(dtype=target_dtype)
            key = name if name not in tensors else f"{file_path.stem}.{name}"
            tensors[key] = value
        del state
        gc.collect()

    if not tensors:
        raise ValueError(f"No floating-point tensors found in safetensors for {model_id}.")
    return tensors


def _result_exponent_metrics(result):
    raw_bytes = result.get("raw_bytes")
    compressed_bytes = result.get("compressed_bytes")
    scope = str(result.get("scope", "")).lower()
    if "exponent" in scope:
        return {
            "exponent_raw_bytes": raw_bytes,
            "exponent_compressed_bytes": compressed_bytes,
            "exponent_compressed_pct": (
                100 * compressed_bytes / raw_bytes if raw_bytes else None
            ),
        }

    stream_results = result.get("stream_results") or ()
    if not stream_results:
        return {
            "exponent_raw_bytes": None,
            "exponent_compressed_bytes": None,
            "exponent_compressed_pct": None,
        }

    # ZipNN byte grouping places the exponent/significant high byte in the last
    # stream for bf16 checkpoints. The large-model experiments use bf16.
    exponent_stream = max(stream_results, key=lambda row: int(row["stream_id"]))
    exponent_raw = int(exponent_stream["raw_bytes"])
    exponent_compressed = int(exponent_stream["compressed_bytes"])
    return {
        "exponent_raw_bytes": exponent_raw,
        "exponent_compressed_bytes": exponent_compressed,
        "exponent_compressed_pct": (
            100 * exponent_compressed / exponent_raw if exponent_raw else None
        ),
    }


def _compress_mapping(model_name, tensor_mapping, *, methods=DEFAULT_METHODS):
    rows = []
    wrapped = MappingModel(model_name=model_name, model=tensor_mapping)
    for compressor_name, method in methods:
        start = time.perf_counter()
        try:
            result = Compressor(method, verify=False).compress(wrapped)
            elapsed = time.perf_counter() - start
            exponent_metrics = _result_exponent_metrics(result)
            rows.append(
                {
                    "model": model_name,
                    "compressor": compressor_name,
                    "method": result.get("method"),
                    "scope": result.get("scope", "full checkpoint"),
                    "status": "ok",
                    "raw_bytes": result.get("raw_bytes"),
                    "compressed_bytes": result.get("compressed_bytes"),
                    "compression_ratio": result.get("compression_ratio"),
                    "compressed_pct": 100 * result.get("compression_ratio"),
                    "full_compressed_pct": 100 * result.get("compression_ratio"),
                    **exponent_metrics,
                    "precompute_seconds": result.get("precompute_seconds"),
                    "encode_seconds": result.get("encode_seconds"),
                    "cold_encode_seconds": result.get(
                        "cold_encode_seconds",
                        elapsed,
                    ),
                    "clusters": result.get("clusters"),
                    "codebooks": (
                        len(result.get("codebooks"))
                        if result.get("codebooks") is not None
                        else None
                    ),
                    "huffman_chunks": result.get("huffman_chunks"),
                    "raw_chunks": result.get("raw_chunks"),
                    "error": None,
                }
            )
        except Exception as error:
            rows.append(
                {
                    "model": model_name,
                    "compressor": compressor_name,
                    "method": method,
                    "scope": None,
                    "status": "failed",
                    "raw_bytes": None,
                    "compressed_bytes": None,
                    "compression_ratio": None,
                    "compressed_pct": None,
                    "full_compressed_pct": None,
                    "exponent_raw_bytes": None,
                    "exponent_compressed_bytes": None,
                    "exponent_compressed_pct": None,
                    "precompute_seconds": None,
                    "encode_seconds": None,
                    "cold_encode_seconds": None,
                    "clusters": None,
                    "codebooks": None,
                    "huffman_chunks": None,
                    "raw_chunks": None,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    return rows


def run_hf_model_suite_experiment(
    output_dir,
    *,
    model_suite,
    experiment,
    methods=DEFAULT_METHODS,
    default_dtype=torch.bfloat16,
    local_files_only=True,
    trust_remote_code=True,
):
    output_dir = _ensure_dir(output_dir)
    result_path = output_dir / f"{experiment.replace('-', '_')}_results.csv"
    rows = []
    for spec in model_suite:
        model_name = _model_spec_value(spec, "name")
        model_id = _model_spec_value(spec, "model_id")
        dtype = _model_spec_value(spec, "dtype", None)
        if dtype is None:
            dtype = _model_spec_value(spec, "default_dtype", default_dtype)
        representation = _model_spec_value(spec, "representation", None)
        category = _model_spec_value(spec, "category", None)
        load_kwargs = dict(_model_spec_value(spec, "load_kwargs", {}) or {})
        model = None
        try:
            try:
                tensors = _load_safetensors_mapping(
                    model_id,
                    dtype=dtype,
                    local_files_only=local_files_only,
                )
                load_backend = "safetensors"
            except Exception as safetensors_error:
                model = _load_transformers_module(
                    model_id,
                    dtype=dtype,
                    local_files_only=local_files_only,
                    trust_remote_code=trust_remote_code,
                    **load_kwargs,
                )
                tensor_dtype = (
                    None
                    if _normalise_dtype(dtype) == "auto"
                    else _normalise_dtype(dtype)
                )
                tensors = _tensor_mapping_from_module(model, dtype=tensor_dtype)
                load_backend = "transformers"
            if not tensors:
                rows.append(
                    {
                        "experiment": experiment,
                        "model": model_name,
                        "model_name": model_name,
                        "model_id": model_id,
                        "representation": representation,
                        "category": category,
                        "status": "no-floating-tensors",
                        "error": "Loaded model exposes no floating-point tensors.",
                        "load_backend": None,
                    }
                )
                continue
            for row in _compress_mapping(model_name, tensors, methods=methods):
                row["experiment"] = experiment
                row["model_name"] = model_name
                row["model_id"] = model_id
                row["representation"] = representation
                row["category"] = category
                row["dtype"] = str(dtype).replace("torch.", "")
                row["tensor_count"] = len(tensors)
                row["load_backend"] = load_backend
                rows.append(row)
        except Exception as error:
            rows.append(
                {
                    "experiment": experiment,
                    "model": model_name,
                    "model_name": model_name,
                    "model_id": model_id,
                    "representation": representation,
                    "category": category,
                    "status": "load-failed",
                    "error": f"{type(error).__name__}: {error}",
                    "load_backend": None,
                }
            )
        finally:
            if model is not None:
                del model
            gc.collect()
            if rows:
                pd.DataFrame(rows).to_csv(result_path, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(result_path, index=False)
    return df


def run_moe_experiment(
    output_dir,
    *,
    model_suite=None,
    methods=DEFAULT_METHODS,
    dtype=torch.bfloat16,
    local_files_only=True,
    trust_remote_code=True,
):
    if model_suite is not None:
        return run_hf_model_suite_experiment(
            output_dir,
            model_suite=model_suite,
            experiment="moe",
            methods=methods,
            default_dtype=dtype,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )

    from transformers import MixtralConfig, MixtralForCausalLM

    output_dir = _ensure_dir(output_dir)
    model_configs = (
        {
            "name": "Tiny-Mixtral-4E",
            "num_local_experts": 4,
            "num_experts_per_tok": 2,
        },
        {
            "name": "Tiny-Mixtral-8E",
            "num_local_experts": 8,
            "num_experts_per_tok": 2,
        },
    )
    rows = []
    for index, spec in enumerate(model_configs):
        torch.manual_seed(1000 + index)
        config = MixtralConfig(
            vocab_size=512,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            num_local_experts=spec["num_local_experts"],
            num_experts_per_tok=spec["num_experts_per_tok"],
            max_position_embeddings=128,
        )
        model = MixtralForCausalLM(config).to(dtype=dtype)
        full = _tensor_mapping_from_module(model, dtype=dtype)
        experts = _tensor_mapping_from_module(
            model,
            dtype=dtype,
            predicate=lambda name: ".experts." in name,
        )
        for scope, tensors in (("full-checkpoint", full), ("experts-only", experts)):
            for row in _compress_mapping(
                f'{spec["name"]}:{scope}',
                tensors,
                methods=methods,
            ):
                row["experiment"] = "moe"
                row["checkpoint_scope"] = scope
                row["num_experts"] = spec["num_local_experts"]
                row["active_experts"] = spec["num_experts_per_tok"]
                row["tensor_count"] = len(tensors)
                rows.append(row)
        del model, full, experts
        gc.collect()

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "moe_results.csv", index=False)
    return df


def run_lossy_lossless_experiment(
    output_dir,
    *,
    model_suite=None,
    methods=DEFAULT_METHODS,
    local_files_only=True,
    trust_remote_code=True,
):
    from transformers import AutoModel

    output_dir = _ensure_dir(output_dir)
    if model_suite is not None and any(isinstance(spec, dict) for spec in model_suite):
        return run_hf_model_suite_experiment(
            output_dir,
            model_suite=model_suite,
            experiment="lossy-lossless",
            methods=methods,
            default_dtype=torch.bfloat16,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )

    if model_suite is None:
        model_suite = (
            ("GPT-2", "gpt2"),
            ("BERT Base", "google-bert/bert-base-uncased"),
        )

    representations = (
        ("fp32-original", torch.float32, FAST_METHODS),
        ("bf16-rounded", torch.bfloat16, methods),
    )

    rows = []
    for model_name, model_id in model_suite:
        for representation, dtype, selected_methods in representations:
            try:
                model = AutoModel.from_pretrained(
                    model_id,
                    dtype=dtype,
                    local_files_only=local_files_only,
                )
            except Exception as error:
                rows.append(
                    {
                        "experiment": "lossy-lossless",
                        "model": model_name,
                        "model_id": model_id,
                        "representation": representation,
                        "status": "load-failed",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                continue
            tensors = _tensor_mapping_from_module(model, dtype=dtype)
            for row in _compress_mapping(
                f"{model_name}:{representation}",
                tensors,
                methods=selected_methods,
            ):
                row["experiment"] = "lossy-lossless"
                row["model_name"] = model_name
                row["model_id"] = model_id
                row["representation"] = representation
                row["dtype"] = str(dtype).replace("torch.", "")
                row["tensor_count"] = len(tensors)
                rows.append(row)
            del model, tensors
            gc.collect()

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "lossy_lossless_results.csv", index=False)
    return df


def _hf_model_files(model_id, *, extensions):
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    files = []
    for item in fs.ls(model_id, detail=True):
        if item.get("type") != "file":
            continue
        name = item["name"]
        if any(str(name).lower().endswith(extension) for extension in extensions):
            files.append(
                {
                    "path": name,
                    "size": int(item.get("size") or 0),
                }
            )
    return fs, files


def _sample_hf_binary_file(
    fs,
    path,
    *,
    file_size,
    max_sample_bytes,
    chunk_size=4 << 20,
):
    suffix = str(path).lower()
    with fs.open(path, "rb") as handle:
        offset = 0
        if suffix.endswith(".safetensors"):
            header_size_bytes = handle.read(8)
            if len(header_size_bytes) == 8:
                header_size = int.from_bytes(header_size_bytes, "little")
                offset = 8 + header_size
        elif suffix.endswith(".gguf"):
            offset = min(4096, int(file_size))

        try:
            handle.seek(offset)
        except Exception:
            if offset:
                handle.read(offset)

        remaining = max(0, int(file_size) - int(offset))
        target = min(int(max_sample_bytes), remaining)
        counts = [0] * 8
        sampled = 0
        while sampled < target:
            data = handle.read(min(int(chunk_size), target - sampled))
            if not data:
                break
            sampled += _update_bitplane_counts(counts, data)
        return counts, sampled


def run_quantized_bitplane_experiment(
    output_dir,
    *,
    model_suite,
    max_sample_bytes_per_file=64 << 20,
    low_entropy_threshold=0.5,
):
    output_dir = _ensure_dir(output_dir)
    rows = []
    extensions = (".safetensors", ".bin", ".pt", ".gguf")

    for spec in model_suite:
        if _model_spec_value(spec, "category") != "quantized":
            continue
        model_name = _model_spec_value(spec, "name")
        model_id = _model_spec_value(spec, "model_id")
        try:
            fs, files = _hf_model_files(model_id, extensions=extensions)
            if not files:
                raise ValueError("No supported model files found.")

            total_counts = [0] * 8
            total_bytes = 0
            sampled_files = 0
            for item in files:
                counts, sampled = _sample_hf_binary_file(
                    fs,
                    item["path"],
                    file_size=item["size"],
                    max_sample_bytes=max_sample_bytes_per_file,
                )
                if sampled <= 0:
                    continue
                sampled_files += 1
                total_bytes += sampled
                for bit, value in enumerate(counts):
                    total_counts[bit] += value

            entropies = [
                _bit_entropy_from_counts(total_counts[bit], total_bytes)
                for bit in range(8)
            ]
            min_entropy = min(entropies) if entropies else None
            low_bits = [
                bit
                for bit, entropy in enumerate(entropies)
                if entropy < float(low_entropy_threshold)
            ]
            row = {
                "experiment": "lossy-lossless-bitplanes",
                "model": model_name,
                "model_name": model_name,
                "model_id": model_id,
                "status": "ok",
                "files": len(files),
                "sampled_files": sampled_files,
                "sampled_bytes": total_bytes,
                "min_bit_entropy": min_entropy,
                "mean_bit_entropy": (
                    sum(entropies) / len(entropies) if entropies else None
                ),
                "low_entropy_threshold": float(low_entropy_threshold),
                "low_entropy_bits": ",".join(str(bit) for bit in low_bits),
                "low_entropy_bit_count": len(low_bits),
                "error": None,
            }
            for bit, entropy in enumerate(entropies):
                row[f"bit_{bit}_entropy"] = entropy
            rows.append(row)
        except Exception as error:
            rows.append(
                {
                    "experiment": "lossy-lossless-bitplanes",
                    "model": model_name,
                    "model_name": model_name,
                    "model_id": model_id,
                    "status": "failed",
                    "files": None,
                    "sampled_files": None,
                    "sampled_bytes": None,
                    "min_bit_entropy": None,
                    "mean_bit_entropy": None,
                    "low_entropy_threshold": float(low_entropy_threshold),
                    "low_entropy_bits": None,
                    "low_entropy_bit_count": None,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
        pd.DataFrame(rows).to_csv(
            output_dir / "quantized_bitplane_entropy.csv",
            index=False,
        )

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "quantized_bitplane_entropy.csv", index=False)
    return df


def build_resnet18_cifar():
    import torch.nn as nn
    from torchvision import models

    model = models.resnet18(weights=None, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def train_resnet18_cifar_checkpoints(
    output_dir,
    *,
    epochs=3,
    train_subset=5000,
    test_subset=1000,
    batch_size=128,
    learning_rate=0.05,
    seed=2026,
    download=True,
    dataset="cifar10",
    data_dir=None,
    evaluate_each_epoch=True,
    device=None,
):
    import torch.nn as nn
    import torch.nn as nn
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms

    output_dir = _ensure_dir(output_dir)
    checkpoint_dir = _ensure_dir(output_dir / "checkpoints")
    if data_dir is None:
        data_dir = output_dir / "data"
    data_dir = _ensure_dir(data_dir)

    torch.manual_seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    dataset_key = str(dataset).lower().replace("-", "_")
    if dataset_key == "cifar10":
        train_data = datasets.CIFAR10(
            root=data_dir,
            train=True,
            download=download,
            transform=transform,
        )
        test_data = datasets.CIFAR10(
            root=data_dir,
            train=False,
            download=download,
            transform=transform,
        )
    elif dataset_key in {"hf_cifar10", "cifar10_hf", "parquet_cifar10"}:
        train_parquet = data_dir / "plain_text" / "train-00000-of-00001.parquet"
        test_parquet = data_dir / "plain_text" / "test-00000-of-00001.parquet"
        if not train_parquet.exists() or not test_parquet.exists():
            raise FileNotFoundError(
                "Hugging Face CIFAR-10 parquet files were not found under "
                f"{data_dir / 'plain_text'}. Download them from "
                "https://huggingface.co/datasets/uoft-cs/cifar10/tree/main/plain_text"
            )
        train_data = _HFCIFAR10Parquet(train_parquet, transform=transform)
        test_data = _HFCIFAR10Parquet(test_parquet, transform=transform)
    elif dataset_key in {"fake", "fake_cifar", "synthetic"}:
        train_data = datasets.FakeData(
            size=int(train_subset),
            image_size=(3, 32, 32),
            num_classes=10,
            transform=transform,
            random_offset=0,
        )
        test_data = datasets.FakeData(
            size=int(test_subset),
            image_size=(3, 32, 32),
            num_classes=10,
            transform=transform,
            random_offset=int(train_subset),
        )
    else:
        raise ValueError("dataset must be 'cifar10', 'hf-cifar10', or 'fake'.")
    train_indices = list(range(min(int(train_subset), len(train_data))))
    test_indices = list(range(min(int(test_subset), len(test_data))))
    train_loader = DataLoader(
        Subset(train_data, train_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        Subset(test_data, test_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )

    model = build_resnet18_cifar().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=5e-4,
    )

    metrics = []

    def evaluate():
        model.eval()
        correct = 0
        total = 0
        loss_total = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(images)
                loss = criterion(logits, labels)
                loss_total += float(loss.item()) * labels.numel()
                correct += int((logits.argmax(dim=1) == labels).sum().item())
                total += int(labels.numel())
        return loss_total / max(total, 1), correct / max(total, 1)

    def save_checkpoint(epoch):
        path = checkpoint_dir / f"resnet18_cifar10_epoch_{epoch:02d}.pt"
        torch.save(
            {
                "epoch": int(epoch),
                "model_state": {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                },
            },
            path,
        )
        return path

    save_checkpoint(0)
    for epoch in range(1, int(epochs) + 1):
        model.train()
        start = time.perf_counter()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += float(loss.item()) * labels.numel()
            train_correct += int((logits.argmax(dim=1) == labels).sum().item())
            train_total += int(labels.numel())

        if evaluate_each_epoch:
            test_loss, test_accuracy = evaluate()
        else:
            test_loss, test_accuracy = None, None
        checkpoint_path = save_checkpoint(epoch)
        metrics.append(
            {
                "epoch": epoch,
                "device": str(device),
                "train_loss": train_loss / max(train_total, 1),
                "train_accuracy": train_correct / max(train_total, 1),
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "seconds": time.perf_counter() - start,
                "checkpoint_path": str(checkpoint_path),
            }
        )
        pd.DataFrame(metrics).to_csv(output_dir / "training_metrics.csv", index=False)
        if test_accuracy is None:
            print(
                f"epoch={epoch} train_loss={metrics[-1]['train_loss']:.4f}",
                flush=True,
            )
        else:
            print(
                f"epoch={epoch} train_loss={metrics[-1]['train_loss']:.4f} "
                f"test_accuracy={test_accuracy:.4f}",
                flush=True,
            )

    df = pd.DataFrame(metrics)
    df.to_csv(output_dir / "training_metrics.csv", index=False)
    return df


def compress_checkpoint_directory(
    checkpoint_dir,
    output_dir,
    *,
    methods=DEFAULT_METHODS,
    dtype=torch.bfloat16,
):
    checkpoint_dir = Path(checkpoint_dir)
    output_dir = _ensure_dir(output_dir)
    rows = []
    for path in sorted(checkpoint_dir.glob("*.pt")):
        checkpoint = torch.load(path, map_location="cpu")
        state = checkpoint.get("model_state", checkpoint)
        tensors = {
            name: tensor.detach().cpu().contiguous().to(dtype=dtype)
            for name, tensor in state.items()
            if torch.is_tensor(tensor) and tensor.is_floating_point()
        }
        for row in _compress_mapping(path.stem, tensors, methods=methods):
            row["experiment"] = "checkpoints"
            row["checkpoint"] = path.name
            row["epoch"] = int(checkpoint.get("epoch", -1))
            row["dtype"] = str(dtype).replace("torch.", "")
            row["tensor_count"] = len(tensors)
            rows.append(row)
        del checkpoint, state, tensors
        gc.collect()

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "checkpoint_compression_results.csv", index=False)
    return df


def _average_floating_states(states):
    averaged = {}
    first = states[0]
    for key, value in first.items():
        if torch.is_tensor(value) and value.is_floating_point():
            stacked = torch.stack([state[key].to(torch.float32) for state in states])
            averaged[key] = stacked.mean(dim=0).to(dtype=value.dtype)
        else:
            averaged[key] = value.clone() if torch.is_tensor(value) else value
    return averaged


def run_federated_resnet_synthetic(
    output_dir,
    *,
    clients=4,
    rounds=2,
    local_batches=8,
    batch_size=64,
    learning_rate=0.02,
    seed=2026,
    device=None,
):
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    output_dir = _ensure_dir(output_dir)
    checkpoint_dir = _ensure_dir(output_dir / "client_checkpoints")

    torch.manual_seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    criterion = nn.CrossEntropyLoss()
    server = build_resnet18_cifar()
    server_state = {
        key: value.detach().cpu().clone()
        for key, value in server.state_dict().items()
    }
    rows = []

    for round_index in range(1, int(rounds) + 1):
        client_states = []
        for client_index in range(int(clients)):
            model = build_resnet18_cifar().to(device)
            model.load_state_dict(server_state, strict=True)
            model.train()
            optimizer = optim.SGD(
                model.parameters(),
                lr=learning_rate,
                momentum=0.9,
                weight_decay=5e-4,
            )
            data = datasets.FakeData(
                size=int(local_batches) * int(batch_size),
                image_size=(3, 32, 32),
                num_classes=10,
                transform=transform,
                random_offset=round_index * 10000 + client_index * 1000,
            )
            loader = DataLoader(
                data,
                batch_size=batch_size,
                shuffle=True,
                num_workers=2,
                pin_memory=device.type == "cuda",
            )

            start = time.perf_counter()
            total_loss = 0.0
            total_correct = 0
            total_examples = 0
            for images, labels in loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                total_loss += float(loss.item()) * labels.numel()
                total_correct += int((logits.argmax(dim=1) == labels).sum().item())
                total_examples += int(labels.numel())

            client_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            client_states.append(client_state)
            checkpoint_path = (
                checkpoint_dir
                / f"round_{round_index:02d}_client_{client_index:02d}.pt"
            )
            torch.save(
                {
                    "round": round_index,
                    "client": client_index,
                    "model_state": client_state,
                },
                checkpoint_path,
            )
            rows.append(
                {
                    "round": round_index,
                    "client": client_index,
                    "device": str(device),
                    "examples": total_examples,
                    "train_loss": total_loss / max(total_examples, 1),
                    "train_accuracy": total_correct / max(total_examples, 1),
                    "seconds": time.perf_counter() - start,
                    "checkpoint_path": str(checkpoint_path),
                }
            )
            del model, data, loader, client_state
            if device.type == "cuda":
                torch.cuda.empty_cache()

        server_state = _average_floating_states(client_states)
        del client_states
        gc.collect()

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "federated_training_metrics.csv", index=False)
    return df


def compress_federated_directory(
    checkpoint_dir,
    output_dir,
    *,
    methods=FAST_METHODS,
    dtype=torch.bfloat16,
):
    checkpoint_dir = Path(checkpoint_dir)
    output_dir = _ensure_dir(output_dir)
    rows = []
    for path in sorted(checkpoint_dir.glob("round_*_client_*.pt")):
        checkpoint = torch.load(path, map_location="cpu")
        state = checkpoint["model_state"]
        tensors = {
            name: tensor.detach().cpu().contiguous().to(dtype=dtype)
            for name, tensor in state.items()
            if torch.is_tensor(tensor) and tensor.is_floating_point()
        }
        for row in _compress_mapping(path.stem, tensors, methods=methods):
            row["experiment"] = "federated-learning"
            row["checkpoint"] = path.name
            row["round"] = int(checkpoint["round"])
            row["client"] = int(checkpoint["client"])
            row["dtype"] = str(dtype).replace("torch.", "")
            row["tensor_count"] = len(tensors)
            rows.append(row)
        del checkpoint, state, tensors
        gc.collect()

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "federated_compression_results.csv", index=False)
    return df


def compress_federated_directory_amortized(
    checkpoint_dir,
    output_dir,
    *,
    dtype=torch.bfloat16,
    chunk_size=256 << 10,
    compression_threshold=0.95,
    clustering="js-distance",
    cluster_js_threshold=0.05,
    max_codebook_sample_bytes=8 << 20,
    test_cluster_compressibility=True,
    calibration_rounds=1,
):
    checkpoint_dir = Path(checkpoint_dir)
    output_dir = _ensure_dir(output_dir)
    checkpoint_paths = sorted(checkpoint_dir.glob("round_*_client_*.pt"))
    if not checkpoint_paths:
        raise ValueError(f"No federated checkpoints found in {checkpoint_dir}.")

    def load_tensors(path):
        checkpoint = torch.load(path, map_location="cpu")
        state = checkpoint["model_state"]
        tensors = {
            name: tensor.detach().cpu().contiguous().to(dtype=dtype)
            for name, tensor in state.items()
            if torch.is_tensor(tensor) and tensor.is_floating_point()
        }
        return checkpoint, tensors

    if calibration_rounds is None:
        calibration_paths = checkpoint_paths[:1]
    else:
        calibration_paths = [
            path
            for path in checkpoint_paths
            if int(path.name.split("_")[1]) <= int(calibration_rounds)
        ]
    if not calibration_paths:
        calibration_paths = checkpoint_paths[:1]

    codec = NeuralZipCodec(
        chunk_size=chunk_size,
        compression_threshold=compression_threshold,
        clustering=clustering,
        cluster_js_threshold=cluster_js_threshold,
        max_codebook_sample_bytes=max_codebook_sample_bytes,
        test_cluster_compressibility=test_cluster_compressibility,
    )
    calibration_named = []
    for path in calibration_paths:
        calibration_checkpoint, calibration_tensors = load_tensors(path)
        calibration_named.extend(list(calibration_tensors.items()))
        del calibration_checkpoint, calibration_tensors
    start = time.perf_counter()
    codec.precompute_codebooks(calibration_named)
    setup_seconds = time.perf_counter() - start
    codebooks = codec.export_codebooks()
    codebook_metadata_bytes = codec.codebook_metadata_bytes()
    total_updates = len(checkpoint_paths)

    rows = []
    for path in checkpoint_paths:
        checkpoint, tensors = load_tensors(path)
        raw_bytes = 0
        payload_bytes = 0
        chunk_metadata_bytes = 0
        huffman_chunks = 0
        raw_chunks = 0
        shared_refs = 0
        start = time.perf_counter()
        for name, tensor in tensors.items():
            compressed = codec.compress_tensor(name, tensor)
            raw_bytes += compressed.raw_bytes
            payload_bytes += compressed.payload_bytes
            chunk_metadata_bytes += compressed.compressed_bytes - compressed.payload_bytes
            huffman_chunks += compressed.huffman_chunks
            raw_chunks += compressed.raw_chunks
            shared_refs += compressed.shared_codebook_refs
        encode_seconds = time.perf_counter() - start
        compressed_bytes = payload_bytes + chunk_metadata_bytes
        amortized_compressed_bytes = (
            compressed_bytes + codebook_metadata_bytes / total_updates
        )
        rows.append(
            {
                "experiment": "federated-learning",
                "checkpoint": path.name,
                "round": int(checkpoint["round"]),
                "client": int(checkpoint["client"]),
                "compressor": "NeuralZip-Fast amortized",
                "method": "neuralzip-fast-amortized",
                "scope": "full checkpoint",
                "status": "ok",
                "raw_bytes": raw_bytes,
                "compressed_bytes": compressed_bytes,
                "amortized_compressed_bytes": amortized_compressed_bytes,
                "compression_ratio": compressed_bytes / raw_bytes if raw_bytes else 0.0,
                "amortized_compression_ratio": (
                    amortized_compressed_bytes / raw_bytes if raw_bytes else 0.0
                ),
                "compressed_pct": (
                    100 * compressed_bytes / raw_bytes if raw_bytes else 0.0
                ),
                "amortized_compressed_pct": (
                    100 * amortized_compressed_bytes / raw_bytes
                    if raw_bytes
                    else 0.0
                ),
                "setup_seconds_once": setup_seconds,
                "encode_seconds": encode_seconds,
                "amortized_cold_seconds": encode_seconds
                + setup_seconds / total_updates,
                "clusters": codec.cluster_count,
                "codebooks": len(codebooks),
                "codebook_metadata_bytes_once": codebook_metadata_bytes,
                "shared_huffman_chunks": shared_refs,
                "huffman_chunks": huffman_chunks,
                "raw_chunks": raw_chunks,
                "dtype": str(dtype).replace("torch.", ""),
                "tensor_count": len(tensors),
                "calibration_updates": len(calibration_paths),
                "calibration_rounds": calibration_rounds,
                "calibration_scope": ", ".join(path.name for path in calibration_paths),
            }
        )
        del checkpoint, tensors
        gc.collect()

    del calibration_named
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "federated_amortized_compression_results.csv", index=False)
    return df


def _floating_tensors_from_state(state, *, dtype):
    return {
        name: tensor.detach().cpu().contiguous().to(dtype=dtype)
        for name, tensor in state.items()
        if torch.is_tensor(tensor) and tensor.is_floating_point()
    }


def _initial_resnet18_cifar_tensors(*, dtype, seed=2026):
    torch.manual_seed(seed)
    model = build_resnet18_cifar()
    tensors = _floating_tensors_from_state(model.state_dict(), dtype=dtype)
    del model
    return tensors


def _parse_federated_checkpoint_path(path):
    parts = Path(path).stem.split("_")
    return int(parts[1]), int(parts[3])


def _load_federated_client_tensors(path, *, dtype):
    checkpoint = torch.load(path, map_location="cpu")
    tensors = _floating_tensors_from_state(checkpoint["model_state"], dtype=dtype)
    round_index = int(checkpoint.get("round", _parse_federated_checkpoint_path(path)[0]))
    client_index = int(
        checkpoint.get("client", _parse_federated_checkpoint_path(path)[1])
    )
    return round_index, client_index, tensors


class _HFCIFAR10Parquet(torch.utils.data.Dataset):
    def __init__(self, parquet_path, *, transform=None):
        import io

        from PIL import Image
        import pyarrow.parquet as pq

        table = pq.read_table(parquet_path, columns=["img", "label"])
        rows = table.to_pylist()
        self._images = [row["img"]["bytes"] for row in rows]
        self._labels = [int(row["label"]) for row in rows]
        self._transform = transform
        self._image_cls = Image
        self._io = io

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, index):
        image = self._image_cls.open(
            self._io.BytesIO(self._images[int(index)])
        ).convert("RGB")
        if self._transform is not None:
            image = self._transform(image)
        return image, self._labels[int(index)]


def _codec_compress_tensors(codec, tensors):
    payloads = []
    raw_bytes = 0
    compressed_bytes = 0
    huffman_chunks = 0
    raw_chunks = 0
    shared_refs = 0

    start = time.perf_counter()
    for name, tensor in tensors.items():
        payload = codec.compress_tensor(name, tensor)
        payloads.append(payload)
        raw_bytes += payload.raw_bytes
        compressed_bytes += payload.compressed_bytes
        huffman_chunks += payload.huffman_chunks
        raw_chunks += payload.raw_chunks
        shared_refs += getattr(payload, "shared_codebook_refs", 0)
    seconds = time.perf_counter() - start

    return {
        "payloads": payloads,
        "raw_bytes": raw_bytes,
        "compressed_bytes": compressed_bytes,
        "seconds": seconds,
        "huffman_chunks": huffman_chunks,
        "raw_chunks": raw_chunks,
        "shared_refs": shared_refs,
    }


def _codec_decompress_tensors(codec, payloads):
    start = time.perf_counter()
    tensors = {payload.name: codec.decompress_tensor(payload) for payload in payloads}
    seconds = time.perf_counter() - start
    return tensors, seconds


def run_federated_codec_protocol(
    checkpoint_dir,
    output_dir,
    *,
    training_metrics_path=None,
    dtype=torch.bfloat16,
    seed=2026,
    chunk_size=256 << 10,
    compression_threshold=0.95,
    clustering="js-distance",
    cluster_js_threshold=0.05,
    max_codebook_sample_bytes=8 << 20,
    test_cluster_compressibility=True,
):
    """Benchmark FedZipNN vs FedNeuralZip as communication protocols.

    The saved checkpoints represent the client updates produced after local
    training. For each round we measure:
    server downlink compression, client downlink decompression, client uplink
    compression, server uplink decompression, and FedAvg aggregation. Training
    time is read from ``federated_training_metrics.csv`` when available.
    """

    checkpoint_dir = Path(checkpoint_dir)
    output_dir = _ensure_dir(output_dir)
    checkpoint_paths = sorted(checkpoint_dir.glob("round_*_client_*.pt"))
    if not checkpoint_paths:
        raise ValueError(f"No federated checkpoints found in {checkpoint_dir}.")

    round_paths = {}
    for path in checkpoint_paths:
        round_index, client_index = _parse_federated_checkpoint_path(path)
        round_paths.setdefault(round_index, {})[client_index] = path

    if training_metrics_path is None:
        training_metrics_path = checkpoint_dir.parent / "federated_training_metrics.csv"
    training_metrics_path = Path(training_metrics_path)
    training_metrics = None
    if training_metrics_path.exists():
        training_metrics = pd.read_csv(training_metrics_path)

    try:
        initial_server = _initial_resnet18_cifar_tensors(dtype=dtype, seed=seed)
    except ModuleNotFoundError:
        first_round = min(round_paths)
        first_round_states = [
            _load_federated_client_tensors(path, dtype=dtype)[2]
            for path in round_paths[first_round].values()
        ]
        initial_server = _average_floating_states(first_round_states)
        del first_round_states

    protocols = []
    protocols.append(
        {
            "name": "FedZipNN",
            "codec": ZipNNCodec(
                chunk_size=chunk_size,
                compression_threshold=compression_threshold,
            ),
            "setup_seconds": 0.0,
            "setup_bytes": 0,
            "clusters": 0,
            "codebooks": 0,
        }
    )

    neuralzip_codec = NeuralZipCodec(
        chunk_size=chunk_size,
        compression_threshold=compression_threshold,
        clustering=clustering,
        cluster_js_threshold=cluster_js_threshold,
        max_codebook_sample_bytes=max_codebook_sample_bytes,
        test_cluster_compressibility=test_cluster_compressibility,
    )
    start = time.perf_counter()
    neuralzip_codec.precompute_codebooks(list(initial_server.items()))
    neuralzip_setup_seconds = time.perf_counter() - start
    neuralzip_codebooks = neuralzip_codec.export_codebooks()
    neuralzip_codec.load_codebooks(neuralzip_codebooks)
    protocols.append(
        {
            "name": "FedNeuralZip",
            "codec": neuralzip_codec,
            "setup_seconds": neuralzip_setup_seconds,
            "setup_bytes": neuralzip_codec.codebook_metadata_bytes(),
            "clusters": neuralzip_codec.cluster_count,
            "codebooks": len(neuralzip_codebooks),
        }
    )

    rows = []
    for protocol in protocols:
        server_tensors = {
            name: tensor.detach().cpu().contiguous().clone()
            for name, tensor in initial_server.items()
        }
        codec = protocol["codec"]

        for round_index in sorted(round_paths):
            client_paths = round_paths[round_index]
            client_count = len(client_paths)

            if training_metrics is not None:
                round_training = training_metrics[
                    training_metrics["round"] == round_index
                ]
                train_client_sum_seconds = float(round_training["seconds"].sum())
                train_wall_proxy_seconds = float(round_training["seconds"].max())
            else:
                train_client_sum_seconds = 0.0
                train_wall_proxy_seconds = 0.0

            downlink = _codec_compress_tensors(codec, server_tensors)

            downlink_decode_seconds = []
            for _ in range(client_count):
                _, seconds = _codec_decompress_tensors(codec, downlink["payloads"])
                downlink_decode_seconds.append(seconds)
            downlink_decode_sum = float(sum(downlink_decode_seconds))
            downlink_decode_wall = float(max(downlink_decode_seconds, default=0.0))

            uplink_encode_seconds = []
            uplink_decode_seconds = []
            uplink_bytes = 0
            uplink_raw_bytes = 0
            client_states = []
            huffman_chunks = downlink["huffman_chunks"]
            raw_chunks = downlink["raw_chunks"]
            shared_refs = downlink["shared_refs"]

            for client_index in sorted(client_paths):
                _, _, client_tensors = _load_federated_client_tensors(
                    client_paths[client_index],
                    dtype=dtype,
                )
                uplink = _codec_compress_tensors(codec, client_tensors)
                decoded, decode_seconds = _codec_decompress_tensors(
                    codec,
                    uplink["payloads"],
                )
                client_states.append(decoded)
                uplink_encode_seconds.append(float(uplink["seconds"]))
                uplink_decode_seconds.append(float(decode_seconds))
                uplink_bytes += int(uplink["compressed_bytes"])
                uplink_raw_bytes += int(uplink["raw_bytes"])
                huffman_chunks += int(uplink["huffman_chunks"])
                raw_chunks += int(uplink["raw_chunks"])
                shared_refs += int(uplink["shared_refs"])

            start = time.perf_counter()
            server_tensors = _average_floating_states(client_states)
            aggregation_seconds = time.perf_counter() - start

            uplink_encode_sum = float(sum(uplink_encode_seconds))
            uplink_decode_sum = float(sum(uplink_decode_seconds))
            uplink_encode_wall = float(max(uplink_encode_seconds, default=0.0))
            codec_serial_seconds = (
                downlink["seconds"]
                + downlink_decode_sum
                + uplink_encode_sum
                + uplink_decode_sum
            )
            codec_wall_proxy_seconds = (
                downlink["seconds"]
                + downlink_decode_wall
                + uplink_encode_wall
                + uplink_decode_sum
            )
            total_serial_seconds = (
                codec_serial_seconds
                + train_client_sum_seconds
                + aggregation_seconds
            )
            total_wall_proxy_seconds = (
                codec_wall_proxy_seconds
                + train_wall_proxy_seconds
                + aggregation_seconds
            )
            message_bytes = downlink["compressed_bytes"] * client_count + uplink_bytes
            raw_message_bytes = downlink["raw_bytes"] * client_count + uplink_raw_bytes

            rows.append(
                {
                    "protocol": protocol["name"],
                    "round": round_index,
                    "clients": client_count,
                    "dtype": str(dtype).replace("torch.", ""),
                    "setup_seconds_once": protocol["setup_seconds"],
                    "setup_bytes_once": protocol["setup_bytes"],
                    "clusters": protocol["clusters"],
                    "codebooks": protocol["codebooks"],
                    "train_client_sum_seconds": train_client_sum_seconds,
                    "train_wall_proxy_seconds": train_wall_proxy_seconds,
                    "downlink_encode_seconds": float(downlink["seconds"]),
                    "downlink_decode_sum_seconds": downlink_decode_sum,
                    "downlink_decode_wall_proxy_seconds": downlink_decode_wall,
                    "uplink_encode_sum_seconds": uplink_encode_sum,
                    "uplink_encode_wall_proxy_seconds": uplink_encode_wall,
                    "uplink_decode_sum_seconds": uplink_decode_sum,
                    "aggregation_seconds": aggregation_seconds,
                    "codec_serial_seconds": codec_serial_seconds,
                    "codec_wall_proxy_seconds": codec_wall_proxy_seconds,
                    "total_serial_seconds": total_serial_seconds,
                    "total_wall_proxy_seconds": total_wall_proxy_seconds,
                    "downlink_bytes": int(downlink["compressed_bytes"]),
                    "uplink_bytes": int(uplink_bytes),
                    "message_bytes": int(message_bytes),
                    "raw_message_bytes": int(raw_message_bytes),
                    "message_compressed_pct": (
                        100 * message_bytes / raw_message_bytes
                        if raw_message_bytes
                        else 0.0
                    ),
                    "huffman_chunks": int(huffman_chunks),
                    "raw_chunks": int(raw_chunks),
                    "shared_codebook_refs": int(shared_refs),
                }
            )

        del server_tensors
        gc.collect()

    rounds = len(round_paths)
    round_df = pd.DataFrame(rows)
    summary = (
        round_df.groupby("protocol", as_index=False)
        .agg(
            rounds=("round", "count"),
            clients=("clients", "mean"),
            setup_seconds_once=("setup_seconds_once", "first"),
            setup_bytes_once=("setup_bytes_once", "first"),
            clusters=("clusters", "first"),
            codebooks=("codebooks", "first"),
            codec_serial_seconds=("codec_serial_seconds", "sum"),
            codec_wall_proxy_seconds=("codec_wall_proxy_seconds", "sum"),
            train_client_sum_seconds=("train_client_sum_seconds", "sum"),
            train_wall_proxy_seconds=("train_wall_proxy_seconds", "sum"),
            aggregation_seconds=("aggregation_seconds", "sum"),
            total_serial_seconds=("total_serial_seconds", "sum"),
            total_wall_proxy_seconds=("total_wall_proxy_seconds", "sum"),
            message_bytes=("message_bytes", "sum"),
            raw_message_bytes=("raw_message_bytes", "sum"),
            huffman_chunks=("huffman_chunks", "sum"),
            raw_chunks=("raw_chunks", "sum"),
            shared_codebook_refs=("shared_codebook_refs", "sum"),
        )
    )
    summary["amortized_total_serial_seconds"] = (
        summary["total_serial_seconds"] + summary["setup_seconds_once"]
    )
    summary["amortized_total_wall_proxy_seconds"] = (
        summary["total_wall_proxy_seconds"] + summary["setup_seconds_once"]
    )
    summary["avg_codec_wall_proxy_seconds"] = (
        summary["codec_wall_proxy_seconds"] / rounds
    )
    summary["avg_total_wall_proxy_seconds"] = (
        summary["total_wall_proxy_seconds"] / rounds
    )
    summary["message_compressed_pct"] = (
        100 * summary["message_bytes"] / summary["raw_message_bytes"]
    )

    round_df.to_csv(output_dir / "federated_protocol_round_results.csv", index=False)
    summary.to_csv(output_dir / "federated_protocol_summary.csv", index=False)
    return round_df, summary


def run_federated_codec_protocol_online(
    output_dir,
    *,
    clients=10,
    rounds=200,
    local_batches=1,
    batch_size=64,
    learning_rate=0.02,
    dtype=torch.bfloat16,
    seed=2026,
    dataset="cifar10",
    data_dir=None,
    download=True,
    device=None,
    chunk_size=256 << 10,
    compression_threshold=0.95,
    clustering="js-distance",
    cluster_js_threshold=0.05,
    max_codebook_sample_bytes=8 << 20,
    test_cluster_compressibility=True,
    num_workers=0,
    progress_every=10,
):
    """Run a FedZipNN vs FedNeuralZip protocol benchmark online.

    This version does not save per-client checkpoints. It executes the FedAvg
    loop and measures the communication codec at each round:

    server compress -> clients decompress -> clients train -> clients compress
    -> server decompress -> server average.
    """

    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms

    output_dir = _ensure_dir(output_dir)
    compression_dir = _ensure_dir(output_dir / "compression")
    if data_dir is None:
        data_dir = output_dir / "data"
    data_dir = _ensure_dir(data_dir)

    torch.manual_seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    criterion = nn.CrossEntropyLoss()
    dataset_key = str(dataset).lower().replace("-", "_")
    if dataset_key == "cifar10":
        train_data = datasets.CIFAR10(
            root=data_dir,
            train=True,
            download=download,
            transform=transform,
        )
    elif dataset_key in {"hf_cifar10", "cifar10_hf", "parquet_cifar10"}:
        parquet_path = data_dir / "plain_text" / "train-00000-of-00001.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(
                "Hugging Face CIFAR-10 parquet file not found at "
                f"{parquet_path}. Download "
                "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/"
                "plain_text/train-00000-of-00001.parquet?download=true"
            )
        train_data = _HFCIFAR10Parquet(parquet_path, transform=transform)
    elif dataset_key in {"fake", "fake_cifar", "synthetic"}:
        train_data = datasets.FakeData(
            size=max(
                int(clients) * int(local_batches) * int(batch_size),
                50_000,
            ),
            image_size=(3, 32, 32),
            num_classes=10,
            transform=transform,
            random_offset=0,
        )
    else:
        raise ValueError("dataset must be 'cifar10', 'hf-cifar10', or 'fake'.")

    split_generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(train_data), generator=split_generator).tolist()
    client_indices = [indices[index :: int(clients)] for index in range(int(clients))]
    client_loaders = [
        DataLoader(
            Subset(train_data, client_indices[client_index]),
            batch_size=batch_size,
            shuffle=True,
            num_workers=int(num_workers),
            pin_memory=device.type == "cuda",
        )
        for client_index in range(int(clients))
    ]
    client_iterators = [iter(loader) for loader in client_loaders]

    def next_client_batches(client_index):
        batches = []
        while len(batches) < int(local_batches):
            try:
                batch = next(client_iterators[client_index])
            except StopIteration:
                client_iterators[client_index] = iter(client_loaders[client_index])
                batch = next(client_iterators[client_index])
            batches.append(batch)
        return batches

    server = build_resnet18_cifar()
    server_state = {
        key: value.detach().cpu().clone()
        for key, value in server.state_dict().items()
    }
    del server

    initial_tensors = _floating_tensors_from_state(server_state, dtype=dtype)

    protocols = [
        {
            "name": "FedZipNN",
            "codec": ZipNNCodec(
                chunk_size=chunk_size,
                compression_threshold=compression_threshold,
            ),
            "setup_seconds": 0.0,
            "setup_bytes": 0,
            "clusters": 0,
            "codebooks": 0,
        }
    ]

    neuralzip_codec = NeuralZipCodec(
        chunk_size=chunk_size,
        compression_threshold=compression_threshold,
        clustering=clustering,
        cluster_js_threshold=cluster_js_threshold,
        max_codebook_sample_bytes=max_codebook_sample_bytes,
        test_cluster_compressibility=test_cluster_compressibility,
    )
    start = time.perf_counter()
    neuralzip_codec.precompute_codebooks(list(initial_tensors.items()))
    neuralzip_setup_seconds = time.perf_counter() - start
    neuralzip_codebooks = neuralzip_codec.export_codebooks()
    neuralzip_codec.load_codebooks(neuralzip_codebooks)
    protocols.append(
        {
            "name": "FedNeuralZip",
            "codec": neuralzip_codec,
            "setup_seconds": neuralzip_setup_seconds,
            "setup_bytes": neuralzip_codec.codebook_metadata_bytes(),
            "clusters": neuralzip_codec.cluster_count,
            "codebooks": len(neuralzip_codebooks),
        }
    )

    rows = []
    for round_index in range(1, int(rounds) + 1):
        server_tensors = _floating_tensors_from_state(server_state, dtype=dtype)
        downlink_metrics = {}
        for protocol in protocols:
            downlink = _codec_compress_tensors(protocol["codec"], server_tensors)
            _, downlink_decode_seconds = _codec_decompress_tensors(
                protocol["codec"],
                downlink["payloads"],
            )
            downlink_metrics[protocol["name"]] = {
                "encode_seconds": float(downlink["seconds"]),
                "decode_one_seconds": float(downlink_decode_seconds),
                "raw_bytes": int(downlink["raw_bytes"]),
                "compressed_bytes": int(downlink["compressed_bytes"]),
                "huffman_chunks": int(downlink["huffman_chunks"]),
                "raw_chunks": int(downlink["raw_chunks"]),
                "shared_refs": int(downlink["shared_refs"]),
            }
            del downlink

        client_states = []
        train_seconds = []
        for client_index in range(int(clients)):
            model = build_resnet18_cifar().to(device)
            model.load_state_dict(server_state, strict=True)
            model.train()
            optimizer = optim.SGD(
                model.parameters(),
                lr=learning_rate,
                momentum=0.9,
                weight_decay=5e-4,
            )
            start = time.perf_counter()
            for images, labels in next_client_batches(client_index):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
            train_seconds.append(time.perf_counter() - start)
            client_states.append(
                {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
            )
            del model, optimizer

        for protocol in protocols:
            codec = protocol["codec"]
            uplink_encode_seconds = []
            uplink_decode_seconds = []
            uplink_bytes = 0
            uplink_raw_bytes = 0
            huffman_chunks = downlink_metrics[protocol["name"]]["huffman_chunks"]
            raw_chunks = downlink_metrics[protocol["name"]]["raw_chunks"]
            shared_refs = downlink_metrics[protocol["name"]]["shared_refs"]

            for client_state in client_states:
                client_tensors = _floating_tensors_from_state(client_state, dtype=dtype)
                uplink = _codec_compress_tensors(codec, client_tensors)
                _, decode_seconds = _codec_decompress_tensors(
                    codec,
                    uplink["payloads"],
                )
                uplink_encode_seconds.append(float(uplink["seconds"]))
                uplink_decode_seconds.append(float(decode_seconds))
                uplink_bytes += int(uplink["compressed_bytes"])
                uplink_raw_bytes += int(uplink["raw_bytes"])
                huffman_chunks += int(uplink["huffman_chunks"])
                raw_chunks += int(uplink["raw_chunks"])
                shared_refs += int(uplink["shared_refs"])
                del client_tensors, uplink

            downlink = downlink_metrics[protocol["name"]]
            downlink_decode_sum = downlink["decode_one_seconds"] * int(clients)
            downlink_decode_wall = downlink["decode_one_seconds"]
            uplink_encode_sum = float(sum(uplink_encode_seconds))
            uplink_decode_sum = float(sum(uplink_decode_seconds))
            uplink_encode_wall = float(max(uplink_encode_seconds, default=0.0))
            codec_serial_seconds = (
                downlink["encode_seconds"]
                + downlink_decode_sum
                + uplink_encode_sum
                + uplink_decode_sum
            )
            codec_wall_proxy_seconds = (
                downlink["encode_seconds"]
                + downlink_decode_wall
                + uplink_encode_wall
                + uplink_decode_sum
            )
            message_bytes = downlink["compressed_bytes"] * int(clients) + uplink_bytes
            raw_message_bytes = downlink["raw_bytes"] * int(clients) + uplink_raw_bytes

            rows.append(
                {
                    "protocol": protocol["name"],
                    "round": round_index,
                    "clients": int(clients),
                    "local_batches": int(local_batches),
                    "batch_size": int(batch_size),
                    "device": str(device),
                    "dtype": str(dtype).replace("torch.", ""),
                    "setup_seconds_once": protocol["setup_seconds"],
                    "setup_bytes_once": protocol["setup_bytes"],
                    "clusters": protocol["clusters"],
                    "codebooks": protocol["codebooks"],
                    "train_client_sum_seconds": float(sum(train_seconds)),
                    "train_wall_proxy_seconds": float(max(train_seconds, default=0.0)),
                    "downlink_encode_seconds": downlink["encode_seconds"],
                    "downlink_decode_sum_seconds": downlink_decode_sum,
                    "downlink_decode_wall_proxy_seconds": downlink_decode_wall,
                    "uplink_encode_sum_seconds": uplink_encode_sum,
                    "uplink_encode_wall_proxy_seconds": uplink_encode_wall,
                    "uplink_decode_sum_seconds": uplink_decode_sum,
                    "codec_serial_seconds": codec_serial_seconds,
                    "codec_wall_proxy_seconds": codec_wall_proxy_seconds,
                    "total_serial_seconds": codec_serial_seconds
                    + float(sum(train_seconds)),
                    "total_wall_proxy_seconds": codec_wall_proxy_seconds
                    + float(max(train_seconds, default=0.0)),
                    "downlink_bytes": downlink["compressed_bytes"],
                    "uplink_bytes": int(uplink_bytes),
                    "message_bytes": int(message_bytes),
                    "raw_message_bytes": int(raw_message_bytes),
                    "message_compressed_pct": (
                        100 * message_bytes / raw_message_bytes
                        if raw_message_bytes
                        else 0.0
                    ),
                    "huffman_chunks": int(huffman_chunks),
                    "raw_chunks": int(raw_chunks),
                    "shared_codebook_refs": int(shared_refs),
                }
            )

        start = time.perf_counter()
        server_state = _average_floating_states(client_states)
        aggregation_seconds = time.perf_counter() - start
        for row in rows[-len(protocols) :]:
            row["aggregation_seconds"] = aggregation_seconds
            row["total_serial_seconds"] += aggregation_seconds
            row["total_wall_proxy_seconds"] += aggregation_seconds

        if device.type == "cuda":
            torch.cuda.empty_cache()
        del client_states, server_tensors

        if progress_every and round_index % int(progress_every) == 0:
            partial = pd.DataFrame(rows)
            partial.to_csv(
                compression_dir / "federated_protocol_online_round_results.csv",
                index=False,
            )
            print(
                f"finished round {round_index}/{rounds}",
                flush=True,
            )

    round_df = pd.DataFrame(rows)
    summary = (
        round_df.groupby("protocol", as_index=False)
        .agg(
            rounds=("round", "count"),
            clients=("clients", "mean"),
            local_batches=("local_batches", "first"),
            batch_size=("batch_size", "first"),
            setup_seconds_once=("setup_seconds_once", "first"),
            setup_bytes_once=("setup_bytes_once", "first"),
            clusters=("clusters", "first"),
            codebooks=("codebooks", "first"),
            codec_serial_seconds=("codec_serial_seconds", "sum"),
            codec_wall_proxy_seconds=("codec_wall_proxy_seconds", "sum"),
            train_client_sum_seconds=("train_client_sum_seconds", "sum"),
            train_wall_proxy_seconds=("train_wall_proxy_seconds", "sum"),
            aggregation_seconds=("aggregation_seconds", "sum"),
            total_serial_seconds=("total_serial_seconds", "sum"),
            total_wall_proxy_seconds=("total_wall_proxy_seconds", "sum"),
            message_bytes=("message_bytes", "sum"),
            raw_message_bytes=("raw_message_bytes", "sum"),
            huffman_chunks=("huffman_chunks", "sum"),
            raw_chunks=("raw_chunks", "sum"),
            shared_codebook_refs=("shared_codebook_refs", "sum"),
        )
    )
    summary["amortized_total_serial_seconds"] = (
        summary["total_serial_seconds"] + summary["setup_seconds_once"]
    )
    summary["amortized_total_wall_proxy_seconds"] = (
        summary["total_wall_proxy_seconds"] + summary["setup_seconds_once"]
    )
    summary["avg_codec_wall_proxy_seconds"] = (
        summary["codec_wall_proxy_seconds"] / summary["rounds"]
    )
    summary["avg_total_wall_proxy_seconds"] = (
        summary["total_wall_proxy_seconds"] / summary["rounds"]
    )
    summary["message_compressed_pct"] = (
        100 * summary["message_bytes"] / summary["raw_message_bytes"]
    )

    round_df.to_csv(
        compression_dir / "federated_protocol_online_round_results.csv",
        index=False,
    )
    summary.to_csv(
        compression_dir / "federated_protocol_online_summary.csv",
        index=False,
    )
    return round_df, summary
