import argparse
import csv
import hashlib
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel

from neuralzip import huffman
from neuralzip.ablations import zipnn_exponent_stream


DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parent
    / "outputs/checkpoints/"
    "resnet18_imagenet1k_cifar10_finetune20_relela05/checkpoints/"
    "resnet18_cifar10_finetune_epoch_020.pt"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs/deferred/mmrepair_original"
CHUNK_SIZE = 256 << 10


PAIR_SPECS = {
    "ResNet-18": (
        "layer4.0.conv2.weight",
        "layer4.1.conv1.weight",
    ),
    "BERT Base uncased": (
        "encoder.layer.1.intermediate.dense.weight",
        "encoder.layer.7.intermediate.dense.weight",
    ),
}


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _load_resnet_tensors(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state", checkpoint)
    return {
        name: tensor
        for name, tensor in state.items()
        if torch.is_tensor(tensor) and tensor.is_floating_point()
    }


def _load_bert_tensors(model_id, *, local_files_only):
    model = AutoModel.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        local_files_only=local_files_only,
    )
    return dict(model.named_parameters())


def _export_pair(output_dir, slug, tensors, names):
    matrices = []
    shapes = []
    for name in names:
        tensor = tensors[name].detach().cpu().contiguous().to(torch.bfloat16)
        stream = np.frombuffer(
            zipnn_exponent_stream(tensor, layout="original"),
            dtype=np.uint8,
        ).copy()
        matrices.append(stream.reshape(int(tensor.shape[0]), -1))
        shapes.append(list(tensor.shape))

    matrix = np.concatenate(matrices, axis=0)
    raw = matrix.tobytes()
    base = output_dir / slug
    raw_path = Path(str(base) + ".exp.u8")
    input_path = Path(str(base) + ".exp.i32")
    raw_path.write_bytes(raw)
    matrix.astype("<i4").tofile(input_path)
    return {
        "base": input_path,
        "raw": raw,
        "raw_path": raw_path,
        "rows": int(matrix.shape[0]),
        "cols": int(matrix.shape[1]),
        "tensor_names": list(names),
        "tensor_shapes": shapes,
        "sha256": _sha256(raw),
    }


def _zipnn_size(raw):
    total = 0
    for start in range(0, len(raw), CHUNK_SIZE):
        chunk = raw[start : start + CHUNK_SIZE]
        raw_cost = 1 + 8 + 8 + len(chunk)
        encoded = huffman.compress(chunk, min_saving_bytes=0)
        if encoded is None:
            total += raw_cost
            continue
        encoded_cost = 1 + 8 + 8 + len(encoded.code_lengths) + len(encoded.payload)
        total += min(raw_cost, encoded_cost)
    return total


def _benchmark_zipnn(raw, repeats):
    times = []
    sizes = []
    for _ in range(repeats):
        start = time.perf_counter()
        sizes.append(_zipnn_size(raw))
        times.append(time.perf_counter() - start)
    if len(set(sizes)) != 1:
        raise RuntimeError(f"ZipNN size changed across runs: {sizes}")
    return sizes[0], times


def _official_size(base, *, dense):
    sequence_ext = ".dv" if dense else ".vc"
    values_ext = ".ivald" if dense else ".ival"
    paths = (
        Path(str(base) + values_ext),
        Path(str(base) + sequence_ext + ".R.iv"),
        Path(str(base) + sequence_ext + ".C.ansf.1"),
    )
    return sum(path.stat().st_size for path in paths)


def _benchmark_mmrepair(matrepair, matrix, repeats, *, dense, log_dir):
    times = []
    sizes = []
    mode = "reans-drv" if dense else "reans-csrv"
    for run in range(repeats):
        command = [
            sys.executable,
            str(matrepair),
            "-r",
            "--i32",
        ]
        if dense:
            command.append("--drv")
        command.extend(
            (
                str(matrix["base"]),
                str(matrix["rows"]),
                str(matrix["cols"]),
            )
        )
        completed = subprocess.run(command, check=True, text=True, capture_output=True)
        output = completed.stdout + completed.stderr
        (log_dir / f"{matrix['base'].stem}.{mode}.run{run + 1}.log").write_text(
            output
        )
        matches = re.findall(r"^Elapsed time:\s+([0-9.]+)$", completed.stdout, re.M)
        if not matches:
            raise RuntimeError(f"Could not parse matrepair time for {matrix['base']}")
        times.append(float(matches[-1]))
        sizes.append(_official_size(matrix["base"], dense=dense))
    if len(set(sizes)) != 1:
        raise RuntimeError(f"MM-RePair size changed across runs: {sizes}")
    return sizes[0], times


def _result_row(model, method, format_name, matrix, size, times):
    raw_bytes = len(matrix["raw"])
    return {
        "model": model,
        "method": method,
        "format": format_name,
        "raw_exponent_bytes": raw_bytes,
        "compressed_bytes": size,
        "compressed_pct": 100 * size / raw_bytes,
        "median_seconds": statistics.median(times),
        "min_seconds": min(times),
        "max_seconds": max(times),
        "rows": matrix["rows"],
        "cols": matrix["cols"],
        "sha256": matrix["sha256"],
    }


def run(args):
    matrepair = args.mm_repair_dir.resolve() / "matrepair"
    if not matrepair.is_file():
        raise FileNotFoundError(f"Official matrepair executable not found: {matrepair}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.output_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    tensor_sets = {
        "ResNet-18": _load_resnet_tensors(args.checkpoint),
        "BERT Base uncased": _load_bert_tensors(
            args.bert_model,
            local_files_only=args.local_files_only,
        ),
    }
    slugs = {
        "ResNet-18": "resnet18",
        "BERT Base uncased": "bert-base-uncased",
    }

    results = []
    metadata = {}
    for model, tensors in tensor_sets.items():
        matrix = _export_pair(
            args.output_dir,
            slugs[model],
            tensors,
            PAIR_SPECS[model],
        )
        metadata[model] = {
            key: value
            for key, value in matrix.items()
            if key not in {"raw", "base", "raw_path"}
        }

        size, times = _benchmark_zipnn(matrix["raw"], args.repeats)
        results.append(
            _result_row(
                model,
                "ZipNN matched matrix",
                "chunked local Huffman",
                matrix,
                size,
                times,
            )
        )
        for dense in (False, True):
            size, times = _benchmark_mmrepair(
                matrepair,
                matrix,
                args.repeats,
                dense=dense,
                log_dir=log_dir,
            )
            results.append(
                _result_row(
                    model,
                    "MM-RePair original",
                    "ReANS-DRV" if dense else "ReANS-CSRV",
                    matrix,
                    size,
                    times,
                )
            )

    csv_path = args.output_dir / "mmrepair_original_results.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    (args.output_dir / "mmrepair_original_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str)
    )
    print(json.dumps(results, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mm-repair-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--bert-model", default="google-bert/bert-base-uncased")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
