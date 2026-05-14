import argparse
import shutil
from pathlib import Path

import numpy as np

from experiment_utils import (
    collect_exponents,
    collect_field,
    clustered_best_exponent_bits,
    ctx_huffman_exponent_bits,
    default_models,
    empirical_conditional_entropy,
    empirical_entropy_order0,
    ensure_irepair,
    exponent_budget_bits,
    global_huffman_exponent_bits,
    huffman_exponent_bits,
    log_bits,
    log_per_tensor_metric,
    official_zipnn_bits,
    pack_non_overlapping_ngrams,
    per_tensor_best_bits,
    per_tensor_ngram_bits,
    run_irepair,
    should_validate_huffman,
    trigram_estimator,
    zipnn_exponent_bits_by_tensor,
    zipnn_huffman_exponent_bits,
    zipnn_trigram_huffman_exponent_bits,
)
from model_catalog import EXPERIMENTS, select_experiments
from model_compression import HuggingFaceModel


def parse_args():
    parser = argparse.ArgumentParser(description="Run model compression experiments.")
    parser.add_argument(
        "--mode",
        choices=["full", "zipnn", "exponent", "entropy", "mm-repair"],
        default="full",
        help="Experiment family to run.",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="bfloat16",
        help="Tensor dtype to request from Hugging Face when loading the model.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model ids or labels to run. Defaults depend on --mode.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate round-trip reconstruction when supported.",
    )
    parser.add_argument(
        "--zipnn-method",
        choices=["AUTO", "HUFFMAN", "ZSTD", "LZ4", "SNAPPY"],
        default="HUFFMAN",
        help="Official ZipNN compression method.",
    )
    parser.add_argument(
        "--ctx-order",
        type=int,
        default=3,
        help="Context order for contextual Huffman experiments.",
    )
    parser.add_argument(
        "--validate-huffman",
        action="store_true",
        help="Validate Huffman round-trip on tensors up to --validate-huffman-limit.",
    )
    parser.add_argument(
        "--validate-huffman-limit",
        type=int,
        default=1_000_000,
        help="Maximum symbols per tensor for full Huffman round-trip validation.",
    )
    parser.add_argument(
        "--only",
        choices=[
            "all",
            "trigram-vs-clustered",
            "zipnn-vs-global-vs-neuralzip",
            "ngram-sweep",
        ],
        default="all",
        help="Focused subset for --mode exponent.",
    )
    parser.add_argument(
        "--field",
        choices=["sign", "exponent", "mantissa"],
        default="exponent",
        help="Floating-point field for --mode entropy.",
    )
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=2_000_000,
        help="Cap symbols for entropy/mm-repair probes. Use 0 for full stream.",
    )
    parser.add_argument(
        "--orders",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3],
        help="Empirical conditional entropy orders for --mode entropy.",
    )
    parser.add_argument(
        "--ngrams",
        type=int,
        nargs="+",
        default=[2, 3],
        help="Non-overlapping block n-gram sizes for --mode entropy.",
    )
    parser.add_argument(
        "--mm-repair-dir",
        type=Path,
        default=Path("/tmp/mm-repair"),
        help="Path to a cloned mm-repair repository for --mode mm-repair.",
    )
    parser.add_argument(
        "--memory-mb",
        type=int,
        default=4096,
        help="Memory hint passed to mm-repair.",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Build mm-repair if the binary is missing.",
    )
    return parser.parse_args()


def load_experiment_model(experiment, args, prefix):
    print(f"\n{prefix} {experiment['label']} ({experiment['model_id']})")
    model = HuggingFaceModel(experiment["model_id"], dtype=args.dtype)
    model.print_model_size()
    return model


def run_neuralzip_default(model, args):
    model.neuralzip_compress(
        method="ctx-huffman-best:3",
        ngram=1,
        cluster=True,
        exp="cluster-ctx-huffman-best-o3-fields",
        validate=args.validate,
        compress_aux_fields=True,
    )


def run_full(experiment, args):
    model = load_experiment_model(experiment, args, "Running")
    model.zipnn_compress()
    run_neuralzip_default(model, args)


def run_zipnn(experiment, args):
    model = load_experiment_model(experiment, args, "Running comparison for")
    zipnn_bits = official_zipnn_bits(model.model, args.zipnn_method)
    model.log_bits(f"Official ZipNN ({args.zipnn_method})", zipnn_bits)
    run_neuralzip_default(model, args)


def run_exponent(experiment, args):
    print(f"\nExponent-only comparison for {experiment['label']} ({experiment['model_id']})")
    model = HuggingFaceModel(experiment["model_id"], dtype=args.dtype)
    layer_exponents = collect_exponents(model)
    original_bits = sum(exponents.size for exponents in layer_exponents) * 8
    log_bits("Raw exponents", original_bits, original_bits)

    if args.only == "trigram-vs-clustered":
        log_per_tensor_metric(
            "NeuralZip canonical trigram-Huffman/tensor",
            layer_exponents,
            original_bits,
            trigram_estimator(model, args),
        )
        log_bits(
            f"NeuralZip exponent clustered-best o{args.ctx_order}",
            clustered_best_exponent_bits(model, layer_exponents, args.ctx_order),
            original_bits,
        )
        return

    if args.only == "zipnn-vs-global-vs-neuralzip":
        log_bits(
            f"Official ZipNN byte/tensor ({args.zipnn_method})",
            zipnn_exponent_bits_by_tensor(layer_exponents, args.zipnn_method),
            original_bits,
        )
        log_bits(
            "Global canonical Huffman",
            global_huffman_exponent_bits(model, layer_exponents),
            original_bits,
        )
        log_per_tensor_metric(
            "NeuralZip canonical trigram-Huffman/tensor",
            layer_exponents,
            original_bits,
            trigram_estimator(model, args),
        )
        return

    if args.only == "ngram-sweep":
        for n in range(2, 6):
            log_bits(
                f"NeuralZip canonical {n}-gram Huffman/tensor",
                per_tensor_ngram_bits(model, layer_exponents, n, args),
                original_bits,
            )
        return

    log_bits(
        f"Official ZipNN byte/tensor ({args.zipnn_method})",
        zipnn_exponent_bits_by_tensor(layer_exponents, args.zipnn_method),
        original_bits,
    )
    tensor_metrics = [
        (
            "NeuralZip exponent Huffman/tensor",
            lambda exponents: huffman_exponent_bits(
                model,
                exponents,
                validate=should_validate_huffman(exponents, args),
            ),
        ),
        (
            "NeuralZip ZipNN-Huffman/tensor",
            lambda exponents: zipnn_huffman_exponent_bits(model, exponents),
        ),
        (
            "NeuralZip exponent trigram-Huffman/tensor",
            trigram_estimator(model, args),
        ),
        (
            "NeuralZip ZipNN-trigram-Huffman/tensor",
            lambda exponents: zipnn_trigram_huffman_exponent_bits(model, exponents),
        ),
        (
            f"NeuralZip exponent ctx-Huffman/tensor o{args.ctx_order}",
            lambda exponents: ctx_huffman_exponent_bits(
                model,
                exponents,
                args.ctx_order,
            ),
        ),
    ]
    for name, estimator in tensor_metrics:
        log_per_tensor_metric(name, layer_exponents, original_bits, estimator)
    log_bits(
        f"NeuralZip exponent tensor-best o{args.ctx_order}",
        per_tensor_best_bits(model, layer_exponents, args.ctx_order),
        original_bits,
    )
    log_bits(
        f"NeuralZip exponent clustered-best o{args.ctx_order}",
        clustered_best_exponent_bits(model, layer_exponents, args.ctx_order),
        original_bits,
    )


def run_entropy(experiment, args):
    print(f"\nRunning entropy probe for {experiment['label']} ({experiment['model_id']})")
    model = HuggingFaceModel(experiment["model_id"], dtype=args.dtype)
    symbols = collect_field(model, args.field, args.max_symbols)
    if symbols.size == 0:
        print("No symbols collected.")
        return

    print(f"Field: {args.field}")
    print(f"Symbols: {symbols.size:,}")
    print(f"Unique symbols: {np.unique(symbols).size:,}")

    for order in args.orders:
        entropy = empirical_conditional_entropy(symbols, order)
        print(f"H_{order}: {entropy:.4f} bits/symbol")

    for ngram in args.ngrams:
        packed = pack_non_overlapping_ngrams(symbols, ngram)
        entropy = empirical_entropy_order0(packed)
        print(
            f"Block H(n={ngram}): {entropy:.4f} bits/block | "
            f"{entropy / ngram:.4f} bits/symbol"
        )


def run_mm_repair(experiment, args, irepair):
    print(f"\nRunning mm-repair probe for {experiment['label']} ({experiment['model_id']})")
    model = HuggingFaceModel(experiment["model_id"], dtype=args.dtype)
    symbols = collect_field(model, "exponent", args.max_symbols)
    if symbols.size == 0:
        print("No exponent symbols collected.")
        return

    exp_bits = exponent_budget_bits(model, symbols.size)
    result = run_irepair(irepair, symbols, args.memory_mb)
    estimated_bits = result["estimated_bytes"] * 8
    actual_files_bits = (
        result["rule_file_bytes"] + result["sequence_file_bytes"]
    ) * 8
    ratio_vs_exp = 100 * estimated_bits / exp_bits if exp_bits else 0.0
    actual_ratio_vs_exp = 100 * actual_files_bits / exp_bits if exp_bits else 0.0

    print(f"Symbols: {symbols.size:,}")
    print(f"Raw int32 stream: {result['raw_bytes'] / 1e6:.2f} MB")
    print(f"Original exponent budget: {exp_bits / 1e6:.2f} Mbits")
    print(
        "mm-repair estimated: "
        f"{estimated_bits / 1e6:.2f} Mbits | {ratio_vs_exp:.2f}% of exponent bits"
    )
    print(
        "mm-repair files: "
        f"{actual_files_bits / 1e6:.2f} Mbits | "
        f"{actual_ratio_vs_exp:.2f}% of exponent bits"
    )
    print(f"Rules: {result['rules']} | Final sequence length: {result['final_length']}")


def main():
    args = parse_args()
    experiments = select_experiments(default_models(args), EXPERIMENTS)

    irepair = None
    if args.mode == "mm-repair":
        if not shutil.which("make") and args.build:
            raise RuntimeError("`make` is required to build mm-repair.")
        irepair = ensure_irepair(args.mm_repair_dir, build=args.build)

    runners = {
        "full": lambda experiment: run_full(experiment, args),
        "zipnn": lambda experiment: run_zipnn(experiment, args),
        "exponent": lambda experiment: run_exponent(experiment, args),
        "entropy": lambda experiment: run_entropy(experiment, args),
        "mm-repair": lambda experiment: run_mm_repair(experiment, args, irepair),
    }
    for experiment in experiments:
        runners[args.mode](experiment)


if __name__ == "__main__":
    main()
