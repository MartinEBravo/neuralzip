import argparse
import gc
import os
import re
import time
from pathlib import Path

from constants import (
    ANALYSIS_DIR,
    COMPRESSION_METHODS,
    DEFAULT_QUANT_BITS,
    MODEL_PRESETS,
    PIPELINE_SUMMARY_CSV,
    PLOTS_DIR,
    RESULTS_CSV,
    get_model_label,
)
from entropy import block_histograms, cluster_blocks, compute_js_matrix, summarize_js_matrix
from quantization import SUPPORTED_PREQUANT_BITS
from results import (
    format_row,
    print_pipeline_summary,
    print_presets,
    print_summary,
    save_csv_rows,
    write_square_matrix_csv,
    write_summary_json,
)


def clear_torch_cache() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def cleanup_memory() -> None:
    gc.collect()
    clear_torch_cache()


def expand_models(explicit_models: list[str] | None, presets: list[str]) -> list[str]:
    models: list[str] = []
    for preset in presets:
        models.extend(MODEL_PRESETS[preset])
    if explicit_models:
        models.extend(explicit_models)
    if not models:
        models.extend(MODEL_PRESETS["baseline"])
    return list(dict.fromkeys(models))


def resolve_hf_token(cli_token: str | None) -> str | None:
    if cli_token:
        return cli_token
    return os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")


def slugify_model_name(model_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", model_name).strip("-").lower() or "model"


def print_model_selection(models: list[str]) -> None:
    print("Selected models:")
    for idx, model_name in enumerate(models):
        label = get_model_label(model_name)
        print(f"  [{idx}] {model_name} ({label})")


def select_job_model(models: list[str], job_index: int | None) -> list[str]:
    if job_index is None:
        return models
    if job_index < 0 or job_index >= len(models):
        raise ValueError(
            f"--job-index must be between 0 and {len(models) - 1} for the current selection."
        )
    return [models[job_index]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the NeuralZip thesis pipeline across models: field H0, exponent JS, "
            "baseline ZipNN vs NeuralZip variants, and Q8/Q12 compression passes."
        )
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Hugging Face model ids or local checkpoint paths to evaluate.",
    )
    parser.add_argument(
        "--preset",
        action="append",
        default=[],
        choices=sorted(MODEL_PRESETS),
        help="Append a predefined model family to the evaluation list.",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="Print the available model presets and exit.",
    )
    parser.add_argument(
        "--show-models",
        action="store_true",
        help="Print the expanded model list with stable job indices and exit.",
    )
    parser.add_argument(
        "--job-index",
        type=int,
        default=None,
        help="Run only one model from the expanded selection, addressed by its 0-based index.",
    )
    parser.add_argument(
        "--loader",
        choices=["auto", "safetensors", "transformers"],
        default="auto",
        help="How to load model weights. 'auto' prefers safetensors streaming.",
    )
    parser.add_argument("--hf-token", default=None, help="Hugging Face token. Falls back to HF_TOKEN.")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only already cached local files instead of downloading model weights.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom model code when the transformers loader is used.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep running the remaining models if one checkpoint fails to load.",
    )
    parser.add_argument(
        "--grouping",
        choices=["module", "tensor"],
        default="module",
        help="How to define exponent JS blocks. 'module' is the default layer-level view.",
    )
    parser.add_argument(
        "--cluster-threshold",
        type=float,
        default=0.05,
        help="Jensen-Shannon distance threshold for NeuralZip block clustering.",
    )
    parser.add_argument(
        "--neuralzip-max-clusters",
        type=int,
        default=32,
        help="Hard cap on the number of NeuralZip block clusters. Use 0 to disable the cap.",
    )
    parser.add_argument(
        "--quant-bits",
        nargs="+",
        type=int,
        default=list(DEFAULT_QUANT_BITS),
        help=(
            "Floating-point quantization stages to test before ZipNN and NeuralZip variants. "
            "Supported formats: Q8=E4M3 and Q12=E5M6."
        ),
    )
    parser.add_argument("--results-csv", type=Path, default=RESULTS_CSV)
    parser.add_argument("--pipeline-summary-csv", type=Path, default=PIPELINE_SUMMARY_CSV)
    parser.add_argument("--analysis-dir", type=Path, default=ANALYSIS_DIR)
    parser.add_argument("--plots-dir", type=Path, default=PLOTS_DIR)
    parser.add_argument(
        "--verify-codecs",
        action="store_true",
        dest="verify_codecs",
        help="Decode real codec artifacts after compression and verify exponent round-trips.",
    )
    args = parser.parse_args()
    if args.neuralzip_max_clusters is not None and args.neuralzip_max_clusters < 0:
        parser.error("--neuralzip-max-clusters must be non-negative.")
    if not args.quant_bits:
        parser.error("--quant-bits requires at least one integer.")
    if any(bits not in SUPPORTED_PREQUANT_BITS for bits in args.quant_bits):
        supported = ", ".join(str(bits) for bits in SUPPORTED_PREQUANT_BITS)
        parser.error(f"--quant-bits values must be one of: {supported}.")
    return args


def resolved_max_clusters(args: argparse.Namespace) -> int | None:
    return None if args.neuralzip_max_clusters == 0 else args.neuralzip_max_clusters


def pipeline_metric_key(method: str, prequant_bits: int | None) -> str:
    prefix = "" if prequant_bits is None else f"q{prequant_bits}_"
    return f"{prefix}{method.lower()}_percent_original"


def run_compression_pass(
    model_name: str,
    args: argparse.Namespace,
    hf_token: str | None,
    max_clusters: int | None,
    prequant_bits: int | None,
) -> tuple[list[dict[str, object]], object, dict[str, float]]:
    from experiment import ModelCompressionExperiment

    experiment = ModelCompressionExperiment(
        model_name=model_name,
        loader=args.loader,
        hf_token=hf_token,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
        block_grouping=args.grouping,
        prequant_bits=prequant_bits,
    )

    scan_start = time.perf_counter()
    summary = experiment.scan()
    scan_seconds = time.perf_counter() - scan_start

    zipnn_start = time.perf_counter()
    zipnn_bits = experiment.zipnn_bits(summary=summary, verify=args.verify_codecs)
    zipnn_seconds = time.perf_counter() - zipnn_start

    neuralzip_start = time.perf_counter()
    neuralzip_bits = experiment.neuralzip_bits(
            summary=summary,
            cluster_threshold=args.cluster_threshold,
            max_clusters=max_clusters,
            verify=args.verify_codecs,
        )
    neuralzip_seconds = time.perf_counter() - neuralzip_start

    neuralzip_full_start = time.perf_counter()
    neuralzip_full_bits = experiment.neuralzip_full_bits(
            summary=summary,
            cluster_threshold=args.cluster_threshold,
            max_clusters=max_clusters,
            verify=args.verify_codecs,
        )
    neuralzip_full_seconds = time.perf_counter() - neuralzip_full_start

    bits_by_method = {
        "ZipNN": zipnn_bits,
        "NeuralZip": neuralzip_bits,
        "NeuralZipFull": neuralzip_full_bits,
    }
    timing_by_method = {
        "scan_seconds": scan_seconds,
        "ZipNN": zipnn_seconds,
        "NeuralZip": neuralzip_seconds,
        "NeuralZipFull": neuralzip_full_seconds,
    }
    rows: list[dict[str, object]] = []
    for method in COMPRESSION_METHODS:
        row = format_row(
            model_name,
            method,
            bits_by_method[method],
            summary.original_bits,
            summary.loader,
            summary.dtype_summary,
            summary.num_blocks,
            timing_by_method[method],
            prequant_bits,
        )
        rows.append(row)
        print(
            f"  {row['label']} {row['variant']} done: "
            f"{row['percent_original']:.2f}% in {timing_by_method[method]:.3f}s"
        )

    return rows, summary, timing_by_method


def build_pipeline_row(
    model_name: str,
    summary,
    js_summary: dict[str, object],
    num_clusters: int,
    scan_seconds: float,
) -> dict[str, object]:
    sign_entropy = summary.field_entropy("sign")
    exponent_entropy = summary.field_entropy("exponent")
    mantissa_entropy = summary.field_entropy("mantissa")
    return {
        "model": model_name,
        "label": get_model_label(model_name),
        "loader": summary.loader,
        "dtype_summary": summary.dtype_summary,
        "num_blocks": summary.num_blocks,
        "sign_h0_bits": sign_entropy.h0_bits,
        "sign_h0_bits_per_symbol": sign_entropy.bits_per_symbol,
        "exponent_h0_bits": exponent_entropy.h0_bits,
        "exponent_h0_bits_per_symbol": exponent_entropy.bits_per_symbol,
        "mantissa_h0_bits": mantissa_entropy.h0_bits,
        "mantissa_h0_bits_per_symbol": mantissa_entropy.bits_per_symbol,
        "average_exponent_js": js_summary["average_js"],
        "median_exponent_js": js_summary["median_js"],
        "min_exponent_js": js_summary["min_js"],
        "max_exponent_js": js_summary["max_js"],
        "num_clusters": num_clusters,
        "baseline_scan_seconds": scan_seconds,
    }


def save_exponent_js_artifacts(
    analysis_dir: Path,
    model_name: str,
    block_infos,
    matrix,
    payload: dict[str, object],
    grouping: str,
) -> None:
    from plots import plot_js_heatmap

    label = get_model_label(model_name)
    slug = slugify_model_name(model_name)
    matrix_path = analysis_dir / f"{slug}.exponent-js.matrix.csv"
    summary_path = analysis_dir / f"{slug}.exponent-js.summary.json"
    heatmap_path = analysis_dir / f"{slug}.exponent-js.heatmap.png"
    labels = [block.name for block in block_infos]
    write_square_matrix_csv(matrix_path, labels, matrix)
    write_summary_json(summary_path, payload)
    plot_js_heatmap(
        matrix=matrix,
        labels=labels,
        path=heatmap_path,
        title=f"{label} | Exponent JS ({grouping})",
    )


def main() -> None:
    args = parse_args()
    if args.list_presets:
        print_presets()
        return

    from plots import plot_gain_bars, plot_ratio_bars

    expanded_models = expand_models(args.models, args.preset)
    if args.show_models:
        print_model_selection(expanded_models)
        return
    selected_models = select_job_model(expanded_models, args.job_index)
    hf_token = resolve_hf_token(args.hf_token)
    quant_bits = list(dict.fromkeys(args.quant_bits))
    max_clusters = resolved_max_clusters(args)

    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    args.plots_dir.mkdir(parents=True, exist_ok=True)

    compression_rows: list[dict[str, object]] = []
    pipeline_rows: list[dict[str, object]] = []
    failures: list[tuple[str, str]] = []

    for model_name in selected_models:
        print(f"Running pipeline for {model_name} ...")
        model_start = time.perf_counter()
        try:
            baseline_rows, baseline_summary, baseline_timings = run_compression_pass(
                model_name, args, hf_token, max_clusters, prequant_bits=None,
            )

            histograms = block_histograms(baseline_summary.block_infos)
            js_matrix = compute_js_matrix(histograms)
            js_summary = summarize_js_matrix(baseline_summary.block_infos, js_matrix)
            clusters = cluster_blocks(
                baseline_summary.block_infos,
                threshold=args.cluster_threshold,
                max_clusters=max_clusters,
            )
            save_exponent_js_artifacts(
                analysis_dir=args.analysis_dir,
                model_name=model_name,
                block_infos=baseline_summary.block_infos,
                matrix=js_matrix,
                payload={
                    "model": model_name,
                    "label": get_model_label(model_name),
                    "grouping": args.grouping,
                    "cluster_threshold": args.cluster_threshold,
                    "max_clusters": max_clusters,
                    "num_clusters": len(clusters),
                    **js_summary,
                },
                grouping=args.grouping,
            )

            model_pipeline_row = build_pipeline_row(
                model_name=model_name,
                summary=baseline_summary,
                js_summary=js_summary,
                num_clusters=len(clusters),
                scan_seconds=baseline_timings["scan_seconds"],
            )
            model_pipeline_row["zipnn_seconds"] = baseline_timings["ZipNN"]
            model_pipeline_row["neuralzip_seconds"] = baseline_timings["NeuralZip"]
            model_pipeline_row["neuralzipfull_seconds"] = baseline_timings["NeuralZipFull"]
            del baseline_summary
            cleanup_memory()

            all_passes = [(None, baseline_rows)]
            for quant_bits_value in quant_bits:
                rows, _, timings = run_compression_pass(
                    model_name, args, hf_token, max_clusters, prequant_bits=quant_bits_value,
                )
                all_passes.append((quant_bits_value, rows))
                model_pipeline_row[f"q{quant_bits_value}_scan_seconds"] = timings["scan_seconds"]
                model_pipeline_row[f"q{quant_bits_value}_zipnn_seconds"] = timings["ZipNN"]
                model_pipeline_row[f"q{quant_bits_value}_neuralzip_seconds"] = timings["NeuralZip"]
                model_pipeline_row[f"q{quant_bits_value}_neuralzipfull_seconds"] = timings["NeuralZipFull"]
                cleanup_memory()

            for prequant_bits, rows in all_passes:
                compression_rows.extend(rows)
                for row in rows:
                    model_pipeline_row[
                        pipeline_metric_key(str(row["method"]), prequant_bits)
                    ] = row["percent_original"]

            model_pipeline_row["total_pipeline_seconds"] = time.perf_counter() - model_start
            pipeline_rows.append(model_pipeline_row)
        except Exception as exc:
            failures.append((model_name, str(exc)))
            if not args.continue_on_error:
                raise
            print(f"  failed: {exc}")
        finally:
            cleanup_memory()

    if not compression_rows:
        raise RuntimeError("No pipeline results were produced.")

    save_csv_rows(compression_rows, args.results_csv)
    save_csv_rows(pipeline_rows, args.pipeline_summary_csv)
    plot_ratio_bars(compression_rows, args.plots_dir / "compression_ratio_by_model.png")
    plot_gain_bars(compression_rows, args.plots_dir / "compression_gain_vs_zipnn.png")
    print_pipeline_summary(pipeline_rows)
    print("")
    print_summary(compression_rows)

    if failures:
        print("\nModels that failed:")
        for model_name, error_message in failures:
            print(f"  {model_name}: {error_message}")


if __name__ == "__main__":
    main()
