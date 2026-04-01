import argparse
import gc
import os
from pathlib import Path

from constants import DEFAULT_METHODS, MODEL_PRESETS, PLOTS_DIR, RESULTS_CSV
from results import format_row, print_presets, print_summary, save_results_csv


def clear_torch_cache() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ZipNN compression experiments across models.")
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
        "--methods",
        nargs="+",
        default=DEFAULT_METHODS,
        choices=DEFAULT_METHODS,
        help="Compression variants to evaluate.",
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
    parser.add_argument("--ngram", type=int, default=2, help="N-gram size for ZipNN++.")
    parser.add_argument(
        "--cluster-threshold",
        type=float,
        default=0.02,
        help="Jensen-Shannon distance threshold for ZipNN++ layer clustering.",
    )
    parser.add_argument("--results-csv", type=Path, default=RESULTS_CSV)
    parser.add_argument("--plots-dir", type=Path, default=PLOTS_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_presets:
        print_presets()
        return

    from experiment import ModelCompressionExperiment
    from plots import plot_gain_bars, plot_ratio_bars

    methods = list(dict.fromkeys(args.methods))
    selected_models = expand_models(args.models, args.preset)
    hf_token = resolve_hf_token(args.hf_token)
    args.plots_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    failures: list[tuple[str, str]] = []

    for model_name in selected_models:
        print(f"Running experiments for {model_name} ...")
        try:
            experiment = ModelCompressionExperiment(
                model_name=model_name,
                loader=args.loader,
                hf_token=hf_token,
                local_files_only=args.local_files_only,
                trust_remote_code=args.trust_remote_code,
            )
            summary = experiment.scan()

            if "ZipNN" in methods:
                rows.append(
                    format_row(
                        model_name,
                        "ZipNN",
                        summary.zipnn_bits,
                        summary.original_bits,
                        summary.loader,
                        summary.dtype_summary,
                        summary.num_tensors,
                    )
                )
            if "ZipNN+" in methods:
                rows.append(
                    format_row(
                        model_name,
                        "ZipNN+",
                        summary.zipnn_plus_bits,
                        summary.original_bits,
                        summary.loader,
                        summary.dtype_summary,
                        summary.num_tensors,
                    )
                )
            if "ZipNN++" in methods:
                rows.append(
                    format_row(
                        model_name,
                        "ZipNN++",
                        experiment.zipnn_plusplus_bits(
                            summary=summary,
                            ngram=args.ngram,
                            cluster_threshold=args.cluster_threshold,
                        ),
                        summary.original_bits,
                        summary.loader,
                        summary.dtype_summary,
                        summary.num_tensors,
                    )
                )
        except Exception as exc:
            failures.append((model_name, str(exc)))
            if not args.continue_on_error:
                raise
            print(f"  failed: {exc}")
        finally:
            gc.collect()
            clear_torch_cache()

    if not rows:
        raise RuntimeError("No experiment results were produced.")

    save_results_csv(rows, args.results_csv)
    plot_ratio_bars(rows, args.plots_dir / "compression_ratio_by_model.png")
    plot_gain_bars(rows, args.plots_dir / "compression_gain_vs_zipnn.png")
    print_summary(rows)

    if failures:
        print("\nModels that failed:")
        for model_name, error_message in failures:
            print(f"  {model_name}: {error_message}")


if __name__ == "__main__":
    main()
