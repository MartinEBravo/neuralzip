import argparse

from model_catalog import EXPERIMENTS, select_experiments
from model_compression import HuggingFaceModel


def parse_args():
    parser = argparse.ArgumentParser(description="Run compression experiments.")
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
        help="Tensor dtype to request from Hugging Face when loading the model.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate sign, mantissa, and exponent round-trip reconstruction.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model ids or labels to run. Defaults to all experiments.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    for experiment in select_experiments(args.models, EXPERIMENTS):
        print(f"\nRunning {experiment['label']} ({experiment['model_id']})")
        model = HuggingFaceModel(experiment["model_id"], dtype=args.dtype)

        model.print_model_size()
        model.zipnn_compress()
        model.neuralzip_compress(
            method="huffman",
            ngram=3,
            cluster=True,
            exp="cluster-huffman-trigram",
            validate=False,
        )
        model.neuralzip_compress(
            method="arith",
            ngram=3,
            cluster=True,
            exp="cluster-arith-trigram",
            validate=args.validate,
        )


if __name__ == "__main__":
    main()
