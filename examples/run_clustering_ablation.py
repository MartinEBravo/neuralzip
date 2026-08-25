from __future__ import annotations

import argparse
import gc
import json
import math
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import torch

from neuralzip.metrics import exponent_distribution, jensen_shannon_distance
from neuralzip.neuralzip_codec import NeuralZipCodec

try:
    from .run_thesis_model_suite import MODEL_SPECS, SPECS_BY_SLUG, load_model
except ImportError:  # Direct execution from the examples directory.
    from run_thesis_model_suite import MODEL_SPECS, SPECS_BY_SLUG, load_model


def normalized_role(model_slug: str, name: str) -> str:
    if model_slug == "gpt2":
        return re.sub(r"(^|\.)h\.\d+\.", r"\1h.#.", name)
    if model_slug == "bert-base":
        return re.sub(r"encoder\.layer\.\d+\.", "encoder.layer.#.", name)
    if model_slug in {"resnet18", "resnet34"}:
        return re.sub(r"layer(\d+)\.\d+\.", r"layer\1.#.", name)
    raise ValueError(f"Unsupported model slug: {model_slug}")


def _transformer_pairs(model_slug: str, names: set[str]):
    if model_slug == "gpt2":
        pattern = re.compile(r"^h\.(\d+)\.mlp\.c_fc\.weight$")
        right_template = "h.{depth}.mlp.c_proj.weight"
        label = "FFN expansion--reduction"
    elif model_slug == "bert-base":
        pattern = re.compile(r"^encoder\.layer\.(\d+)\.intermediate\.dense\.weight$")
        right_template = "encoder.layer.{depth}.output.dense.weight"
        label = "FFN expansion--reduction"
    else:
        return None

    pairs = []
    for name in sorted(names):
        match = pattern.match(name)
        if not match:
            continue
        depth = int(match.group(1))
        right = right_template.format(depth=depth)
        if right in names:
            pairs.append((depth, name, right))
    return label, pairs


def _resnet_pairs(names: set[str]):
    pattern = re.compile(r"^layer(\d+)\.(\d+)\.conv2\.weight$")
    pairs = []
    for name in sorted(names):
        match = pattern.match(name)
        if not match:
            continue
        stage = int(match.group(1))
        block = int(match.group(2))
        right = f"layer{stage}.{block + 1}.conv1.weight"
        if right in names:
            pairs.append((stage * 100 + block, name, right))
    return "Adjacent block convolutions", pairs


def coupled_pairs(model_slug: str, names: set[str]):
    transformer = _transformer_pairs(model_slug, names)
    if transformer is not None:
        return transformer
    if model_slug in {"resnet18", "resnet34"}:
        return _resnet_pairs(names)
    raise ValueError(f"Unsupported model slug: {model_slug}")


def _entropy(labels) -> float:
    counts = Counter(labels)
    total = sum(counts.values())
    return -sum(
        (count / total) * math.log(count / total) for count in counts.values() if count
    )


def _log_comb(n: int, k: int) -> float:
    if k < 0 or k > n:
        return float("-inf")
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def adjusted_mutual_information(left, right) -> float:
    left = list(left)
    right = list(right)
    if len(left) != len(right) or not left:
        raise ValueError("AMI requires two non-empty label sequences of equal length.")

    total = len(left)
    left_counts = Counter(left)
    right_counts = Counter(right)
    joint = Counter(zip(left, right))
    mutual_information = 0.0
    for (left_label, right_label), count in joint.items():
        mutual_information += (count / total) * math.log(
            count * total / (left_counts[left_label] * right_counts[right_label])
        )

    expected_mutual_information = 0.0
    for left_count in left_counts.values():
        for right_count in right_counts.values():
            lower = max(1, left_count + right_count - total)
            upper = min(left_count, right_count)
            for count in range(lower, upper + 1):
                probability = math.exp(
                    _log_comb(left_count, count)
                    + _log_comb(total - left_count, right_count - count)
                    - _log_comb(total, right_count)
                )
                expected_mutual_information += (
                    count
                    / total
                    * math.log(count * total / (left_count * right_count))
                    * probability
                )

    normalizer = 0.5 * (_entropy(left) + _entropy(right))
    denominator = normalizer - expected_mutual_information
    numerator = mutual_information - expected_mutual_information
    if abs(denominator) < 1e-12:
        equivalent = all(
            (left[i] == left[j]) == (right[i] == right[j])
            for i in range(total)
            for j in range(total)
        )
        return 1.0 if equivalent else 0.0
    return float(numerator / denominator)


def summarize_clusters(assignments: pd.DataFrame):
    rows = []
    for cluster_id, group in assignments.groupby("js_cluster", sort=True):
        role_counts = Counter(group["normalized_role"])
        dominant_role, dominant_count = role_counts.most_common(1)[0]
        rows.append(
            {
                "js_cluster": str(cluster_id),
                "tensors": len(group),
                "roles": len(role_counts),
                "dominant_role": dominant_role,
                "role_purity": float(dominant_count / len(group)),
                "members": " | ".join(group["name"]),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["tensors", "role_purity"], ascending=[False, False]
    )


def run_model(spec, model_root: Path, output_root: Path, threshold: float):
    print(f"\n=== {spec.label} ===", flush=True)
    view = load_model(spec, model_root=model_root)
    model = view.model
    model.requires_grad_(False)
    named_tensors = sorted(
        (
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.is_floating_point()
        ),
        key=lambda item: item[0],
    )

    codec = NeuralZipCodec(
        cluster_js_threshold=threshold,
        ngram_values=(),
        packed_source_budget_ratio=0.0,
    )
    codec.prepare_clusters(named_tensors)

    distributions = {}
    assignment_rows = []
    for name, tensor in named_tensors:
        distributions[name] = exponent_distribution(tensor)
        assignment_rows.append(
            {
                "name": name,
                "normalized_role": normalized_role(spec.key, name),
                "numel": int(tensor.numel()),
                "js_cluster": str(codec.cluster_id(name)),
            }
        )
    assignments = pd.DataFrame(assignment_rows)
    assignment_by_name = assignments.set_index("name")
    pair_label, selected_pairs = coupled_pairs(spec.key, set(assignments["name"]))
    if not selected_pairs:
        raise RuntimeError(f"No representative pairs found for {spec.label}.")

    pair_rows = []
    left_clusters = []
    right_clusters = []
    for depth, left_name, right_name in selected_pairs:
        left_cluster = str(assignment_by_name.loc[left_name, "js_cluster"])
        right_cluster = str(assignment_by_name.loc[right_name, "js_cluster"])
        left_clusters.append(left_cluster)
        right_clusters.append(right_cluster)
        pair_rows.append(
            {
                "depth": depth,
                "left": left_name,
                "right": right_name,
                "left_cluster": left_cluster,
                "right_cluster": right_cluster,
                "same_cluster": left_cluster == right_cluster,
                "js_distance": jensen_shannon_distance(
                    distributions[left_name], distributions[right_name]
                ),
            }
        )

    pairs = pd.DataFrame(pair_rows)
    clusters = summarize_clusters(assignments)
    summary = {
        "model_slug": spec.key,
        "model": spec.label,
        "dtype": "bfloat16",
        "js_threshold": float(threshold),
        "tensors": len(assignments),
        "js_clusters": int(assignments["js_cluster"].nunique()),
        "representative_pair": pair_label,
        "paired_depths": len(pairs),
        "same_cluster_pct": float(100.0 * pairs["same_cluster"].mean()),
        "adjusted_mutual_information": adjusted_mutual_information(
            left_clusters, right_clusters
        ),
        "mean_pair_js": float(pairs["js_distance"].mean()),
    }

    output_dir = output_root / spec.key
    output_dir.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(output_dir / "assignments.csv", index=False)
    pairs.to_csv(output_dir / "coupled_pairs.csv", index=False)
    clusters.to_csv(output_dir / "cluster_summary.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)

    del codec, model, view, named_tensors, distributions
    gc.collect()
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["all"])
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--threads", type=int, default=12)
    args = parser.parse_args()

    if args.models == ["all"]:
        specs = MODEL_SPECS
    else:
        unknown = sorted(set(args.models) - set(SPECS_BY_SLUG))
        if unknown:
            parser.error(f"Unknown model slugs: {', '.join(unknown)}")
        specs = tuple(SPECS_BY_SLUG[slug] for slug in args.models)

    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = [
        run_model(spec, args.model_root, args.output_dir, args.threshold)
        for spec in specs
    ]
    pd.DataFrame(summaries).to_csv(args.output_dir / "model_summary.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
