from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from itertools import combinations
from pathlib import Path

import pandas as pd
import torch
from transformers import Qwen3_5Config, Qwen3_5TextModel

from neuralzip.neuralzip_codec import (
    NeuralZipCodec,
    _exponent_distribution,
    _jensen_shannon,
)


LAYER_PATTERN = re.compile(r"(^|\.)layers\.(\d+)\.")


def normalized_role(name: str) -> str:
    return LAYER_PATTERN.sub(lambda match: f"{match.group(1)}layers.#.", str(name))


def layer_index(name: str):
    match = LAYER_PATTERN.search(str(name))
    return int(match.group(2)) if match else None


def summarize_roles(assignments: pd.DataFrame, distributions):
    role_rows = []
    pair_rows = []
    for role, group in assignments.groupby("normalized_role", sort=True):
        if len(group) < 2:
            continue
        cluster_counts = group["js_cluster"].value_counts()
        distances = []
        same_cluster_pairs = 0
        for (_, left), (_, right) in combinations(group.iterrows(), 2):
            distance = _jensen_shannon(
                distributions[left["name"]], distributions[right["name"]]
            )
            same_cluster = left["js_cluster"] == right["js_cluster"]
            same_cluster_pairs += int(same_cluster)
            distances.append(distance)
            pair_rows.append(
                {
                    "normalized_role": role,
                    "left": left["name"],
                    "right": right["name"],
                    "left_layer": left["layer_index"],
                    "right_layer": right["layer_index"],
                    "js_distance": distance,
                    "same_js_cluster": same_cluster,
                }
            )
        role_rows.append(
            {
                "normalized_role": role,
                "tensors": int(len(group)),
                "js_clusters": int(len(cluster_counts)),
                "modal_cluster_fraction": float(cluster_counts.iloc[0] / len(group)),
                "pair_same_cluster_fraction": float(
                    same_cluster_pairs / len(distances)
                ),
                "mean_within_role_js": float(sum(distances) / len(distances)),
                "min_within_role_js": float(min(distances)),
                "max_within_role_js": float(max(distances)),
                "cluster_layers": " | ".join(
                    f"{cluster}:"
                    + ",".join(
                        str(int(index))
                        for index in group.loc[
                            group["js_cluster"] == cluster, "layer_index"
                        ].dropna()
                    )
                    for cluster in cluster_counts.index
                ),
            }
        )
    roles = pd.DataFrame(role_rows).sort_values(
        ["modal_cluster_fraction", "tensors", "normalized_role"],
        ascending=[False, False, True],
    )
    pairs = pd.DataFrame(pair_rows)
    return roles, pairs


def summarize_clusters(assignments: pd.DataFrame):
    rows = []
    for cluster_id, group in assignments.groupby("js_cluster", sort=True):
        role_counts = Counter(group["normalized_role"])
        dominant_role, dominant_count = role_counts.most_common(1)[0]
        rows.append(
            {
                "js_cluster": cluster_id,
                "tensors": int(len(group)),
                "roles": int(len(role_counts)),
                "dominant_role": dominant_role,
                "role_purity": float(dominant_count / len(group)),
                "role_counts": " | ".join(
                    f"{role}:{count}" for role, count in role_counts.most_common()
                ),
                "members": " | ".join(group["name"]),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["tensors", "role_purity"], ascending=[False, False]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.05)
    args = parser.parse_args()

    torch.set_num_threads(32)
    config = Qwen3_5Config.from_pretrained(
        args.model_path, local_files_only=True
    ).text_config
    model = Qwen3_5TextModel.from_pretrained(
        args.model_path,
        config=config,
        dtype=torch.bfloat16,
        local_files_only=True,
    ).eval()
    named_tensors = sorted(
        (
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.is_floating_point()
        ),
        key=lambda item: item[0],
    )

    codec = NeuralZipCodec(
        cluster_js_threshold=args.threshold,
        ngram_values=(),
        packed_source_budget_ratio=0.0,
    )
    codec.prepare_clusters(named_tensors)

    distributions = {}
    assignment_rows = []
    for name, tensor in named_tensors:
        distributions[name] = _exponent_distribution(tensor)
        assignment_rows.append(
            {
                "name": name,
                "layer_index": layer_index(name),
                "normalized_role": normalized_role(name),
                "numel": int(tensor.numel()),
                "js_cluster": codec.cluster_id(name),
            }
        )
    assignments = pd.DataFrame(assignment_rows)
    roles, pairs = summarize_roles(assignments, distributions)
    clusters = summarize_clusters(assignments)

    repeated_roles = roles[roles["tensors"] >= 2]
    recovered = repeated_roles[repeated_roles["js_clusters"] == 1]
    fragmented = repeated_roles[repeated_roles["js_clusters"] > 1]
    mixed_clusters = clusters[(clusters["tensors"] >= 2) & (clusters["roles"] > 1)]
    pure_clusters = clusters[(clusters["tensors"] >= 2) & (clusters["roles"] == 1)]
    summary = {
        "model": "Qwen 3.5 0.8B",
        "dtype": "bfloat16",
        "js_threshold": float(args.threshold),
        "tensors": int(len(assignments)),
        "transformer_layers": int(
            assignments["layer_index"].dropna().nunique()
        ),
        "js_clusters": int(assignments["js_cluster"].nunique()),
        "repeated_roles": int(len(repeated_roles)),
        "roles_fully_recovered": int(len(recovered)),
        "roles_fragmented": int(len(fragmented)),
        "non_singleton_pure_role_clusters": int(len(pure_clusters)),
        "non_singleton_mixed_role_clusters": int(len(mixed_clusters)),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(args.output_dir / "assignments.csv", index=False)
    roles.to_csv(args.output_dir / "role_summary.csv", index=False)
    clusters.to_csv(args.output_dir / "cluster_summary.csv", index=False)
    pairs.sort_values("js_distance").to_csv(
        args.output_dir / "within_role_pairs.csv", index=False
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print("\nFully recovered repeated roles", flush=True)
    print(recovered.to_string(index=False), flush=True)
    print("\nMost coherent fragmented roles", flush=True)
    print(fragmented.head(20).to_string(index=False), flush=True)
    print("\nLargest non-singleton clusters", flush=True)
    print(clusters[clusters["tensors"] >= 2].head(25).to_string(index=False), flush=True)
    print("\nMost distant pairs with the same architectural role", flush=True)
    print(
        pairs.sort_values("js_distance", ascending=False).head(20).to_string(
            index=False
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
