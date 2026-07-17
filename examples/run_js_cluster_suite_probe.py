from __future__ import annotations

import argparse
import gc
import json
import re
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import pandas as pd
import torch
from transformers import (
    Gemma3TextConfig,
    Gemma3TextModel,
    Qwen3_5Config,
    Qwen3_5TextModel,
    SiglipConfig,
    SiglipVisionModel,
    WhisperModel,
)

from neuralzip.neuralzip_codec import (
    NeuralZipCodec,
    _exponent_distribution,
    _jensen_shannon,
)


LAYER_PATTERN = re.compile(r"(^|\.)layers\.(\d+)\.")


@dataclass(frozen=True)
class ModelSpec:
    slug: str
    label: str
    relative_path: str
    kind: str


MODEL_SPECS = (
    ModelSpec("qwen35-08b", "Qwen 3.5 0.8B", "qwen35-08b", "qwen_text"),
    ModelSpec("gemma3-1b", "Gemma 3 1B", "gemma3-1b-pt", "gemma_text"),
    ModelSpec(
        "siglip2-so400m",
        "SigLIP 2 So400m",
        "siglip2-so400m-384",
        "siglip_vision",
    ),
    ModelSpec(
        "whisper-v3-turbo",
        "Whisper v3 Turbo",
        "whisper-large-v3-turbo",
        "whisper",
    ),
)
SPECS_BY_SLUG = {spec.slug: spec for spec in MODEL_SPECS}


def normalized_role(name: str) -> str:
    return LAYER_PATTERN.sub(lambda match: f"{match.group(1)}layers.#.", str(name))


def layer_index(name: str):
    match = LAYER_PATTERN.search(str(name))
    return int(match.group(2)) if match else None


def layer_stack(name: str):
    match = LAYER_PATTERN.search(str(name))
    if not match:
        return None
    prefix = str(name)[: match.start()].rstrip(".")
    return prefix or "root"


def load_model(spec: ModelSpec, model_root: Path):
    local_path = model_root / spec.relative_path
    common = {"dtype": torch.bfloat16, "local_files_only": True}
    if spec.kind == "qwen_text":
        config = Qwen3_5Config.from_pretrained(
            local_path, local_files_only=True
        ).text_config
        return Qwen3_5TextModel.from_pretrained(
            local_path, config=config, **common
        ).eval()
    if spec.kind == "gemma_text":
        config = Gemma3TextConfig.from_pretrained(local_path, local_files_only=True)
        return Gemma3TextModel.from_pretrained(
            local_path, config=config, **common
        ).eval()
    if spec.kind == "siglip_vision":
        config = SiglipConfig.from_pretrained(
            local_path, local_files_only=True
        ).vision_config
        return SiglipVisionModel.from_pretrained(
            local_path, config=config, **common
        ).eval()
    if spec.kind == "whisper":
        return WhisperModel.from_pretrained(local_path, **common).eval()
    raise ValueError(f"Unsupported model kind: {spec.kind}")


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


def run_model(spec: ModelSpec, model_root: Path, output_root: Path, threshold: float):
    print(f"\n=== {spec.label} ===", flush=True)
    model = load_model(spec, model_root)
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
        distributions[name] = _exponent_distribution(tensor)
        assignment_rows.append(
            {
                "name": name,
                "layer_stack": layer_stack(name),
                "layer_index": layer_index(name),
                "normalized_role": normalized_role(name),
                "numel": int(tensor.numel()),
                "js_cluster": codec.cluster_id(name),
            }
        )
    assignments = pd.DataFrame(assignment_rows)
    roles, pairs = summarize_roles(assignments, distributions)
    clusters = summarize_clusters(assignments)

    recovered = roles[roles["js_clusters"] == 1]
    mixed_clusters = clusters[(clusters["tensors"] >= 2) & (clusters["roles"] > 1)]
    pure_clusters = clusters[(clusters["tensors"] >= 2) & (clusters["roles"] == 1)]
    same_pairs = int(pairs["same_js_cluster"].sum())
    summary = {
        "model_slug": spec.slug,
        "model": spec.label,
        "dtype": "bfloat16",
        "js_threshold": float(threshold),
        "tensors": int(len(assignments)),
        "transformer_blocks": int(
            assignments.dropna(subset=["layer_index"])[
                ["layer_stack", "layer_index"]
            ].drop_duplicates().shape[0]
        ),
        "js_clusters": int(assignments["js_cluster"].nunique()),
        "repeated_roles": int(len(roles)),
        "roles_fully_recovered": int(len(recovered)),
        "roles_fragmented": int(len(roles) - len(recovered)),
        "non_singleton_pure_role_clusters": int(len(pure_clusters)),
        "non_singleton_mixed_role_clusters": int(len(mixed_clusters)),
        "within_role_pairs": int(len(pairs)),
        "within_role_pairs_same_cluster": same_pairs,
        "within_role_pair_same_cluster_fraction": float(same_pairs / len(pairs)),
        "mean_within_role_js": float(pairs["js_distance"].mean()),
        "median_role_modal_cluster_fraction": float(
            roles["modal_cluster_fraction"].median()
        ),
    }

    output_dir = output_root / spec.slug
    output_dir.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(output_dir / "assignments.csv", index=False)
    roles.to_csv(output_dir / "role_summary.csv", index=False)
    clusters.to_csv(output_dir / "cluster_summary.csv", index=False)
    pairs.sort_values("js_distance").to_csv(
        output_dir / "within_role_pairs.csv", index=False
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print("\nMost coherent repeated roles", flush=True)
    print(roles.head(12).to_string(index=False), flush=True)
    print("\nLargest non-singleton clusters", flush=True)
    print(
        clusters[clusters["tensors"] >= 2].head(12).to_string(index=False),
        flush=True,
    )

    del codec, model, named_tensors, distributions
    gc.collect()
    return summary, roles.assign(model_slug=spec.slug, model=spec.label)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["all"])
    parser.add_argument("--threshold", type=float, default=0.05)
    args = parser.parse_args()

    if args.models == ["all"]:
        specs = MODEL_SPECS
    else:
        unknown = sorted(set(args.models) - set(SPECS_BY_SLUG))
        if unknown:
            parser.error(f"Unknown model slugs: {', '.join(unknown)}")
        specs = tuple(SPECS_BY_SLUG[slug] for slug in args.models)

    torch.set_num_threads(32)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    role_frames = []
    for spec in specs:
        summary, roles = run_model(
            spec, args.model_root, args.output_dir, args.threshold
        )
        summaries.append(summary)
        role_frames.append(roles)

    pd.DataFrame(summaries).to_csv(args.output_dir / "model_summary.csv", index=False)
    pd.concat(role_frames, ignore_index=True).to_csv(
        args.output_dir / "model_role_summary.csv", index=False
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
