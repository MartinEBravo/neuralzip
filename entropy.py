import numpy as np
from scipy.spatial.distance import jensenshannon

from tensors import BlockInfo, FieldEntropySummary, natural_sort_key


def _normalize_histogram(counts: np.ndarray, name: str = "counts") -> np.ndarray:
    total = int(counts.sum())
    if total <= 0:
        raise RuntimeError(f"Block {name!r} has an empty exponent histogram.")
    return counts.astype(np.float64) / total


def entropy_bits_per_symbol(counts: np.ndarray) -> float:
    total = int(counts.sum())
    if total <= 0:
        return 0.0
    probabilities = counts[counts > 0].astype(np.float64) / total
    return float(-(probabilities * np.log2(probabilities)).sum())


def summarize_field_entropy(field: str, grouped_counts: list[np.ndarray]) -> FieldEntropySummary:
    total_symbols = 0
    total_bits = 0.0
    for counts in grouped_counts:
        field_total = int(counts.sum())
        if field_total <= 0:
            continue
        total_symbols += field_total
        total_bits += field_total * entropy_bits_per_symbol(counts)
    bits_per_symbol = 0.0 if total_symbols <= 0 else total_bits / total_symbols
    return FieldEntropySummary(
        field=field,
        total_symbols=total_symbols,
        h0_bits=float(total_bits),
        bits_per_symbol=float(bits_per_symbol),
    )


def block_histograms(block_infos: list[BlockInfo]) -> list[np.ndarray]:
    return [_normalize_histogram(block.exponent_counts, block.name) for block in block_infos]


def compute_js_matrix(histograms: list[np.ndarray]) -> np.ndarray:
    size = len(histograms)
    matrix = np.zeros((size, size), dtype=np.float64)
    for i in range(size):
        for j in range(i + 1, size):
            distance = float(jensenshannon(histograms[i], histograms[j]))
            matrix[i, j] = distance
            matrix[j, i] = distance
    return matrix


def summarize_js_matrix(block_infos: list[BlockInfo], matrix: np.ndarray) -> dict[str, object]:
    size = len(block_infos)
    if size == 0:
        raise RuntimeError("No blocks were collected for JS analysis.")

    if size == 1:
        pairwise = np.array([], dtype=np.float64)
        mean_per_block = np.zeros(1, dtype=np.float64)
    else:
        pairwise = matrix[np.triu_indices(size, k=1)]
        mean_per_block = matrix.sum(axis=1) / (size - 1)

    ranking = sorted(
        (
            {
                "name": block_infos[idx].name,
                "mean_js": float(mean_per_block[idx]),
                "num_weights": int(block_infos[idx].num_weights),
            }
            for idx in range(size)
        ),
        key=lambda item: (-item["mean_js"], natural_sort_key(str(item["name"]))),
    )

    return {
        "num_blocks": size,
        "average_js": float(pairwise.mean()) if pairwise.size else 0.0,
        "median_js": float(np.median(pairwise)) if pairwise.size else 0.0,
        "max_js": float(pairwise.max()) if pairwise.size else 0.0,
        "min_js": float(pairwise.min()) if pairwise.size else 0.0,
        "top_blocks_by_mean_js": ranking[:10],
    }


def cluster_blocks(
    block_infos: list[BlockInfo],
    threshold: float,
    max_clusters: int | None = None,
) -> list[list[int]]:
    weighted_histograms: list[tuple[int, np.ndarray, int]] = []
    for block_idx, block in enumerate(block_infos):
        if int(block.exponent_counts.sum()) <= 0:
            continue
        histogram = _normalize_histogram(block.exponent_counts, block.name)
        weighted_histograms.append((block_idx, histogram, block.num_weights))

    weighted_histograms.sort(key=lambda item: item[2], reverse=True)

    clusters: list[dict[str, object]] = []
    for block_idx, histogram, weight in weighted_histograms:
        best_idx = None
        best_distance = None
        for cluster_idx, cluster in enumerate(clusters):
            distance = float(jensenshannon(histogram, cluster["centroid"]))
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_idx = cluster_idx

        force_assignment = max_clusters is not None and len(clusters) >= max_clusters and best_idx is not None
        if best_idx is not None and (force_assignment or (best_distance is not None and best_distance < threshold)):
            cluster = clusters[best_idx]
            cluster_weight = int(cluster["weight"])
            merged_weight = cluster_weight + weight
            cluster["centroid"] = (
                cluster_weight * cluster["centroid"] + weight * histogram
            ) / merged_weight
            cluster["weight"] = merged_weight
            cluster["members"].append(block_idx)
            continue

        clusters.append(
            {
                "centroid": histogram.copy(),
                "weight": weight,
                "members": [block_idx],
            }
        )

    return [sorted(cluster["members"]) for cluster in clusters]
