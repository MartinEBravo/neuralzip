import heapq
import math
from collections import defaultdict

import numpy as np
from scipy.spatial.distance import jensenshannon


def huffman_weighted_length(counts: np.ndarray) -> int:
    if counts.size <= 1:
        return 0
    heap = [int(x) for x in counts if int(x) > 0]
    if len(heap) <= 1:
        return 0
    heapq.heapify(heap)
    total = 0
    while len(heap) > 1:
        a = heapq.heappop(heap)
        b = heapq.heappop(heap)
        merged = a + b
        total += merged
        heapq.heappush(heap, merged)
    return total


def pack_ngrams_uint(exponents: np.ndarray, n: int) -> np.ndarray:
    usable = (len(exponents) // n) * n
    if usable == 0:
        return np.array([], dtype=np.uint32)
    packed_input = exponents[:usable].reshape(-1, n).astype(np.uint32, copy=False)
    packed = np.zeros(packed_input.shape[0], dtype=np.uint32)
    for i in range(n):
        packed |= (packed_input[:, i] & 0xFF) << (8 * i)
    return packed


def first_order_arithmetic_bits_from_counts(
    symbol_counts: dict[int, int],
    prev_counts: dict[int, int],
    pair_counts: dict[int, int],
    eps: float = 1.0,
) -> float:
    if sum(symbol_counts.values()) <= 1:
        return 0.0
    vocab_size = len(symbol_counts)
    total_bits = 0.0
    for pair_key, pair_count in pair_counts.items():
        prev_symbol = int(pair_key >> 32)
        row_total = prev_counts.get(prev_symbol, 0)
        prob = (pair_count + eps) / (row_total + eps * vocab_size)
        total_bits += pair_count * -math.log2(prob)
    return total_bits


def cluster_layers(layer_histograms: list[np.ndarray], threshold: float) -> list[list[int]]:
    histograms = [hist for hist in layer_histograms if hist.sum() > 0]
    original_indices = [i for i, hist in enumerate(layer_histograms) if hist.sum() > 0]

    clusters: list[dict[str, object]] = []
    for layer_idx, hist in zip(original_indices, histograms):
        placed = False
        for cluster in clusters:
            centroid = cluster["centroid"]
            if float(jensenshannon(hist, centroid)) < threshold:
                cluster["members"].append(layer_idx)
                count = len(cluster["members"])
                cluster["centroid"] = ((count - 1) * centroid + hist) / count
                placed = True
                break
        if not placed:
            clusters.append({"members": [layer_idx], "centroid": hist.copy()})
    return [cluster["members"] for cluster in clusters]


def accumulate_counts(target: dict[int, int], keys: np.ndarray, counts: np.ndarray) -> None:
    for key, count in zip(keys.tolist(), counts.tolist()):
        target[int(key)] += int(count)


class PackedSequenceStats:
    def __init__(self, ngram: int):
        self.ngram = ngram
        self.pending = np.array([], dtype=np.uint8)
        self.symbol_counts: dict[int, int] = defaultdict(int)
        self.prev_counts: dict[int, int] = defaultdict(int)
        self.pair_counts: dict[int, int] = defaultdict(int)
        self.previous_symbol: int | None = None

    def update_exponents(self, exponents: np.ndarray) -> None:
        stream = exponents.astype(np.uint8, copy=False)
        if self.pending.size:
            stream = np.concatenate([self.pending, stream])
        usable = (stream.size // self.ngram) * self.ngram
        if usable == 0:
            self.pending = stream
            return
        packed = pack_ngrams_uint(stream[:usable], self.ngram)
        self.pending = stream[usable:]
        self.update_packed_symbols(packed)

    def update_packed_symbols(self, packed: np.ndarray) -> None:
        if packed.size == 0:
            return

        symbols, symbol_counts = np.unique(packed, return_counts=True)
        accumulate_counts(self.symbol_counts, symbols, symbol_counts)

        if self.previous_symbol is not None:
            boundary_key = (int(self.previous_symbol) << 32) | int(packed[0])
            self.prev_counts[int(self.previous_symbol)] += 1
            self.pair_counts[boundary_key] += 1

        if packed.size > 1:
            previous = packed[:-1]
            previous_symbols, previous_counts = np.unique(previous, return_counts=True)
            accumulate_counts(self.prev_counts, previous_symbols, previous_counts)

            pair_keys = (previous.astype(np.uint64) << 32) | packed[1:].astype(np.uint64)
            unique_pairs, pair_counts = np.unique(pair_keys, return_counts=True)
            accumulate_counts(self.pair_counts, unique_pairs, pair_counts)

        self.previous_symbol = int(packed[-1])

    def bit_length(self) -> float:
        return first_order_arithmetic_bits_from_counts(self.symbol_counts, self.prev_counts, self.pair_counts)
