# Dependency note:
# Install once in the environment before running this script:
#   python -m pip install zstandard

import argparse
import torch
import numpy as np
from transformers import AutoModelForCausalLM
from tqdm import tqdm
import heapq
import zstandard as zstd
from scipy.spatial.distance import jensenshannon

# implementation: https://www.geeksforgeeks.org/dsa/huffman-coding-in-python/
class Node:
    def __init__(self, symbol=None, frequency=None):
        self.symbol = symbol
        self.frequency = frequency
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.frequency < other.frequency

def build_huffman_tree(chars, freq):
    # Create a priority queue of nodes
    priority_queue = [Node(char, f) for char, f in zip(chars, freq)]
    heapq.heapify(priority_queue)

    # Build the Huffman tree
    while len(priority_queue) > 1:
        left_child = heapq.heappop(priority_queue)
        right_child = heapq.heappop(priority_queue)
        merged_node = Node(frequency=left_child.frequency + right_child.frequency)
        merged_node.left = left_child
        merged_node.right = right_child
        heapq.heappush(priority_queue, merged_node)

    return priority_queue[0]

def generate_huffman_codes(node, code="", huffman_codes={}):
    if node is not None:
        if node.symbol is not None:
            huffman_codes[node.symbol] = code
        generate_huffman_codes(node.left, code + "0", huffman_codes)
        generate_huffman_codes(node.right, code + "1", huffman_codes)

    return huffman_codes

def extract_fp32_parts(arr):
    bits = arr.view(np.uint32)
    sign = (bits >> 31) & 0x1
    exponents = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    return sign.astype(np.uint32), exponents.astype(np.uint32), mantissa.astype(np.uint32)

class HuggingFaceModel:
    def __init__(self, model_name):
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32
        )
        self._params = [p.detach().to(torch.float32).cpu().numpy().flatten()
                        for p in self._model.parameters()]
        self._weights = np.concatenate(self._params)

        self._exp_offsets = None
        self._exp_codes = None
        self._exp_block_size = None
        self._exp_total_bits = None
        self._exp_ngram = None

    def log_bits(self, name, total_bits):
        mbits = total_bits / 1e6
        mbytes = total_bits / 8e6
        ratio = 100 * (total_bits / (self._weights.size * 32))
        print(f"[{name}] {mbits:8.2f} Mbits | {mbytes:8.2f} MB | {ratio:5.2f}% of original")

    def print_model_size(self):
        total_bits = self._weights.size * 32
        self.log_bits("Original", total_bits)
        return total_bits

    def compute_arithmetic_bits(self, arr, order=1):
        """Adaptive arithmetic coding bits estimation (order-1)."""
        if arr.size < 2:
            return 0.0
        A = 256
        eps = 1
        arr = arr.astype(np.uint8)
        counts = np.ones((A, A)) * eps
        totals = np.ones(A) * (A * eps)
        bits = 0.0
        prev = int(arr[0])
        for x in arr[1:]:
            p = counts[prev, x] / totals[prev]
            bits += -np.log2(p)
            counts[prev, x] += 1
            totals[prev] += 1
            prev = x
        return bits

    def compute_sigma_huffman_bits(self, arr):
        arr = arr.astype(np.uint8)
        if arr.size < 2:
            return 0.0

        A = 256
        counts = np.zeros((A, A), dtype=np.int64)
        prev = int(arr[0])
        for x in arr[1:]:
            x = int(x)
            counts[prev, x] += 1
            prev = x

        total_bits = 0.0
        for prev in range(A):
            row = counts[prev]
            row_sum = int(row.sum())
            if row_sum == 0:
                continue

            symbols = np.nonzero(row)[0]
            freqs = row[symbols]

            root = build_huffman_tree(symbols.tolist(), freqs.tolist())
            codes = generate_huffman_codes(root, code="", huffman_codes={})

            for s, f in zip(symbols, freqs):
                total_bits += int(f) * len(codes[int(s)])

        return total_bits

    def compute_zipnn_bits(self, arr, method="huffman"):
        """
        Compute expected bits using:
        - 'huffman' : Huffman global (orden-0)
        - 'arith'   : modelo aritmético adaptativo (orden-1)
        - 'sigma'   : σ-Huffman (orden-1 con Huffman por contexto)
        """
        if arr.size == 0:
            return 0.0
        if method == "arith":
            return self.compute_arithmetic_bits(arr)
        if method == "sigma":
            return self.compute_sigma_huffman_bits(arr)

        # método clásico: Huffman global
        unique, counts = np.unique(arr, return_counts=True)
        probs = counts / counts.sum()
        bits = np.sum(counts * np.ceil(-np.log2(probs)))
        return bits

    def pack_ngrams_uint(self, exps, n=2):
        """Pack consecutive exponent symbols into 16–24-bit ngrams."""
        L = (len(exps) // n) * n
        x = exps[:L].reshape(-1, n).astype(np.uint32)
        packed = np.zeros(x.shape[0], dtype=np.uint32)
        for i in range(n):
            packed |= (x[:, i] & 0xFF) << (8 * i)
        return packed

    def cluster_layers(self, layer_arrays, threshold=0.02):
        """Group layers with similar exponent histograms (Jensen–Shannon)."""
        hists = []
        for arr in layer_arrays:
            _, exps, _ = extract_fp32_parts(arr)
            h = np.bincount(exps, minlength=256).astype(np.float64)
            h /= h.sum() if h.sum() > 0 else 1
            hists.append(h)
        clusters = []
        for i, h in enumerate(hists):
            placed = False
            for c in clusters:
                if jensenshannon(h, c["centroid"]) < threshold:
                    c["idx"].append(i)
                    c["centroid"] = (c["centroid"] + h) / 2
                    placed = True
                    break
            if not placed:
                clusters.append({"idx": [i], "centroid": h})
        return [c["idx"] for c in clusters]

    def build_huffman_index(self, exps, block_size=4096, ngram=1):
        if ngram > 1:
            exps = self.pack_ngrams_uint(exps, n=ngram).astype(np.uint32)

        unique, counts = np.unique(exps, return_counts=True)
        root = build_huffman_tree(unique.tolist(), counts.tolist())
        codes = generate_huffman_codes(root)

        offsets = []
        bit_offset = 0
        for i, sym in enumerate(exps):
            if i % block_size == 0:
                offsets.append(bit_offset)
            bit_offset += len(codes[sym])

        return np.array(offsets, dtype=np.uint64), codes, bit_offset

    def build_global_exponent_index(self, block_size=4096, ngram=1):
        _, exps, mant = extract_fp32_parts(self._weights)
        offsets, codes, total_bits_exp = self.build_huffman_index(
            exps, block_size=block_size, ngram=ngram
        )

        self._exp_offsets = offsets
        self._exp_codes = codes
        self._exp_block_size = block_size
        self._exp_total_bits = total_bits_exp
        self._exp_ngram = ngram

        total_bits = total_bits_exp + mant.size * 23 + exps.size
        self.log_bits(
            f"Exp bitstream indexed (block={block_size}, n={ngram})",
            total_bits,
        )
        return total_bits

    def exp_block_start_bit(self, position):
        if self._exp_offsets is None:
            raise RuntimeError("Index not built. Call build_global_exponent_index first.")
        b = position // self._exp_block_size
        if b >= len(self._exp_offsets):
            raise IndexError("position out of range for exponent stream")
        return self._exp_offsets[b]

    def zipnn_compress(self, verbose=1):
        """Classic ZipNN: compress only exponent field globally (orden-0)."""
        sign, exps, mant = extract_fp32_parts(self._weights)
        bits_exp = self.compute_zipnn_bits(exps, method="huffman")
        total_bits = bits_exp + mant.size * 23 + sign.size
        if verbose:
            self.log_bits("ZipNN", total_bits)
        return total_bits

    def neuralzip_compress(self, verbose=5, ngram=1, method="huffman",
                          cluster=False, direct_block=0, exp=""):
        total_bits = 0
        layer_arrays = [p.detach().to(torch.float32).cpu().numpy().flatten()
                        for _, p in self._model.named_parameters()]

        if cluster:
            clusters = self.cluster_layers(layer_arrays)
            for cluster_ids in tqdm(clusters, desc="NeuralZip clustered"):
                all_exps_model = []
                total_floats_cluster = 0

                for idx in cluster_ids:
                    arr = layer_arrays[idx]
                    _, exps_raw, mant_raw = extract_fp32_parts(arr)
                    total_floats_cluster += exps_raw.size   # = mant_raw.size

                    if method == "sigma":
                        exps_model = exps_raw.astype(np.uint8)
                    else:
                        exps_model = (self.pack_ngrams_uint(exps_raw, n=ngram)
                                      if ngram > 1 else exps_raw)
                    if exps_model.size == 0:
                        continue
                    all_exps_model.append(exps_model)

                if not all_exps_model:
                    total_bits += total_floats_cluster * (23 + 1)
                    continue
                exps_all = np.concatenate(all_exps_model)

                bits_exp = self.compute_zipnn_bits(exps_all, method=method)
                bits_total = bits_exp + total_floats_cluster * (23 + 1)
                total_bits += bits_total
        else:
            for name, param in tqdm(self._model.named_parameters(),
                                    desc="NeuralZip", total=len(layer_arrays)):
                arr = param.detach().to(torch.float32).cpu().numpy().flatten()
                if arr.size == 0:
                    continue
                sign, exps, mant = extract_fp32_parts(arr)

                if method == "sigma":
                    exps_model = exps.astype(np.uint8)
                else:
                    exps_model = (self.pack_ngrams_uint(exps, n=ngram)
                                  if ngram > 1 else exps)

                bits_exp = self.compute_zipnn_bits(exps_model, method=method)
                bits_mant = self.compute_zipnn_bits(mant, method="huffman")
                bits_total = bits_exp + bits_mant + sign.size
                total_bits += bits_total

        if verbose:
            self.log_bits(f"NeuralZip ({exp}) [n={ngram}, {method}, cluster={cluster}]",
                          total_bits)
        return total_bits


    def zstd_compress(self, level=19, verbose=1):
        raw = self._weights.tobytes()
        cctx = zstd.ZstdCompressor(level=level, threads=8)
        compressed = cctx.compress(raw)
        bits = len(compressed) * 8
        if verbose:
            self.log_bits(f"Zstd/level={level}", bits)
        return bits

def parse_args():
    parser = argparse.ArgumentParser(description="Run compression experiments.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model ids or labels to run. Defaults to all experiments.",
    )
    return parser.parse_args()


def select_experiments(model_selectors, experiments):
    if not model_selectors:
        return experiments

    lookup = {}
    for experiment in experiments:
        lookup[experiment["model_id"].lower()] = experiment
        lookup[experiment["label"].lower()] = experiment

    selected = []
    missing = []
    for selector in model_selectors:
        experiment = lookup.get(selector.lower())
        if experiment is None:
            missing.append(selector)
            continue
        if experiment not in selected:
            selected.append(experiment)

    if missing:
        raise ValueError(f"Unknown model selectors: {', '.join(missing)}")

    return selected


def main():
    args = parse_args()
    experiments = [
        {
            "label": "GPT-2",
            "model_id": "gpt2",
        },
        {
            "label": "GPT-2 Nano",
            "model_id": "sshleifer/tiny-gpt2",
        },
        {
            "label": "Gemma 4 E2B",
            "model_id": "google/gemma-4-E2B",
        },
        {
            "label": "Gemma 4 E2B IT",
            "model_id": "google/gemma-4-E2B-it",
        },
        {
            "label": "Gemma 4 E4B",
            "model_id": "google/gemma-4-E4B",
        },
        {
            "label": "Gemma 4 E4B IT",
            "model_id": "google/gemma-4-E4B-it",
        },
        {
            "label": "Llama 3.2 1B",
            "model_id": "meta-llama/Llama-3.2-1B",
        },
        {
            "label": "Llama 3.2 1B Instruct",
            "model_id": "meta-llama/Llama-3.2-1B-Instruct",
        },
        {
            "label": "Llama 3.2 3B",
            "model_id": "meta-llama/Llama-3.2-3B",
        },
        {
            "label": "Llama 3.2 3B Instruct",
            "model_id": "meta-llama/Llama-3.2-3B-Instruct",
        },
        {
            "label": "Llama 3.2 11B Vision",
            "model_id": "meta-llama/Llama-3.2-11B-Vision",
        },
        {
            "label": "Llama 3.2 11B Vision Instruct",
            "model_id": "meta-llama/Llama-3.2-11B-Vision-Instruct",
        },
        {
            "label": "Phi-4-mini",
            "model_id": "microsoft/Phi-4-mini-instruct",
        },
        {
            "label": "Qwen 3.5 0.8B",
            "model_id": "Qwen/Qwen3.5-0.8B",
        },
        {
            "label": "Qwen 3.5 2B",
            "model_id": "Qwen/Qwen3.5-2B",
        },
        {
            "label": "Qwen 3.5 4B",
            "model_id": "Qwen/Qwen3.5-4B",
        },
        {
            "label": "Qwen 3.5 9B",
            "model_id": "Qwen/Qwen3.5-9B",
        },
    ]

    for experiment in select_experiments(args.models, experiments):
        print(f"\nRunning {experiment['label']} ({experiment['model_id']})")
        model = HuggingFaceModel(experiment["model_id"])

        model.print_model_size()
        model.zipnn_compress()
        model.neuralzip_compress(
            method="arith",
            ngram=3,
            cluster=True,
            direct_block=4096,
            exp="cluster-arith-trigram-direct",
        )


if __name__ == "__main__":
    main()
