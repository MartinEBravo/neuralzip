import numpy as np
import torch
from transformers import AutoModel

from .float_utils import extract_float_parts
from .formats import FLOAT_FORMATS


DEFAULT_NGRAM_SIZES = (1, 2, 4, 8)
DEFAULT_MAX_HEATMAP_TENSORS = 48
DEFAULT_JS_THRESHOLD = 0.02


def resolve_torch_dtype(dtype):
    if dtype is None or dtype == "auto":
        return "auto"
    if isinstance(dtype, torch.dtype):
        return dtype

    dtype_key = str(dtype).lower().replace("torch.", "").replace("-", "_")
    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "f16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "f32": torch.float32,
    }
    if dtype_key not in dtype_map:
        choices = ", ".join(("auto", *sorted(dtype_map)))
        raise ValueError(f"Unsupported dtype {dtype!r}. Expected one of: {choices}.")
    return dtype_map[dtype_key]


def float_field_widths(dtype):
    fmt = FLOAT_FORMATS[dtype]
    return {
        "sign_bits": 1,
        "exponent_bits": fmt["total_bits"] - fmt["mantissa_bits"] - 1,
        "mantissa_bits": fmt["mantissa_bits"],
    }


def entropy_from_counts(counts):
    counts = np.asarray(list(counts.values()), dtype=np.float64)
    total = counts.sum()
    if total == 0:
        return 0.0
    probabilities = counts / total
    return float(-np.sum(probabilities * np.log2(probabilities)))


def empirical_entropy(values):
    values = np.asarray(values).reshape(-1)
    if values.size == 0:
        return 0.0
    _, counts = np.unique(values, return_counts=True)
    probabilities = counts.astype(np.float64) / float(values.size)
    return float(-np.sum(probabilities * np.log2(probabilities)))


def aggregate_symbol_counts(streams):
    counts = {}
    total = 0
    for stream in streams:
        values = stream.reshape(-1).astype(np.uint64, copy=False)
        total += values.size
        symbols, symbol_counts = np.unique(values, return_counts=True)
        for symbol, count in zip(symbols.tolist(), symbol_counts.tolist()):
            counts[symbol] = counts.get(symbol, 0) + int(count)
    return counts, total


def packed_ngram_counts(exponent_streams, n, exponent_bits):
    counts = {}
    ngram_count = 0
    tail_count = 0
    total_count = 0

    for stream in exponent_streams:
        values = stream.reshape(-1).astype(np.uint64, copy=False)
        total_count += values.size
        full_length = (values.size // n) * n
        tail_count += values.size - full_length
        if full_length == 0:
            continue

        grams = values[:full_length].reshape(-1, n)
        packed = np.zeros(grams.shape[0], dtype=np.uint64)
        for offset in range(n):
            packed |= grams[:, offset] << np.uint64(offset * exponent_bits)

        symbols, symbol_counts = np.unique(packed, return_counts=True)
        ngram_count += packed.size
        for symbol, count in zip(symbols.tolist(), symbol_counts.tolist()):
            counts[symbol] = counts.get(symbol, 0) + int(count)

    return counts, ngram_count, tail_count, total_count


def normalized_exponent_histogram(stream, exponent_bits):
    values = stream.reshape(-1).astype(np.int64, copy=False)
    counts = np.bincount(values, minlength=2**exponent_bits).astype(np.float64)
    total = counts.sum()
    return counts / total if total else counts


def kl_divergence(p, q):
    mask = p > 0
    return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))


def js_distance(p, q):
    midpoint = 0.5 * (p + q)
    return float(
        np.sqrt(
            0.5 * kl_divergence(p, midpoint)
            + 0.5 * kl_divergence(q, midpoint)
        )
    )


def pairwise_js_distance(histograms):
    n_histograms = len(histograms)
    distances = np.zeros((n_histograms, n_histograms), dtype=np.float64)
    for i in range(n_histograms):
        for j in range(i):
            distances[i, j] = js_distance(histograms[i], histograms[j])
            distances[j, i] = distances[i, j]
    return distances


def greedy_js_clusters(histograms, threshold):
    clusters = []
    centroids = []
    for index, histogram in enumerate(histograms):
        assigned = False
        for cluster_index, centroid in enumerate(centroids):
            if js_distance(histogram, centroid) <= threshold:
                clusters[cluster_index].append(index)
                centroids[cluster_index] = np.mean(
                    [histograms[i] for i in clusters[cluster_index]],
                    axis=0,
                )
                assigned = True
                break
        if not assigned:
            clusters.append([index])
            centroids.append(histogram.copy())
    return clusters


def selected_tensor_indices(metadata, max_tensors):
    largest = sorted(
        range(len(metadata)),
        key=lambda index: metadata[index]["numel"],
        reverse=True,
    )[:max_tensors]
    return sorted(largest)


class HFModel:
    def __init__(self, model_name, dtype="auto", **kwargs):
        self.model_name = model_name
        self.dtype = resolve_torch_dtype(dtype)
        self._model = self._load_model(model_name, self.dtype, **kwargs)
        self.linear_projection = []
        self.sign_fields = []
        self.exponent_fields = []
        self.mantissa_fields = []
        self._fields_split = False

    @property
    def model(self):
        return self._model

    def _load_model(self, model_name, dtype, **kwargs):
        load_kwargs = dict(kwargs)
        if "dtype" not in load_kwargs and "torch_dtype" not in load_kwargs:
            load_kwargs["dtype"] = dtype
        return AutoModel.from_pretrained(model_name, **load_kwargs)

    def split_fields(self):
        """Split every floating-point parameter into sign, exponent, and mantissa fields."""
        if self._fields_split:
            return {
                "sign": self.sign_fields,
                "exponent": self.exponent_fields,
                "mantissa": self.mantissa_fields,
            }

        self.linear_projection = []
        self.sign_fields = []
        self.exponent_fields = []
        self.mantissa_fields = []

        for name, tensor in self._model.named_parameters():
            if not torch.is_tensor(tensor) or not tensor.is_floating_point():
                continue

            sign, exponent, mantissa = extract_float_parts(tensor)
            self.linear_projection.append(
                {
                    "name": name,
                    "shape": tuple(int(dim) for dim in tensor.shape),
                    "dtype": str(tensor.dtype).replace("torch.", ""),
                    "numel": int(tensor.numel()),
                }
            )
            self.sign_fields.append(sign)
            self.exponent_fields.append(exponent)
            self.mantissa_fields.append(mantissa)

        self._fields_split = True
        return {
            "sign": self.sign_fields,
            "exponent": self.exponent_fields,
            "mantissa": self.mantissa_fields,
        }

    def compute_zero_order_empirical_entropy(self, field="exponent"):
        """Compute the zero-order empirical entropy of one split field."""
        field_map = {
            "sign": "sign_fields",
            "exponent": "exponent_fields",
            "mantissa": "mantissa_fields",
        }
        if field not in field_map:
            choices = ", ".join(sorted(field_map))
            raise ValueError(f"Unsupported field {field!r}. Expected one of: {choices}.")
        attr_name = field_map[field]
        self.split_fields()
        arrays = [
            arr.reshape(-1).astype(np.uint64, copy=False)
            for arr in getattr(self, attr_name)
        ]
        if not arrays:
            return 0.0
        values = np.concatenate(arrays)
        if values.size == 0:
            return 0.0
        _, counts = np.unique(values, return_counts=True)
        probabilities = counts.astype(np.float64) / float(values.size)
        return float(-np.sum(probabilities * np.log2(probabilities)))

    def compute_zero_order_entropy_summary(self):
        return {
            **float_field_widths(self.dtype),
            "zero_order_entropy_sign": self.compute_zero_order_empirical_entropy(
                field="sign"
            ),
            "zero_order_entropy_exponents": self.compute_zero_order_empirical_entropy(
                field="exponent"
            ),
            "zero_order_entropy_mantissa": self.compute_zero_order_empirical_entropy(
                field="mantissa"
            ),
        }

    def compute_exponent_ngram_entropy(self, ngram_sizes=DEFAULT_NGRAM_SIZES):
        self.split_fields()
        exponent_bits = float_field_widths(self.dtype)["exponent_bits"]
        results = []
        for n in ngram_sizes:
            if n * exponent_bits > 64:
                continue

            counts, ngram_count, tail_count, total_count = packed_ngram_counts(
                self.exponent_fields,
                n=n,
                exponent_bits=exponent_bits,
            )
            entropy_per_ngram = entropy_from_counts(counts)
            ideal_bits = ngram_count * entropy_per_ngram + tail_count * exponent_bits
            bits_per_exponent = ideal_bits / total_count if total_count else 0.0
            results.append(
                {
                    "n": n,
                    "entropy_per_ngram": entropy_per_ngram,
                    "bits_per_exponent": bits_per_exponent,
                    "tail_fraction": tail_count / total_count if total_count else 0.0,
                    "unique_ngrams": len(counts),
                }
            )
        return results

    def compute_blockwise_heterogeneity(
        self,
        max_heatmap_tensors=DEFAULT_MAX_HEATMAP_TENSORS,
        js_threshold=DEFAULT_JS_THRESHOLD,
    ):
        self.split_fields()
        exponent_bits = float_field_widths(self.dtype)["exponent_bits"]
        exponent_counts, total_exponents = aggregate_symbol_counts(
            self.exponent_fields
        )
        global_entropy = entropy_from_counts(exponent_counts)
        tensorwise_entropy = (
            sum(
                stream.size * empirical_entropy(stream)
                for stream in self.exponent_fields
            )
            / total_exponents
            if total_exponents
            else 0.0
        )

        indices = selected_tensor_indices(self.linear_projection, max_heatmap_tensors)
        histograms = [
            normalized_exponent_histogram(self.exponent_fields[index], exponent_bits)
            for index in indices
        ]
        clusters = greedy_js_clusters(histograms, js_threshold)
        clustered_order = [index for cluster in clusters for index in cluster]
        js_matrix = pairwise_js_distance(histograms)
        js_matrix = js_matrix[np.ix_(clustered_order, clustered_order)]

        lower_triangle_values = js_matrix[np.tril_indices_from(js_matrix, k=-1)]
        mean_js = (
            float(np.mean(lower_triangle_values))
            if lower_triangle_values.size
            else 0.0
        )
        max_js = (
            float(np.max(lower_triangle_values))
            if lower_triangle_values.size
            else 0.0
        )

        return {
            "summary": {
                "global_entropy": global_entropy,
                "tensorwise_entropy": tensorwise_entropy,
                "tensorwise_gain": global_entropy - tensorwise_entropy,
                "mean_js_distance": mean_js,
                "max_js_distance": max_js,
                "n_tensors": len(self.exponent_fields),
                "n_heatmap_tensors": len(indices),
                "n_js_clusters": len(clusters),
                "total_exponents": total_exponents,
            },
            "js_matrix": js_matrix,
        }

    def compress(
        self,
        method,
        **kwargs,
    ):
        from .compressor import Compressor

        return Compressor(method, **kwargs).compress(self)
