import math
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from neuralzip import huff0
from neuralzip.dtypes import DTYPES
from neuralzip.payloads import (
    PACKED_CHUNK_METADATA_BYTES,
    RAW_CHUNK_METADATA_BYTES,
    SHARED_CHUNK_METADATA_BYTES,
    CodebookPayload,
)
from neuralzip.utils import (
    _floating_tensors,
    _model_tensors,
    _stream_chunk_size,
    _symbol_dtype,
    _tensor_raw_bytes,
    exponent_distribution,
    jensen_shannon_distance,
    time_call,
)


@dataclass(frozen=True)
class NeuralZipSetupReport:
    clustering_seconds: float
    codebook_setup_seconds: float

    @property
    def setup_seconds(self):
        return self.clustering_seconds + self.codebook_setup_seconds


class NeuralZipSetup:
    """Learn clustered raw/shared/packed-pair modes without compression probes.

    Exponent byte histograms are complete; other byte and exponent pair
    histograms use bounded samples. Size estimates approximate chunk fallback
    at group level, while the encoder decides the final mode for each chunk.
    """

    def __init__(
        self,
        *,
        min_saving_bytes=0,
        chunk_size=256 << 10,
        compression_threshold=0.95,
        threads=0,
        prior_count=1,
        exponent_prior_count=1,
        cluster_map=None,
        cluster_js_threshold=0.05,
        max_codebook_sample_bytes=8 << 20,
        packed_min_gain=0.02,
        enable_shared_codebooks=True,
        enable_packed_codebooks=True,
        direct_batch_bytes=128 << 20,
    ):
        self.min_saving_bytes = int(min_saving_bytes)
        self.chunk_size = int(chunk_size)
        self.compression_threshold = float(compression_threshold)
        self.threads = int(threads)
        self.prior_count = int(prior_count)
        self.exponent_prior_count = int(exponent_prior_count)
        self.cluster_map = {
            str(name): str(value) for name, value in (cluster_map or {}).items()
        }
        self.cluster_js_threshold = float(cluster_js_threshold)
        self.max_codebook_sample_bytes = int(max_codebook_sample_bytes)
        self.packed_min_gain = float(packed_min_gain)
        if not 0 <= self.packed_min_gain < 1:
            raise ValueError("packed_min_gain must be in [0, 1).")
        self.enable_shared_codebooks = bool(enable_shared_codebooks)
        self.enable_packed_codebooks = bool(enable_packed_codebooks)
        self.direct_batch_bytes = int(direct_batch_bytes)

        self.codebooks = ()
        self.tensor_codebooks = {}
        self.global_codebooks = {}
        self.setup_report = None

    def build(self, model):
        named_tensors = _model_tensors(model)
        clusters, clustering_seconds = time_call(self._cluster_tensors, named_tensors)
        (codebooks, tensor_modes), codebook_setup_seconds = time_call(
            self._build_codebooks, named_tensors, clusters
        )
        self.codebooks = tuple(codebooks)
        self.tensor_codebooks = {
            key: None if mode["kind"] == "raw" else mode["codebook_id"]
            for key, mode in tensor_modes.items()
        }
        self.global_codebooks = {}
        self.setup_report = NeuralZipSetupReport(
            clustering_seconds, codebook_setup_seconds
        )
        return self

    @property
    def setup_seconds(self):
        return None if self.setup_report is None else self.setup_report.setup_seconds

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle, protocol=5)
        return path

    @classmethod
    def load(cls, value):
        """Load an existing setup object or a trusted setup pickle file."""
        if isinstance(value, cls):
            return value
        with Path(value).open("rb") as handle:
            setup = pickle.load(handle)
        if not isinstance(setup, cls):
            raise TypeError("The file does not contain a NeuralZip setup.")
        return setup

    def _cluster_tensors(self, named_tensors):
        clusters = []
        computed = {}
        for name, tensor in _floating_tensors(named_tensors):
            if name in self.cluster_map:
                continue

            distribution = exponent_distribution(tensor)
            weight = int(tensor.numel())
            best_index = None
            best_distance = float("inf")
            for index, cluster in enumerate(clusters):
                distance = jensen_shannon_distance(distribution, cluster["centroid"])
                if distance < best_distance:
                    best_index = index
                    best_distance = distance

            if best_index is None or best_distance > self.cluster_js_threshold:
                cluster_id = f"js_{len(clusters):03d}"
                clusters.append(
                    {
                        "cluster_id": cluster_id,
                        "centroid": distribution,
                        "weight": weight,
                    }
                )
                computed[name] = cluster_id
                continue

            cluster = clusters[best_index]
            total_weight = cluster["weight"] + weight
            cluster["centroid"] = (
                cluster["centroid"] * cluster["weight"] + distribution * weight
            ) / total_weight
            cluster["weight"] = total_weight
            computed[name] = cluster["cluster_id"]
        return {**computed, **self.cluster_map}

    def _build_codebooks(self, named_tensors, clusters):
        named_tensors = list(named_tensors)
        groups = self._collect_histograms(named_tensors, clusters)
        codebooks, mode_by_key = [], {}
        for key, group in sorted(groups.items()):
            selected = self._choose_mode(key, group)
            mode_by_key[key] = self._finalize_mode(key, selected, codebooks)
        return codebooks, self._prepare_tensor_modes(
            named_tensors, mode_by_key, clusters
        )

    def _collect_histograms(self, named_tensors, clusters):
        groups = {}
        for name, tensor in _floating_tensors(named_tensors):
            original = tensor.detach().cpu().contiguous()
            config = DTYPES[original.dtype]
            reordered = config.reorder_bytes(_tensor_raw_bytes(original))
            raw_step = _stream_chunk_size(self.chunk_size, config.element_size)
            shared_step = min(raw_step, huff0.MAX_BLOCK_SIZE)
            packed_step = max(2, min(raw_step, 2 * huff0.MAX_BLOCK_SIZE) // 2 * 2)
            for stream_id in config.stream_ids:
                key = self._codebook_key(name, original, stream_id, clusters)
                is_exponent = stream_id == config.exponent_stream
                if key not in groups:
                    groups[key] = {
                        "is_exponent": is_exponent,
                        "bytes": np.zeros(256, dtype=np.uint64),
                        "pairs": np.zeros(65536, dtype=np.uint64)
                        if is_exponent and self.enable_packed_codebooks
                        else None,
                        "size": 0,
                        "pair_count": 0,
                        "tails": 0,
                        "raw_chunks": 0,
                        "shared_chunks": 0,
                        "packed_chunks": 0,
                        "sample_bytes": 0,
                        "sample_pairs": 0,
                    }
                group = groups[key]
                stream = reordered[stream_id :: config.element_size]
                size = len(stream)
                group["size"] += size
                group["pair_count"] += size // 2
                group["tails"] += size % 2
                for kind, step in (
                    ("raw", raw_step),
                    ("shared", shared_step),
                    ("packed", packed_step),
                ):
                    group[f"{kind}_chunks"] += (size + step - 1) // step
                if is_exponent:
                    group["bytes"] += self._counts(stream, 256, np.uint8)
                    if not self.enable_packed_codebooks:
                        continue
                remaining = self.max_codebook_sample_bytes - group["sample_bytes"]
                for start, end in self._sample_ranges(size, remaining):
                    part = memoryview(stream)[start:end]
                    group["sample_bytes"] += len(part)
                    if not is_exponent:
                        group["bytes"] += self._counts(part, 256, np.uint8)
                    elif len(part) >= 2:
                        paired = part[: len(part) // 2 * 2]
                        group["pairs"] += self._counts(paired, 65536, "<u2")
                        group["sample_pairs"] += len(paired) // 2
        return groups

    @staticmethod
    def _counts(data, alphabet_size, dtype):
        return np.bincount(
            np.frombuffer(data, dtype=dtype), minlength=alphabet_size
        ).astype(np.uint64)

    @staticmethod
    def _sample_ranges(size, budget):
        """Even starts preserve pair alignment within tensors and packed chunks.

        Packed2 chunks have even length, so no aligned pair crosses a chunk
        boundary. Each segment is counted separately; segments are never joined.
        """
        budget = min(size, max(0, int(budget)))
        if not budget:
            return ()
        if size <= budget:
            return ((0, size),)
        segments = min(8, max(1, budget // 2))
        width = max(1, budget // segments)
        if width > 1:
            width -= width % 2
        starts = np.linspace(0, size - width, segments, dtype=np.int64)
        return tuple(
            (int(start) // 2 * 2, int(start) // 2 * 2 + width) for start in starts
        )

    def _table(self, counts, *, is_exponent, packed=False, vocab=b""):
        if not int(counts.sum()):
            return None
        try:
            codebook = huff0.build_codebook_from_counts(
                counts,
                prior_count=self.exponent_prior_count
                if is_exponent
                else self.prior_count,
                max_bits=12 if is_exponent and not packed else 11,
            )
        except (RuntimeError, ValueError):
            return None
        return {
            "kind": "packed_huff0" if packed else "shared_huff0",
            "ngram_n": 2 if packed else 1,
            "vocab": vocab,
            "codebook": codebook,
        }

    @staticmethod
    def _huffman_bytes(mode, counts, scale, chunks):
        lengths, _, _ = huff0.export_code_table(mode["codebook"].codebook, max_bits=12)
        lengths = np.frombuffer(lengths, dtype=np.uint8).astype(np.uint64)
        if np.any((counts > 0) & (lengths == 0)):
            return math.inf
        bits = int(np.dot(counts, lengths))
        return math.ceil(bits * scale / 8) + 10 * chunks

    def _choose_mode(self, key, group):
        total = group["size"]
        raw_cost = total + RAW_CHUNK_METADATA_BYTES * group["raw_chunks"]
        selected, best_cost = {"kind": "raw"}, raw_cost
        if self.enable_shared_codebooks:
            shared = self._table(group["bytes"], is_exponent=group["is_exponent"])
            if shared is not None:
                chunks = group["shared_chunks"]
                scale = (
                    1 if group["is_exponent"] else total / max(1, group["sample_bytes"])
                )
                body = self._huffman_bytes(shared, group["bytes"], scale, chunks)
                compressed = body + SHARED_CHUNK_METADATA_BYTES * chunks
                viable = (
                    body + self.min_saving_bytes * chunks < total
                    and compressed < self.compression_threshold * total
                )
                shared_cost = self._candidate_payload(key, shared).metadata_bytes + (
                    compressed if viable else total + RAW_CHUNK_METADATA_BYTES * chunks
                )
                if shared_cost < best_cost:
                    selected, best_cost = shared, shared_cost

        if (
            self.enable_packed_codebooks
            and group["is_exponent"]
            and group["sample_pairs"]
        ):
            pairs = group["pairs"]
            observed = np.flatnonzero(pairs)
            order = np.argsort(-pairs[observed].astype(np.int64), kind="stable")
            vocabulary = np.sort(observed[order[:254]])
            token_counts = np.zeros(256, dtype=np.uint64)
            token_counts[: len(vocabulary)] = pairs[vocabulary]
            token_counts[255] = pairs.sum() - pairs[vocabulary].sum()
            packed = self._table(
                token_counts,
                is_exponent=True,
                packed=True,
                vocab=vocabulary.astype(_symbol_dtype(2)).tobytes(),
            )
            if packed is not None:
                chunks = group["packed_chunks"]
                scale = group["pair_count"] / group["sample_pairs"]
                body = self._huffman_bytes(packed, token_counts, scale, chunks)
                escapes = math.ceil(2 * int(token_counts[255]) * scale)
                compressed = (
                    body
                    + escapes
                    + group["tails"]
                    + PACKED_CHUNK_METADATA_BYTES * chunks
                )
                viable = (
                    body + self.min_saving_bytes * chunks < group["pair_count"]
                    and compressed < self.compression_threshold * total
                )
                packed_cost = self._candidate_payload(key, packed).metadata_bytes + (
                    compressed if viable else total + RAW_CHUNK_METADATA_BYTES * chunks
                )
                if packed_cost < best_cost * (1 - self.packed_min_gain):
                    selected, best_cost = packed, packed_cost

        return selected

    def _finalize_mode(self, key, selected, codebooks):
        if selected["kind"] == "raw":
            return selected
        codebook_id = len(codebooks)
        selected["codebook_id"] = codebook_id
        codebooks.append(
            self._candidate_payload(key, selected, codebook_id=codebook_id)
        )
        return selected

    def _candidate_payload(self, key, mode, codebook_id=-1):
        _, dtype, stream_id = key
        return CodebookPayload(
            codebook_id=int(codebook_id),
            stream_id=stream_id,
            dtype=dtype,
            mode=mode["kind"],
            codebook=mode["codebook"].codebook,
            ngram_n=int(mode.get("ngram_n", 1)),
            vocab=bytes(mode.get("vocab", b"")),
        )

    def _prepare_tensor_modes(self, named_tensors, mode_by_key, clusters):
        tensor_modes = {}
        for name, tensor in _floating_tensors(named_tensors):
            config = DTYPES[tensor.dtype]
            for stream_id in config.stream_ids:
                selected = mode_by_key[
                    self._codebook_key(name, tensor, stream_id, clusters)
                ]
                tensor_modes[self._transfer_plan_key(name, tensor, stream_id)] = (
                    selected
                )
        return tensor_modes

    @staticmethod
    def _codebook_key(name, tensor, stream_id, clusters):
        return clusters.get(name), DTYPES[tensor.dtype].name, int(stream_id)

    @staticmethod
    def _transfer_plan_key(name, tensor, stream_id):
        return str(name), DTYPES[tensor.dtype].name, int(stream_id)
