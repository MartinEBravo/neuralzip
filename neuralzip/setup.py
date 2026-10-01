import pickle
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import numpy as np

from neuralzip import huff0
from neuralzip.dtypes import DTYPES
from neuralzip.payloads import CHUNK_METADATA_BYTES, CodebookPayload
from neuralzip.utils import (
    _floating_tensors,
    _model_tensors,
    _stream_chunk_size,
    _tensor_raw_bytes,
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
    def __init__(
        self, *, min_saving_bytes=0, chunk_size=256 << 10,
        compression_threshold=0.95, prior_count=1, cluster_map=None,
        cluster_js_threshold=0.05, max_codebook_sample_bytes=8 << 20,
        enable_shared_codebooks=True, enable_packed_codebooks=True,
    ):
        if chunk_size < 4 or max_codebook_sample_bytes < 1 or prior_count < 1:
            raise ValueError("Chunk size >= 4, sample budget >= 1, and prior >= 1 are required.")
        self.min_saving_bytes = int(min_saving_bytes)
        self.chunk_size = int(chunk_size)
        self.compression_threshold = float(compression_threshold)
        self.prior_count = int(prior_count)
        self.cluster_map = dict(cluster_map or {})
        self.cluster_js_threshold = float(cluster_js_threshold)
        self.max_codebook_sample_bytes = int(max_codebook_sample_bytes)
        self.enable_shared_codebooks = bool(enable_shared_codebooks)
        self.enable_packed_codebooks = bool(enable_packed_codebooks)
        self.codebooks = ()
        self.chunk_codebooks = {}
        self.tensor_codebooks = {}
        self.global_codebooks = {}
        self.packing_report = ()
        self.setup_report = NeuralZipSetupReport(0.0, 0.0)

    @property
    def setup_seconds(self):
        return self.setup_report.setup_seconds

    def build(self, model):
        (groups, assignments, parents), clustering_seconds = time_call(
            self._cluster_chunks, _model_tensors(model)
        )
        (books, chunks, tensors, report), codebook_seconds = time_call(
            self._build_codebooks, groups, assignments, parents
        )
        self.codebooks = tuple(books)
        self.chunk_codebooks = chunks
        self.tensor_codebooks = tensors
        self.global_codebooks = {}
        self.packing_report = tuple(report)
        self.setup_report = NeuralZipSetupReport(clustering_seconds, codebook_seconds)
        return self

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle, protocol=5)
        return path

    @classmethod
    def load(cls, value):
        if not isinstance(value, cls):
            with Path(value).open("rb") as handle:
                value = pickle.load(handle)
        if not isinstance(value, cls) or not hasattr(value, "chunk_codebooks"):
            raise TypeError("Expected a current chunk-level NeuralZip setup.")
        return value

    def _cluster_chunks(self, named_tensors):
        groups, assignments, parents = [], {}, {}
        for name, tensor in _floating_tensors(named_tensors):
            spec = DTYPES[tensor.dtype]
            raw = _tensor_raw_bytes(tensor)
            stream_id = spec.exponent_stream
            if stream_id is None:
                stream_id = spec.element_size - 1
            stream = np.frombuffer(spec.reorder_bytes(raw), dtype=np.uint8)[
                stream_id::spec.element_size
            ]
            exponents = spec.extract_exponents(raw)
            step = _stream_chunk_size(self.chunk_size, spec.element_size)
            key = (str(name), spec.name, stream_id)
            parents[key] = np.bincount(stream, minlength=256).astype(np.uint64)
            ids = []
            for index, start in enumerate(range(0, len(stream), step)):
                part = stream[start:start + step]
                counts = np.bincount(exponents[start:start + step], minlength=256)
                distribution = counts / counts.sum()
                forced = self.cluster_map.get((name, index), self.cluster_map.get(name))
                compatible = [
                    i for i, group in enumerate(groups)
                    if group["dtype"] == spec.name and group["forced"] == forced
                ] if self.enable_shared_codebooks else []
                # ponytail: linear centroid scan; batch distances if setup profiling warrants it.
                distance, nearest = min(
                    ((jensen_shannon_distance(distribution, groups[i]["centroid"]), i) for i in compatible),
                    default=(float("inf"), None),
                )
                if nearest is None or (forced is None and distance > self.cluster_js_threshold):
                    nearest = len(groups)
                    groups.append({
                        "dtype": spec.name, "stream": stream_id, "forced": forced,
                        "centroid": distribution, "weight": 0, "samples": [], "sampled": 0,
                    })
                group = groups[nearest]
                weight = len(part)
                total = group["weight"] + weight
                group["centroid"] = (
                    group["centroid"] * group["weight"] + distribution * weight
                ) / total
                group["weight"] = total
                budget = self.max_codebook_sample_bytes - group["sampled"]
                if budget > 0:
                    sample = part[:budget].tobytes()
                    group["samples"].append(sample)
                    group["sampled"] += len(sample)
                ids.append(nearest)
            assignments[(*key, tensor.numel())] = tuple(ids)
        return groups, assignments, parents

    def _build_codebooks(self, groups, assignments, parents):
        books, cache, report = [], {}, []
        specs = {spec.name: spec for spec in DTYPES.values()}

        def add(book, dtype, stream, vocab=b""):
            if book is None or not (self.enable_shared_codebooks or self.enable_packed_codebooks):
                return None
            mode = "mixed_huffman" if vocab else "shared_huff0"
            key = dtype, stream, mode, book.codebook, vocab
            if not self.enable_shared_codebooks:
                key = (*key, len(books))
            if key not in cache:
                cache[key] = len(books)
                books.append(CodebookPayload(len(books), stream, dtype, mode, book.codebook, vocab=vocab))
            return cache[key]

        cluster_ids = []
        for group in groups:
            book, vocab, scores = self._prepare_codebook(group, specs[group["dtype"]])
            if scores is not None:
                report.append({"cluster": len(cluster_ids), **scores})
            cluster_ids.append(add(book, group["dtype"], group["stream"], vocab))
        chunks = {key: tuple(cluster_ids[i] for i in ids) for key, ids in assignments.items()}
        # Pool original counts once: shape-independent transfer and the byte-only CUDA ABI.
        tensors = {key: add(self._singleton_codebook(counts), key[1], key[2]) for key, counts in parents.items()}
        return books, chunks, tensors, report

    def _singleton_codebook(self, counts):
        if not (self.enable_shared_codebooks or self.enable_packed_codebooks):
            return None
        try:
            return huff0.build_codebook_from_counts(counts, prior_count=self.prior_count, max_bits=12)
        except RuntimeError:
            return None  # Huff0 cannot serialize some uniform alphabets; store raw.

    def _prepare_codebook(self, group, spec):
        if not (self.enable_shared_codebooks or self.enable_packed_codebooks):
            return None, b"", None
        samples = group["samples"]
        singleton = self._singleton_codebook(huff0.count_mixed_tokens(samples, b""))
        if not self.enable_packed_codebooks or spec.exponent_stream is None:
            return singleton, b"", None
        pairs = np.zeros(65536, dtype=np.int64)
        for sample in samples:
            values = np.frombuffer(sample, dtype=np.uint8).astype(np.uint16)
            pairs += np.bincount(values[:-1] | (values[1:] << 8), minlength=65536)
        observed = np.flatnonzero(pairs)
        order = np.argsort(-pairs[observed], kind="stable")
        selected = np.sort(observed[order[:254]]).astype("<u2")
        candidate_pairs = len(selected)
        while len(selected):
            vocab = selected.tobytes()
            mixed = huff0.build_mixed_codebook(
                huff0.count_mixed_tokens(samples, vocab), vocab, prior_count=self.prior_count,
            )
            lengths = np.frombuffer(mixed.codebook[4:], dtype=np.uint8).astype(np.uint16)
            keep = lengths[256:] <= lengths[selected & 255] + lengths[selected >> 8]
            if keep.all():
                scores = self._compare_codebooks(group, spec, singleton, mixed, vocab)
                scores.update(candidate_pairs=candidate_pairs, kept_pairs=len(selected))
                return (mixed, vocab, scores) if scores["mode"] == "mixed_huffman" else (singleton, b"", scores)
            selected = selected[keep]
        return singleton, b"", None

    def _compare_codebooks(self, group, spec, singleton, mixed, vocab):
        # Spread four source chunks across the sample, preserving their boundaries.
        available = group["samples"]
        samples = [available[i] for i in np.linspace(0, len(available) - 1, min(4, len(available)), dtype=int)]
        data = []
        for sample in samples:
            lanes = np.zeros(len(sample) * spec.element_size, dtype=np.uint8)
            lanes[group["stream"]::spec.element_size] = np.frombuffer(sample, dtype=np.uint8)
            raw = spec.reorder_bytes(lanes.tobytes(), restore=True)
            data.append(np.frombuffer(raw, dtype=np.uint8))
        options = (singleton, mixed)
        modes = ("shared_huff0" if singleton is not None else "raw", "mixed_huffman")
        specs = [tuple((raw, spec, group["stream"],
                        _stream_chunk_size(self.chunk_size, spec.element_size),
                        ({"kind": mode, "codebook": book, "codebook_id": 0} if book is not None
                         else {"kind": "raw"},))
                       for raw in data) for book, mode in zip(options, modes)]
        timings = [[], []]
        sizes = [0, 0]
        used = [False, False]
        for repeat in range(6):
            for index in (0, 1) if repeat % 2 == 0 else (1, 0):
                (output, groups), seconds = time_call(
                    huff0.compress_direct_storage, specs[index], b"", threads=1,
                    min_saving_bytes=self.min_saving_bytes, compression_threshold=self.compression_threshold,
                )
                if repeat:
                    timings[index].append(seconds)
                sizes[index] = sum(chunk[4] + CHUNK_METADATA_BYTES[chunk[0]]
                                   for streams in groups for chunk in streams[group["stream"]])
                used[index] = any(chunk[0] != "raw" for streams in groups for chunk in streams[group["stream"]])
                del output
        sampled = sum(map(len, samples))
        for index, (book, mode, vocabulary) in enumerate(zip(options, modes, (b"", vocab))):
            overhead = 0 if book is None or not used[index] else CodebookPayload(
                0, group["stream"], spec.name, mode, book.codebook, vocab=vocabulary,
            ).metadata_bytes
            sizes[index] = sizes[index] * group["weight"] / sampled + overhead
        times = tuple(map(median, timings))
        # A 5% timing margin avoids choosing packing on a near-parity sample.
        keep = sizes[1] < sizes[0] and times[1] < 0.95 * times[0]
        return {"mode": "mixed_huffman" if keep else modes[0],
                "singleton_bytes_estimate": sizes[0], "mixed_bytes_estimate": sizes[1],
                "singleton_seconds": times[0], "mixed_seconds": times[1]}
