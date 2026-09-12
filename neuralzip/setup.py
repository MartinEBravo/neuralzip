import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from neuralzip import huff0
from neuralzip.payloads import (
    PACKED_CHUNK_METADATA_BYTES,
    RAW_CHUNK_METADATA_BYTES,
    SHARED_CHUNK_METADATA_BYTES,
    CodebookPayload,
)
from neuralzip.utils import (
    DTYPES,
    _byte_group_reordered_bytes,
    _floating_tensors,
    _model_tensors,
    estimate_mode,
    exponent_distribution,
    jensen_shannon_distance,
    select_fastest_within_ratio,
    select_packed_time_budget,
    time_call,
    tokenize_ngrams,
    topk_vocab,
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
        ngram_values=(2, 3, 4),
        top_k=254,
        packed_source_budget_ratio=0.30,
        packed_time_budget_ratio=1.0,
        speed_ratio_tolerance=0.03,
        enable_shared_codebooks=True,
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
        self.ngram_values = tuple(sorted({int(value) for value in ngram_values}))
        self.top_k = int(top_k)
        self.packed_source_budget_ratio = float(packed_source_budget_ratio)
        self.packed_time_budget_ratio = float(packed_time_budget_ratio)
        self.speed_ratio_tolerance = float(speed_ratio_tolerance)
        self.enable_shared_codebooks = bool(enable_shared_codebooks)
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
        sample_parts, total_sizes, stream_info, exponent_counts, chunk_counts = (
            self._collect_codebook_samples(named_tensors, clusters)
        )

        evaluations = []
        for key in sorted(sample_parts):
            element_size, is_exponent = stream_info[key]
            parts = tuple(sample_parts[key])
            evaluations.append(
                self._evaluate_modes(
                    key,
                    b"".join(parts),
                    total_size=total_sizes[key],
                    element_size=element_size,
                    is_exponent=is_exponent,
                    sample_parts=parts,
                    symbol_counts=exponent_counts.get(key),
                    chunk_counts=chunk_counts[key],
                )
            )

        codebooks, mode_by_key = self._select_modes(evaluations)
        tensor_modes = self._prepare_tensor_modes(named_tensors, mode_by_key, clusters)
        return codebooks, tensor_modes

    def _collect_codebook_samples(self, named_tensors, clusters):
        """Collect bounded timing probes and complete exponent histograms."""
        sample_parts = {}
        sample_sizes = {}
        total_sizes = {}
        stream_info = {}
        exponent_counts = {}
        chunk_counts = {}
        for name, tensor in _floating_tensors(named_tensors):
            original = tensor.detach().cpu().contiguous()
            config = DTYPES[original.dtype]
            reordered = _byte_group_reordered_bytes(original, config)
            element_size = int(config["element_size"])
            exponent_stream = element_size - 1 if config["bit_reorder"] else -1
            for stream_id in range(element_size):
                is_exponent = stream_id == exponent_stream
                stream = reordered[stream_id::element_size]
                key = self._codebook_key(name, original, stream_id, clusters)
                total_sizes[key] = total_sizes.get(key, 0) + len(stream)
                stream_info[key] = (element_size, is_exponent)
                raw_step = self._stream_chunk_size(element_size)
                shared_step = min(raw_step, huff0.MAX_BLOCK_SIZE)
                raw_chunks, shared_chunks = chunk_counts.get(key, (0, 0))
                chunk_counts[key] = (
                    raw_chunks + (len(stream) + raw_step - 1) // raw_step,
                    shared_chunks + (len(stream) + shared_step - 1) // shared_step,
                )
                if is_exponent:
                    counts = np.bincount(
                        np.frombuffer(stream, dtype=np.uint8), minlength=256
                    ).astype(np.uint64, copy=False)
                    if key in exponent_counts:
                        exponent_counts[key] += counts
                    else:
                        exponent_counts[key] = counts
                remaining = self.max_codebook_sample_bytes - sample_sizes.get(key, 0)
                if remaining <= 0:
                    continue
                part = self._bounded_stream_sample(stream, remaining)
                if not part:
                    continue
                sample_parts.setdefault(key, []).append(bytes(part))
                sample_sizes[key] = sample_sizes.get(key, 0) + len(part)
        return sample_parts, total_sizes, stream_info, exponent_counts, chunk_counts

    @staticmethod
    def _bounded_stream_sample(stream, max_bytes, segments=8):
        stream = bytes(stream)
        max_bytes = min(len(stream), int(max_bytes))
        if max_bytes <= 0:
            return b""
        if len(stream) <= max_bytes:
            return stream
        segment_count = min(int(segments), max_bytes)
        segment_size = max(1, max_bytes // segment_count)
        last_start = len(stream) - segment_size
        starts = np.linspace(0, last_start, segment_count, dtype=np.int64)
        sample = b"".join(
            stream[int(start) : int(start) + segment_size] for start in starts
        )
        return sample[:max_bytes]

    def _evaluate_modes(
        self,
        key,
        sample,
        *,
        total_size,
        element_size,
        is_exponent,
        sample_parts,
        symbol_counts=None,
        chunk_counts,
    ):
        raw_mode = {"kind": "raw", "fixed_cost": 0}
        self._score_mode(raw_mode, element_size, sample_parts=sample_parts)
        raw_mode["full_cost"] = (
            int(total_size) + int(chunk_counts[0]) * RAW_CHUNK_METADATA_BYTES
        )
        prior_count = self.exponent_prior_count if is_exponent else self.prior_count

        shared = None
        if self.enable_shared_codebooks:
            shared = self._build_candidate(
                sample,
                kind="shared_huff0",
                prior_count=prior_count,
                symbol_counts=symbol_counts,
                max_bits=12 if is_exponent else 11,
            )
        if shared is not None:
            payload = self._candidate_payload(key, shared)
            if symbol_counts is not None:
                shared["fixed_cost"] = payload.metadata_bytes
                _, shared["sample_seconds"] = self._measure_direct_mode(
                    shared, element_size, tuple(sample_parts)
                )
                shared["full_cost"] = self._shared_histogram_cost(
                    shared,
                    symbol_counts,
                    total_size=total_size,
                    shared_chunk_count=chunk_counts[1],
                    codebook_payload=payload,
                )
            else:
                self._score_mode(shared, element_size, payload, sample_parts)

        base_candidates = [raw_mode]
        if shared is not None:
            base_candidates.append(shared)
        scale = total_size / len(sample)
        base, base_cost, base_seconds = select_fastest_within_ratio(
            (estimate_mode(candidate, scale) for candidate in base_candidates),
            ratio_tolerance=self.speed_ratio_tolerance,
        )

        packed_candidates = []
        if is_exponent:
            for n in self.ngram_values:
                packed = self._build_candidate(
                    sample, kind="packed_huff0", prior_count=prior_count, ngram_n=n
                )
                if packed is None:
                    continue
                payload = self._candidate_payload(key, packed)
                self._score_mode(packed, element_size, payload, sample_parts)
                packed_candidates.append(packed)

        packed_scores = (
            estimate_mode(candidate, scale) for candidate in packed_candidates
        )
        improving = [score for score in packed_scores if score[1] < base_cost]
        packed = None
        if improving:
            packed, packed_cost, packed_seconds = select_fastest_within_ratio(
                improving,
                ratio_tolerance=self.speed_ratio_tolerance,
            )
        opportunity = None
        if packed is not None:
            saved_bytes = base_cost - packed_cost
            opportunity = {
                "mode": packed,
                "saved_bytes": saved_bytes,
                "source_bytes": int(total_size),
                "savings_per_source_byte": saved_bytes / total_size,
                "extra_seconds": max(0.0, packed_seconds - base_seconds),
            }
        return {
            "key": key,
            "base": base,
            "base_seconds": base_seconds,
            "total_size": int(total_size),
            "is_exponent": bool(is_exponent),
            "opportunity": opportunity,
        }

    def _build_candidate(
        self,
        sample,
        *,
        kind,
        prior_count,
        ngram_n=1,
        symbol_counts=None,
        max_bits=11,
    ):
        if kind == "shared_huff0":
            tokens = sample
            vocab = b""
        else:
            vocab = topk_vocab(sample, ngram_n, self.top_k)
            if not vocab:
                return None
            tokens, _, _ = tokenize_ngrams(sample, ngram_n, vocab)
        if not tokens:
            return None
        counts = symbol_counts
        if counts is None or kind != "shared_huff0":
            counts = np.bincount(np.frombuffer(tokens, dtype=np.uint8), minlength=256)
        try:
            codebook = huff0.build_codebook_from_counts(
                counts, prior_count=prior_count, max_bits=max_bits
            )
        except (RuntimeError, ValueError):
            return None
        return {
            "kind": kind,
            "ngram_n": int(ngram_n),
            "vocab": vocab,
            "codebook": codebook,
        }

    def _score_mode(self, mode, element_size, codebook_payload=None, sample_parts=()):
        sample_cost, sample_seconds = self._measure_direct_mode(
            mode, element_size, tuple(sample_parts)
        )
        mode["sample_cost"] = sample_cost
        mode["fixed_cost"] = (
            codebook_payload.metadata_bytes if codebook_payload is not None else 0
        )
        mode["sample_cost"] += mode["fixed_cost"]
        mode["sample_seconds"] = sample_seconds
        return mode

    def _measure_direct_mode(self, mode, element_size, sample_parts):
        """Measure a mode with the same native primitive used by ``encode``."""
        element_size = int(element_size)
        raw_mode = {"kind": "raw"}
        stream_modes = (mode,) + (raw_mode,) * (element_size - 1)
        tensor_specs = []
        for part in sample_parts:
            part = bytes(part)
            if not part:
                continue
            interleaved = bytearray(len(part) * element_size)
            interleaved[0::element_size] = part
            tensor_specs.append(
                (
                    interleaved,
                    element_size,
                    0,
                    self._stream_chunk_size(element_size),
                    stream_modes,
                )
            )
        if not tensor_specs:
            return 0, 0.0

        def encode_once():
            return huff0.compress_mixed_streams_many(
                tuple(tensor_specs),
                min_saving_bytes=self.min_saving_bytes,
                threads=self.threads,
                compression_threshold=self.compression_threshold,
            )

        native_batch, _ = time_call(encode_once)
        seconds = min(time_call(encode_once)[1] for _ in range(3))
        metadata_by_mode = {
            "raw": RAW_CHUNK_METADATA_BYTES,
            "shared_huff0": SHARED_CHUNK_METADATA_BYTES,
            "packed_huff0": PACKED_CHUNK_METADATA_BYTES,
        }
        encoded_bytes = 0
        for native_groups in native_batch:
            for final_kind, body, _, escapes, tail in native_groups[0]:
                encoded_bytes += (
                    metadata_by_mode[str(final_kind)]
                    + len(body)
                    + len(escapes)
                    + len(tail)
                )
        return encoded_bytes, seconds

    def _shared_histogram_cost(
        self,
        mode,
        symbol_counts,
        *,
        total_size,
        shared_chunk_count,
        codebook_payload,
    ):
        """Bound shared size from exact counts plus Huff0's 10-byte 4X overhead."""
        code_lengths, _, _ = huff0.export_code_table(
            mode["codebook"].codebook, max_bits=12
        )
        symbol_bits = sum(
            int(count) * int(length)
            for count, length in zip(symbol_counts, code_lengths, strict=True)
        )
        symbol_bytes = (symbol_bits + 7) // 8
        shared_chunks = (
            symbol_bytes
            + 10 * int(shared_chunk_count)
            + int(shared_chunk_count) * SHARED_CHUNK_METADATA_BYTES
        )
        fallback_chunks = (
            int(total_size) + int(shared_chunk_count) * RAW_CHUNK_METADATA_BYTES
        )
        return int(codebook_payload.metadata_bytes) + min(
            shared_chunks, fallback_chunks
        )

    def _select_modes(self, evaluations):
        exponent_bytes = sum(
            row["total_size"] for row in evaluations if row["is_exponent"]
        )
        source_budget = int(exponent_bytes * self.packed_source_budget_ratio)
        time_selected = select_packed_time_budget(
            evaluations, packed_time_budget_ratio=self.packed_time_budget_ratio
        )
        opportunities = [
            row
            for row in evaluations
            if row["opportunity"] is not None and row["key"] in time_selected
        ]
        opportunities.sort(
            key=lambda row: (
                -row["opportunity"]["savings_per_source_byte"],
                -row["opportunity"]["saved_bytes"],
                row["key"],
            )
        )
        packed_used = 0
        selected_packed = set()
        for row in opportunities:
            source_bytes = row["opportunity"]["source_bytes"]
            if packed_used + source_bytes > source_budget:
                continue
            selected_packed.add(row["key"])
            packed_used += source_bytes

        codebooks = []
        mode_by_key = {}
        for row in evaluations:
            selected = (
                row["opportunity"]["mode"]
                if row["key"] in selected_packed
                else row["base"]
            )
            mode_by_key[row["key"]] = self._finalize_mode(
                row["key"], selected, codebooks
            )
        return codebooks, mode_by_key

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
            for stream_id in range(int(config["element_size"])):
                selected = mode_by_key[
                    self._codebook_key(name, tensor, stream_id, clusters)
                ]
                tensor_modes[self._transfer_plan_key(name, tensor, stream_id)] = (
                    selected
                )
        return tensor_modes

    @staticmethod
    def _codebook_key(name, tensor, stream_id, clusters):
        return clusters.get(name), DTYPES[tensor.dtype]["name"], int(stream_id)

    @staticmethod
    def _transfer_plan_key(name, tensor, stream_id):
        return str(name), DTYPES[tensor.dtype]["name"], int(stream_id)

    def _stream_chunk_size(self, element_size):
        return max(1, self.chunk_size // int(element_size))
