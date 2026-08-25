import re
import time

import numpy as np
import torch

from . import huff0
from .dtypes import dtype_name
from .packed_exponents import (
    restore_2d_stream,
    topk_vocab,
    tokenize_ngrams,
    transpose_2d_stream,
)
from .mode_selection import (
    select_fastest_within_ratio,
    select_packed_time_budget,
)
from .metrics import jensen_shannon_distance
from .payloads import (
    ByteGroupPayload,
    CodebookPayload,
    PACKED_CHUNK_METADATA_BYTES,
    SHARED_CHUNK_METADATA_BYTES,
    StreamChunk,
    TensorPayload,
)
from .utils import split_fields
from .zipnn_codec import (
    DTYPE_BY_NAME,
    ZipNNCodec,
    _tensor_from_raw_bytes,
    _zipnn_dtype_config,
    _zipnn_reordered_bytes,
    _zipnn_restored_bytes,
)


def _layer_signature(name):
    return ".".join(re.sub(r"\d+", "#", part) for part in str(name).split("."))


def _exponent_distribution(tensor):
    _, exponent, _ = split_fields(tensor.detach().cpu().contiguous())
    stream = exponent.reshape(-1).to(torch.int64).numpy()
    counts = np.bincount(stream, minlength=256).astype(np.float64)
    total = counts.sum()
    return counts / total if total else counts


class NeuralZipCodec(ZipNNCodec):
    """JS-clustered Huffman with deterministic packed-exponent selection."""

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
        clustering="js-distance",
        cluster_js_threshold=0.05,
        max_codebook_sample_bytes=8 << 20,
        ngram_values=(2, 3, 4),
        top_k=254,
        exponent_layout="original",
        packed_source_budget_ratio=0.30,
        packed_time_budget_ratio=1.0,
        speed_ratio_tolerance=0.03,
        enable_shared_codebooks=True,
        direct_batch_bytes=128 << 20,
    ):
        super().__init__(
            min_saving_bytes=min_saving_bytes,
            chunk_size=chunk_size,
            compression_threshold=compression_threshold,
            threads=threads,
        )
        self.prior_count = int(prior_count)
        self.exponent_prior_count = int(exponent_prior_count)
        self.cluster_map = {
            str(name): str(value) for name, value in (cluster_map or {}).items()
        }
        self.clustering = str(clustering).lower().replace("_", "-")
        self.cluster_js_threshold = float(cluster_js_threshold)
        self.max_codebook_sample_bytes = int(max_codebook_sample_bytes)
        self.ngram_values = tuple(sorted({int(value) for value in ngram_values}))
        self.top_k = int(top_k)
        self.exponent_layout = str(exponent_layout)
        self.packed_source_budget_ratio = float(packed_source_budget_ratio)
        self.packed_time_budget_ratio = float(packed_time_budget_ratio)
        self.speed_ratio_tolerance = float(speed_ratio_tolerance)
        self.enable_shared_codebooks = bool(enable_shared_codebooks)
        self.direct_batch_bytes = int(direct_batch_bytes)

        if self.clustering not in {
            "js-distance",
            "jensen-shannon",
            "jensen-shannon-distance",
        }:
            raise ValueError("NeuralZip clustering must use Jensen-Shannon distance.")
        if self.prior_count < 0 or self.exponent_prior_count < 0:
            raise ValueError("Huffman prior counts must be non-negative.")
        if self.cluster_js_threshold < 0:
            raise ValueError("cluster_js_threshold must be non-negative.")
        if self.max_codebook_sample_bytes <= 0:
            raise ValueError("max_codebook_sample_bytes must be positive.")
        if self.direct_batch_bytes <= 0:
            raise ValueError("direct_batch_bytes must be positive.")
        if any(value < 2 or value > 8 for value in self.ngram_values):
            raise ValueError("ngram_values must contain only values in [2, 8].")
        if not 1 <= self.top_k <= 254:
            raise ValueError("top_k must be in [1, 254].")
        if self.exponent_layout not in {"original", "transpose2d"}:
            raise ValueError("exponent_layout must be 'original' or 'transpose2d'.")
        if not 0 <= self.packed_source_budget_ratio <= 1:
            raise ValueError("packed_source_budget_ratio must be in [0, 1].")
        if self.packed_time_budget_ratio < 0:
            raise ValueError("packed_time_budget_ratio must be non-negative.")
        if self.speed_ratio_tolerance < 0:
            raise ValueError("speed_ratio_tolerance must be non-negative.")

        self.reset_compression()
        self.reset_decompression()

    def reset_compression(self):
        self._mode_by_key = {}
        self._codebook_payloads = []
        self._codebook_payload_by_id = {}
        self._computed_cluster_map = {}
        self._cluster_centroids = {}
        self._clusters_prepared = False
        self._next_codebook_id = 0
        self._selected_mode_counts = {}
        self._selection_stats = {}
        self._selection_diagnostics = ()
        self._mode_by_tensor_stream = {}
        self._transfer_mode_by_tensor_stream = {}
        self._transfer_cluster_by_name = {}
        self._transfer_stats = {}
        self._transfer_active = False

    def reset_decompression(self):
        self._decode_codebooks = {}
        self._codebook_payload_by_id = {}

    def export_codebooks(self):
        return tuple(self._codebook_payloads)

    def load_codebooks(self, codebooks):
        self._codebook_payload_by_id = {
            int(codebook.codebook_id): codebook for codebook in codebooks or ()
        }
        self._decode_codebooks = {}
        for codebook in codebooks or ():
            codebook_id = int(codebook.codebook_id)
            if codebook.codec != "huff0":
                raise ValueError(f"Unsupported codebook codec {codebook.codec!r}.")
            self._decode_codebooks[codebook_id] = huff0.build_decodebook(
                codebook.codebook
            )

    def codebook_metadata_bytes(self):
        return sum(codebook.metadata_bytes for codebook in self._codebook_payloads)

    @property
    def cluster_count(self):
        return len(set(self.cluster_map.values()) | set(self._computed_cluster_map.values()))

    @property
    def selected_modes(self):
        return dict(self._selected_mode_counts)

    @property
    def selection_stats(self):
        return dict(self._selection_stats)

    @property
    def selection_diagnostics(self):
        return tuple(self._selection_diagnostics)

    @property
    def transfer_stats(self):
        return dict(self._transfer_stats)

    @property
    def setup_memory_bytes(self):
        """Logical native bytes retained by selected shared/packed CTables."""
        seen = set()
        total = 0
        for mode in self._mode_by_key.values():
            codebook = mode.get("codebook")
            if codebook is None or id(codebook.handle) in seen:
                continue
            seen.add(id(codebook.handle))
            total += huff0.codebook_memory_bytes(codebook)
        return int(total)

    def cluster_id(self, name):
        if name in self._transfer_cluster_by_name:
            return self._transfer_cluster_by_name[name]
        if name in self.cluster_map:
            return self.cluster_map[name]
        if name in self._computed_cluster_map:
            return self._computed_cluster_map[name]
        return _layer_signature(name)

    def _prepare_clusters(self, named_tensors):
        named_tensors = list(named_tensors)
        clusters = []
        computed = {}
        distributions = {}
        for name, tensor in sorted(named_tensors, key=lambda item: item[0]):
            if (
                not torch.is_tensor(tensor)
                or not tensor.is_floating_point()
                or tensor.numel() == 0
            ):
                continue

            distribution = _exponent_distribution(tensor)
            weight = int(tensor.numel())
            distributions[name] = (distribution, weight)
            if name in self.cluster_map:
                continue
            best_index = None
            best_distance = float("inf")
            for index, cluster in enumerate(clusters):
                distance = jensen_shannon_distance(
                    distribution,
                    cluster["centroid"],
                )
                if distance < best_distance:
                    best_index = index
                    best_distance = distance

            if best_index is None or best_distance > self.cluster_js_threshold:
                cluster_id = f"js_{len(clusters):03d}"
                clusters.append(
                    {"cluster_id": cluster_id, "centroid": distribution, "weight": weight}
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
        self._computed_cluster_map = computed
        weighted_centroids = {}
        cluster_weights = {}
        for name, (distribution, weight) in distributions.items():
            cluster_id = self.cluster_id(name)
            weighted_centroids[cluster_id] = weighted_centroids.get(
                cluster_id,
                np.zeros(256, dtype=np.float64),
            ) + distribution * weight
            cluster_weights[cluster_id] = cluster_weights.get(cluster_id, 0) + weight
        self._cluster_centroids = {
            cluster_id: weighted / cluster_weights[cluster_id]
            for cluster_id, weighted in weighted_centroids.items()
        }

    def _codebook_key(self, name, tensor, stream_id):
        return self.cluster_id(name), dtype_name(tensor.dtype), int(stream_id)

    @staticmethod
    def _is_exponent_stream(config, stream_id):
        return bool(config["bit_reorder"]) and int(stream_id) == (
            int(config["element_size"]) - 1
        )

    def _apply_exponent_layout(self, tensor, stream, is_exponent):
        if not is_exponent or self.exponent_layout == "original":
            return bytes(stream), "original"
        return transpose_2d_stream(stream, tuple(int(dim) for dim in tensor.shape))

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

    def _source_chunk_size(self, element_size, mode):
        base = self._stream_chunk_size(element_size)
        if mode["kind"] == "packed_huff0":
            n = int(mode["ngram_n"])
            limit = min(base, huff0.max_block_size() * n)
            return max(n, (limit // n) * n)
        if mode["kind"] == "shared_huff0":
            return min(base, huff0.max_block_size())
        return base

    @staticmethod
    def _raw_chunk(chunk):
        return StreamChunk(mode="raw", symbol_count=len(chunk), payload=bytes(chunk))

    def _encode_chunk(self, chunk, mode):
        if mode["kind"] == "shared_huff0":
            max_payload_bytes = max(
                0,
                int(len(chunk) * self.compression_threshold)
                - SHARED_CHUNK_METADATA_BYTES,
            )
            encoder = (
                huff0.encode_with_codebook_4x
                if mode.get("frame_4x", False)
                else huff0.encode_with_codebook
            )
            encoded = encoder(
                chunk,
                mode["codebook"],
                min_saving_bytes=self.min_saving_bytes,
                max_output_bytes=max_payload_bytes,
            )
            if encoded is None:
                return None
            candidate = StreamChunk(
                mode="shared_huff0",
                symbol_count=len(chunk),
                payload=encoded.payload,
                codebook_id=mode.get("codebook_id"),
            )
        elif mode["kind"] == "packed_huff0":
            max_payload_bytes = max(
                0,
                int(len(chunk) * self.compression_threshold)
                - PACKED_CHUNK_METADATA_BYTES,
            )
            encoder = (
                huff0.encode_ngram_with_codebook_4x
                if mode.get("frame_4x", False)
                else huff0.encode_ngram_with_codebook
            )
            encoded = encoder(
                chunk,
                mode["codebook"],
                mode["vocab"],
                n=mode["ngram_n"],
                min_saving_bytes=self.min_saving_bytes,
                max_output_bytes=max_payload_bytes,
            )
            if encoded is None:
                return None
            candidate = StreamChunk(
                mode="packed_huff0",
                symbol_count=len(chunk),
                payload=encoded.payload,
                codebook_id=mode.get("codebook_id"),
                token_count=encoded.symbol_count,
                escape_payload=encoded.escape_bytes,
                tail_payload=encoded.tail,
                ngram_n=int(mode["ngram_n"]),
                ngram_hits=encoded.hits,
                ngram_escapes=encoded.misses,
            )
        else:
            raise ValueError(f"Cannot encode mode {mode['kind']!r}.")

        if candidate.compressed_bytes >= len(chunk) * self.compression_threshold:
            return None
        return candidate

    def _mode_cost(
        self,
        sample,
        mode,
        element_size,
        codebook_payload=None,
        sample_parts=None,
    ):
        parts = tuple(sample_parts) if sample_parts is not None else (sample,)
        if mode["kind"] == "local_huff0":
            total = 0
            for part in parts:
                group = super()._compress_stream(
                    part,
                    "sample",
                    chunk_size=self._stream_chunk_size(element_size),
                )
                total += sum(chunk.compressed_bytes for chunk in group.chunks)
            return total

        total = (
            codebook_payload.metadata_bytes if codebook_payload is not None else 0
        )
        step = self._source_chunk_size(element_size, mode)
        for part in parts:
            for start in range(0, len(part), step):
                chunk = part[start : start + step]
                encoded = (
                    None if mode["kind"] == "raw" else self._encode_chunk(chunk, mode)
                )
                if encoded is not None:
                    total += encoded.compressed_bytes
                elif mode["kind"] == "raw":
                    total += self._raw_chunk(chunk).compressed_bytes
                else:
                    # Runtime shared/packed rejection first tries the exact
                    # local Huff0 fallback and stores raw only if local coding
                    # also loses.  Candidate sizing must model that same path.
                    fallback = super()._compress_local_huff0_chunks((chunk,))[0]
                    total += fallback.compressed_bytes
        return total

    def _measure_direct_mode_seconds(self, mode, element_size, sample_parts):
        """Measure target-dependent work in the uncached direct encoder.

        The sampled logical byte lane is placed in stream zero of a synthetic
        tensor before the timer; the remaining lanes use raw mode and therefore
        contribute the same background work to every candidate.  The timed
        call is the model-wide mixed primitive itself, so shared/packed
        rejection, local-tree construction, the strict compression threshold,
        and final raw fallback are identical to the real direct encoder.
        """
        element_size = int(element_size)
        raw_mode = {"kind": "raw"}
        stream_modes = (mode,) + (raw_mode,) * (element_size - 1)
        tensor_specs = []
        for part in sample_parts:
            part = bytes(part)
            if not part:
                continue
            # Sampling produced a logical (already reordered) byte lane.  A
            # no-bit-reorder synthetic tensor preserves it exactly in stream
            # zero while letting the production primitive perform extraction,
            # chunking, encoding, and every fallback decision.
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
            return 0.0

        def encode_once():
            started = time.perf_counter()
            huff0.compress_zipnn_mixed_many(
                tuple(tensor_specs),
                min_saving_bytes=self.min_saving_bytes,
                threads=self.threads,
                compression_threshold=self.compression_threshold,
            )
            return time.perf_counter() - started

        # Warm code/data caches once, then use the best of three complete
        # measurements. The minimum estimates algorithmic work without letting
        # scheduler jitter change the frozen family plan.
        encode_once()
        return min(encode_once() for _ in range(3))

    def _score_mode(
        self,
        sample,
        mode,
        element_size,
        codebook_payload=None,
        sample_parts=None,
    ):
        mode["sample_cost"] = self._mode_cost(
            sample,
            mode,
            element_size,
            codebook_payload,
            sample_parts,
        )
        mode["fixed_cost"] = (
            codebook_payload.metadata_bytes if codebook_payload is not None else 0
        )
        parts = tuple(sample_parts) if sample_parts is not None else (sample,)
        mode["sample_seconds"] = self._measure_direct_mode_seconds(
            mode,
            element_size,
            parts,
        )
        return mode

    def _build_candidate(self, sample, *, kind, prior_count, ngram_n=1):
        if kind == "shared_huff0":
            tokens = sample
            vocab = b""
            escapes = b""
            tail = b""
            hits = 0
            misses = 0
        else:
            vocab = topk_vocab(sample, ngram_n, self.top_k)
            if not vocab:
                return None
            tokens, escapes, tail, hits, misses = tokenize_ngrams(
                sample,
                ngram_n,
                vocab,
            )
        if not tokens:
            return None
        try:
            codebook = huff0.build_codebook(tokens, prior_count=prior_count)
        except ValueError:
            return None
        return {
            "kind": kind,
            "ngram_n": int(ngram_n),
            "vocab": vocab,
            "codebook": codebook,
            "sample_token_count": len(tokens),
            "sample_ngram_hits": int(hits),
            "sample_ngram_misses": int(misses),
            "sample_escape_bytes": len(escapes),
            "sample_tail_bytes": len(tail),
            # New codebooks use the same four-stream Huff0 payload shape as
            # ZipNN local blocks.  The archive mode records this explicitly so
            # legacy one-stream archives remain decodable.
            "frame_4x": True,
        }

    def _candidate_payload(self, key, mode, codebook_id=-1):
        cluster_id, dtype, stream_id = key
        codec = "huff0"
        codebook = mode["codebook"].codebook
        return CodebookPayload(
            codebook_id=int(codebook_id),
            cluster_id=cluster_id,
            stream_id=stream_id,
            dtype=dtype,
            codec=codec,
            mode=(
                f"{mode['kind']}_4x"
                if mode.get("frame_4x", False)
                else mode["kind"]
            ),
            codebook=codebook,
            ngram_n=int(mode.get("ngram_n", 1)),
            vocab=bytes(mode.get("vocab", b"")),
        )

    @staticmethod
    def _estimated_metric(mode, sample_size, total_size, metric):
        if sample_size <= 0:
            return 0.0
        scale = total_size / sample_size
        if metric == "sample_cost":
            fixed_cost = float(mode.get("fixed_cost", 0))
            variable_cost = max(0.0, float(mode[metric]) - fixed_cost)
            return fixed_cost + variable_cost * scale
        return float(mode[metric]) * scale

    def _evaluate_modes(
        self,
        key,
        sample,
        *,
        total_size,
        element_size,
        is_exponent,
        sample_parts=None,
    ):
        if sample_parts is None:
            sample_parts = (sample,)
        raw_mode = {"kind": "raw", "fixed_cost": 0}
        self._score_mode(
            sample,
            raw_mode,
            element_size,
            sample_parts=sample_parts,
        )
        local_mode = {"kind": "local_huff0", "fixed_cost": 0}
        self._score_mode(
            sample,
            local_mode,
            element_size,
            sample_parts=sample_parts,
        )
        prior_count = self.exponent_prior_count if is_exponent else self.prior_count

        shared = None
        if self.enable_shared_codebooks:
            shared = self._build_candidate(
                sample,
                kind="shared_huff0",
                prior_count=prior_count,
            )
        if shared is not None:
            payload = self._candidate_payload(key, shared)
            self._score_mode(
                sample,
                shared,
                element_size,
                payload,
                sample_parts,
            )

        base_candidates = [raw_mode, local_mode]
        if shared is not None:
            base_candidates.append(shared)
        sample_size = len(sample)

        def estimated_cost(candidate):
            return self._estimated_metric(
                candidate,
                sample_size,
                total_size,
                "sample_cost",
            )

        def estimated_seconds(candidate):
            return self._estimated_metric(
                candidate,
                sample_size,
                total_size,
                "sample_seconds",
            )

        base_selection = select_fastest_within_ratio(
            base_candidates,
            estimated_bytes=estimated_cost,
            estimated_seconds=estimated_seconds,
            ratio_tolerance=self.speed_ratio_tolerance,
        )
        base = base_selection.selected
        base_cost = estimated_cost(base)
        base_seconds = estimated_seconds(base)

        packed_candidates = []
        if is_exponent:
            for n in self.ngram_values:
                packed = self._build_candidate(
                    sample,
                    kind="packed_huff0",
                    prior_count=prior_count,
                    ngram_n=n,
                )
                if packed is None:
                    continue
                payload = self._candidate_payload(key, packed)
                self._score_mode(
                    sample,
                    packed,
                    element_size,
                    payload,
                    sample_parts,
                )
                packed_candidates.append(packed)

        ratio_best_packed = (
            min(packed_candidates, key=estimated_cost)
            if packed_candidates
            else None
        )
        improving_packed_candidates = [
            candidate
            for candidate in packed_candidates
            if estimated_cost(candidate) < base_cost
        ]
        packed_selection = None
        if improving_packed_candidates:
            # Reject non-saving packed modes before applying the speed
            # tolerance. Otherwise a faster n-gram that is slightly larger
            # than the selected base can hide another n that saves bytes. The
            # surviving opportunity remains time-aware here and is then
            # subject to the model-wide packed-time budget below.
            packed_selection = select_fastest_within_ratio(
                improving_packed_candidates,
                estimated_bytes=estimated_cost,
                estimated_seconds=estimated_seconds,
                ratio_tolerance=self.speed_ratio_tolerance,
            )
        packed = packed_selection.selected if packed_selection is not None else None
        ratio_best_base_cost = float(
            base_selection.metrics["ratio_best_estimated_bytes"]
        )
        packed_candidate_diagnostics = tuple(
            {
                "ngram_n": int(candidate["ngram_n"]),
                "top_k": int(self.top_k),
                "estimated_bytes": float(estimated_cost(candidate)),
                "estimated_seconds": float(estimated_seconds(candidate)),
                "saved_bytes_vs_selected_base": float(
                    base_cost - estimated_cost(candidate)
                ),
                "saved_bytes_vs_ratio_best_base": float(
                    ratio_best_base_cost - estimated_cost(candidate)
                ),
                "improves_selected_base": bool(
                    estimated_cost(candidate) < base_cost
                ),
                "improves_ratio_best_base": bool(
                    estimated_cost(candidate) < ratio_best_base_cost
                ),
                "ratio_best": bool(candidate is ratio_best_packed),
                "selected_for_time_budget": bool(candidate is packed),
                "sample_token_count": int(candidate.get("sample_token_count", 0)),
                "sample_ngram_hits": int(candidate.get("sample_ngram_hits", 0)),
                "sample_ngram_misses": int(
                    candidate.get("sample_ngram_misses", 0)
                ),
                "sample_ngram_hit_rate": (
                    candidate.get("sample_ngram_hits", 0)
                    / (
                        candidate.get("sample_ngram_hits", 0)
                        + candidate.get("sample_ngram_misses", 0)
                    )
                    if candidate.get("sample_ngram_hits", 0)
                    + candidate.get("sample_ngram_misses", 0)
                    else 0.0
                ),
                "sample_escape_bytes": int(candidate.get("sample_escape_bytes", 0)),
                "sample_tail_bytes": int(candidate.get("sample_tail_bytes", 0)),
            }
            for candidate in packed_candidates
        )
        opportunity = None
        if packed is not None:
            packed_cost = estimated_cost(packed)
            packed_seconds = estimated_seconds(packed)
            saved_bytes = base_cost - packed_cost
            if saved_bytes > 0:
                opportunity = {
                    "mode": packed,
                    "saved_bytes": saved_bytes,
                    "source_bytes": int(total_size),
                    "savings_per_source_byte": saved_bytes / total_size,
                    "packed_seconds": packed_seconds,
                    "extra_seconds": max(0.0, packed_seconds - base_seconds),
                }
        return {
            "key": key,
            "base": base,
            "base_cost": base_cost,
            "base_seconds": base_seconds,
            "base_selection": dict(base_selection.metrics),
            "packed_selection": (
                dict(packed_selection.metrics)
                if packed_selection is not None
                else None
            ),
            "packed_candidate_diagnostics": packed_candidate_diagnostics,
            "total_size": int(total_size),
            "is_exponent": bool(is_exponent),
            "opportunity": opportunity,
        }

    def _finalize_mode(self, key, selected):
        if selected["kind"] in {"raw", "local_huff0"}:
            return selected

        codebook_id = self._next_codebook_id
        self._next_codebook_id += 1
        selected["codebook_id"] = codebook_id
        payload = self._candidate_payload(key, selected, codebook_id=codebook_id)
        self._codebook_payloads.append(payload)
        self._codebook_payload_by_id[codebook_id] = payload
        self._decode_codebooks[codebook_id] = huff0.build_decodebook(
            payload.codebook
        )
        return selected

    def _record_selected_mode(self, mode):
        kind = mode["kind"]
        if kind == "packed_huff0":
            kind = f"packed_huff0_n{mode['ngram_n']}"
        self._selected_mode_counts[kind] = (
            self._selected_mode_counts.get(kind, 0) + 1
        )

    def prepare_clusters(self, named_tensors):
        self._prepare_clusters(named_tensors)
        self._clusters_prepared = True

    @staticmethod
    def _transfer_plan_key(name, tensor, stream_id):
        return str(name), dtype_name(tensor.dtype), int(stream_id)

    def _compress_stream_with_mode(
        self,
        stream,
        stream_kind,
        *,
        element_size,
        mode,
    ):
        if mode["kind"] == "local_huff0":
            return super()._compress_stream(
                stream,
                stream_kind,
                chunk_size=self._stream_chunk_size(element_size),
            )

        step = (
            self._stream_chunk_size(element_size)
            if mode["kind"] == "raw"
            else self._source_chunk_size(element_size, mode)
        )
        source_chunks = tuple(
            stream[start : start + step]
            for start in range(0, len(stream), step)
            if stream[start : start + step]
        )
        if mode["kind"] == "raw":
            chunks = [self._raw_chunk(chunk) for chunk in source_chunks]
            return ByteGroupPayload(
                stream_kind=stream_kind,
                symbol_count=len(stream),
                chunks=tuple(chunks),
            )

        encoded_chunks = [None] * len(source_chunks)
        if mode["kind"] == "shared_huff0":
            encoder = (
                huff0.encode_many_with_codebook_4x
                if mode.get("frame_4x", False)
                else huff0.encode_many_with_codebook
            )
            payloads = encoder(
                source_chunks,
                mode["codebook"],
                min_saving_bytes=self.min_saving_bytes,
                compression_threshold=self.compression_threshold,
                metadata_bytes=SHARED_CHUNK_METADATA_BYTES,
                threads=self.threads,
            )
            for index, (chunk, payload) in enumerate(
                zip(source_chunks, payloads, strict=True)
            ):
                if payload is not None:
                    encoded_chunks[index] = StreamChunk(
                        mode="shared_huff0",
                        symbol_count=len(chunk),
                        payload=payload,
                        codebook_id=mode.get("codebook_id"),
                    )
        elif mode["kind"] == "packed_huff0":
            encoder = (
                huff0.encode_many_ngram_with_codebook_4x
                if mode.get("frame_4x", False)
                else huff0.encode_many_ngram_with_codebook
            )
            payloads = encoder(
                source_chunks,
                mode["codebook"],
                mode["vocab"],
                n=mode["ngram_n"],
                min_saving_bytes=self.min_saving_bytes,
                compression_threshold=self.compression_threshold,
                metadata_bytes=PACKED_CHUNK_METADATA_BYTES,
                threads=self.threads,
            )
            for index, (chunk, payload) in enumerate(
                zip(source_chunks, payloads, strict=True)
            ):
                if payload is not None:
                    encoded_chunks[index] = StreamChunk(
                        mode="packed_huff0",
                        symbol_count=len(chunk),
                        payload=payload.payload,
                        codebook_id=mode.get("codebook_id"),
                        token_count=payload.symbol_count,
                        escape_payload=payload.escape_bytes,
                        tail_payload=payload.tail,
                        ngram_n=int(mode["ngram_n"]),
                        ngram_hits=payload.hits,
                        ngram_escapes=payload.misses,
                    )
        else:
            raise ValueError(f"Cannot encode mode {mode['kind']!r}.")

        return self._finish_stream_group(
            source_chunks,
            encoded_chunks,
            stream_kind=stream_kind,
            symbol_count=len(stream),
        )

    def _finish_stream_group(
        self,
        source_chunks,
        encoded_chunks,
        *,
        stream_kind,
        symbol_count,
    ):
        """Apply the normal local fallback and assemble one byte group."""
        fallback_indices = [
            index for index, encoded in enumerate(encoded_chunks) if encoded is None
        ]
        if fallback_indices:
            fallback_chunks = super()._compress_local_huff0_chunks(
                source_chunks[index] for index in fallback_indices
            )
            for index, fallback in zip(
                fallback_indices, fallback_chunks, strict=True
            ):
                encoded_chunks[index] = fallback
        return ByteGroupPayload(
            stream_kind=stream_kind,
            symbol_count=int(symbol_count),
            chunks=tuple(encoded_chunks),
        )

    def _select_transfer_mode(self, sample, source_mode, *, element_size, stream_kind):
        raw_mode = {"kind": "raw", "fixed_cost": 0}
        local_mode = {"kind": "local_huff0", "fixed_cost": 0}
        raw_cost = self._compress_stream_with_mode(
            sample,
            stream_kind,
            element_size=element_size,
            mode=raw_mode,
        ).compressed_bytes
        local_cost = self._compress_stream_with_mode(
            sample,
            stream_kind,
            element_size=element_size,
            mode=local_mode,
        ).compressed_bytes
        if local_cost < raw_cost:
            selected = local_mode
            selected_cost = local_cost
        else:
            selected = raw_mode
            selected_cost = raw_cost

        transfer_cost = None
        if source_mode is not None and source_mode["kind"] != "raw":
            transfer_cost = self._compress_stream_with_mode(
                sample,
                stream_kind,
                element_size=element_size,
                mode=source_mode,
            ).compressed_bytes
            if transfer_cost < selected_cost:
                selected = source_mode
                selected_cost = transfer_cost

        return selected, {
            "raw_cost": int(raw_cost),
            "local_cost": int(local_cost),
            "transfer_cost": (
                int(transfer_cost) if transfer_cost is not None else None
            ),
            "selected_cost": int(selected_cost),
            "uses_source_codebook": selected is source_mode,
        }

    def _prepare_tensor_modes(self, named_tensors):
        used_codebook_ids = set()
        mode_counts = {}
        mode_symbols = {}
        probe_bytes = 0

        for name, tensor in sorted(named_tensors, key=lambda item: item[0]):
            if (
                not torch.is_tensor(tensor)
                or not tensor.is_floating_point()
                or tensor.numel() == 0
            ):
                continue

            original = tensor.detach().cpu().contiguous()
            config = _zipnn_dtype_config(original.dtype)
            for stream_id in range(int(config["num_buffers"])):
                key = self._codebook_key(name, original, stream_id)
                # The source cluster decision already compared compressed size
                # and direct-encoding time under the configured tolerance.
                # Re-probing every tensor with a ratio-only local baseline here
                # would undo that decision and prune all shared codebooks.
                selected = self._mode_by_key[key]

                plan_key = self._transfer_plan_key(name, original, stream_id)
                self._mode_by_tensor_stream[plan_key] = selected
                kind = selected["kind"]
                mode_counts[kind] = mode_counts.get(kind, 0) + 1
                mode_symbols[kind] = mode_symbols.get(kind, 0) + int(original.numel())
                codebook_id = selected.get("codebook_id")
                if codebook_id is not None:
                    used_codebook_ids.add(int(codebook_id))

        removed_codebook_ids = {
            int(payload.codebook_id)
            for payload in self._codebook_payloads
            if int(payload.codebook_id) not in used_codebook_ids
        }
        if removed_codebook_ids:
            self._codebook_payloads = [
                payload
                for payload in self._codebook_payloads
                if int(payload.codebook_id) in used_codebook_ids
            ]
            for codebook_id in removed_codebook_ids:
                self._decode_codebooks.pop(codebook_id, None)
                self._codebook_payload_by_id.pop(codebook_id, None)
            for key, mode in tuple(self._mode_by_key.items()):
                if mode.get("codebook_id") in removed_codebook_ids:
                    self._mode_by_key[key] = {
                        "kind": "local_huff0",
                        "fixed_cost": 0,
                    }

        self._selected_mode_counts = {}
        for mode in self._mode_by_key.values():
            self._record_selected_mode(mode)
        self._selection_stats.update(
            {
                "tensor_probe_bytes": probe_bytes,
                "tensor_selected_modes": mode_counts,
                "tensor_selected_symbols": mode_symbols,
                "pruned_codebooks": len(removed_codebook_ids),
            }
        )

    def prepare_direct_transfer_target(
        self,
        named_tensors,
        *,
        require_full_match=True,
    ):
        """Map a family target to the frozen source plan by exact tensor name.

        This path reads only target metadata (name, dtype, size, and number of
        byte streams). It does not inspect tensor values, compute exponent
        histograms or Jensen--Shannon distances, or evaluate alternative modes.
        The target can encode immediately from its own tensor buffers while
        retaining the source model's modes, vocabularies, and codebooks. No
        target data are copied or retained by this mapping step.
        """
        if not self._clusters_prepared:
            raise RuntimeError("Source clusters must be prepared before transfer.")
        if not self._mode_by_tensor_stream:
            raise RuntimeError(
                "Source codebooks and modes must be built before transfer."
            )

        named_tensors = list(named_tensors)
        transfer_modes = {}
        transfer_clusters = {}
        missing = []
        selected_modes = {}
        selected_symbols = {}
        target_tensors = 0
        matched_tensors = 0
        target_symbols = 0
        matched_symbols = 0

        for name, tensor in sorted(named_tensors, key=lambda item: item[0]):
            if (
                not torch.is_tensor(tensor)
                or not tensor.is_floating_point()
                or tensor.numel() == 0
            ):
                continue

            config = _zipnn_dtype_config(tensor.dtype)
            plan = []
            for stream_id in range(int(config["num_buffers"])):
                plan_key = self._transfer_plan_key(name, tensor, stream_id)
                source_mode = self._mode_by_tensor_stream.get(plan_key)
                if source_mode is None:
                    missing.append(plan_key)
                    source_mode = {"kind": "local_huff0", "fixed_cost": 0}
                plan.append((plan_key, source_mode))

            tensor_matched = all(
                plan_key in self._mode_by_tensor_stream
                for plan_key, _ in plan
            )
            symbols = int(tensor.numel())
            target_tensors += 1
            target_symbols += symbols
            if tensor_matched:
                matched_tensors += 1
                matched_symbols += symbols

            source_cluster = self.cluster_map.get(str(name))
            if source_cluster is None:
                source_cluster = self._computed_cluster_map.get(str(name))
            transfer_clusters[str(name)] = (
                str(source_cluster)
                if tensor_matched and source_cluster is not None
                else f"transfer-local:{name}"
            )

            for plan_key, selected in plan:
                transfer_modes[plan_key] = selected
                kind = str(selected["kind"])
                selected_modes[kind] = selected_modes.get(kind, 0) + 1
                selected_symbols[kind] = selected_symbols.get(kind, 0) + symbols

        if missing and require_full_match:
            preview = ", ".join(
                f"{name}:{dtype}:stream{stream_id}"
                for name, dtype, stream_id in missing[:8]
            )
            suffix = "" if len(missing) <= 8 else f" (+{len(missing) - 8} more)"
            raise ValueError(
                "Direct transfer requires an exact source mode for every target "
                f"stream; missing {preview}{suffix}."
            )

        self._transfer_mode_by_tensor_stream = transfer_modes
        self._transfer_cluster_by_name = transfer_clusters
        self._transfer_stats = {
            "mapping": "exact source tensor name, dtype, and stream",
            "target_tensors": target_tensors,
            "matched_tensors": matched_tensors,
            "matched_tensor_pct": (
                100.0 * matched_tensors / target_tensors if target_tensors else 0.0
            ),
            "target_exponents": target_symbols,
            "matched_exponents": matched_symbols,
            "matched_exponent_pct": (
                100.0 * matched_symbols / target_symbols if target_symbols else 0.0
            ),
            "missing_streams": len(missing),
            "selected_modes": selected_modes,
            "selected_symbols": selected_symbols,
            "target_histograms_computed": False,
            "target_js_matching": False,
            "target_mode_probing": False,
        }
        self._transfer_active = True
        return self.transfer_stats

    def prepare_transfer_target(self, named_tensors, *, js_threshold=None):
        """Prepare a target model against this codec's frozen source codebooks."""
        if not self._clusters_prepared or not self._cluster_centroids:
            raise RuntimeError("Source clusters must be prepared before transfer.")
        if not self._mode_by_key:
            raise RuntimeError("Source codebooks must be built before transfer.")

        threshold = (
            self.cluster_js_threshold
            if js_threshold is None
            else float(js_threshold)
        )
        if threshold < 0:
            raise ValueError("Transfer JS threshold must be non-negative.")

        self._transfer_mode_by_tensor_stream = {}
        self._transfer_cluster_by_name = {}
        mode_counts = {}
        mode_symbols = {}
        distances = []
        matched_tensors = 0
        target_tensors = 0
        target_symbols = 0
        matched_symbols = 0
        probe_bytes = 0

        for name, tensor in sorted(named_tensors, key=lambda item: item[0]):
            if (
                not torch.is_tensor(tensor)
                or not tensor.is_floating_point()
                or tensor.numel() == 0
            ):
                continue

            original = tensor.detach().cpu().contiguous()
            config = _zipnn_dtype_config(original.dtype)
            dtype = dtype_name(original.dtype)
            exponent_stream_id = int(config["element_size"]) - 1
            candidates = [
                (cluster_id, centroid)
                for cluster_id, centroid in self._cluster_centroids.items()
                if (cluster_id, dtype, exponent_stream_id) in self._mode_by_key
            ]
            distribution = _exponent_distribution(original)
            if candidates:
                cluster_id, distance = min(
                    (
                        (
                            cluster_id,
                            jensen_shannon_distance(distribution, centroid),
                        )
                        for cluster_id, centroid in candidates
                    ),
                    key=lambda item: (item[1], item[0]),
                )
            else:
                cluster_id, distance = None, float("inf")

            matched = cluster_id is not None and distance <= threshold
            target_tensors += 1
            tensor_symbols = int(original.numel())
            target_symbols += tensor_symbols
            distances.append(float(distance))
            if matched:
                matched_tensors += 1
                matched_symbols += tensor_symbols
                self._transfer_cluster_by_name[str(name)] = str(cluster_id)
            else:
                self._transfer_cluster_by_name[str(name)] = f"transfer-local:{name}"

            reordered = _zipnn_reordered_bytes(original, config)
            element_size = int(config["element_size"])
            for stream_id in range(int(config["num_buffers"])):
                is_exponent = self._is_exponent_stream(config, stream_id)
                stream = reordered[stream_id::element_size]
                stream, _ = self._apply_exponent_layout(original, stream, is_exponent)
                sample = stream[: self.max_codebook_sample_bytes]
                source_mode = (
                    self._mode_by_key.get((cluster_id, dtype, int(stream_id)))
                    if matched
                    else None
                )
                selected, _ = self._select_transfer_mode(
                    sample,
                    source_mode,
                    element_size=element_size,
                    stream_kind="exponent" if is_exponent else "byte_group",
                )
                plan_key = self._transfer_plan_key(name, original, stream_id)
                self._transfer_mode_by_tensor_stream[plan_key] = selected
                kind = selected["kind"]
                mode_counts[kind] = mode_counts.get(kind, 0) + 1
                mode_symbols[kind] = mode_symbols.get(kind, 0) + len(stream)
                probe_bytes += len(sample)

        finite_distances = [distance for distance in distances if np.isfinite(distance)]
        self._transfer_stats = {
            "mapping": "target Jensen-Shannon match and bounded mode probe",
            "js_threshold": threshold,
            "target_tensors": target_tensors,
            "matched_tensors": matched_tensors,
            "matched_tensor_pct": (
                100.0 * matched_tensors / target_tensors if target_tensors else 0.0
            ),
            "target_exponents": target_symbols,
            "matched_exponents": matched_symbols,
            "matched_exponent_pct": (
                100.0 * matched_symbols / target_symbols if target_symbols else 0.0
            ),
            "mean_nearest_js": (
                float(np.mean(finite_distances)) if finite_distances else None
            ),
            "max_nearest_js": (
                float(np.max(finite_distances)) if finite_distances else None
            ),
            "probe_bytes": probe_bytes,
            "selected_modes": mode_counts,
            "selected_symbols": mode_symbols,
            "target_histograms_computed": True,
            "target_js_matching": True,
            "target_mode_probing": True,
        }
        self._transfer_active = True
        return self.transfer_stats

    def _collect_codebook_samples(self, named_tensors):
        """Collect the bounded, reordered streams used to fit codebooks."""

        sample_parts = {}
        sample_sizes = {}
        total_sizes = {}
        stream_info = {}
        for name, tensor in sorted(named_tensors, key=lambda item: item[0]):
            if not torch.is_tensor(tensor) or not tensor.is_floating_point():
                continue
            original = tensor.detach().cpu().contiguous()
            config = _zipnn_dtype_config(original.dtype)
            reordered = _zipnn_reordered_bytes(original, config)
            element_size = int(config["element_size"])
            for stream_id in range(int(config["num_buffers"])):
                is_exponent = self._is_exponent_stream(config, stream_id)
                stream = reordered[stream_id::element_size]
                stream, _ = self._apply_exponent_layout(original, stream, is_exponent)
                key = self._codebook_key(name, original, stream_id)
                total_sizes[key] = total_sizes.get(key, 0) + len(stream)
                remaining = self.max_codebook_sample_bytes - sample_sizes.get(key, 0)
                if remaining <= 0:
                    continue
                part = self._bounded_stream_sample(stream, remaining)
                if not part:
                    continue
                sample_parts.setdefault(key, []).append(bytes(part))
                sample_sizes[key] = sample_sizes.get(key, 0) + len(part)
                stream_info[key] = (element_size, is_exponent)
        return sample_parts, total_sizes, stream_info

    def build_codebooks_and_select_modes(self, named_tensors):
        if not self._clusters_prepared:
            raise RuntimeError(
                "Clusters must be prepared before building NeuralZip codebooks."
            )
        named_tensors = list(named_tensors)
        sample_parts, total_sizes, stream_info = self._collect_codebook_samples(
            named_tensors
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
                )
            )

        exponent_source_bytes = sum(
            evaluation["total_size"]
            for evaluation in evaluations
            if evaluation["is_exponent"]
        )
        packed_source_budget = int(
            exponent_source_bytes * self.packed_source_budget_ratio
        )
        time_selection = select_packed_time_budget(
            evaluations,
            packed_time_budget_ratio=self.packed_time_budget_ratio,
        )
        packed_used = 0
        packed_saved = 0.0
        selected_packed_keys = set()
        opportunities = [
            evaluation
            for evaluation in evaluations
            if evaluation["opportunity"] is not None
            and evaluation["key"] in time_selection.selected_keys
        ]
        opportunities.sort(
            key=lambda evaluation: (
                -evaluation["opportunity"]["savings_per_source_byte"],
                -evaluation["opportunity"]["saved_bytes"],
                evaluation["key"],
            )
        )
        for evaluation in opportunities:
            if self.packed_source_budget_ratio == 0:
                break
            opportunity = evaluation["opportunity"]
            source_bytes = opportunity["source_bytes"]
            if packed_used + source_bytes > packed_source_budget:
                continue
            selected_packed_keys.add(evaluation["key"])
            packed_used += source_bytes
            packed_saved += opportunity["saved_bytes"]

        selection_diagnostics = []
        for evaluation in evaluations:
            if not evaluation["is_exponent"]:
                continue
            key = evaluation["key"]
            candidate_diagnostics = evaluation[
                "packed_candidate_diagnostics"
            ]
            opportunity = evaluation["opportunity"]
            if not candidate_diagnostics:
                decision = "no-packed-mode"
            elif opportunity is None:
                decision = "no-ratio-saving"
            elif key not in time_selection.selected_keys:
                decision = "time-budget"
            elif key not in selected_packed_keys:
                decision = "source-budget"
            else:
                decision = "selected"
            cluster_id, dtype, stream_id = key
            selection_diagnostics.append(
                {
                    "cluster_id": str(cluster_id),
                    "dtype": str(dtype),
                    "stream_id": int(stream_id),
                    "source_bytes": int(evaluation["total_size"]),
                    "base_kind": str(evaluation["base"]["kind"]),
                    "base_estimated_bytes": float(evaluation["base_cost"]),
                    "ratio_best_base_kind": str(
                        evaluation["base_selection"]["ratio_best_kind"]
                    ),
                    "ratio_best_base_estimated_bytes": float(
                        evaluation["base_selection"][
                            "ratio_best_estimated_bytes"
                        ]
                    ),
                    "base_estimated_seconds": float(
                        evaluation["base_seconds"]
                    ),
                    "selected_candidate_ngram_n": (
                        int(opportunity["mode"]["ngram_n"])
                        if opportunity is not None
                        else None
                    ),
                    "ratio_best_ngram_n": next(
                        (
                            int(candidate["ngram_n"])
                            for candidate in candidate_diagnostics
                            if candidate["ratio_best"]
                        ),
                        None,
                    ),
                    "decision": decision,
                    "candidates": candidate_diagnostics,
                }
            )
        self._selection_diagnostics = tuple(selection_diagnostics)

        for evaluation in evaluations:
            selected = evaluation["base"]
            if evaluation["key"] in selected_packed_keys:
                selected = evaluation["opportunity"]["mode"]
            selected = self._finalize_mode(evaluation["key"], selected)
            self._mode_by_key[evaluation["key"]] = selected
            self._record_selected_mode(selected)

        base_selections = [evaluation["base_selection"] for evaluation in evaluations]
        self._selection_stats = {
            **dict(time_selection.metrics),
            "selection_strategy": "fastest-within-ratio-and-time-budget",
            "speed_ratio_tolerance": self.speed_ratio_tolerance,
            "base_modes_avoiding_local_huff0": sum(
                bool(selection["avoided_local_huff0"])
                for selection in base_selections
            ),
            "base_estimated_ratio_overhead_bytes": sum(
                float(selection["selected_byte_overhead"])
                for selection in base_selections
            ),
            "exponent_source_bytes": exponent_source_bytes,
            "packed_source_budget_ratio": self.packed_source_budget_ratio,
            "packed_source_budget_bytes": packed_source_budget,
            "packed_source_bytes": packed_used,
            "packed_estimated_bytes_saved": packed_saved,
            "packed_modes_evaluated": sum(
                len(evaluation["packed_candidate_diagnostics"])
                for evaluation in evaluations
                if evaluation["is_exponent"]
            ),
            "packed_improving_selected_base_modes": sum(
                int(candidate["improves_selected_base"])
                for evaluation in evaluations
                if evaluation["is_exponent"]
                for candidate in evaluation["packed_candidate_diagnostics"]
            ),
            "packed_improving_ratio_best_base_modes": sum(
                int(candidate["improves_ratio_best_base"])
                for evaluation in evaluations
                if evaluation["is_exponent"]
                for candidate in evaluation["packed_candidate_diagnostics"]
            ),
            "packed_rejected_no_ratio_saving_modes": sum(
                not candidate["improves_selected_base"]
                for evaluation in evaluations
                if evaluation["is_exponent"]
                for candidate in evaluation["packed_candidate_diagnostics"]
            ),
            "packed_opportunities": sum(
                evaluation["opportunity"] is not None
                for evaluation in evaluations
                if evaluation["is_exponent"]
            ),
            "packed_keys_rejected_no_ratio_saving": sum(
                bool(evaluation["packed_candidate_diagnostics"])
                and evaluation["opportunity"] is None
                for evaluation in evaluations
                if evaluation["is_exponent"]
            ),
            "packed_time_selected": len(time_selection.selected_keys),
            "packed_rejected_by_source_budget": (
                len(time_selection.selected_keys) - len(selected_packed_keys)
            ),
            "packed_selected": len(selected_packed_keys),
        }

        self._prepare_tensor_modes(named_tensors)
        self._selection_stats["packed_selected"] = sum(
            mode["kind"] == "packed_huff0" for mode in self._mode_by_key.values()
        )
        self._selection_stats["actual_packed_source_bytes"] = (
            self._selection_stats["tensor_selected_symbols"].get(
                "packed_huff0",
                0,
            )
        )
        self.load_codebooks(self._codebook_payloads)

    def _mode_for_tensor_stream(self, name, tensor, stream_id):
        """Return the prepared mode for one concrete tensor byte stream."""
        if self._transfer_active:
            plan_key = self._transfer_plan_key(name, tensor, stream_id)
            return self._transfer_mode_by_tensor_stream.get(
                plan_key,
                {"kind": "local_huff0", "fixed_cost": 0},
            )

        key = self._codebook_key(name, tensor, stream_id)
        plan_key = self._transfer_plan_key(name, tensor, stream_id)
        mode = self._mode_by_tensor_stream.get(plan_key)
        if mode is None:
            mode = self._mode_by_key.get(key)
        if mode is None:
            if not self._clusters_prepared:
                raise RuntimeError(
                    "Clusters must be prepared before NeuralZip compression."
                )
            raise RuntimeError(
                "Codebooks and modes must be built before NeuralZip compression."
            )
        return mode

    def compress_tensor(self, name, tensor):
        if not torch.is_tensor(tensor) or not tensor.is_floating_point():
            raise TypeError("NeuralZipCodec only compresses floating tensors.")
        config = _zipnn_dtype_config(tensor.dtype)
        original = tensor.detach().cpu().contiguous()
        return self._compress_neuralzip(name, original, config)

    def _compress_direct_native_batch(self, tensor_specs):
        """Encode one input-size-windowed tensor batch through one job queue."""
        native_specs = [
            (
                tensor.view(torch.uint8).reshape(-1).numpy(),
                int(config["element_size"]),
                int(config["bit_reorder"]),
                self._stream_chunk_size(int(config["element_size"])),
                tensor_modes,
            )
            for _, tensor, config, tensor_modes in tensor_specs
        ]
        native_batch = huff0.compress_zipnn_mixed_many(
            native_specs,
            min_saving_bytes=self.min_saving_bytes,
            threads=self.threads,
            compression_threshold=self.compression_threshold,
        )
        if len(native_batch) != len(tensor_specs):
            raise RuntimeError("Mixed ZipNN returned an invalid tensor count.")
        return [
            self._tensor_payload_from_native_mixed(
                name,
                tensor,
                config,
                tensor_modes,
                native_groups,
            )
            for (name, tensor, config, tensor_modes), native_groups in zip(
                tensor_specs,
                native_batch,
                strict=True,
            )
        ]

    def compress_tensors(self, named_tensors):
        """Compress a model without retaining target-derived data."""
        named_tensors = list(named_tensors)

        # Build transient input views and submit input-size-windowed batches to
        # the fused native encoder. Transposed exponent layouts use the Python
        # stream scheduler below because they require a tensor-dependent
        # transform before coding.
        if self.exponent_layout == "original":
            outputs = []
            batch = []
            batch_bytes = 0
            for name, tensor in named_tensors:
                if not torch.is_tensor(tensor) or not tensor.is_floating_point():
                    raise TypeError("NeuralZipCodec only compresses floating tensors.")
                config = _zipnn_dtype_config(tensor.dtype)
                raw_bytes = int(tensor.numel()) * int(config["element_size"])
                if batch and batch_bytes + raw_bytes > self.direct_batch_bytes:
                    outputs.extend(self._compress_direct_native_batch(batch))
                    batch = []
                    batch_bytes = 0
                # Copy a GPU/non-contiguous tensor only after the preceding
                # window has released its native scratch buffers.
                original = tensor.detach().cpu().contiguous()
                stream_count = int(config["num_buffers"])
                tensor_modes = tuple(
                    self._mode_for_tensor_stream(name, original, stream_id)
                    for stream_id in range(stream_count)
                )
                batch.append(
                    (name, original, config, tensor_modes)
                )
                batch_bytes += raw_bytes
                # One oversized tensor is still bounded to that tensor rather
                # than being retained alongside the following tensors.
                if batch_bytes >= self.direct_batch_bytes:
                    outputs.extend(self._compress_direct_native_batch(batch))
                    batch = []
                    batch_bytes = 0
            if batch:
                outputs.extend(self._compress_direct_native_batch(batch))
            return tuple(outputs)

        outputs = [None] * len(named_tensors)
        tensor_specs = {}
        packed_batches = {}

        for tensor_index, (name, tensor) in enumerate(named_tensors):
            if not torch.is_tensor(tensor) or not tensor.is_floating_point():
                raise TypeError("NeuralZipCodec only compresses floating tensors.")

            config = _zipnn_dtype_config(tensor.dtype)
            original = tensor.detach().cpu().contiguous()
            element_size = int(config["element_size"])
            stream_count = int(config["num_buffers"])
            tensor_modes = tuple(
                self._mode_for_tensor_stream(name, original, stream_id)
                for stream_id in range(stream_count)
            )

            if not any(mode.get("kind") == "packed_huff0" for mode in tensor_modes):
                outputs[tensor_index] = self._compress_neuralzip(
                    name,
                    original,
                    config,
                )
                continue

            reordered = _zipnn_reordered_bytes(original, config)
            groups = [None] * stream_count
            tensor_layout = "original"
            tensor_specs[tensor_index] = {
                "name": name,
                "tensor": original,
                "config": config,
                "groups": groups,
                "tensor_layout": tensor_layout,
            }

            for stream_id, mode in enumerate(tensor_modes):
                is_exponent = self._is_exponent_stream(config, stream_id)
                stream = reordered[stream_id::element_size]
                stream, layout = self._apply_exponent_layout(
                    original,
                    stream,
                    is_exponent,
                )
                if is_exponent:
                    tensor_specs[tensor_index]["tensor_layout"] = layout
                stream_kind = "exponent" if is_exponent else "byte_group"

                if mode.get("kind") != "packed_huff0":
                    groups[stream_id] = self._compress_stream_with_mode(
                        stream,
                        stream_kind,
                        element_size=element_size,
                        mode=mode,
                    )
                    continue

                step = self._source_chunk_size(element_size, mode)
                source_chunks = tuple(
                    stream[start : start + step]
                    for start in range(0, len(stream), step)
                    if stream[start : start + step]
                )
                # Identical serialized codebooks and vocabularies produce the
                # same token and bit streams even when their archive ids differ.
                batch_key = (
                    bytes(mode["codebook"].codebook),
                    bytes(mode["vocab"]),
                    int(mode["ngram_n"]),
                    bool(mode.get("frame_4x", False)),
                )
                batch = packed_batches.setdefault(
                    batch_key,
                    {"mode": mode, "requests": []},
                )
                batch["requests"].append(
                    {
                        "tensor_index": tensor_index,
                        "stream_id": stream_id,
                        "stream_kind": stream_kind,
                        "symbol_count": len(stream),
                        "source_chunks": source_chunks,
                        "mode": mode,
                    }
                )

        for batch in packed_batches.values():
            mode = batch["mode"]
            flat_chunks = tuple(
                chunk
                for request in batch["requests"]
                for chunk in request["source_chunks"]
            )
            encoder = (
                huff0.encode_many_ngram_with_codebook_4x
                if mode.get("frame_4x", False)
                else huff0.encode_many_ngram_with_codebook
            )
            flat_payloads = encoder(
                flat_chunks,
                mode["codebook"],
                mode["vocab"],
                n=mode["ngram_n"],
                min_saving_bytes=self.min_saving_bytes,
                compression_threshold=self.compression_threshold,
                metadata_bytes=PACKED_CHUNK_METADATA_BYTES,
                threads=self.threads,
            )

            offset = 0
            for request in batch["requests"]:
                source_chunks = request["source_chunks"]
                payloads = flat_payloads[offset : offset + len(source_chunks)]
                offset += len(source_chunks)
                request_mode = request["mode"]
                encoded_chunks = [None] * len(source_chunks)
                for index, (chunk, payload) in enumerate(
                    zip(source_chunks, payloads, strict=True)
                ):
                    if payload is not None:
                        candidate = StreamChunk(
                            mode="packed_huff0",
                            symbol_count=len(chunk),
                            payload=payload.payload,
                            codebook_id=request_mode.get("codebook_id"),
                            token_count=payload.symbol_count,
                            escape_payload=payload.escape_bytes,
                            tail_payload=payload.tail,
                            ngram_n=int(request_mode["ngram_n"]),
                            ngram_hits=payload.hits,
                            ngram_escapes=payload.misses,
                        )
                        if (
                            candidate.compressed_bytes
                            < len(chunk) * self.compression_threshold
                        ):
                            encoded_chunks[index] = candidate
                group = self._finish_stream_group(
                    source_chunks,
                    encoded_chunks,
                    stream_kind=request["stream_kind"],
                    symbol_count=request["symbol_count"],
                )
                tensor_specs[request["tensor_index"]]["groups"][
                    request["stream_id"]
                ] = group

            if offset != len(flat_payloads):
                raise RuntimeError("Packed batch result count did not match its inputs.")

        for tensor_index, spec in tensor_specs.items():
            tensor = spec["tensor"]
            config = spec["config"]
            groups = spec["groups"]
            if any(group is None for group in groups):
                raise RuntimeError("Packed batch left an uncompressed tensor stream.")
            outputs[tensor_index] = TensorPayload(
                name=spec["name"],
                shape=tuple(int(dim) for dim in tensor.shape),
                dtype=dtype_name(tensor.dtype),
                element_size=int(config["element_size"]),
                numel=int(tensor.numel()),
                strategy="neuralzip",
                groups=tuple(groups),
                byte_reorder=int(config["byte_reorder"]),
                bit_reorder=int(config["bit_reorder"]),
                num_buffers=int(config["num_buffers"]),
                cluster_id=self.cluster_id(spec["name"]),
                exponent_layout=spec["tensor_layout"],
            )

        if any(payload is None for payload in outputs):
            raise RuntimeError("Model batch compression did not produce every tensor.")
        return tuple(outputs)

    def _compress_neuralzip_mixed(self, name, tensor, config, tensor_modes):
        """Encode one tensor from its original buffer without retained target data.

        The native backend performs ZipNN bit reordering, byte-lane extraction,
        packed tokenization, Huff0 encoding, and local/raw fallback in one
        transient pipeline.  Only the final archive payload survives the call.
        """
        element_size = int(config["element_size"])
        stream_count = int(config["num_buffers"])
        if stream_count != element_size:
            raise RuntimeError("Mixed ZipNN expects one stream per element byte.")

        raw = tensor.view(torch.uint8).reshape(-1).numpy()
        native_groups = huff0.compress_zipnn_mixed(
            raw,
            element_size=element_size,
            bit_reorder=int(config["bit_reorder"]),
            chunk_size=self._stream_chunk_size(element_size),
            stream_modes=tensor_modes,
            min_saving_bytes=self.min_saving_bytes,
            threads=self.threads,
            compression_threshold=self.compression_threshold,
        )
        return self._tensor_payload_from_native_mixed(
            name,
            tensor,
            config,
            tensor_modes,
            native_groups,
        )

    def _tensor_payload_from_native_mixed(
        self,
        name,
        tensor,
        config,
        tensor_modes,
        native_groups,
    ):
        """Assemble the stable archive objects returned by the native engine."""
        element_size = int(config["element_size"])
        stream_count = int(config["num_buffers"])
        if len(native_groups) != stream_count:
            raise RuntimeError("Mixed ZipNN returned an invalid stream count.")

        baseline_format = all(
            mode.get("kind") in {"local_huff0", "raw"}
            for mode in tensor_modes
        )
        allowed_finals = {
            "raw": {"raw"},
            "local_huff0": {"local_huff0", "raw"},
            "shared_huff0": {"shared_huff0", "local_huff0", "raw"},
            "packed_huff0": {"packed_huff0", "local_huff0", "raw"},
        }
        groups = []
        for stream_id, (planned_mode, native_chunks) in enumerate(
            zip(tensor_modes, native_groups, strict=True)
        ):
            planned_kind = str(planned_mode.get("kind"))
            if planned_kind not in allowed_finals:
                raise ValueError(f"Cannot encode mode {planned_kind!r} natively.")
            chunks = []
            symbols = 0
            for native_chunk in native_chunks:
                (
                    final_kind,
                    payload,
                    symbol_count,
                    token_count,
                    escapes,
                    tail,
                    ngram_n,
                    hits,
                    misses,
                ) = native_chunk
                final_kind = str(final_kind)
                symbol_count = int(symbol_count)
                if final_kind not in allowed_finals[planned_kind]:
                    raise RuntimeError(
                        f"Mixed ZipNN changed {planned_kind!r} into the invalid "
                        f"mode {final_kind!r}."
                    )
                symbols += symbol_count
                if final_kind == "shared_huff0":
                    chunks.append(
                        StreamChunk(
                            mode=final_kind,
                            symbol_count=symbol_count,
                            payload=payload,
                            codebook_id=planned_mode.get("codebook_id"),
                        )
                    )
                elif final_kind == "packed_huff0":
                    chunks.append(
                        StreamChunk(
                            mode=final_kind,
                            symbol_count=symbol_count,
                            payload=payload,
                            codebook_id=planned_mode.get("codebook_id"),
                            token_count=int(token_count),
                            escape_payload=escapes,
                            tail_payload=tail,
                            ngram_n=int(ngram_n),
                            ngram_hits=int(hits),
                            ngram_escapes=int(misses),
                        )
                    )
                else:
                    chunks.append(
                        StreamChunk(
                            mode=final_kind,
                            symbol_count=symbol_count,
                            payload=payload,
                        )
                    )
            if symbols != int(tensor.numel()):
                raise RuntimeError(
                    "Mixed ZipNN stream symbol count does not match the tensor."
                )
            is_exponent = self._is_exponent_stream(config, stream_id)
            stream_kind = (
                "byte_group"
                if baseline_format
                else ("exponent" if is_exponent else "byte_group")
            )
            groups.append(
                ByteGroupPayload(
                    stream_kind=stream_kind,
                    symbol_count=symbols,
                    chunks=tuple(chunks),
                )
            )

        return TensorPayload(
            name=name,
            shape=tuple(int(dim) for dim in tensor.shape),
            dtype=dtype_name(tensor.dtype),
            element_size=element_size,
            numel=int(tensor.numel()),
            strategy="neuralzip",
            groups=tuple(groups),
            byte_reorder=int(config["byte_reorder"]),
            bit_reorder=int(config["bit_reorder"]),
            num_buffers=stream_count,
            cluster_id=self.cluster_id(name),
            exponent_layout="original",
        )

    def _compress_neuralzip(self, name, tensor, config):
        # When precomputation selects only the baseline local/raw modes, use
        # ZipNNCodec's native batched Huff0 path directly.  This preserves the
        # exact baseline representation while avoiding NeuralZip's per-stream
        # Python dispatch overhead; advanced shared/packed modes continue below.
        # Inspect only the plans belonging to this tensor.  Looking at the
        # global plan made a single packed/shared stream disable the fast path
        # for every otherwise-baseline tensor in the model.
        element_size = int(config["element_size"])
        stream_count = int(config["num_buffers"])
        tensor_modes = tuple(
            self._mode_for_tensor_stream(name, tensor, stream_id)
            for stream_id in range(stream_count)
        )
        baseline_format = (
            self.exponent_layout == "original"
            and tensor_modes
            and all(
                mode.get("kind") in {"local_huff0", "raw"}
                for mode in tensor_modes
            )
        )
        if self.exponent_layout == "original" and any(
            mode.get("kind") != "local_huff0" for mode in tensor_modes
        ):
            return self._compress_neuralzip_mixed(
                name,
                tensor,
                config,
                tensor_modes,
            )
        if baseline_format:
            baseline = super()._compress_zipnn(name, tensor, config)
            return TensorPayload(
                name=name,
                shape=tuple(int(dim) for dim in tensor.shape),
                dtype=dtype_name(tensor.dtype),
                element_size=element_size,
                numel=int(tensor.numel()),
                strategy="neuralzip",
                groups=baseline.groups,
                byte_reorder=int(config["byte_reorder"]),
                bit_reorder=int(config["bit_reorder"]),
                num_buffers=stream_count,
                cluster_id=self.cluster_id(name),
                exponent_layout="original",
            )

        # Advanced tensors are encoded directly from one reordered buffer.
        # Running a complete ZipNN baseline first would compress the packed or
        # shared stream only to discard it immediately afterwards.
        reordered = _zipnn_reordered_bytes(tensor, config)
        groups = []
        tensor_layout = "original"
        for stream_id, mode in enumerate(tensor_modes):
            is_exponent = self._is_exponent_stream(config, stream_id)
            stream = reordered[stream_id::element_size]
            stream, layout = self._apply_exponent_layout(
                tensor,
                stream,
                is_exponent,
            )
            if is_exponent:
                tensor_layout = layout
            # The fused ZipNN baseline historically labels every reordered
            # lane as ``byte_group``.  Preserve that archive representation
            # when setup selected the baseline local/raw representation.
            stream_kind = (
                "byte_group"
                if baseline_format
                else ("exponent" if is_exponent else "byte_group")
            )
            groups.append(
                self._compress_stream_with_mode(
                    stream,
                    stream_kind,
                    element_size=element_size,
                    mode=mode,
                )
            )
        return TensorPayload(
            name=name,
            shape=tuple(int(dim) for dim in tensor.shape),
            dtype=dtype_name(tensor.dtype),
            element_size=element_size,
            numel=int(tensor.numel()),
            strategy="neuralzip",
            groups=tuple(groups),
            byte_reorder=int(config["byte_reorder"]),
            bit_reorder=int(config["bit_reorder"]),
            num_buffers=stream_count,
            cluster_id=self.cluster_id(name),
            exponent_layout=tensor_layout,
        )

    def _decompress_cluster_stream(self, group):
        output = bytearray()
        for chunk in group.chunks:
            if chunk.mode == "raw":
                decoded = chunk.payload
            elif chunk.mode == "zero":
                decoded = b"\x00" * chunk.symbol_count
            elif chunk.mode == "local_huff0":
                decoded = huff0.decompress_block(chunk.payload, chunk.symbol_count)
            elif chunk.mode in {"shared_huff0", "packed_huff0"}:
                codebook = self._codebook_payload_by_id.get(chunk.codebook_id)
                decodebook = self._decode_codebooks.get(chunk.codebook_id)
                if codebook is None or decodebook is None:
                    raise ValueError(f"Missing Huff0 codebook {chunk.codebook_id}.")
                if chunk.mode == "shared_huff0":
                    if codebook.mode not in {
                        "shared_huff0",
                        "shared_huff0_4x",
                    }:
                        raise ValueError(
                            "Shared Huff0 chunk references a non-shared codebook."
                        )
                    decoder = (
                        huff0.decompress_with_decodebook_4x
                        if codebook.mode == "shared_huff0_4x"
                        else huff0.decompress_with_decodebook
                    )
                    decoded = decoder(
                        huff0.Huff0Payload(
                            payload=chunk.payload,
                            codebook=codebook.codebook,
                            symbol_count=chunk.symbol_count,
                        ),
                        decodebook,
                    )
                else:
                    if chunk.token_count is None:
                        raise ValueError("Packed Huff0 chunk is missing its token count.")
                    if codebook.mode not in {
                        "packed_huff0",
                        "packed_huff0_4x",
                    }:
                        raise ValueError("Packed Huff0 chunk references a non-packed codebook.")
                    if int(codebook.ngram_n) != int(chunk.ngram_n):
                        raise ValueError("Packed Huff0 chunk and codebook disagree on n.")
                    decoder = (
                        huff0.decompress_ngram_with_decodebook_4x
                        if codebook.mode == "packed_huff0_4x"
                        else huff0.decompress_ngram_with_decodebook
                    )
                    decoded = decoder(
                        huff0.Huff0NGramPayload(
                            payload=chunk.payload,
                            codebook=codebook.codebook,
                            symbol_count=int(chunk.token_count),
                            escape_bytes=chunk.escape_payload,
                            tail=chunk.tail_payload,
                            hits=chunk.ngram_hits,
                            misses=chunk.ngram_escapes,
                        ),
                        decodebook,
                        codebook.vocab,
                        n=chunk.ngram_n,
                    )
            else:
                raise ValueError(f"Unsupported stream chunk mode {chunk.mode!r}.")
            if len(decoded) != chunk.symbol_count:
                raise ValueError(
                    f"Chunk decoded to {len(decoded)} bytes; expected {chunk.symbol_count}."
                )
            output.extend(decoded)
        if len(output) != group.symbol_count:
            raise ValueError(
                f"Stream decoded to {len(output)} bytes; expected {group.symbol_count}."
            )
        return bytes(output)

    def decompress_tensor(self, compressed):
        if compressed.dtype not in DTYPE_BY_NAME:
            raise ValueError(f"Unsupported dtype in payload: {compressed.dtype!r}")
        if compressed.strategy != "neuralzip":
            return super().decompress_tensor(compressed)

        dtype = DTYPE_BY_NAME[compressed.dtype]
        config = _zipnn_dtype_config(dtype)
        if compressed.element_size != int(config["element_size"]):
            raise ValueError("Payload element size does not match its dtype.")
        if len(compressed.groups) != int(config["num_buffers"]):
            raise ValueError("Payload byte-group count does not match its dtype.")

        raw = bytearray(compressed.raw_bytes)
        for stream_id, group in enumerate(compressed.groups):
            stream = self._decompress_cluster_stream(group)
            if self._is_exponent_stream(config, stream_id):
                stream = restore_2d_stream(
                    stream,
                    compressed.shape,
                    compressed.exponent_layout,
                )
            expected = len(raw[stream_id::compressed.element_size])
            if len(stream) != expected:
                raise ValueError(
                    f"Byte group {stream_id} has {len(stream)} bytes; expected {expected}."
                )
            raw[stream_id::compressed.element_size] = stream

        restored = _zipnn_restored_bytes(bytes(raw), dtype, config)
        return _tensor_from_raw_bytes(restored, dtype, compressed.shape)
