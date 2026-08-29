import time

import numpy as np
import torch

from . import huff0
from .codec_primitives import (
    DTYPE_BY_NAME,
    _byte_group_dtype_config,
    _byte_group_reordered_bytes,
    _tensor_from_raw_bytes,
    dtype_name,
    exponent_distribution,
    jensen_shannon_distance,
    select_fastest_within_ratio,
    select_packed_time_budget,
    tokenize_ngrams,
    topk_vocab,
)
from .payloads import (
    PACKED_CHUNK_METADATA_BYTES,
    RAW_CHUNK_METADATA_BYTES,
    SHARED_CHUNK_METADATA_BYTES,
    ByteGroupPayload,
    CodebookPayload,
    StreamChunk,
    TensorPayload,
)

_TRANSFER_NAME_FALLBACKS = {
    # Tied-output models such as Qwen store this tensor only under the input
    # embedding name. Untied targets may expose the equivalent output head as
    # a second tensor, whose streams can safely reuse the embedding modes.
    "lm_head.weight": ("model.embed_tokens.weight",),
}


def _floating_tensors(named_tensors):
    for name, tensor in sorted(named_tensors, key=lambda item: item[0]):
        if torch.is_tensor(tensor) and tensor.is_floating_point() and tensor.numel():
            yield name, tensor


class NeuralZipCodec:
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

        self._codebook_payloads = []
        self._codebook_payload_by_id = {}
        self._decode_codebooks = {}
        self._computed_cluster_map = {}
        self._clusters_prepared = False
        self._mode_by_tensor_stream = {}
        self._transfer_mode_by_tensor_stream = {}

    def _stream_chunk_size(self, element_size):
        return max(1, self.chunk_size // int(element_size))

    def export_codebooks(self):
        return tuple(self._codebook_payloads)

    def load_codebooks(self, codebooks):
        self._codebook_payload_by_id = {
            int(codebook.codebook_id): codebook for codebook in codebooks
        }
        self._decode_codebooks = {}

    def prepare_clusters(self, named_tensors):
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
        self._computed_cluster_map = computed
        self._clusters_prepared = True

    def _codebook_key(self, name, tensor, stream_id):
        cluster_id = self.cluster_map.get(name, self._computed_cluster_map.get(name))
        return cluster_id, dtype_name(tensor.dtype), int(stream_id)

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

    def _measure_direct_mode(self, mode, element_size, sample_parts):
        """Return exact bytes and time from the production native encoder.

        The sampled logical byte lane is placed in stream zero of a synthetic
        tensor before the timer; the remaining lanes use raw mode and therefore
        contribute the same background work to every candidate.  The timed
        call is the model-wide mixed primitive itself, so shared/packed
        rejection, the strict compression threshold, and direct raw fallback
        are identical to the real direct encoder.
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
            return 0, 0.0

        def encode_once():
            started = time.perf_counter()
            output = huff0.compress_mixed_streams_many(
                tuple(tensor_specs),
                min_saving_bytes=self.min_saving_bytes,
                threads=self.threads,
                compression_threshold=self.compression_threshold,
            )
            return output, time.perf_counter() - started

        # Warm code/data caches once, then use the best of three complete
        # measurements. The minimum estimates algorithmic work without letting
        # scheduler jitter change the frozen family setup.
        native_batch, _ = encode_once()
        seconds = min(encode_once()[1] for _ in range(3))

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

    def _score_mode(
        self,
        mode,
        element_size,
        codebook_payload=None,
        sample_parts=(),
    ):
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
            tokens, _, _ = tokenize_ngrams(
                sample,
                ngram_n,
                vocab,
            )
        if not tokens:
            return None
        counts = symbol_counts
        if counts is None or kind != "shared_huff0":
            counts = np.bincount(np.frombuffer(tokens, dtype=np.uint8), minlength=256)
        try:
            codebook = huff0.build_codebook_from_counts(
                counts,
                prior_count=prior_count,
                max_bits=max_bits,
            )
        except (RuntimeError, ValueError):
            return None
        return {
            "kind": kind,
            "ngram_n": int(ngram_n),
            "vocab": vocab,
            "codebook": codebook,
        }

    def _candidate_payload(self, key, mode, codebook_id=-1):
        _, dtype, stream_id = key
        codebook = mode["codebook"].codebook
        return CodebookPayload(
            codebook_id=int(codebook_id),
            stream_id=stream_id,
            dtype=dtype,
            mode=mode["kind"],
            codebook=codebook,
            ngram_n=int(mode.get("ngram_n", 1)),
            vocab=bytes(mode.get("vocab", b"")),
        )

    def _shared_histogram_cost(
        self,
        mode,
        symbol_counts,
        *,
        total_size,
        shared_chunk_count,
        codebook_payload,
    ):
        """Return a sample-independent upper bound for shared Huff0 bytes.

        The symbol contribution is exact for the complete histogram. Huff0's
        4X frame adds a six-byte jump table and at most one padded byte for
        each of its four bitstreams, hence the ten-byte per-chunk bound. Raw
        fallback can only make the emitted chunks smaller than this bound.
        """

        table = huff0.export_code_table(mode["codebook"].codebook, max_bits=12)
        symbol_bits = sum(
            int(count) * int(length)
            for count, length in zip(
                symbol_counts,
                table.code_lengths,
                strict=True,
            )
        )
        symbol_bytes = (symbol_bits + 7) // 8
        frame_overhead = 10 * int(shared_chunk_count)
        shared_chunks = (
            symbol_bytes
            + frame_overhead
            + int(shared_chunk_count) * SHARED_CHUNK_METADATA_BYTES
        )
        fallback_chunks = (
            int(total_size) + int(shared_chunk_count) * RAW_CHUNK_METADATA_BYTES
        )
        return int(codebook_payload.metadata_bytes) + min(
            shared_chunks,
            fallback_chunks,
        )

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
        self._score_mode(
            raw_mode,
            element_size,
            sample_parts=sample_parts,
        )
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
                self._score_mode(
                    shared,
                    element_size,
                    payload,
                    sample_parts,
                )

        # A frozen setup only selects precomputed modes or raw storage.
        base_candidates = [raw_mode]
        if shared is not None:
            base_candidates.append(shared)
        scale = total_size / len(sample)

        def estimated_cost(candidate):
            if "full_cost" in candidate:
                return float(candidate["full_cost"])
            fixed_cost = float(candidate.get("fixed_cost", 0))
            return (
                fixed_cost
                + max(0.0, float(candidate["sample_cost"]) - fixed_cost) * scale
            )

        def estimated_seconds(candidate):
            return float(candidate["sample_seconds"]) * scale

        base = select_fastest_within_ratio(
            base_candidates,
            estimated_bytes=estimated_cost,
            estimated_seconds=estimated_seconds,
            ratio_tolerance=self.speed_ratio_tolerance,
        )
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
                    packed,
                    element_size,
                    payload,
                    sample_parts,
                )
                packed_candidates.append(packed)

        improving_packed_candidates = [
            candidate
            for candidate in packed_candidates
            if estimated_cost(candidate) < base_cost
        ]
        packed = None
        if improving_packed_candidates:
            # Reject non-saving packed modes before applying the speed
            # tolerance. Otherwise a faster n-gram that is slightly larger
            # than the selected base can hide another n that saves bytes. The
            # surviving opportunity remains time-aware here and is then
            # subject to the model-wide packed-time budget below.
            packed = select_fastest_within_ratio(
                improving_packed_candidates,
                estimated_bytes=estimated_cost,
                estimated_seconds=estimated_seconds,
                ratio_tolerance=self.speed_ratio_tolerance,
            )
        opportunity = None
        if packed is not None:
            packed_cost = estimated_cost(packed)
            packed_seconds = estimated_seconds(packed)
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

    def _finalize_mode(self, key, selected):
        if selected["kind"] == "raw":
            return selected
        codebook_id = len(self._codebook_payloads)
        selected["codebook_id"] = codebook_id
        payload = self._candidate_payload(key, selected, codebook_id=codebook_id)
        self._codebook_payloads.append(payload)
        return selected

    @staticmethod
    def _transfer_plan_key(name, tensor, stream_id):
        return str(name), dtype_name(tensor.dtype), int(stream_id)

    def _prepare_tensor_modes(self, named_tensors, mode_by_key):
        for name, tensor in _floating_tensors(named_tensors):
            config = _byte_group_dtype_config(tensor.dtype)
            for stream_id in range(int(config["element_size"])):
                key = self._codebook_key(name, tensor, stream_id)
                # The source cluster decision already compared compressed size
                # and direct-encoding time under the configured tolerance.
                # Re-probing every tensor with a ratio-only local fallback here
                # would undo that decision and prune all shared codebooks.
                selected = mode_by_key[key]

                plan_key = self._transfer_plan_key(name, tensor, stream_id)
                self._mode_by_tensor_stream[plan_key] = selected

    def prepare_direct_transfer_target(self, named_tensors):
        """Map a family target to a reusable source setup.

        This path reads only target metadata (name, dtype, size, and number of
        byte streams). It does not inspect tensor values, compute exponent
        histograms or Jensen--Shannon distances, or evaluate alternative modes.
        The target can encode immediately from its own tensor buffers while
        retaining the source model's modes, vocabularies, and codebooks. No
        target data are copied or retained by this mapping step. Exact tensor
        names are preferred; explicit structural aliases cover equivalent
        tensors represented differently by related architectures.
        """
        if not self._clusters_prepared:
            raise RuntimeError("Source clusters must be prepared before transfer.")
        if not self._mode_by_tensor_stream:
            raise RuntimeError(
                "Source codebooks and modes must be built before transfer."
            )

        transfer_modes = {}
        target_tensors = 0
        fallback_tensors = 0
        target_symbols = 0
        fallback_symbols = 0
        fallback_assignments = {}

        for name, tensor in _floating_tensors(named_tensors):
            config = _byte_group_dtype_config(tensor.dtype)
            stream_ids = tuple(range(int(config["element_size"])))
            tensor_name = str(name)
            dtype = dtype_name(tensor.dtype)
            candidates = (tensor_name, *_TRANSFER_NAME_FALLBACKS.get(tensor_name, ()))
            source_name = next(
                (
                    candidate
                    for candidate in candidates
                    if all(
                        (candidate, dtype, stream_id) in self._mode_by_tensor_stream
                        for stream_id in stream_ids
                    )
                ),
                None,
            )
            if source_name is None:
                raise ValueError(
                    "Direct transfer has no compatible source tensor for "
                    f"{tensor_name}:{dtype}."
                )

            for stream_id in stream_ids:
                plan_key = (tensor_name, dtype, stream_id)
                transfer_modes[plan_key] = self._mode_by_tensor_stream[
                    (source_name, dtype, stream_id)
                ]
                if source_name != tensor_name:
                    fallback_assignments[f"{tensor_name}:{dtype}:stream{stream_id}"] = (
                        f"{source_name}:{dtype}:stream{stream_id}"
                    )

            symbols = int(tensor.numel())
            target_tensors += 1
            target_symbols += symbols
            if source_name != tensor_name:
                fallback_tensors += 1
                fallback_symbols += symbols

        self._transfer_mode_by_tensor_stream = transfer_modes
        stats = {
            "mapping": (
                "exact source tensor name, dtype, and stream"
                if not fallback_assignments
                else "exact name or explicit structural alias, dtype, and stream"
            ),
            "target_tensors": target_tensors,
            "matched_tensors": target_tensors,
            "exact_tensors": target_tensors - fallback_tensors,
            "fallback_tensors": fallback_tensors,
            "target_exponents": target_symbols,
            "matched_exponents": target_symbols,
            "exact_exponents": target_symbols - fallback_symbols,
            "fallback_exponents": fallback_symbols,
            "missing_streams": 0,
            "fallback_streams": len(fallback_assignments),
            "fallback_assignments": dict(sorted(fallback_assignments.items())),
            "target_histograms_computed": False,
            "target_js_matching": False,
            "target_mode_probing": False,
        }
        return stats

    def _collect_codebook_samples(self, named_tensors):
        """Collect bounded probes and complete exponent histograms.

        Bounded byte samples remain only for direct-time probes and packed
        vocabulary/cost discovery. Shared exponent codebooks and their symbol
        costs use all 256 bins, so their ratio decision no longer depends on
        the sample budget or tensor traversal position.
        """

        sample_parts = {}
        sample_sizes = {}
        total_sizes = {}
        stream_info = {}
        exponent_counts = {}
        chunk_counts = {}
        for name, tensor in _floating_tensors(named_tensors):
            original = tensor.detach().cpu().contiguous()
            config = _byte_group_dtype_config(original.dtype)
            reordered = _byte_group_reordered_bytes(original, config)
            element_size = int(config["element_size"])
            exponent_stream = element_size - 1 if config["bit_reorder"] else -1
            for stream_id in range(int(config["element_size"])):
                is_exponent = stream_id == exponent_stream
                stream = reordered[stream_id::element_size]
                key = self._codebook_key(name, original, stream_id)
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
                        np.frombuffer(stream, dtype=np.uint8),
                        minlength=256,
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
        return (
            sample_parts,
            total_sizes,
            stream_info,
            exponent_counts,
            chunk_counts,
        )

    def build_codebooks_and_select_modes(self, named_tensors):
        if not self._clusters_prepared:
            raise RuntimeError(
                "Clusters must be prepared before building NeuralZip codebooks."
            )
        named_tensors = list(named_tensors)
        (
            sample_parts,
            total_sizes,
            stream_info,
            exponent_counts,
            chunk_counts,
        ) = self._collect_codebook_samples(named_tensors)

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

        exponent_source_bytes = sum(
            evaluation["total_size"]
            for evaluation in evaluations
            if evaluation["is_exponent"]
        )
        packed_source_budget = int(
            exponent_source_bytes * self.packed_source_budget_ratio
        )
        time_selected_keys = select_packed_time_budget(
            evaluations,
            packed_time_budget_ratio=self.packed_time_budget_ratio,
        )
        packed_used = 0
        selected_packed_keys = set()
        opportunities = [
            evaluation
            for evaluation in evaluations
            if evaluation["opportunity"] is not None
            and evaluation["key"] in time_selected_keys
        ]
        opportunities.sort(
            key=lambda evaluation: (
                -evaluation["opportunity"]["savings_per_source_byte"],
                -evaluation["opportunity"]["saved_bytes"],
                evaluation["key"],
            )
        )
        for evaluation in opportunities:
            opportunity = evaluation["opportunity"]
            source_bytes = opportunity["source_bytes"]
            if packed_used + source_bytes > packed_source_budget:
                continue
            selected_packed_keys.add(evaluation["key"])
            packed_used += source_bytes

        mode_by_key = {}
        for evaluation in evaluations:
            selected = evaluation["base"]
            if evaluation["key"] in selected_packed_keys:
                selected = evaluation["opportunity"]["mode"]
            selected = self._finalize_mode(evaluation["key"], selected)
            mode_by_key[evaluation["key"]] = selected

        self._prepare_tensor_modes(named_tensors, mode_by_key)

    def _mode_for_tensor_stream(self, name, tensor, stream_id):
        """Return the prepared mode for one concrete tensor byte stream."""
        plan_key = self._transfer_plan_key(name, tensor, stream_id)
        return self._transfer_mode_by_tensor_stream[plan_key]

    def _compress_direct_native_batch(self, tensor_specs):
        """Encode one input-size-windowed tensor batch through one job queue."""
        native_specs = [
            (
                tensor.reshape(-1).view(torch.uint8).numpy(),
                int(config["element_size"]),
                int(config["bit_reorder"]),
                self._stream_chunk_size(int(config["element_size"])),
                tensor_modes,
            )
            for _, tensor, config, tensor_modes in tensor_specs
        ]
        native_batch = huff0.compress_mixed_streams_many(
            native_specs,
            min_saving_bytes=self.min_saving_bytes,
            threads=self.threads,
            compression_threshold=self.compression_threshold,
        )
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
        # the fused native encoder. Every public encoding entrypoint reaches
        # this path, so setup selection and runtime fallback share one backend.
        outputs = []
        batch = []
        batch_bytes = 0

        def flush_batch():
            nonlocal batch, batch_bytes
            if batch:
                outputs.extend(self._compress_direct_native_batch(batch))
                batch = []
                batch_bytes = 0

        for name, tensor in named_tensors:
            config = _byte_group_dtype_config(tensor.dtype)
            raw_bytes = int(tensor.numel()) * int(config["element_size"])
            if batch and batch_bytes + raw_bytes > self.direct_batch_bytes:
                flush_batch()
            # Copy a GPU/non-contiguous tensor only after the preceding
            # window has released its native scratch buffers.
            original = tensor.detach().cpu().contiguous()
            stream_count = int(config["element_size"])
            if original.numel() == 0:
                tensor_modes = ({"kind": "raw"},) * stream_count
            else:
                tensor_modes = tuple(
                    self._mode_for_tensor_stream(name, original, stream_id)
                    for stream_id in range(stream_count)
                )
            batch.append((name, original, config, tensor_modes))
            batch_bytes += raw_bytes
            # One oversized tensor is still bounded to that tensor rather
            # than being retained alongside the following tensors.
            if batch_bytes >= self.direct_batch_bytes:
                flush_batch()
        flush_batch()
        return tuple(outputs)

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
        groups = []
        for planned_mode, native_chunks in zip(
            tensor_modes, native_groups, strict=True
        ):
            chunks = []
            for native_chunk in native_chunks:
                final_kind, payload, symbol_count, escapes, tail = native_chunk
                final_kind = str(final_kind)
                chunks.append(
                    StreamChunk(
                        mode=final_kind,
                        symbol_count=int(symbol_count),
                        payload=payload,
                        codebook_id=(
                            planned_mode.get("codebook_id")
                            if final_kind != "raw"
                            else None
                        ),
                        escape_payload=escapes,
                        tail_payload=tail,
                    )
                )
            groups.append(
                ByteGroupPayload(
                    chunks=tuple(chunks),
                )
            )

        return TensorPayload(
            name=name,
            shape=tuple(int(dim) for dim in tensor.shape),
            dtype=dtype_name(tensor.dtype),
            element_size=element_size,
            numel=int(tensor.numel()),
            groups=tuple(groups),
            bit_reorder=int(config["bit_reorder"]),
        )

    def _decode_assets(self, chunk):
        codebook = self._codebook_payload_by_id[chunk.codebook_id]
        decodebook = self._decode_codebooks.get(chunk.codebook_id)
        if decodebook is None:
            decodebook = huff0.build_decodebook(codebook.codebook)
            self._decode_codebooks[chunk.codebook_id] = decodebook
        return codebook, decodebook

    def _native_decode_spec(self, compressed):
        """Translate one stable payload into the native mixed-stream ABI."""

        dtype = DTYPE_BY_NAME[compressed.dtype]
        config = _byte_group_dtype_config(dtype)
        native_groups = []
        for group in compressed.groups:
            native_chunks = []
            for chunk in group.chunks:
                mode = str(chunk.mode)
                decodebook_handle = None
                vocab = b""
                ngram_n = 1
                escape_payload = b""
                tail_payload = b""
                if mode in {"shared_huff0", "packed_huff0"}:
                    codebook, decodebook = self._decode_assets(chunk)
                    decodebook_handle = decodebook
                    if mode == "packed_huff0":
                        vocab = codebook.vocab
                        ngram_n = int(codebook.ngram_n)
                        escape_payload = chunk.escape_payload
                        tail_payload = chunk.tail_payload
                native_chunks.append(
                    (
                        mode,
                        chunk.payload,
                        int(chunk.symbol_count),
                        decodebook_handle,
                        vocab,
                        ngram_n,
                        escape_payload,
                        tail_payload,
                    )
                )
            native_groups.append(tuple(native_chunks))
        return (
            int(config["element_size"]),
            bool(config["bit_reorder"]),
            int(compressed.numel),
            tuple(native_groups),
        ), dtype

    def _decompress_direct_native_batch(self, payloads):
        prepared = [self._native_decode_spec(compressed) for compressed in payloads]
        raw_buffers = huff0.decompress_mixed_streams_many(
            tuple(spec for spec, _ in prepared),
            threads=self.threads,
        )
        return [
            _tensor_from_raw_bytes(raw, dtype, compressed.shape)
            for compressed, (_, dtype), raw in zip(
                payloads, prepared, raw_buffers, strict=True
            )
        ]

    def decompress_tensors(self, payloads):
        """Decode tensors through input-size-windowed native queues."""

        outputs = []
        batch = []
        batch_bytes = 0

        def flush_batch():
            nonlocal batch, batch_bytes
            if batch:
                outputs.extend(self._decompress_direct_native_batch(batch))
                batch = []
                batch_bytes = 0

        for compressed in payloads:
            raw_bytes = int(compressed.raw_bytes)
            if batch and batch_bytes + raw_bytes > self.direct_batch_bytes:
                flush_batch()
            batch.append(compressed)
            batch_bytes += raw_bytes
            if batch_bytes >= self.direct_batch_bytes:
                flush_batch()
        flush_batch()
        return tuple(outputs)
