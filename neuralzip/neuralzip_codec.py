import re
import time

import numpy as np
import torch

from . import huff0
from .codec_primitives import (
    DTYPE_BY_NAME,
    _byte_group_dtype_config,
    _byte_group_reordered_bytes,
    _byte_group_restored_bytes,
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


def _layer_signature(name):
    return ".".join(re.sub(r"\d+", "#", part) for part in str(name).split("."))


def _packed_token_count(symbol_count, tail_payload, ngram_n):
    symbol_count = int(symbol_count)
    ngram_n = int(ngram_n)
    if symbol_count < 0 or not 2 <= ngram_n <= 8:
        raise ValueError("Packed Huff0 symbol count and tail are inconsistent.")
    token_count, tail_bytes = divmod(symbol_count, ngram_n)
    if tail_bytes != len(tail_payload):
        raise ValueError("Packed Huff0 symbol count and tail are inconsistent.")
    return token_count


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
        if self.min_saving_bytes < 0:
            raise ValueError("min_saving_bytes must be non-negative.")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        if not 0 < self.compression_threshold <= 1:
            raise ValueError("compression_threshold must be in (0, 1].")
        if self.threads < 0:
            raise ValueError("threads must be non-negative.")

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

        if self.prior_count <= 0 or self.exponent_prior_count <= 0:
            raise ValueError("Transferable Huffman prior counts must be positive.")
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
        if not 0 <= self.packed_source_budget_ratio <= 1:
            raise ValueError("packed_source_budget_ratio must be in [0, 1].")
        if self.packed_time_budget_ratio < 0:
            raise ValueError("packed_time_budget_ratio must be non-negative.")
        if self.speed_ratio_tolerance < 0:
            raise ValueError("speed_ratio_tolerance must be non-negative.")

        self.reset_compression()
        self.reset_decompression()

    def _stream_chunk_size(self, element_size):
        return max(1, self.chunk_size // int(element_size))

    def reset_compression(self):
        self._mode_by_key = {}
        self._codebook_payloads = []
        self._codebook_payload_by_id = {}
        self._computed_cluster_map = {}
        self._clusters_prepared = False
        self._next_codebook_id = 0
        self._mode_by_tensor_stream = {}
        self._transfer_mode_by_tensor_stream = {}
        self._transfer_cluster_by_name = {}
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
            self._decode_codebooks[codebook_id] = huff0.build_decodebook(
                codebook.codebook
            )

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
        for name, tensor in sorted(named_tensors, key=lambda item: item[0]):
            if (
                not torch.is_tensor(tensor)
                or not tensor.is_floating_point()
                or tensor.numel() == 0
            ):
                continue

            distribution = exponent_distribution(tensor)
            weight = int(tensor.numel())
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

    def _codebook_key(self, name, tensor, stream_id):
        return self.cluster_id(name), dtype_name(tensor.dtype), int(stream_id)

    @staticmethod
    def _is_exponent_stream(config, stream_id):
        return bool(config["bit_reorder"]) and int(stream_id) == (
            int(config["element_size"]) - 1
        )

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
            encoded = huff0.encode_with_codebook(
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
            encoded = huff0.encode_ngram_with_codebook(
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
                escape_payload=encoded.escape_bytes,
                tail_payload=encoded.tail,
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
        total = codebook_payload.metadata_bytes if codebook_payload is not None else 0
        step = self._source_chunk_size(element_size, mode)
        for part in parts:
            for start in range(0, len(part), step):
                chunk = part[start : start + step]
                encoded = (
                    None if mode["kind"] == "raw" else self._encode_chunk(chunk, mode)
                )
                if encoded is not None:
                    total += encoded.compressed_bytes
                else:
                    # Post-setup encoding never builds a target-local tree.
                    # Raw is the exact runtime fallback for every rejected
                    # precomputed shared or packed representation.
                    total += self._raw_chunk(chunk).compressed_bytes
        return total

    def _measure_direct_mode_seconds(self, mode, element_size, sample_parts):
        """Measure target-dependent work in the uncached direct encoder.

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
            return 0.0

        def encode_once():
            started = time.perf_counter()
            huff0.compress_mixed_streams_many(
                tuple(tensor_specs),
                min_saving_bytes=self.min_saving_bytes,
                threads=self.threads,
                compression_threshold=self.compression_threshold,
            )
            return time.perf_counter() - started

        # Warm code/data caches once, then use the best of three complete
        # measurements. The minimum estimates algorithmic work without letting
        # scheduler jitter change the frozen family setup.
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
        try:
            if symbol_counts is not None and kind == "shared_huff0":
                codebook = huff0.build_codebook_from_counts(
                    symbol_counts,
                    prior_count=prior_count,
                    max_bits=max_bits,
                )
            else:
                codebook = huff0.build_codebook(
                    tokens,
                    prior_count=prior_count,
                    max_bits=max_bits,
                )
        except ValueError:
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

        table = huff0.export_code_table(mode["codebook"], max_bits=12)
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
        symbol_counts=None,
        chunk_counts=None,
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
        if chunk_counts is not None:
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
            if symbol_counts is not None and chunk_counts is not None:
                shared["fixed_cost"] = payload.metadata_bytes
                shared["sample_seconds"] = self._measure_direct_mode_seconds(
                    shared,
                    element_size,
                    tuple(sample_parts),
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
                    sample,
                    shared,
                    element_size,
                    payload,
                    sample_parts,
                )

        # A frozen setup only selects precomputed modes or raw storage.
        base_candidates = [raw_mode]
        if shared is not None:
            base_candidates.append(shared)
        sample_size = len(sample)

        def estimated_cost(candidate):
            if "full_cost" in candidate:
                return float(candidate["full_cost"])
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
                    sample,
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
            if saved_bytes > 0:
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
        codebook_id = self._next_codebook_id
        self._next_codebook_id += 1
        selected["codebook_id"] = codebook_id
        payload = self._candidate_payload(key, selected, codebook_id=codebook_id)
        self._codebook_payloads.append(payload)
        self._codebook_payload_by_id[codebook_id] = payload
        self._decode_codebooks[codebook_id] = huff0.build_decodebook(payload.codebook)
        return selected

    def prepare_clusters(self, named_tensors):
        self._prepare_clusters(named_tensors)
        self._clusters_prepared = True

    @staticmethod
    def _transfer_plan_key(name, tensor, stream_id):
        return str(name), dtype_name(tensor.dtype), int(stream_id)

    def _prepare_tensor_modes(self, named_tensors):
        used_codebook_ids = set()
        for name, tensor in sorted(named_tensors, key=lambda item: item[0]):
            if (
                not torch.is_tensor(tensor)
                or not tensor.is_floating_point()
                or tensor.numel() == 0
            ):
                continue

            original = tensor.detach().cpu().contiguous()
            config = _byte_group_dtype_config(original.dtype)
            for stream_id in range(int(config["element_size"])):
                key = self._codebook_key(name, original, stream_id)
                # The source cluster decision already compared compressed size
                # and direct-encoding time under the configured tolerance.
                # Re-probing every tensor with a ratio-only local fallback here
                # would undo that decision and prune all shared codebooks.
                selected = self._mode_by_key[key]

                plan_key = self._transfer_plan_key(name, original, stream_id)
                self._mode_by_tensor_stream[plan_key] = selected
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
                        "kind": "raw",
                        "fixed_cost": 0,
                    }

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

        named_tensors = list(named_tensors)
        transfer_modes = {}
        transfer_clusters = {}
        missing = []
        target_tensors = 0
        matched_tensors = 0
        exact_tensors = 0
        fallback_tensors = 0
        target_symbols = 0
        matched_symbols = 0
        exact_symbols = 0
        fallback_symbols = 0
        fallback_assignments = {}

        for name, tensor in sorted(named_tensors, key=lambda item: item[0]):
            if (
                not torch.is_tensor(tensor)
                or not tensor.is_floating_point()
                or tensor.numel() == 0
            ):
                continue

            config = _byte_group_dtype_config(tensor.dtype)
            stream_ids = tuple(range(int(config["element_size"])))
            source_name = str(name)
            exact_keys = tuple(
                self._transfer_plan_key(name, tensor, stream_id)
                for stream_id in stream_ids
            )
            tensor_exact = all(key in self._mode_by_tensor_stream for key in exact_keys)
            if not tensor_exact:
                for candidate in _TRANSFER_NAME_FALLBACKS.get(str(name), ()):
                    candidate_keys = tuple(
                        (candidate, dtype_name(tensor.dtype), stream_id)
                        for stream_id in stream_ids
                    )
                    if all(
                        key in self._mode_by_tensor_stream for key in candidate_keys
                    ):
                        source_name = candidate
                        break

            plan = []
            for stream_id in stream_ids:
                plan_key = self._transfer_plan_key(name, tensor, stream_id)
                source_key = (source_name, plan_key[1], plan_key[2])
                source_mode = self._mode_by_tensor_stream.get(source_key)
                if source_mode is None:
                    missing.append(plan_key)
                    source_mode = {"kind": "raw", "fixed_cost": 0}
                elif source_name != str(name):
                    fallback_assignments[f"{name}:{plan_key[1]}:stream{stream_id}"] = (
                        f"{source_name}:{plan_key[1]}:stream{stream_id}"
                    )
                plan.append((plan_key, source_mode))

            tensor_matched = all(plan_key not in missing for plan_key, _ in plan)
            tensor_exact = tensor_matched and source_name == str(name)
            symbols = int(tensor.numel())
            target_tensors += 1
            target_symbols += symbols
            if tensor_matched:
                matched_tensors += 1
                matched_symbols += symbols
                if tensor_exact:
                    exact_tensors += 1
                    exact_symbols += symbols
                else:
                    fallback_tensors += 1
                    fallback_symbols += symbols

            cluster_name = str(name)
            if tensor_matched and not tensor_exact:
                cluster_name = source_name
            source_cluster = self.cluster_map.get(cluster_name)
            if source_cluster is None:
                source_cluster = self._computed_cluster_map.get(cluster_name)
            transfer_clusters[str(name)] = (
                str(source_cluster)
                if tensor_matched and source_cluster is not None
                else f"transfer-local:{name}"
            )

            for plan_key, selected in plan:
                transfer_modes[plan_key] = selected

        if missing:
            preview = ", ".join(
                f"{name}:{dtype}:stream{stream_id}"
                for name, dtype, stream_id in missing[:8]
            )
            suffix = "" if len(missing) <= 8 else f" (+{len(missing) - 8} more)"
            raise ValueError(
                "Direct transfer requires a compatible source mode for every target "
                f"stream; missing {preview}{suffix}."
            )

        self._transfer_mode_by_tensor_stream = transfer_modes
        self._transfer_cluster_by_name = transfer_clusters
        stats = {
            "mapping": (
                "exact source tensor name, dtype, and stream"
                if not fallback_assignments
                else "exact name or explicit structural alias, dtype, and stream"
            ),
            "target_tensors": target_tensors,
            "matched_tensors": matched_tensors,
            "exact_tensors": exact_tensors,
            "fallback_tensors": fallback_tensors,
            "target_exponents": target_symbols,
            "matched_exponents": matched_symbols,
            "exact_exponents": exact_symbols,
            "fallback_exponents": fallback_symbols,
            "missing_streams": len(missing),
            "fallback_streams": len(fallback_assignments),
            "fallback_assignments": dict(sorted(fallback_assignments.items())),
            "target_histograms_computed": False,
            "target_js_matching": False,
            "target_mode_probing": False,
        }
        self._transfer_active = True
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
        for name, tensor in sorted(named_tensors, key=lambda item: item[0]):
            if not torch.is_tensor(tensor) or not tensor.is_floating_point():
                continue
            original = tensor.detach().cpu().contiguous()
            config = _byte_group_dtype_config(original.dtype)
            reordered = _byte_group_reordered_bytes(original, config)
            element_size = int(config["element_size"])
            for stream_id in range(int(config["element_size"])):
                is_exponent = self._is_exponent_stream(config, stream_id)
                stream = reordered[stream_id::element_size]
                key = self._codebook_key(name, original, stream_id)
                total_sizes[key] = total_sizes.get(key, 0) + len(stream)
                stream_info[key] = (element_size, is_exponent)
                raw_step = self._stream_chunk_size(element_size)
                shared_step = min(raw_step, huff0.max_block_size())
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
                    chunk_counts=chunk_counts.get(key),
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
            if self.packed_source_budget_ratio == 0:
                break
            opportunity = evaluation["opportunity"]
            source_bytes = opportunity["source_bytes"]
            if packed_used + source_bytes > packed_source_budget:
                continue
            selected_packed_keys.add(evaluation["key"])
            packed_used += source_bytes

        for evaluation in evaluations:
            selected = evaluation["base"]
            if evaluation["key"] in selected_packed_keys:
                selected = evaluation["opportunity"]["mode"]
            selected = self._finalize_mode(evaluation["key"], selected)
            self._mode_by_key[evaluation["key"]] = selected

        self._prepare_tensor_modes(named_tensors)
        self.load_codebooks(self._codebook_payloads)

    def _mode_for_tensor_stream(self, name, tensor, stream_id):
        """Return the prepared mode for one concrete tensor byte stream."""
        if self._transfer_active:
            plan_key = self._transfer_plan_key(name, tensor, stream_id)
            return self._transfer_mode_by_tensor_stream.get(
                plan_key,
                {"kind": "raw", "fixed_cost": 0},
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
        if len(native_batch) != len(tensor_specs):
            raise RuntimeError("Mixed-stream encoder returned an invalid tensor count.")
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
        for name, tensor in named_tensors:
            if not torch.is_tensor(tensor) or not tensor.is_floating_point():
                raise TypeError("NeuralZipCodec only compresses floating tensors.")
            config = _byte_group_dtype_config(tensor.dtype)
            raw_bytes = int(tensor.numel()) * int(config["element_size"])
            if batch and batch_bytes + raw_bytes > self.direct_batch_bytes:
                outputs.extend(self._compress_direct_native_batch(batch))
                batch = []
                batch_bytes = 0
            # Copy a GPU/non-contiguous tensor only after the preceding
            # window has released its native scratch buffers.
            original = tensor.detach().cpu().contiguous()
            stream_count = int(config["element_size"])
            tensor_modes = tuple(
                self._mode_for_tensor_stream(name, original, stream_id)
                for stream_id in range(stream_count)
            )
            batch.append((name, original, config, tensor_modes))
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
        stream_count = element_size
        if len(native_groups) != stream_count:
            raise RuntimeError("Mixed-stream encoder returned an invalid stream count.")

        allowed_finals = {
            "raw": {"raw"},
            "shared_huff0": {"shared_huff0", "raw"},
            "packed_huff0": {"packed_huff0", "raw"},
        }
        groups = []
        for planned_mode, native_chunks in zip(
            tensor_modes, native_groups, strict=True
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
                ) = native_chunk
                final_kind = str(final_kind)
                symbol_count = int(symbol_count)
                if final_kind not in allowed_finals[planned_kind]:
                    raise RuntimeError(
                        f"Mixed-stream encoder changed {planned_kind!r} into the invalid "
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
                    codebook_id = planned_mode.get("codebook_id")
                    codebook = self._codebook_payload_by_id.get(codebook_id)
                    if codebook is None:
                        raise RuntimeError(
                            f"Packed Huff0 mode references missing codebook {codebook_id}."
                        )
                    if int(ngram_n) != int(codebook.ngram_n):
                        raise RuntimeError(
                            "Mixed-stream encoder returned an n-gram width that "
                            "disagrees with its codebook."
                        )
                    expected_tokens = _packed_token_count(
                        symbol_count, tail, codebook.ngram_n
                    )
                    if int(token_count) != expected_tokens:
                        raise RuntimeError(
                            "Mixed-stream encoder returned an invalid packed token "
                            "count."
                        )
                    chunks.append(
                        StreamChunk(
                            mode=final_kind,
                            symbol_count=symbol_count,
                            payload=payload,
                            codebook_id=codebook_id,
                            escape_payload=escapes,
                            tail_payload=tail,
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
                    "Mixed-stream symbol count does not match the tensor."
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

    def _decompress_cluster_stream(self, group):
        output = bytearray()
        for chunk in group.chunks:
            if chunk.mode == "raw":
                decoded = chunk.payload
            elif chunk.mode in {"shared_huff0", "packed_huff0"}:
                codebook = self._codebook_payload_by_id.get(chunk.codebook_id)
                decodebook = self._decode_codebooks.get(chunk.codebook_id)
                if codebook is None or decodebook is None:
                    raise ValueError(f"Missing Huff0 codebook {chunk.codebook_id}.")
                if chunk.mode == "shared_huff0":
                    if codebook.mode != "shared_huff0":
                        raise ValueError(
                            "Shared Huff0 chunk references a non-shared codebook."
                        )
                    decoded = huff0.decompress_with_decodebook(
                        huff0.Huff0Payload(
                            payload=chunk.payload,
                            symbol_count=chunk.symbol_count,
                        ),
                        decodebook,
                    )
                else:
                    if codebook.mode != "packed_huff0":
                        raise ValueError(
                            "Packed Huff0 chunk references a non-packed codebook."
                        )
                    token_count = _packed_token_count(
                        chunk.symbol_count,
                        chunk.tail_payload,
                        codebook.ngram_n,
                    )
                    decoded = huff0.decompress_ngram_with_decodebook(
                        huff0.Huff0NGramPayload(
                            payload=chunk.payload,
                            symbol_count=token_count,
                            escape_bytes=chunk.escape_payload,
                            tail=chunk.tail_payload,
                        ),
                        decodebook,
                        codebook.vocab,
                        n=codebook.ngram_n,
                    )
            else:
                raise ValueError(f"Unsupported stream chunk mode {chunk.mode!r}.")
            if len(decoded) != chunk.symbol_count:
                raise ValueError(
                    f"Chunk decoded to {len(decoded)} bytes; expected {chunk.symbol_count}."
                )
            output.extend(decoded)
        return bytes(output)

    @staticmethod
    def _native_decode_eligible(compressed):
        """Return whether the first-stage native decoder supports a payload."""

        supported = {"raw", "shared_huff0"}
        return all(
            chunk.mode in supported
            for group in compressed.groups
            for chunk in group.chunks
        )

    def _validate_decompression_payload(self, compressed):
        if compressed.dtype not in DTYPE_BY_NAME:
            raise ValueError(f"Unsupported dtype in payload: {compressed.dtype!r}")

        dtype = DTYPE_BY_NAME[compressed.dtype]
        config = _byte_group_dtype_config(dtype)
        if compressed.element_size != int(config["element_size"]):
            raise ValueError("Payload element size does not match its dtype.")
        if compressed.bit_reorder != int(config["bit_reorder"]):
            raise ValueError("Payload bit reorder does not match its dtype.")
        if len(compressed.groups) != int(config["element_size"]):
            raise ValueError("Payload byte-group count does not match its dtype.")
        if int(compressed.numel) < 0:
            raise ValueError("Payload element count must be non-negative.")
        for stream_id, group in enumerate(compressed.groups):
            symbols = sum(int(chunk.symbol_count) for chunk in group.chunks)
            if symbols != int(compressed.numel):
                raise ValueError(
                    f"Byte group {stream_id} describes {symbols} symbols; "
                    f"expected {compressed.numel}."
                )
        return dtype, config

    def _native_decode_spec(self, compressed):
        """Translate one stable payload into the sharing-only native ABI."""

        dtype, config = self._validate_decompression_payload(compressed)
        native_groups = []
        for group in compressed.groups:
            native_chunks = []
            for chunk in group.chunks:
                mode = str(chunk.mode)
                decodebook_handle = None
                if mode == "shared_huff0":
                    codebook = self._codebook_payload_by_id.get(chunk.codebook_id)
                    decodebook = self._decode_codebooks.get(chunk.codebook_id)
                    if codebook is None or decodebook is None:
                        raise ValueError(f"Missing Huff0 codebook {chunk.codebook_id}.")
                    if codebook.mode != "shared_huff0":
                        raise ValueError(
                            "Shared Huff0 chunk references a non-shared codebook."
                        )
                    decodebook_handle = decodebook.handle
                native_chunks.append(
                    (
                        mode,
                        chunk.payload,
                        int(chunk.symbol_count),
                        decodebook_handle,
                    )
                )
            native_groups.append(tuple(native_chunks))
        return (
            int(config["element_size"]),
            int(config["bit_reorder"]),
            int(compressed.numel),
            tuple(native_groups),
        ), dtype

    def _decompress_direct_native_batch(self, payloads):
        prepared = [
            (compressed, *self._native_decode_spec(compressed))
            for compressed in payloads
        ]
        raw_buffers = huff0.decompress_mixed_streams_many(
            tuple(spec for _, spec, _ in prepared),
            threads=self.threads,
        )
        if len(raw_buffers) != len(prepared):
            raise RuntimeError("Mixed-stream decoder returned an invalid tensor count.")
        outputs = []
        for (compressed, _, dtype), raw in zip(
            prepared,
            raw_buffers,
            strict=True,
        ):
            if len(raw) != int(compressed.raw_bytes):
                raise RuntimeError(
                    f"Decoded tensor {compressed.name!r} has {len(raw)} bytes; "
                    f"expected {compressed.raw_bytes}."
                )
            outputs.append(_tensor_from_raw_bytes(raw, dtype, compressed.shape))
        return outputs

    def decompress_tensors(self, payloads):
        """Decode supported tensors through input-size-windowed native queues."""

        payloads = tuple(payloads)
        outputs = []
        batch = []
        batch_bytes = 0

        def flush_native_batch():
            nonlocal batch, batch_bytes
            if batch:
                outputs.extend(self._decompress_direct_native_batch(batch))
                batch = []
                batch_bytes = 0

        for compressed in payloads:
            if not self._native_decode_eligible(compressed):
                flush_native_batch()
                outputs.append(self._decompress_tensor_python(compressed))
                continue

            raw_bytes = int(compressed.raw_bytes)
            if batch and batch_bytes + raw_bytes > self.direct_batch_bytes:
                flush_native_batch()
            batch.append(compressed)
            batch_bytes += raw_bytes
            if batch_bytes >= self.direct_batch_bytes:
                flush_native_batch()
        flush_native_batch()
        return tuple(outputs)

    def decompress_tensor(self, compressed):
        if self._native_decode_eligible(compressed):
            return self.decompress_tensors((compressed,))[0]
        return self._decompress_tensor_python(compressed)

    def _decompress_tensor_python(self, compressed):
        """Decode current packed-mode payloads outside the sharing-only ABI."""

        dtype, config = self._validate_decompression_payload(compressed)

        raw = bytearray(compressed.raw_bytes)
        for stream_id, group in enumerate(compressed.groups):
            stream = self._decompress_cluster_stream(group)
            expected = int(compressed.numel)
            if len(stream) != expected:
                raise ValueError(
                    f"Byte group {stream_id} has {len(stream)} bytes; expected {expected}."
                )
            raw[stream_id :: compressed.element_size] = stream

        restored = _byte_group_restored_bytes(bytes(raw), dtype, config)
        return _tensor_from_raw_bytes(restored, dtype, compressed.shape)
