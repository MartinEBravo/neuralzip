import re

import numpy as np
import torch

from . import huff0, huffman
from .packed_exponents import (
    restore_2d_stream,
    topk_vocab,
    tokenize_ngrams,
    transpose_2d_stream,
)
from .utils import split_fields
from .zipnn_codec import (
    DTYPE_BY_NAME,
    ByteGroupPayload,
    CodebookPayload,
    StreamChunk,
    TensorPayload,
    ZipNNCodec,
    _dtype_name,
    _tensor_from_raw_bytes,
    _zipnn_dtype_config,
    _zipnn_reordered_bytes,
    _zipnn_restored_bytes,
)


def _layer_signature(name):
    return ".".join(re.sub(r"\d+", "#", part) for part in str(name).split("."))


def _jensen_shannon(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    middle = 0.5 * (left + right)
    left_mask = left > 0
    right_mask = right > 0
    divergence = 0.5 * np.sum(
        left[left_mask] * np.log2(left[left_mask] / middle[left_mask])
    )
    divergence += 0.5 * np.sum(
        right[right_mask] * np.log2(right[right_mask] / middle[right_mask])
    )
    return float(np.sqrt(divergence))


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
        max_tensor_probe_bytes=256 << 10,
        ngram_values=(2, 3, 4),
        top_k=254,
        exponent_layout="original",
        packed_source_budget_ratio=0.30,
        enable_shared_codebooks=True,
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
        self.max_tensor_probe_bytes = int(max_tensor_probe_bytes)
        self.ngram_values = tuple(sorted({int(value) for value in ngram_values}))
        self.top_k = int(top_k)
        self.exponent_layout = str(exponent_layout)
        self.packed_source_budget_ratio = float(packed_source_budget_ratio)
        self.enable_shared_codebooks = bool(enable_shared_codebooks)

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
        if self.max_tensor_probe_bytes <= 0:
            raise ValueError("max_tensor_probe_bytes must be positive.")
        if any(value < 2 or value > 8 for value in self.ngram_values):
            raise ValueError("ngram_values must contain only values in [2, 8].")
        if not 1 <= self.top_k <= 254:
            raise ValueError("top_k must be in [1, 254].")
        if self.exponent_layout not in {"original", "transpose2d"}:
            raise ValueError("exponent_layout must be 'original' or 'transpose2d'.")
        if not 0 <= self.packed_source_budget_ratio <= 1:
            raise ValueError("packed_source_budget_ratio must be in [0, 1].")

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
            if codebook.codec == "huff0":
                self._decode_codebooks[codebook_id] = huff0.build_decodebook(
                    codebook.codebook
                )
            elif codebook.codec == "canonical-huffman":
                self._decode_codebooks[codebook_id] = huffman.build_decodebook(
                    codebook.code_lengths
                )
            else:
                raise ValueError(f"Unsupported codebook codec {codebook.codec!r}.")

    def codebook_metadata_bytes(self):
        return sum(codebook.metadata_bytes for codebook in self._codebook_payloads)

    @property
    def raw_codebook_keys(self):
        return sum(mode["kind"] == "raw" for mode in self._mode_by_key.values())

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
    def transfer_stats(self):
        return dict(self._transfer_stats)

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
                distance = _jensen_shannon(distribution, cluster["centroid"])
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
        return self.cluster_id(name), _dtype_name(tensor.dtype), int(stream_id)

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
                int(len(chunk) * self.compression_threshold) - (1 + 8 + 8 + 8),
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
                - (1 + 8 + 8 + 8 + 8 + 1 + 8 + 8),
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
                total += (encoded or self._raw_chunk(chunk)).compressed_bytes
        return total

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
        return mode

    def _build_candidate(self, sample, *, kind, prior_count, ngram_n=1):
        if kind == "shared_huff0":
            tokens = sample
            vocab = b""
        else:
            vocab = topk_vocab(sample, ngram_n, self.top_k)
            if not vocab:
                return None
            tokens, _, _, _, _ = tokenize_ngrams(sample, ngram_n, vocab)
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
        }

    def _candidate_payload(self, key, mode, codebook_id=-1):
        cluster_id, dtype, stream_id = key
        codec = "huff0"
        code_lengths = b""
        codebook = mode["codebook"].codebook
        return CodebookPayload(
            codebook_id=int(codebook_id),
            cluster_id=cluster_id,
            stream_id=stream_id,
            dtype=dtype,
            codec=codec,
            mode=mode["kind"],
            code_lengths=code_lengths,
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

        base = min(base_candidates, key=estimated_cost)

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

        packed = (
            min(packed_candidates, key=estimated_cost)
            if packed_candidates
            else None
        )
        base_cost = estimated_cost(base)
        opportunity = None
        if packed is not None:
            packed_cost = estimated_cost(packed)
            saved_bytes = base_cost - packed_cost
            if saved_bytes > 0:
                opportunity = {
                    "mode": packed,
                    "saved_bytes": saved_bytes,
                    "source_bytes": int(total_size),
                    "savings_per_source_byte": saved_bytes / total_size,
                }
        return {
            "key": key,
            "base": base,
            "base_cost": base_cost,
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
        return str(name), _dtype_name(tensor.dtype), int(stream_id)

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
            payloads = self._encode_huff0_chunks(
                source_chunks,
                mode["codebook"],
                metadata_bytes=1 + 8 + 8 + 8,
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
        else:
            for index, chunk in enumerate(source_chunks):
                encoded_chunks[index] = self._encode_chunk(chunk, mode)

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
        chunks = encoded_chunks
        return ByteGroupPayload(
            stream_kind=stream_kind,
            symbol_count=len(stream),
            chunks=tuple(chunks),
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
            reordered = _zipnn_reordered_bytes(original, config)
            element_size = int(config["element_size"])
            for stream_id in range(int(config["num_buffers"])):
                is_exponent = self._is_exponent_stream(config, stream_id)
                stream = reordered[stream_id::element_size]
                stream, _ = self._apply_exponent_layout(
                    original,
                    stream,
                    is_exponent,
                )
                key = self._codebook_key(name, original, stream_id)
                source_mode = self._mode_by_key[key]
                if source_mode["kind"] in {
                    "raw",
                    "local_huff0",
                    "packed_huff0",
                }:
                    selected = source_mode
                else:
                    sample = self._bounded_stream_sample(
                        stream,
                        self.max_tensor_probe_bytes,
                    )
                    selected, _ = self._select_transfer_mode(
                        sample,
                        source_mode,
                        element_size=element_size,
                        stream_kind=("exponent" if is_exponent else "byte_group"),
                    )
                    probe_bytes += len(sample)

                plan_key = self._transfer_plan_key(name, original, stream_id)
                self._mode_by_tensor_stream[plan_key] = selected
                kind = selected["kind"]
                mode_counts[kind] = mode_counts.get(kind, 0) + 1
                mode_symbols[kind] = mode_symbols.get(kind, 0) + len(stream)
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
            dtype = _dtype_name(original.dtype)
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
                        (cluster_id, _jensen_shannon(distribution, centroid))
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
                selected, probe = self._select_transfer_mode(
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
        }
        self._transfer_active = True
        return self.transfer_stats

    def build_codebooks_and_select_modes(self, named_tensors):
        if not self._clusters_prepared:
            raise RuntimeError(
                "Clusters must be prepared before building NeuralZip codebooks."
            )
        named_tensors = list(named_tensors)
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
        packed_budget = int(
            exponent_source_bytes * self.packed_source_budget_ratio
        )
        packed_used = 0
        packed_saved = 0.0
        selected_packed_keys = set()
        opportunities = [
            evaluation
            for evaluation in evaluations
            if evaluation["opportunity"] is not None
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
            if packed_used + source_bytes > packed_budget:
                continue
            selected_packed_keys.add(evaluation["key"])
            packed_used += source_bytes
            packed_saved += opportunity["saved_bytes"]

        for evaluation in evaluations:
            selected = evaluation["base"]
            if evaluation["key"] in selected_packed_keys:
                selected = evaluation["opportunity"]["mode"]
            selected = self._finalize_mode(evaluation["key"], selected)
            self._mode_by_key[evaluation["key"]] = selected
            self._record_selected_mode(selected)

        self._selection_stats = {
            "selection_strategy": "source-byte-budget",
            "exponent_source_bytes": exponent_source_bytes,
            "packed_source_budget_bytes": packed_budget,
            "packed_source_bytes": packed_used,
            "packed_estimated_bytes_saved": packed_saved,
            "packed_candidates": len(opportunities),
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

    def _compress_cluster_stream(
        self,
        name,
        tensor,
        stream,
        stream_kind,
        stream_id,
        *,
        is_exponent,
    ):
        if self._transfer_active:
            plan_key = self._transfer_plan_key(name, tensor, stream_id)
            mode = self._transfer_mode_by_tensor_stream.get(
                plan_key,
                {"kind": "local_huff0", "fixed_cost": 0},
            )
            return self._compress_stream_with_mode(
                stream,
                stream_kind,
                element_size=tensor.element_size(),
                mode=mode,
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

        return self._compress_stream_with_mode(
            stream,
            stream_kind,
            element_size=tensor.element_size(),
            mode=mode,
        )

    def compress_tensor(self, name, tensor):
        if not torch.is_tensor(tensor) or not tensor.is_floating_point():
            raise TypeError("NeuralZipCodec only compresses floating tensors.")
        original = tensor.detach().cpu().contiguous()
        return self._compress_neuralzip(
            name,
            original,
            _zipnn_dtype_config(original.dtype),
        )

    def _compress_neuralzip(self, name, tensor, config):
        reordered = _zipnn_reordered_bytes(tensor, config)
        element_size = int(config["element_size"])
        groups = []
        tensor_layout = "original"
        for stream_id in range(int(config["num_buffers"])):
            is_exponent = self._is_exponent_stream(config, stream_id)
            stream = reordered[stream_id::element_size]
            stream, layout = self._apply_exponent_layout(tensor, stream, is_exponent)
            if is_exponent:
                tensor_layout = layout
            groups.append(
                self._compress_cluster_stream(
                    name,
                    tensor,
                    stream,
                    "exponent" if is_exponent else "byte_group",
                    stream_id,
                    is_exponent=is_exponent,
                )
            )
        return TensorPayload(
            name=name,
            shape=tuple(int(dim) for dim in tensor.shape),
            dtype=_dtype_name(tensor.dtype),
            element_size=element_size,
            numel=int(tensor.numel()),
            strategy="neuralzip",
            groups=tuple(groups),
            byte_reorder=int(config["byte_reorder"]),
            bit_reorder=int(config["bit_reorder"]),
            num_buffers=int(config["num_buffers"]),
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
            elif chunk.mode == "huffman":
                decoded = huffman.decompress(
                    huffman.HuffmanPayload(
                        payload=chunk.payload,
                        code_lengths=chunk.code_lengths,
                        symbol_count=chunk.symbol_count,
                    )
                )
            elif chunk.mode == "local_huff0":
                decoded = huff0.decompress_block(chunk.payload, chunk.symbol_count)
            elif chunk.mode == "shared_huffman":
                codebook = self._codebook_payload_by_id.get(chunk.codebook_id)
                decodebook = self._decode_codebooks.get(chunk.codebook_id)
                if codebook is None or decodebook is None:
                    raise ValueError(f"Missing shared codebook {chunk.codebook_id}.")
                decoded = huffman.decompress_with_decodebook(
                    huffman.HuffmanPayload(
                        payload=chunk.payload,
                        code_lengths=codebook.code_lengths,
                        symbol_count=chunk.symbol_count,
                    ),
                    decodebook,
                )
            elif chunk.mode in {"shared_huff0", "packed_huff0"}:
                codebook = self._codebook_payload_by_id.get(chunk.codebook_id)
                decodebook = self._decode_codebooks.get(chunk.codebook_id)
                if codebook is None or decodebook is None:
                    raise ValueError(f"Missing Huff0 codebook {chunk.codebook_id}.")
                if chunk.mode == "shared_huff0":
                    decoded = huff0.decompress_with_decodebook(
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
                    if codebook.mode != "packed_huff0":
                        raise ValueError("Packed Huff0 chunk references a non-packed codebook.")
                    if int(codebook.ngram_n) != int(chunk.ngram_n):
                        raise ValueError("Packed Huff0 chunk and codebook disagree on n.")
                    decoded = huff0.decompress_ngram_with_decodebook(
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
