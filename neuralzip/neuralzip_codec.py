import re
import time

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
    """JS-clustered shared Huffman with time-budgeted packed exponent modes."""

    def __init__(
        self,
        *,
        min_saving_bytes=0,
        chunk_size=256 << 10,
        compression_threshold=0.95,
        prior_count=1,
        exponent_prior_count=1,
        cluster_map=None,
        clustering="js-distance",
        cluster_js_threshold=0.05,
        max_codebook_sample_bytes=8 << 20,
        ngram_values=(2, 3, 4),
        top_k=254,
        exponent_layout="original",
        packed_time_budget_ratio=0.10,
    ):
        super().__init__(
            min_saving_bytes=min_saving_bytes,
            chunk_size=chunk_size,
            compression_threshold=compression_threshold,
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
        self.packed_time_budget_ratio = float(packed_time_budget_ratio)

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
        if any(value < 2 or value > 4 for value in self.ngram_values):
            raise ValueError("ngram_values must contain only values in [2, 4].")
        if not 1 <= self.top_k <= 254:
            raise ValueError("top_k must be in [1, 254].")
        if self.exponent_layout not in {"original", "transpose2d"}:
            raise ValueError("exponent_layout must be 'original' or 'transpose2d'.")
        if self.packed_time_budget_ratio < 0:
            raise ValueError("packed_time_budget_ratio must be non-negative.")

        self.reset_compression()
        self.reset_decompression()

    def reset_compression(self):
        self._mode_by_key = {}
        self._codebook_payloads = []
        self._codebook_payload_by_id = {}
        self._computed_cluster_map = {}
        self._next_codebook_id = 0
        self._selected_mode_counts = {}
        self._selection_stats = {}

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

    def cluster_id(self, name):
        if name in self.cluster_map:
            return self.cluster_map[name]
        if name in self._computed_cluster_map:
            return self._computed_cluster_map[name]
        return _layer_signature(name)

    def _prepare_clusters(self, named_tensors):
        clusters = []
        computed = {}
        for name, tensor in sorted(named_tensors, key=lambda item: item[0]):
            if name in self.cluster_map:
                continue
            if (
                not torch.is_tensor(tensor)
                or not tensor.is_floating_point()
                or tensor.numel() == 0
            ):
                continue

            distribution = _exponent_distribution(tensor)
            weight = int(tensor.numel())
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
        if mode["kind"] == "shared_huffman":
            max_payload_bytes = max(
                0,
                int(len(chunk) * self.compression_threshold) - (1 + 8 + 8 + 8),
            )
            try:
                encoded = huffman.encode_with_codebook(
                    chunk,
                    mode["codebook"],
                    min_saving_bytes=self.min_saving_bytes,
                    max_output_bytes=max_payload_bytes,
                )
            except ValueError as error:
                if "missing" not in str(error):
                    raise
                encoded = None
            if encoded is None:
                return None
            candidate = StreamChunk(
                mode="shared_huffman",
                symbol_count=len(chunk),
                payload=encoded.payload,
                codebook_id=mode.get("codebook_id"),
            )
        elif mode["kind"] == "shared_huff0":
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

    def _mode_cost(self, sample, mode, element_size, codebook_payload):
        total = codebook_payload.metadata_bytes
        step = self._source_chunk_size(element_size, mode)
        for start in range(0, len(sample), step):
            chunk = sample[start : start + step]
            encoded = self._encode_chunk(chunk, mode)
            total += (encoded or self._raw_chunk(chunk)).compressed_bytes
        return total

    def _measure_mode(self, sample, mode, element_size, codebook_payload):
        started = time.perf_counter()
        mode["sample_cost"] = self._mode_cost(
            sample,
            mode,
            element_size,
            codebook_payload,
        )
        mode["sample_seconds"] = time.perf_counter() - started
        mode["fixed_cost"] = codebook_payload.metadata_bytes
        return mode

    def _build_candidate(self, sample, *, kind, prior_count, ngram_n=1):
        if kind in {"shared_huffman", "shared_huff0"}:
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
            if kind == "shared_huffman":
                codebook = huffman.build_codebook(tokens, prior_count=prior_count)
            else:
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
        if mode["kind"] == "shared_huffman":
            codec = "canonical-huffman"
            code_lengths = mode["codebook"].code_lengths
            codebook = b""
        else:
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
    ):
        raw_mode = {"kind": "raw", "fixed_cost": 0}
        raw_step = self._stream_chunk_size(element_size)
        started = time.perf_counter()
        raw_mode["sample_cost"] = sum(
            self._raw_chunk(sample[start : start + raw_step]).compressed_bytes
            for start in range(0, len(sample), raw_step)
        )
        raw_mode["sample_seconds"] = time.perf_counter() - started
        prior_count = self.exponent_prior_count if is_exponent else self.prior_count

        shared = self._build_candidate(
            sample,
            kind="shared_huffman",
            prior_count=prior_count,
        )
        if shared is not None:
            payload = self._candidate_payload(key, shared)
            self._measure_mode(
                sample,
                shared,
                element_size,
                payload,
            )

        base = min(
            (raw_mode, shared) if shared is not None else (raw_mode,),
            key=lambda candidate: int(candidate["sample_cost"]),
        )

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
                self._measure_mode(
                    sample,
                    packed,
                    element_size,
                    payload,
                )
                packed_candidates.append(packed)

        packed = (
            min(packed_candidates, key=lambda candidate: int(candidate["sample_cost"]))
            if packed_candidates
            else None
        )
        sample_size = len(sample)
        base_cost = self._estimated_metric(
            base,
            sample_size,
            total_size,
            "sample_cost",
        )
        base_seconds = self._estimated_metric(
            base,
            sample_size,
            total_size,
            "sample_seconds",
        )
        opportunity = None
        if packed is not None:
            packed_cost = self._estimated_metric(
                packed,
                sample_size,
                total_size,
                "sample_cost",
            )
            packed_seconds = self._estimated_metric(
                packed,
                sample_size,
                total_size,
                "sample_seconds",
            )
            saved_bytes = base_cost - packed_cost
            extra_seconds = max(0.0, packed_seconds - base_seconds)
            if saved_bytes > 0:
                opportunity = {
                    "mode": packed,
                    "saved_bytes": saved_bytes,
                    "extra_seconds": extra_seconds,
                    "savings_per_second": (
                        float("inf")
                        if extra_seconds == 0
                        else saved_bytes / extra_seconds
                    ),
                }
        return {
            "key": key,
            "base": base,
            "base_cost": base_cost,
            "base_seconds": base_seconds,
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
        if payload.codec == "huff0":
            self._decode_codebooks[codebook_id] = huff0.build_decodebook(
                payload.codebook
            )
        else:
            self._decode_codebooks[codebook_id] = huffman.build_decodebook(
                payload.code_lengths
            )
        return selected

    def _record_selected_mode(self, mode):
        kind = mode["kind"]
        if kind == "packed_huff0":
            kind = f"packed_huff0_n{mode['ngram_n']}"
        self._selected_mode_counts[kind] = (
            self._selected_mode_counts.get(kind, 0) + 1
        )

    def precompute_codebooks(self, named_tensors):
        self._prepare_clusters(named_tensors)
        samples = {}
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
                part = stream[:remaining]
                if not part:
                    continue
                samples.setdefault(key, bytearray()).extend(part)
                sample_sizes[key] = sample_sizes.get(key, 0) + len(part)
                stream_info[key] = (element_size, is_exponent)

        evaluations = []
        for key in sorted(samples):
            element_size, is_exponent = stream_info[key]
            evaluations.append(
                self._evaluate_modes(
                    key,
                    bytes(samples[key]),
                    total_size=total_sizes[key],
                    element_size=element_size,
                    is_exponent=is_exponent,
                )
            )

        base_seconds = sum(evaluation["base_seconds"] for evaluation in evaluations)
        packed_budget = base_seconds * self.packed_time_budget_ratio
        packed_used = 0.0
        packed_saved = 0.0
        selected_packed_keys = set()
        opportunities = [
            evaluation
            for evaluation in evaluations
            if evaluation["opportunity"] is not None
        ]
        opportunities.sort(
            key=lambda evaluation: (
                -evaluation["opportunity"]["savings_per_second"],
                evaluation["key"],
            )
        )
        for evaluation in opportunities:
            if self.packed_time_budget_ratio == 0:
                break
            opportunity = evaluation["opportunity"]
            extra_seconds = opportunity["extra_seconds"]
            if extra_seconds > 0 and packed_used + extra_seconds > packed_budget:
                continue
            selected_packed_keys.add(evaluation["key"])
            packed_used += extra_seconds
            packed_saved += opportunity["saved_bytes"]

        for evaluation in evaluations:
            selected = evaluation["base"]
            if evaluation["key"] in selected_packed_keys:
                selected = evaluation["opportunity"]["mode"]
            selected = self._finalize_mode(evaluation["key"], selected)
            self._mode_by_key[evaluation["key"]] = selected
            self._record_selected_mode(selected)

        self._selection_stats = {
            "base_estimated_seconds": base_seconds,
            "packed_time_budget_seconds": packed_budget,
            "packed_extra_seconds": packed_used,
            "packed_estimated_bytes_saved": packed_saved,
            "packed_candidates": len(opportunities),
            "packed_selected": len(selected_packed_keys),
        }

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
        key = self._codebook_key(name, tensor, stream_id)
        mode = self._mode_by_key.get(key)
        if mode is None:
            evaluation = self._evaluate_modes(
                key,
                stream[: self.max_codebook_sample_bytes],
                total_size=len(stream),
                element_size=tensor.element_size(),
                is_exponent=is_exponent,
            )
            mode = evaluation["base"]
            opportunity = evaluation["opportunity"]
            if opportunity is not None and self.packed_time_budget_ratio > 0:
                budget = (
                    evaluation["base_seconds"] * self.packed_time_budget_ratio
                )
                if opportunity["extra_seconds"] <= budget:
                    mode = opportunity["mode"]
            mode = self._finalize_mode(key, mode)
            self._mode_by_key[key] = mode
            self._record_selected_mode(mode)

        step = (
            self._stream_chunk_size(tensor.element_size())
            if mode["kind"] == "raw"
            else self._source_chunk_size(tensor.element_size(), mode)
        )
        chunks = []
        for start in range(0, len(stream), step):
            chunk = stream[start : start + step]
            encoded = None if mode["kind"] == "raw" else self._encode_chunk(chunk, mode)
            chunks.append(encoded or self._raw_chunk(chunk))
        return ByteGroupPayload(
            stream_kind=stream_kind,
            symbol_count=len(stream),
            chunks=tuple(chunks),
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
