from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from constants import MODEL_LABELS
from entropy import PackedSequenceStats, cluster_layers, huffman_weighted_length
from loaders import (
    SafetensorEntries,
    discover_safetensor_entries,
    iter_tensors_from_safetensors,
    iter_tensors_from_transformers,
    resolve_safetensor_source,
)
from schemas import LayerInfo, ScanSummary
from tensors import FLOAT_LAYOUTS, iter_tensor_exponent_chunks, summarize_dtypes, tensor_exponent_stats


class ModelCompressionExperiment:
    def __init__(
        self,
        model_name: str,
        loader: str,
        hf_token: str | None,
        local_files_only: bool,
        trust_remote_code: bool,
    ):
        self.model_name = model_name
        self.loader_preference = loader
        self.hf_token = hf_token
        self.local_files_only = local_files_only
        self.trust_remote_code = trust_remote_code
        self.loader_name, self.loader_details = self._resolve_loader()

    def _resolve_loader(self) -> tuple[str, tuple[Path, SafetensorEntries] | None]:
        if self.loader_preference == "transformers":
            return "transformers", None

        try:
            source_path = resolve_safetensor_source(
                model_name=self.model_name,
                hf_token=self.hf_token,
                local_files_only=self.local_files_only,
            )
            entries = discover_safetensor_entries(source_path)
            return "safetensors", (source_path, entries)
        except Exception:
            if self.loader_preference == "safetensors":
                raise
            return "transformers", None

    def iter_tensors(self):
        if self.loader_name == "safetensors":
            if self.loader_details is None:
                raise RuntimeError("Safetensors loader details were not initialized.")
            _, entries = self.loader_details
            yield from iter_tensors_from_safetensors(entries)
            return

        yield from iter_tensors_from_transformers(
            model_name=self.model_name,
            hf_token=self.hf_token,
            trust_remote_code=self.trust_remote_code,
        )

    def _tensor_progress(self, desc: str):
        total = None
        if self.loader_name == "safetensors" and self.loader_details is not None:
            _, entries = self.loader_details
            total = sum(len(keys) for _, keys in entries)
        model_label = MODEL_LABELS.get(self.model_name, self.model_name)
        return tqdm(
            self.iter_tensors(),
            total=total,
            desc=f"{model_label} | {desc}",
            unit="tensor",
            dynamic_ncols=True,
        )

    def scan(self) -> ScanSummary:
        global_exponent_counts = np.zeros(256, dtype=np.int64)
        payload_bits = 0
        original_bits = 0
        zipnn_plus_bits = 0
        dtype_weights: dict[str, int] = defaultdict(int)
        layer_infos: list[LayerInfo] = []

        with self._tensor_progress("ZipNN / ZipNN+") as progress:
            for name, tensor in progress:
                stats = tensor_exponent_stats(tensor)
                if stats is None:
                    del tensor
                    continue
                counts, num_weights, body_bits, tensor_original_bits, dtype_name = stats
                payload_bits += body_bits
                original_bits += tensor_original_bits
                zipnn_plus_bits += huffman_weighted_length(counts) + body_bits
                global_exponent_counts += counts
                dtype_weights[dtype_name] += num_weights

                histogram = counts.astype(np.float64)
                total = histogram.sum()
                if total > 0:
                    histogram = histogram / total
                layer_infos.append(LayerInfo(name=name, exponent_histogram=histogram))
                del tensor

        if not layer_infos:
            raise RuntimeError(f"No supported floating-point tensors were found in {self.model_name}.")

        zipnn_bits = huffman_weighted_length(global_exponent_counts) + payload_bits
        dtype_summary = summarize_dtypes(dtype_weights)
        return ScanSummary(
            layer_infos=layer_infos,
            original_bits=original_bits,
            payload_bits=payload_bits,
            zipnn_bits=zipnn_bits,
            zipnn_plus_bits=zipnn_plus_bits,
            dtype_summary=dtype_summary,
            num_tensors=len(layer_infos),
            loader=self.loader_name,
        )

    def zipnn_plusplus_bits(self, summary: ScanSummary, ngram: int, cluster_threshold: float) -> int:
        clusters = cluster_layers(
            [layer.exponent_histogram for layer in summary.layer_infos],
            threshold=cluster_threshold,
        )
        cluster_membership = {
            summary.layer_infos[layer_idx].name: cluster_idx
            for cluster_idx, member_ids in enumerate(clusters)
            for layer_idx in member_ids
        }
        packed_stats = [PackedSequenceStats(ngram=ngram) for _ in clusters]

        with self._tensor_progress("ZipNN++") as progress:
            for name, tensor in progress:
                if tensor.numel() == 0 or not tensor.is_floating_point() or tensor.dtype not in FLOAT_LAYOUTS:
                    del tensor
                    continue
                cluster_idx = cluster_membership[name]
                for exponents in iter_tensor_exponent_chunks(tensor):
                    packed_stats[cluster_idx].update_exponents(exponents)
                del tensor

        total_exp_bits = sum(cluster.bit_length() for cluster in packed_stats)
        return int(round(total_exp_bits + summary.payload_bits))
