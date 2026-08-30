from neuralzip._codec_setup import _floating_tensors, _SetupMixin
from neuralzip._codec_storage import _StorageMixin
from neuralzip.codec_primitives import DTYPES

_TRANSFER_NAME_FALLBACKS = {
    "lm_head.weight": ("model.embed_tokens.weight",),
}


class NeuralZipCodec(_SetupMixin, _StorageMixin):
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

    @staticmethod
    def _transfer_plan_key(name, tensor, stream_id):
        return str(name), DTYPES[tensor.dtype]["name"], int(stream_id)

    def prepare_direct_transfer_target(self, named_tensors):
        """Map target names, dtypes and streams to a reusable source setup."""
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
            config = DTYPES[tensor.dtype]
            stream_ids = tuple(range(int(config["element_size"])))
            tensor_name = str(name)
            dtype = config["name"]
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
                    f"Direct transfer has no compatible source tensor for {tensor_name}:{dtype}."
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
        return {
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

    def _mode_for_tensor_stream(self, name, tensor, stream_id):
        """Return the prepared mode for one concrete tensor byte stream."""
        return self._transfer_mode_by_tensor_stream[
            self._transfer_plan_key(name, tensor, stream_id)
        ]
