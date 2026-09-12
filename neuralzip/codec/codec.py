from neuralzip.codec.setup import _floating_tensors, _SetupMixin
from neuralzip.codec.storage import _StorageMixin
from neuralzip.utils import DTYPES

BF16_NAME = "bfloat16"
EXPONENT_STREAM = 1

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
        self._global_modes = {}

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

    def prepare_gpu_transfer_target(self, named_tensors):
        """Project a full setup onto the sharing-only BF16 GPU encoder."""
        global_mode = self._global_modes[(BF16_NAME, EXPONENT_STREAM)]
        transfer_modes = {}

        for name, tensor in _floating_tensors(named_tensors):
            tensor_name = str(name)
            key = (tensor_name, BF16_NAME, EXPONENT_STREAM)
            candidates = (tensor_name, *_TRANSFER_NAME_FALLBACKS.get(tensor_name, ()))
            source_name = next(
                (
                    candidate
                    for candidate in candidates
                    if (
                        candidate,
                        BF16_NAME,
                        EXPONENT_STREAM,
                    )
                    in self._mode_by_tensor_stream
                ),
                None,
            )
            source_mode = (
                self._mode_by_tensor_stream[(source_name, BF16_NAME, EXPONENT_STREAM)]
                if source_name is not None
                else None
            )
            if source_mode is not None and source_mode.get("kind") == "shared_huff0":
                transfer_modes[key] = source_mode
            else:
                transfer_modes[key] = global_mode

        self._transfer_mode_by_tensor_stream = transfer_modes

    def prepare_storage_transfer_target(self, named_tensors):
        """Map a target onto the full storage codec with lossless fallbacks."""
        transfer_modes = {}
        for name, tensor in _floating_tensors(named_tensors):
            config = DTYPES[tensor.dtype]
            tensor_name = str(name)
            dtype = config["name"]
            candidates = (tensor_name, *_TRANSFER_NAME_FALLBACKS.get(tensor_name, ()))
            for stream_id in range(int(config["element_size"])):
                source_key = next(
                    (
                        (candidate, dtype, stream_id)
                        for candidate in candidates
                        if (candidate, dtype, stream_id) in self._mode_by_tensor_stream
                    ),
                    None,
                )
                target_key = (tensor_name, dtype, stream_id)
                if source_key is not None:
                    transfer_modes[target_key] = self._mode_by_tensor_stream[source_key]
                else:
                    transfer_modes[target_key] = self._global_modes.get(
                        (dtype, stream_id), {"kind": "raw"}
                    )
        self._transfer_mode_by_tensor_stream = transfer_modes

    def _mode_for_tensor_stream(self, name, tensor, stream_id):
        """Return the prepared mode for one concrete tensor byte stream."""
        return self._transfer_mode_by_tensor_stream[
            self._transfer_plan_key(name, tensor, stream_id)
        ]
