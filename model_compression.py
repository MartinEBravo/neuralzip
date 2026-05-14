import torch
from transformers import AutoModelForCausalLM

from compression_codecs import CodecMixin
from float_utils import get_float_format
from neuralzip_model import NeuralZipMixin


class HuggingFaceModel(CodecMixin, NeuralZipMixin):
    def __init__(self, model_name, dtype="auto"):
        dtype_map = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype_map[dtype],
        )
        self._named_params = list(self._model.named_parameters())
        self._original_bits = sum(
            param.numel() * get_float_format(param.dtype)["total_bits"]
            for _, param in self._named_params
        )

        self._exp_offsets = None
        self._exp_codes = None
        self._exp_block_size = None
        self._exp_total_bits = None
        self._exp_ngram = None

    @property
    def model(self):
        return self._model

    def log_bits(self, name, total_bits):
        mbits = total_bits / 1e6
        mbytes = total_bits / 8e6
        ratio = 100 * (total_bits / self._original_bits)
        print(
            f"[{name}] {mbits:8.2f} Mbits | {mbytes:8.2f} MB | {ratio:5.2f}% of original"
        )

    def print_model_size(self):
        total_bits = self._original_bits
        self.log_bits("Original", total_bits)
        return total_bits
