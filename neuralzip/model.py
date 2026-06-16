import torch
from transformers import AutoModel


def resolve_torch_dtype(dtype):
    if dtype is None or dtype == "auto":
        return "auto"
    if isinstance(dtype, torch.dtype):
        return dtype

    dtype_key = str(dtype).lower().replace("torch.", "").replace("-", "_")
    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "f16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "f32": torch.float32,
    }
    if dtype_key not in dtype_map:
        choices = ", ".join(("auto", *sorted(dtype_map)))
        raise ValueError(f"Unsupported dtype {dtype!r}. Expected one of: {choices}.")
    return dtype_map[dtype_key]


class HFModel:
    def __init__(self, model_name, dtype="auto", **kwargs):
        self.model_name = model_name
        self.dtype = resolve_torch_dtype(dtype)
        self._model = self._load_model(model_name, self.dtype, **kwargs)

    @property
    def model(self):
        return self._model

    def _load_model(self, model_name, dtype, **kwargs):
        load_kwargs = dict(kwargs)
        if "dtype" not in load_kwargs and "torch_dtype" not in load_kwargs:
            load_kwargs["dtype"] = dtype
        return AutoModel.from_pretrained(model_name, **load_kwargs)

    def compress(self, method="zipnn", **kwargs):
        from .compressor import Compressor

        return Compressor(method, **kwargs).compress(self)
