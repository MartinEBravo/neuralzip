from transformers import AutoModel

from . import dtypes


class HFModel:
    def __init__(self, model_name, dtype="auto", **kwargs):
        self.model_name = model_name
        self.dtype = dtypes.resolve_torch_dtype(dtype)
        self._model = self._load_model(model_name, self.dtype, **kwargs)

    @property
    def model(self):
        return self._model

    def _load_model(self, model_name, dtype, **kwargs):
        load_kwargs = dict(kwargs)
        if "dtype" not in load_kwargs and "torch_dtype" not in load_kwargs:
            load_kwargs["dtype"] = dtype
        return AutoModel.from_pretrained(model_name, **load_kwargs)
