import torch
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageClassification,
)


def resolve_torch_dtype(dtype):
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype is None:
        return "auto"

    dtype_key = str(dtype).lower().replace("torch.", "").replace("-", "_")
    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32
    }
    try:
        return dtype_map[dtype_key]
    except KeyError as exc:
        choices = ", ".join(sorted(dtype_map))
        raise ValueError(
            f"Unsupported dtype {dtype!r}. Expected one of: {choices}."
        ) from exc


class HFModel:
    def __init__(self, model_name, dtype="auto", **kwargs):
        self.model_name = model_name
        self.dtype = resolve_torch_dtype(dtype)
        self._model = self._load_model(model_name, self.dtype, **kwargs)

    @property
    def model(self):
        return self._model

    def _load_model(self, model_name, dtype, **kwargs):
        loaders = (
            AutoModelForCausalLM,
            AutoModelForImageClassification,
            AutoModel,
        )
        load_kwargs = dict(kwargs)
        if dtype != "auto":
            load_kwargs["torch_dtype"] = dtype
        errors = []
        for loader in loaders:
            try:
                return loader.from_pretrained(model_name, **load_kwargs)
            except (OSError, ValueError) as exc:
                errors.append(f"{loader.__name__}: {exc}")
        raise ValueError(
            f"Could not load {model_name} with supported model loaders:\n"
            + "\n".join(errors)
        )


class Compressor:
    def __init__(self, method):
        self.method = method

