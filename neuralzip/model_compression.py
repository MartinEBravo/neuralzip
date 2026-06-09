import torch
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageClassification,
)


class HFModel:
    def __init__(self, model_name, dtype="auto"):
        dtype_map = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        try:
            torch_dtype = dtype_map[dtype]
        except KeyError as exc:
            choices = ", ".join(sorted(dtype_map))
            raise ValueError(
                f"Unsupported dtype {dtype!r}. Expected one of: {choices}."
            ) from exc
        self._model = self._load_model(model_name, torch_dtype)
        super().__init__(self._model.named_parameters())

    @property
    def model(self):
        return self._model

    def _load_model(self, model_name, dtype):
        loaders = (
            AutoModelForCausalLM,
            AutoModelForImageClassification,
            AutoModel,
        )
        errors = []
        for loader in loaders:
            try:
                return loader.from_pretrained(model_name, dtype=dtype)
            except (OSError, ValueError) as exc:
                errors.append(f"{loader.__name__}: {exc}")
        raise ValueError(
            f"Could not load {model_name} with supported model loaders:\n"
            + "\n".join(errors)
        )


class Compressor:
    def __init__(self, method):
        self.method = method
