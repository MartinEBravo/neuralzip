import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForImageClassification

from compression_codecs import CodecMixin
from float_utils import get_float_format
from neuralzip_model import NeuralZipMixin


class PipelineModule(torch.nn.Module):
    def __init__(self, pipeline):
        super().__init__()
        modules = {}
        for name, component in pipeline.components.items():
            if isinstance(component, torch.nn.Module) and any(
                param.requires_grad or param.numel() for param in component.parameters()
            ):
                modules[name] = component
        if not modules:
            raise ValueError("Diffusion pipeline does not expose torch modules with parameters.")
        self.components = torch.nn.ModuleDict(modules)


class HuggingFaceModel(CodecMixin, NeuralZipMixin):
    def __init__(self, model_name, dtype="auto"):
        dtype_map = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        self._model = self._load_model(model_name, dtype_map[dtype])
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
        try:
            return self._load_diffusion_pipeline(model_name, dtype)
        except (ImportError, OSError, ValueError) as exc:
            errors.append(f"DiffusionPipeline: {exc}")
        raise ValueError(
            f"Could not load {model_name} with supported model loaders:\n"
            + "\n".join(errors)
        )

    def _load_diffusion_pipeline(self, model_name, dtype):
        try:
            from diffusers import DiffusionPipeline
        except ImportError as exc:
            raise ImportError(
                "The diffusers package is required to load diffusion pipelines."
            ) from exc

        kwargs = {
            "safety_checker": None,
            "requires_safety_checker": False,
        }
        if dtype != "auto":
            kwargs["torch_dtype"] = dtype
        pipeline = DiffusionPipeline.from_pretrained(model_name, **kwargs)
        return PipelineModule(pipeline)

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
