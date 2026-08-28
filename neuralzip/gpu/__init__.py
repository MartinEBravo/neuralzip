"""Compressed GPU inference for NeuralZip models."""

__all__ = [
    "NeuralZipModel",
    "NeuralZipRuntime",
    "export_model_for_gpu",
    "load_gpu_pretrained",
    "save_gpu_pretrained",
]


def __getattr__(name):
    if name in {"NeuralZipModel", "NeuralZipRuntime", "export_model_for_gpu"}:
        from .inference import (
            NeuralZipModel,
            NeuralZipRuntime,
            export_model_for_gpu,
        )

        return {
            "NeuralZipModel": NeuralZipModel,
            "NeuralZipRuntime": NeuralZipRuntime,
            "export_model_for_gpu": export_model_for_gpu,
        }[name]
    if name in {"load_gpu_pretrained", "save_gpu_pretrained"}:
        from .pretrained import load_gpu_pretrained, save_gpu_pretrained

        return {
            "load_gpu_pretrained": load_gpu_pretrained,
            "save_gpu_pretrained": save_gpu_pretrained,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
