__all__ = [
    "Compressor",
    "NeuralZipModel",
    "NeuralZipRuntime",
    "export_model_for_gpu",
    "load_gpu_pretrained",
    "save_gpu_pretrained",
]


def __getattr__(name):
    if name == "Compressor":
        from .compressor import Compressor

        return Compressor
    if name == "NeuralZipModel":
        from .gpu_inference import NeuralZipModel

        return NeuralZipModel
    if name == "NeuralZipRuntime":
        from .gpu_inference import NeuralZipRuntime

        return NeuralZipRuntime
    if name == "export_model_for_gpu":
        from .gpu_inference import export_model_for_gpu

        return export_model_for_gpu
    if name == "load_gpu_pretrained":
        from .gpu_pretrained import load_gpu_pretrained

        return load_gpu_pretrained
    if name == "save_gpu_pretrained":
        from .gpu_pretrained import save_gpu_pretrained

        return save_gpu_pretrained
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
