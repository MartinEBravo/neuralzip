__all__ = ["Compressor", "HFModel", "NeuralZipPlan"]


def __getattr__(name):
    if name == "Compressor":
        from .compressor import Compressor

        return Compressor
    if name == "HFModel":
        from .model import HFModel

        return HFModel
    if name == "NeuralZipPlan":
        from .plan import NeuralZipPlan

        return NeuralZipPlan
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
