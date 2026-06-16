__all__ = ["Compressor", "HFModel"]


def __getattr__(name):
    if name == "Compressor":
        from .compressor import Compressor

        return Compressor
    if name == "HFModel":
        from .model import HFModel

        return HFModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
