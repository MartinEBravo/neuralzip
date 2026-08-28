"""NeuralZip's public API."""

__all__ = ["NeuralZip"]


def __getattr__(name):
    if name == "NeuralZip":
        from .core import NeuralZip

        return NeuralZip
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
