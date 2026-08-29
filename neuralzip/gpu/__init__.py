"""Advanced GPU runtime primitives for NeuralZip."""

__all__ = [
    "NeuralZipRuntime",
]


def __getattr__(name):
    if name == "NeuralZipRuntime":
        from .inference import NeuralZipRuntime

        return NeuralZipRuntime
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
