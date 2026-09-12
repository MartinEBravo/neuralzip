from neuralzip.gpu.pretrained import enable
from neuralzip.setup import build_setup


class _NeuralZip:
    def __call__(self, setup=None, *, device=None):
        return enable(setup, device=device)

    def setup(self, model, **options):
        return build_setup(model, **options)


neuralzip = _NeuralZip()


__all__ = ["neuralzip"]
