from neuralzip.gpu.pretrained import enable
from neuralzip.setup import NeuralZipSetup


class _NeuralZip:
    def __call__(self, setup=None, *, device=None):
        return enable(setup, device=device)

    def setup(self, model, **options):
        return NeuralZipSetup(**options).build(model)


neuralzip = _NeuralZip()


__all__ = ["neuralzip"]
