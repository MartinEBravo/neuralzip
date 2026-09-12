class _NeuralZip:
    def __call__(self, setup=None, *, device=None):
        from neuralzip.gpu.pretrained import enable

        return enable(setup, device=device)

    def setup(self, model, **options):
        from neuralzip._setup import build_setup

        return build_setup(model, **options)


neuralzip = _NeuralZip()


__all__ = ["neuralzip"]
