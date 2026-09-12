import neuralzip as api


def test_call(monkeypatch):
    setup = object()
    expected = object()

    def fake_enable(received_setup, *, device):
        assert received_setup is setup
        assert device == "cuda"
        return expected

    monkeypatch.setattr(api, "enable", fake_enable)

    result = api.neuralzip(setup, device="cuda")
    assert result is expected


def test_setup(monkeypatch):
    model = object()
    options = {"option1": 1, "option2": 2}
    expected = object()

    class FakeSetup:
        def __init__(self, **received_options):
            assert received_options == options

        def build(self, received_model):
            assert received_model is model
            return expected

    monkeypatch.setattr(api, "NeuralZipSetup", FakeSetup)

    result = api.neuralzip.setup(model, **options)
    assert result is expected
