import pytest
import torch

from neuralzip import Compressor


def _weights(offset=0):
    pattern = torch.tensor([1.0, 2.0, 1.0, 2.0], dtype=torch.bfloat16).repeat(8192)
    return {
        "layers.0.weight": (pattern + offset).contiguous(),
        "layers.1.weight": (pattern.flip(0) + offset).contiguous(),
    }


def _assert_exact(expected, actual):
    assert tuple(actual) == tuple(expected)
    for name, tensor in expected.items():
        assert torch.equal(
            tensor.contiguous().view(torch.uint8),
            actual[name].contiguous().view(torch.uint8),
        )


def test_failed_refit_preserves_the_previous_setup_transactionally():
    source = _weights()
    compressor = Compressor("neuralzip")
    setup = compressor.fit(source)
    expected_digest = compressor.setup_digest
    expected_payloads = compressor.encode(source, setup=setup).payloads

    with pytest.raises(TypeError):
        compressor.fit({"unsupported": torch.ones(8, dtype=torch.float64)})

    assert compressor.setup is setup
    assert compressor.setup_digest == expected_digest
    assert compressor.encode(source).payloads == expected_payloads


def test_decoding_an_unrelated_archive_cannot_poison_a_reusable_setup():
    source = _weights()
    compressor = Compressor("neuralzip", keep_payloads=True)
    setup = compressor.fit(source)
    encoded = compressor.encode(source, setup=setup)

    unrelated = Compressor(
        "neuralzip",
        keep_payloads=True,
        cluster_js_threshold=0.0,
    ).compress(_weights(offset=4))
    compressor.decompress(unrelated)

    _assert_exact(source, compressor.decode(encoded))
    assert compressor.compress(source)["roundtrip_ok"] is True


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"threads": -1}, "threads must be non-negative"),
        ({"direct_batch_bytes": 0}, "direct_batch_bytes must be positive"),
    ],
)
def test_loaded_setup_does_not_bypass_runtime_option_validation(
    tmp_path, options, message
):
    compressor = Compressor("neuralzip")
    compressor.fit(_weights())
    path = compressor.save_setup(tmp_path / "weights.setup")

    with pytest.raises(ValueError, match=message):
        Compressor("neuralzip", setup=path, **options)


def test_failed_explicit_setup_switch_restores_the_active_setup():
    source = _weights()
    compressor = Compressor("neuralzip")
    original_setup = compressor.fit(source)
    original_digest = compressor.setup_digest
    incompatible = Compressor("neuralzip").fit(
        {"different.weight": source["layers.0.weight"]}
    )

    with pytest.raises(ValueError, match="Direct transfer requires"):
        compressor.encode(source, setup=incompatible)

    assert compressor.setup is original_setup
    assert compressor.setup_digest == original_digest
    restored = compressor.decode(compressor.encode(source))
    _assert_exact(source, restored)


def test_loaded_setup_applies_the_new_compressor_runtime_options(tmp_path):
    fitted = Compressor("neuralzip")
    fitted.fit(_weights())
    path = fitted.save_setup(tmp_path / "weights.setup")

    loaded = Compressor(
        "neuralzip",
        setup=path,
        threads=3,
        min_saving_bytes=123,
        chunk_size=64 << 10,
        compression_threshold=0.5,
        direct_batch_bytes=2 << 20,
    )

    assert loaded.codec.threads == 3
    assert loaded.codec.min_saving_bytes == 123
    assert loaded.codec.chunk_size == 64 << 10
    assert loaded.codec.compression_threshold == 0.5
    assert loaded.codec.direct_batch_bytes == 2 << 20


def test_loaded_setup_preserves_omitted_framing_options(tmp_path):
    fitted = Compressor(
        "neuralzip",
        min_saving_bytes=17,
        chunk_size=4096,
        compression_threshold=0.5,
        direct_batch_bytes=123456,
    )
    fitted.fit(_weights())
    path = fitted.save_setup(tmp_path / "weights.setup")

    loaded = Compressor("neuralzip", setup=path)

    assert loaded.codec.min_saving_bytes == 17
    assert loaded.codec.chunk_size == 4096
    assert loaded.codec.compression_threshold == 0.5
    assert loaded.codec.direct_batch_bytes == 123456
