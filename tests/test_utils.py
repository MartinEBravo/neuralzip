from neuralzip.utils import time_call


def test_time_call():
    def sample_function(x, y):
        return x + y

    result, elapsed_time = time_call(sample_function, 3, 4)
    assert result == 7
    assert elapsed_time >= 0

    # Test with a function with a parameter called "func"
    def target(*, func):
        return func

    result, elapsed_time = time_call(target, func=sample_function)
    assert result == sample_function
    assert elapsed_time >= 0
