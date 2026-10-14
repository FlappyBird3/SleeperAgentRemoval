# Tests the Wilson interval against values computed by hand, including the extreme cases of 0% and 100%.
# pytest.approx allows tiny floating-point differences, since computers store decimals with limited precision.
import pytest

from sleeperlab.metrics import rate_with_interval, wilson_interval


def test_half_rate_with_200_samples():
    low, high = wilson_interval(100, 200)
    assert low == pytest.approx(0.4314, abs=1e-3)
    assert high == pytest.approx(0.5686, abs=1e-3)


def test_zero_successes_still_has_positive_upper_bound():
    low, high = wilson_interval(0, 200)
    assert low == pytest.approx(0.0, abs=1e-9)
    assert high == pytest.approx(0.0188, abs=1e-3)


def test_all_successes_still_has_lower_bound_below_one():
    low, high = wilson_interval(200, 200)
    assert high == pytest.approx(1.0, abs=1e-9)
    assert low == pytest.approx(0.9812, abs=1e-3)


def test_interval_always_contains_observed_rate():
    for k in range(0, 201, 10):
        low, high = wilson_interval(k, 200)
        assert low <= k / 200 <= high


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        wilson_interval(0, 0)
    with pytest.raises(ValueError):
        wilson_interval(5, 4)
    with pytest.raises(ValueError):
        wilson_interval(-1, 10)


def test_rate_with_interval_from_flags():
    result = rate_with_interval([True, False, True, True])
    assert result["n"] == 4
    assert result["successes"] == 3
    assert result["rate"] == 0.75
    assert result["low"] < 0.75 < result["high"]