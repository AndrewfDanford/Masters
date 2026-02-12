import numpy as np

from src.eval.statistics import bootstrap_ci, paired_bootstrap_mean_diff_ci


def test_bootstrap_ci_constant_series() -> None:
    mean, low, high = bootstrap_ci([2.0] * 20, num_resamples=300, seed=5)
    assert np.isclose(mean, 2.0)
    assert np.isclose(low, 2.0)
    assert np.isclose(high, 2.0)


def test_paired_bootstrap_mean_diff() -> None:
    a = [0.9, 0.8, 0.7, 0.6]
    b = [0.7, 0.7, 0.7, 0.7]
    mean_diff, low, high = paired_bootstrap_mean_diff_ci(a, b, num_resamples=400, seed=7)
    assert mean_diff > 0
    assert low <= mean_diff <= high
