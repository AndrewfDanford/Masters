from __future__ import annotations

import numpy as np


def _to_1d(values: np.ndarray | list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        raise ValueError("expected a non-empty numeric array")
    return array


def bootstrap_ci(
    values: np.ndarray | list[float],
    num_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 17,
) -> tuple[float, float, float]:
    if num_resamples <= 0:
        raise ValueError("num_resamples must be positive")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be between 0 and 1")

    values_array = _to_1d(values)
    rng = np.random.default_rng(seed)
    n = values_array.size
    resample_means = np.empty(num_resamples, dtype=float)
    for index in range(num_resamples):
        resample = values_array[rng.integers(0, n, size=n)]
        resample_means[index] = float(np.mean(resample))

    lower_q = alpha / 2.0
    upper_q = 1.0 - alpha / 2.0
    return (
        float(np.mean(values_array)),
        float(np.quantile(resample_means, lower_q)),
        float(np.quantile(resample_means, upper_q)),
    )


def paired_bootstrap_mean_diff_ci(
    values_a: np.ndarray | list[float],
    values_b: np.ndarray | list[float],
    num_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 17,
) -> tuple[float, float, float]:
    a = _to_1d(values_a)
    b = _to_1d(values_b)
    if a.size != b.size:
        raise ValueError("values_a and values_b must have equal length")

    diffs = a - b
    return bootstrap_ci(diffs, num_resamples=num_resamples, alpha=alpha, seed=seed)
