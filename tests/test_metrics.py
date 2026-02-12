import numpy as np

from src.utils.metrics import (
    binary_auprc,
    binary_auroc,
    binary_brier,
    expected_calibration_error,
)


def test_binary_auroc_perfect_ranking() -> None:
    y_true = [0, 0, 1, 1]
    y_prob = [0.1, 0.2, 0.8, 0.9]
    assert np.isclose(binary_auroc(y_true, y_prob), 1.0)


def test_binary_auprc_reasonable_range() -> None:
    y_true = [0, 1, 0, 1, 1]
    y_prob = [0.1, 0.7, 0.2, 0.8, 0.9]
    score = binary_auprc(y_true, y_prob)
    assert 0.0 <= score <= 1.0


def test_brier_and_ece_non_negative() -> None:
    y_true = [0, 1, 1, 0, 1]
    y_prob = [0.2, 0.6, 0.9, 0.3, 0.7]
    assert binary_brier(y_true, y_prob) >= 0.0
    assert expected_calibration_error(y_true, y_prob, num_bins=5) >= 0.0

