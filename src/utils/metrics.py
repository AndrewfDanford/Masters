from __future__ import annotations

import numpy as np


def _to_binary_array(values: np.ndarray | list[float] | list[int]) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        raise ValueError("expected a non-empty array")
    if np.any((array != 0) & (array != 1)):
        raise ValueError("binary targets must contain only 0/1 values")
    return array


def _to_prob_array(values: np.ndarray | list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        raise ValueError("expected a non-empty array")
    return np.clip(array, 0.0, 1.0)


def binary_auroc(y_true: np.ndarray | list[int], y_score: np.ndarray | list[float]) -> float:
    truth = _to_binary_array(y_true)
    score = _to_prob_array(y_score)
    if truth.size != score.size:
        raise ValueError("y_true and y_score must have the same length")

    positives = truth == 1
    negatives = truth == 0
    num_pos = int(np.sum(positives))
    num_neg = int(np.sum(negatives))
    if num_pos == 0 or num_neg == 0:
        return float("nan")

    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]

    ranks = np.empty_like(sorted_score, dtype=float)
    start = 0
    n = sorted_score.size
    while start < n:
        end = start + 1
        while end < n and sorted_score[end] == sorted_score[start]:
            end += 1
        mean_rank = (start + end - 1) / 2.0 + 1.0
        ranks[start:end] = mean_rank
        start = end

    ranked = np.empty_like(ranks)
    ranked[order] = ranks
    rank_sum_pos = float(np.sum(ranked[positives]))
    u_stat = rank_sum_pos - (num_pos * (num_pos + 1) / 2.0)
    return float(u_stat / (num_pos * num_neg))


def binary_auprc(y_true: np.ndarray | list[int], y_score: np.ndarray | list[float]) -> float:
    truth = _to_binary_array(y_true)
    score = _to_prob_array(y_score)
    if truth.size != score.size:
        raise ValueError("y_true and y_score must have the same length")

    num_pos = int(np.sum(truth == 1))
    if num_pos == 0:
        return float("nan")

    order = np.argsort(-score, kind="mergesort")
    sorted_truth = truth[order]

    tp = np.cumsum(sorted_truth)
    fp = np.cumsum(1.0 - sorted_truth)
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / num_pos

    precision = np.concatenate(([1.0], precision))
    recall = np.concatenate(([0.0], recall))
    return float(np.trapezoid(precision, recall))


def binary_brier(y_true: np.ndarray | list[int], y_score: np.ndarray | list[float]) -> float:
    truth = _to_binary_array(y_true)
    score = _to_prob_array(y_score)
    if truth.size != score.size:
        raise ValueError("y_true and y_score must have the same length")
    return float(np.mean((score - truth) ** 2))


def expected_calibration_error(
    y_true: np.ndarray | list[int],
    y_score: np.ndarray | list[float],
    num_bins: int = 10,
) -> float:
    if num_bins <= 1:
        raise ValueError("num_bins must be >= 2")

    truth = _to_binary_array(y_true)
    score = _to_prob_array(y_score)
    if truth.size != score.size:
        raise ValueError("y_true and y_score must have the same length")

    bins = np.linspace(0.0, 1.0, num_bins + 1)
    bin_ids = np.digitize(score, bins[1:-1], right=False)

    n = score.size
    ece = 0.0
    for bin_id in range(num_bins):
        in_bin = bin_ids == bin_id
        count = int(np.sum(in_bin))
        if count == 0:
            continue
        avg_conf = float(np.mean(score[in_bin]))
        avg_acc = float(np.mean(truth[in_bin]))
        ece += (count / n) * abs(avg_acc - avg_conf)
    return float(ece)

