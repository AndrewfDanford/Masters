from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _to_1d(values: np.ndarray | list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        raise ValueError("expected a non-empty numeric array")
    return array


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Return average ranks for ties, 1-indexed."""
    values = _to_1d(values)
    order = np.argsort(values, kind="mergesort")
    sorted_vals = values[order]

    ranks = np.empty_like(sorted_vals, dtype=float)
    start = 0
    n = sorted_vals.size
    while start < n:
        end = start + 1
        while end < n and sorted_vals[end] == sorted_vals[start]:
            end += 1
        avg_rank = (start + end - 1) / 2.0 + 1.0
        ranks[start:end] = avg_rank
        start = end

    output = np.empty_like(ranks)
    output[order] = ranks
    return output


def _pearson_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = _to_1d(a)
    b = _to_1d(b)
    if a.size != b.size:
        raise ValueError("arrays must have the same length")

    a_centered = a - a.mean()
    b_centered = b - b.mean()
    a_std = np.sqrt(np.dot(a_centered, a_centered))
    b_std = np.sqrt(np.dot(b_centered, b_centered))

    if a_std == 0 or b_std == 0:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.dot(a_centered, b_centered) / (a_std * b_std))


def spearman_similarity(values_a: np.ndarray | list[float], values_b: np.ndarray | list[float]) -> float:
    """Return Spearman rank similarity mapped from [-1, 1] to [0, 1]."""
    rank_a = _rankdata(_to_1d(values_a))
    rank_b = _rankdata(_to_1d(values_b))
    rho = _pearson_correlation(rank_a, rank_b)
    return _clip01((rho + 1.0) / 2.0)


def cosine_similarity(values_a: np.ndarray | list[float], values_b: np.ndarray | list[float]) -> float:
    a = _to_1d(values_a)
    b = _to_1d(values_b)
    if a.size != b.size:
        raise ValueError("arrays must have the same length")
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0:
        return 1.0 if np.allclose(a, b) else 0.0
    similarity = float(np.dot(a, b) / denominator)
    return _clip01((similarity + 1.0) / 2.0)


def saliency_similarity(map_a: np.ndarray, map_b: np.ndarray) -> float:
    return spearman_similarity(np.asarray(map_a).reshape(-1), np.asarray(map_b).reshape(-1))


def topk_overlap_similarity(
    scores_a: np.ndarray | list[float],
    scores_b: np.ndarray | list[float],
    k_fraction: float = 0.1,
) -> float:
    if not (0.0 < k_fraction <= 1.0):
        raise ValueError("k_fraction must be in (0, 1]")
    a = _to_1d(scores_a)
    b = _to_1d(scores_b)
    if a.size != b.size:
        raise ValueError("arrays must have the same length")

    k = max(1, int(round(a.size * k_fraction)))
    idx_a = set(np.argpartition(a, -k)[-k:])
    idx_b = set(np.argpartition(b, -k)[-k:])
    union = len(idx_a.union(idx_b))
    if union == 0:
        return 0.0
    return float(len(idx_a.intersection(idx_b)) / union)


def sanity_score_from_similarities(similarities: np.ndarray | list[float]) -> float:
    """
    Convert original-vs-randomized explanation similarities into a sanity score.
    Lower similarity after randomization implies better faithfulness.
    """
    sims = _to_1d(similarities)
    sims = np.clip(sims, 0.0, 1.0)
    return _clip01(1.0 - float(np.mean(sims)))


def deletion_insertion_score(
    deletion_curve: np.ndarray | list[float],
    insertion_curve: np.ndarray | list[float],
    fractions: np.ndarray | list[float] | None = None,
) -> float:
    deletion = np.clip(_to_1d(deletion_curve), 0.0, 1.0)
    insertion = np.clip(_to_1d(insertion_curve), 0.0, 1.0)
    if deletion.size != insertion.size:
        raise ValueError("deletion and insertion curves must have the same length")

    if fractions is None:
        x = np.linspace(0.0, 1.0, deletion.size)
    else:
        x = _to_1d(fractions)
        if x.size != deletion.size:
            raise ValueError("fractions must match curve length")

    deletion_auc = float(np.trapezoid(deletion, x))
    insertion_auc = float(np.trapezoid(insertion, x))
    deletion_component = 1.0 - deletion_auc
    insertion_component = insertion_auc
    return _clip01((deletion_component + insertion_component) / 2.0)


def prediction_stability_mask(
    base_predictions: np.ndarray | list[float],
    perturbed_predictions: np.ndarray | list[float],
    abs_tolerance: float = 0.05,
) -> np.ndarray:
    if abs_tolerance < 0:
        raise ValueError("abs_tolerance must be non-negative")
    base = _to_1d(base_predictions)
    perturbed = _to_1d(perturbed_predictions)
    if base.size != perturbed.size:
        raise ValueError("base and perturbed arrays must have the same length")
    return np.abs(base - perturbed) <= abs_tolerance


@dataclass(frozen=True)
class StabilityResult:
    score: float
    num_stable: int
    stable_fraction: float


def nuisance_robustness_score(
    explanation_similarities: np.ndarray | list[float],
    stable_prediction_mask: np.ndarray | list[bool],
) -> StabilityResult:
    similarities = np.clip(_to_1d(explanation_similarities), 0.0, 1.0)
    mask = np.asarray(stable_prediction_mask, dtype=bool).reshape(-1)
    if similarities.size != mask.size:
        raise ValueError("similarities and mask must have the same length")

    stable_count = int(mask.sum())
    stable_fraction = float(stable_count / mask.size) if mask.size else 0.0
    if stable_count == 0:
        return StabilityResult(score=0.0, num_stable=0, stable_fraction=stable_fraction)

    score = float(np.mean(similarities[mask]))
    return StabilityResult(score=_clip01(score), num_stable=stable_count, stable_fraction=stable_fraction)


def normalized_faithfulness_index(
    sanity_score: float,
    perturbation_score: float,
    stability_score: float,
    weights: tuple[float, float, float] | None = None,
) -> float:
    s = _clip01(sanity_score)
    p = _clip01(perturbation_score)
    r = _clip01(stability_score)

    if weights is None:
        return (s + p + r) / 3.0

    w = np.asarray(weights, dtype=float).reshape(-1)
    if w.size != 3:
        raise ValueError("weights must contain exactly three values")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    if np.sum(w) == 0:
        raise ValueError("weights must sum to a positive value")
    w = w / np.sum(w)
    return float(np.dot(np.array([s, p, r]), w))
