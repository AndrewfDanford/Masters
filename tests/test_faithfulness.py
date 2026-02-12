import numpy as np

from src.eval.faithfulness import (
    deletion_insertion_score,
    normalized_faithfulness_index,
    nuisance_robustness_score,
    prediction_stability_mask,
    saliency_similarity,
    sanity_score_from_similarities,
    topk_overlap_similarity,
)


def test_sanity_score_drops_with_high_randomized_similarity() -> None:
    low_degradation = sanity_score_from_similarities([0.9, 0.85, 0.95])
    high_degradation = sanity_score_from_similarities([0.2, 0.3, 0.1])
    assert high_degradation > low_degradation


def test_deletion_insertion_score_prefers_informative_curves() -> None:
    informative = deletion_insertion_score(
        deletion_curve=[1.0, 0.6, 0.35, 0.2, 0.1],
        insertion_curve=[0.1, 0.35, 0.6, 0.82, 0.95],
    )
    flat = deletion_insertion_score(
        deletion_curve=[0.5, 0.5, 0.5, 0.5, 0.5],
        insertion_curve=[0.5, 0.5, 0.5, 0.5, 0.5],
    )
    assert informative > flat


def test_prediction_stability_mask_and_robustness_conditioning() -> None:
    base = [0.8, 0.9, 0.4, 0.7]
    pert = [0.82, 0.89, 0.52, 0.69]
    mask = prediction_stability_mask(base, pert, abs_tolerance=0.05)
    assert mask.tolist() == [True, True, False, True]

    stability = nuisance_robustness_score(
        explanation_similarities=[0.9, 0.8, 0.2, 0.7],
        stable_prediction_mask=mask,
    )
    assert stability.num_stable == 3
    assert np.isclose(stability.score, np.mean([0.9, 0.8, 0.7]))


def test_nfi_weighted_and_unweighted() -> None:
    unweighted = normalized_faithfulness_index(0.6, 0.7, 0.8)
    weighted = normalized_faithfulness_index(0.6, 0.7, 0.8, weights=(1, 2, 1))
    assert np.isclose(unweighted, (0.6 + 0.7 + 0.8) / 3.0)
    assert weighted != unweighted


def test_similarity_helpers() -> None:
    map_a = np.array([[0.1, 0.2], [0.7, 0.8]])
    map_b = np.array([[0.1, 0.2], [0.7, 0.8]])
    assert np.isclose(saliency_similarity(map_a, map_b), 1.0)

    overlap = topk_overlap_similarity([0.9, 0.8, 0.1, 0.0], [0.95, 0.85, 0.1, 0.0], k_fraction=0.5)
    assert np.isclose(overlap, 1.0)
