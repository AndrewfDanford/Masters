from __future__ import annotations

import json
from pathlib import Path

from .faithfulness import (
    deletion_insertion_score,
    normalized_faithfulness_index,
    nuisance_robustness_score,
    prediction_stability_mask,
    sanity_score_from_similarities,
)
from .quality_gates import FaithfulnessScores, evaluate_quality_gates, load_thresholds


def main() -> None:
    randomized_similarities = [0.12, 0.21, 0.18]
    sanity_score = sanity_score_from_similarities(randomized_similarities)

    deletion_curve = [1.0, 0.74, 0.49, 0.27, 0.12]
    insertion_curve = [0.08, 0.31, 0.56, 0.81, 0.94]
    perturbation_score = deletion_insertion_score(deletion_curve, insertion_curve)

    base_predictions = [0.91, 0.77, 0.62, 0.80, 0.55]
    perturbed_predictions = [0.89, 0.79, 0.66, 0.65, 0.58]
    explanation_similarities = [0.93, 0.90, 0.84, 0.51, 0.82]
    stable_mask = prediction_stability_mask(base_predictions, perturbed_predictions, abs_tolerance=0.05)
    stability = nuisance_robustness_score(explanation_similarities, stable_mask)

    nfi = normalized_faithfulness_index(
        sanity_score=sanity_score,
        perturbation_score=perturbation_score,
        stability_score=stability.score,
    )

    thresholds = load_thresholds(path=Path("configs/eval/faithfulness_thresholds.json"))
    scores = FaithfulnessScores(
        sanity=sanity_score,
        perturbation=perturbation_score,
        robustness=stability.score,
        nfi=nfi,
        text_contradiction=0.12,
    )
    gates = evaluate_quality_gates(scores=scores, thresholds=thresholds)

    output = {
        "scores": {
            "sanity": round(sanity_score, 4),
            "perturbation": round(perturbation_score, 4),
            "robustness": round(stability.score, 4),
            "nfi": round(nfi, 4),
            "stable_fraction": round(stability.stable_fraction, 4),
        },
        "quality_gates": gates,
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
