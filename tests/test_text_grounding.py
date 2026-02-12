import numpy as np

from src.eval.text_grounding import (
    concept_alignment_score,
    contradiction_rate,
    finding_consistency_rate,
    semantic_token_jaccard,
)


def test_concept_alignment_score() -> None:
    score = concept_alignment_score(
        mentioned_concepts=["Consolidation", "Pleural Effusion"],
        activated_concepts=["pleural effusion", "edema"],
    )
    assert np.isclose(score, 1.0 / 3.0)


def test_finding_consistency_and_contradiction() -> None:
    predicted = {"Pneumonia": True, "Edema": False}
    rationale = {"pneumonia": True, "edema": True}

    consistency = finding_consistency_rate(predicted, rationale)
    contradiction = contradiction_rate(predicted, rationale)
    assert np.isclose(consistency, 0.5)
    assert np.isclose(contradiction, 0.5)


def test_semantic_token_jaccard() -> None:
    a = "Consolidation in right lower lobe"
    b = "Right lower lobe consolidation is present"
    c = "No pleural effusion seen"

    assert semantic_token_jaccard(a, b) > semantic_token_jaccard(a, c)
