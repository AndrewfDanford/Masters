from src.eval.quality_gates import (
    FaithfulnessScores,
    FaithfulnessThresholds,
    evaluate_quality_gates,
)


def test_quality_gate_pass_case() -> None:
    thresholds = FaithfulnessThresholds(
        sanity_min=0.5,
        perturbation_min=0.5,
        robustness_min=0.5,
        nfi_min=0.6,
        text_contradiction_max=0.2,
    )
    scores = FaithfulnessScores(
        sanity=0.7,
        perturbation=0.65,
        robustness=0.72,
        nfi=0.69,
        text_contradiction=0.1,
    )
    gates = evaluate_quality_gates(scores, thresholds)
    assert gates["overall_pass"] is True


def test_quality_gate_fail_case() -> None:
    thresholds = FaithfulnessThresholds(
        sanity_min=0.6,
        perturbation_min=0.6,
        robustness_min=0.6,
        nfi_min=0.65,
    )
    scores = FaithfulnessScores(
        sanity=0.7,
        perturbation=0.5,
        robustness=0.7,
        nfi=0.62,
    )
    gates = evaluate_quality_gates(scores, thresholds)
    assert gates["overall_pass"] is False
    assert gates["perturbation_pass"] is False
