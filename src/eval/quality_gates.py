from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FaithfulnessScores:
    sanity: float
    perturbation: float
    robustness: float
    nfi: float
    text_contradiction: float | None = None


@dataclass(frozen=True)
class FaithfulnessThresholds:
    sanity_min: float
    perturbation_min: float
    robustness_min: float
    nfi_min: float
    text_contradiction_max: float | None = None


def load_thresholds(path: Path) -> FaithfulnessThresholds:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    return FaithfulnessThresholds(
        sanity_min=float(data["sanity_min"]),
        perturbation_min=float(data["perturbation_min"]),
        robustness_min=float(data["robustness_min"]),
        nfi_min=float(data["nfi_min"]),
        text_contradiction_max=(
            float(data["text_contradiction_max"])
            if data.get("text_contradiction_max") is not None
            else None
        ),
    )


def evaluate_quality_gates(
    scores: FaithfulnessScores,
    thresholds: FaithfulnessThresholds,
) -> dict[str, bool]:
    gates = {
        "sanity_pass": scores.sanity >= thresholds.sanity_min,
        "perturbation_pass": scores.perturbation >= thresholds.perturbation_min,
        "robustness_pass": scores.robustness >= thresholds.robustness_min,
        "nfi_pass": scores.nfi >= thresholds.nfi_min,
    }

    if thresholds.text_contradiction_max is not None and scores.text_contradiction is not None:
        gates["text_contradiction_pass"] = scores.text_contradiction <= thresholds.text_contradiction_max

    gates["overall_pass"] = all(gates.values())
    return gates
