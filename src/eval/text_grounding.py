from __future__ import annotations

import re
from typing import Mapping


_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def _normalize_label(label: str) -> str:
    return label.strip().lower()


def concept_alignment_score(mentioned_concepts: list[str], activated_concepts: list[str]) -> float:
    mentioned = {_normalize_label(value) for value in mentioned_concepts if value.strip()}
    activated = {_normalize_label(value) for value in activated_concepts if value.strip()}

    if not mentioned and not activated:
        return 0.0
    union = len(mentioned.union(activated))
    if union == 0:
        return 0.0
    return len(mentioned.intersection(activated)) / union


def finding_consistency_rate(
    predicted_findings: Mapping[str, bool],
    rationale_assertions: Mapping[str, bool],
) -> float:
    normalized_pred = {_normalize_label(key): bool(value) for key, value in predicted_findings.items()}
    normalized_rat = {_normalize_label(key): bool(value) for key, value in rationale_assertions.items()}
    overlap = set(normalized_pred).intersection(normalized_rat)
    if not overlap:
        return 0.0
    matches = sum(1 for key in overlap if normalized_pred[key] == normalized_rat[key])
    return matches / len(overlap)


def contradiction_rate(
    predicted_findings: Mapping[str, bool],
    rationale_assertions: Mapping[str, bool],
) -> float:
    consistency = finding_consistency_rate(predicted_findings, rationale_assertions)
    if consistency == 0.0 and not set(predicted_findings).intersection(rationale_assertions):
        return 0.0
    return 1.0 - consistency


def semantic_token_jaccard(text_a: str, text_b: str) -> float:
    tokens_a = set(_TOKEN_RE.findall(text_a.lower()))
    tokens_b = set(_TOKEN_RE.findall(text_b.lower()))
    if not tokens_a and not tokens_b:
        return 1.0
    union = len(tokens_a.union(tokens_b))
    if union == 0:
        return 0.0
    return len(tokens_a.intersection(tokens_b)) / union
