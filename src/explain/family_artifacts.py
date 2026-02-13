from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.data.audit import DEFAULT_FINDINGS, load_concepts_table
from src.eval.text_grounding import concept_alignment_score, contradiction_rate, finding_consistency_rate


_CURVE_SPLIT_RE = r"[,;|\s]+"

_CORE_REQUIRED_COLUMNS = [
    "method",
    "study_id",
    "sanity_similarity",
    "deletion_curve",
    "insertion_curve",
    "base_prediction",
    "perturbed_prediction",
    "nuisance_similarity",
]


@dataclass(frozen=True)
class MethodProfile:
    method_name: str
    sanity_mu: float
    nuisance_mu: float
    quality_mu: float
    prediction_shift_sigma: float
    concept_drop_prob: float
    concept_add_prob: float
    rationale_flip_prob: float


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _clean_identifier(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        return text[:-2]
    return text


def _format_curve(values: np.ndarray) -> str:
    return ",".join(f"{float(v):.6f}" for v in values.reshape(-1))


def _parse_curve(value: object) -> np.ndarray:
    if isinstance(value, np.ndarray):
        curve = value.astype(float).reshape(-1)
    elif isinstance(value, (list, tuple)):
        curve = np.asarray(value, dtype=float).reshape(-1)
    elif isinstance(value, str):
        stripped = value.strip().strip("[]")
        if not stripped:
            return np.asarray([], dtype=float)
        tokens = [token for token in pd.Series([stripped]).str.split(_CURVE_SPLIT_RE, regex=True).iloc[0] if token]
        curve = np.asarray([float(token) for token in tokens], dtype=float)
    else:
        return np.asarray([], dtype=float)
    return curve


def _monotonic_nonincreasing(values: np.ndarray) -> np.ndarray:
    output = values.copy()
    for index in range(1, output.size):
        output[index] = min(output[index - 1], output[index])
    return output


def _monotonic_nondecreasing(values: np.ndarray) -> np.ndarray:
    output = values.copy()
    for index in range(1, output.size):
        output[index] = max(output[index - 1], output[index])
    return output


def _build_curves(
    base_prediction: float,
    quality: float,
    curve_steps: int,
    rng: np.random.Generator,
) -> tuple[str, str]:
    fractions = np.linspace(0.0, 1.0, curve_steps + 1)
    deletion = base_prediction * (1.0 - quality * fractions) + rng.normal(0.0, 0.02, size=fractions.shape[0])
    insertion = (base_prediction * 0.18) + (quality * fractions * 0.78) + rng.normal(0.0, 0.02, size=fractions.shape[0])
    deletion = _monotonic_nonincreasing(np.clip(deletion, 0.0, 1.0))
    insertion = _monotonic_nondecreasing(np.clip(insertion, 0.0, 1.0))
    return _format_curve(deletion), _format_curve(insertion)


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], frame_name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{frame_name} missing required columns: {', '.join(missing)}")


def _build_concepts_map(
    concepts_file: Path | None,
    study_col: str,
    concept_col: str,
) -> dict[str, set[str]]:
    if concepts_file is None:
        return {}
    concepts_df = load_concepts_table(concepts_path=concepts_file, study_col=study_col, concept_col=concept_col)
    mapping: dict[str, set[str]] = {}
    for _, row in concepts_df.iterrows():
        study_id = _clean_identifier(row[study_col])
        concept = str(row[concept_col]).strip().lower().replace(" ", "_")
        if not concept:
            continue
        mapping.setdefault(study_id, set()).add(concept)
    return mapping


def _available_labels(frame: pd.DataFrame, candidate_labels: list[str]) -> list[str]:
    return [label for label in candidate_labels if label in frame.columns]


def _predicted_findings_from_row(row: pd.Series, labels: list[str]) -> dict[str, bool]:
    result: dict[str, bool] = {}
    for label in labels:
        value = pd.to_numeric(pd.Series([row.get(label)]), errors="coerce").iloc[0]
        result[label] = bool(pd.notna(value) and float(value) > 0.0)
    return result


def validate_artifact_frame(
    frame: pd.DataFrame,
    expected_method_name: str,
    require_text_contradiction: bool,
    curve_steps: int,
) -> None:
    required_columns = list(_CORE_REQUIRED_COLUMNS)
    if require_text_contradiction:
        required_columns.append("text_contradiction")
    _require_columns(frame, required_columns, "artifact frame")

    if frame.empty:
        raise ValueError("artifact frame is empty")

    if not frame["method"].astype(str).eq(expected_method_name).all():
        raise ValueError("artifact frame contains rows with unexpected method names")

    numeric_columns = [
        "sanity_similarity",
        "base_prediction",
        "perturbed_prediction",
        "nuisance_similarity",
    ]
    if require_text_contradiction:
        numeric_columns.append("text_contradiction")

    for column in numeric_columns:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.isna().any():
            raise ValueError(f"column {column} has non-numeric values")
        if ((numeric < 0.0) | (numeric > 1.0)).any():
            raise ValueError(f"column {column} has values outside [0, 1]")

    expected_curve_len = curve_steps + 1
    for column in ["deletion_curve", "insertion_curve"]:
        for value in frame[column].tolist():
            curve = _parse_curve(value)
            if curve.size != expected_curve_len:
                raise ValueError(f"{column} values must contain exactly {expected_curve_len} points")
            if ((curve < 0.0) | (curve > 1.0)).any():
                raise ValueError(f"{column} values must stay within [0, 1]")


def _mentioned_concepts(
    activated: set[str],
    concept_pool: list[str],
    drop_prob: float,
    add_prob: float,
    rng: np.random.Generator,
) -> list[str]:
    kept = [concept for concept in sorted(activated) if rng.uniform() >= drop_prob]
    additions = [concept for concept in concept_pool if concept not in activated and rng.uniform() < add_prob]
    return sorted(set(kept + additions))


def generate_family_artifacts(
    cohort_df: pd.DataFrame,
    profile: MethodProfile,
    labels: list[str],
    study_col: str,
    split_col: str,
    target_split: str,
    curve_steps: int,
    seed: int,
    include_text_columns: bool,
    include_concept_intervention: bool,
    concepts_map: dict[str, set[str]] | None = None,
) -> pd.DataFrame:
    _require_columns(cohort_df, [study_col, split_col], "cohort")
    concepts_map = concepts_map or {}

    if curve_steps <= 1:
        raise ValueError("curve_steps must be > 1")

    available_labels = _available_labels(cohort_df, labels)
    if not available_labels:
        raise ValueError("cohort does not contain any requested finding labels")

    frame = cohort_df.copy()
    if target_split:
        frame = frame[frame[split_col].astype(str) == target_split].copy()
    if frame.empty:
        raise ValueError("cohort filtering produced an empty frame")

    rng = np.random.default_rng(seed)
    concept_pool = sorted({label.lower().replace(" ", "_") for label in available_labels})
    rows: list[dict[str, object]] = []

    for _, row in frame.iterrows():
        study_id_clean = _clean_identifier(row[study_col])
        predicted_findings = _predicted_findings_from_row(row, available_labels)

        positive_count = sum(predicted_findings.values())
        unknown_count = sum(
            int(pd.isna(pd.to_numeric(pd.Series([row.get(label)]), errors="coerce").iloc[0])) for label in available_labels
        )

        base_prediction = _clip01(0.34 + (0.09 * positive_count) - (0.01 * unknown_count) + rng.normal(0.0, 0.03))
        perturbed_prediction = _clip01(base_prediction + rng.normal(0.0, profile.prediction_shift_sigma))
        sanity_similarity = _clip01(profile.sanity_mu + rng.normal(0.0, 0.03))
        nuisance_similarity = _clip01(profile.nuisance_mu + rng.normal(0.0, 0.04))
        quality = _clip01(profile.quality_mu + rng.normal(0.0, 0.04))
        deletion_curve, insertion_curve = _build_curves(
            base_prediction=base_prediction,
            quality=quality,
            curve_steps=curve_steps,
            rng=rng,
        )

        activated_concepts = set(concept_pool_label for concept_pool_label, present in predicted_findings.items() if present)
        activated_concepts = {value.lower().replace(" ", "_") for value in activated_concepts}
        activated_concepts.update(concepts_map.get(study_id_clean, set()))

        mentioned = _mentioned_concepts(
            activated=activated_concepts,
            concept_pool=concept_pool,
            drop_prob=profile.concept_drop_prob,
            add_prob=profile.concept_add_prob,
            rng=rng,
        )
        concept_alignment = float(concept_alignment_score(mentioned_concepts=mentioned, activated_concepts=sorted(activated_concepts)))

        output_row: dict[str, object] = {
            "method": profile.method_name,
            "study_id": study_id_clean,
            "sanity_similarity": sanity_similarity,
            "deletion_curve": deletion_curve,
            "insertion_curve": insertion_curve,
            "base_prediction": base_prediction,
            "perturbed_prediction": perturbed_prediction,
            "nuisance_similarity": nuisance_similarity,
            "concept_alignment": concept_alignment,
            "generator_mode": "proxy",
            "split": str(row.get(split_col, "")),
        }

        if include_concept_intervention:
            intervention_effect = _clip01(0.16 + (0.40 * quality) + (0.20 * concept_alignment) + rng.normal(0.0, 0.03))
            output_row["concept_intervention_effect"] = intervention_effect

        if include_text_columns:
            rationale_assertions: dict[str, bool] = {}
            for label in available_labels:
                predicted = predicted_findings[label]
                flip = bool(rng.uniform() < profile.rationale_flip_prob)
                rationale_assertions[label] = (not predicted) if flip else predicted
            finding_consistency = float(
                finding_consistency_rate(
                    predicted_findings=predicted_findings,
                    rationale_assertions=rationale_assertions,
                )
            )
            text_contradiction = float(
                contradiction_rate(
                    predicted_findings=predicted_findings,
                    rationale_assertions=rationale_assertions,
                )
            )
            output_row["finding_consistency"] = finding_consistency
            output_row["text_contradiction"] = _clip01(text_contradiction)

        rows.append(output_row)

    output = pd.DataFrame(rows)
    output = output.sort_values(by=["study_id"]).reset_index(drop=True)
    validate_artifact_frame(
        frame=output,
        expected_method_name=profile.method_name,
        require_text_contradiction=include_text_columns,
        curve_steps=curve_steps,
    )
    return output


def build_common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--cohort-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--meta-json", type=Path, required=True)
    parser.add_argument("--study-col", default="study_id")
    parser.add_argument("--split-col", default="split")
    parser.add_argument("--target-split", default="test")
    parser.add_argument("--labels", nargs="+", default=DEFAULT_FINDINGS)
    parser.add_argument("--curve-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--concepts-file", type=Path, default=None)
    parser.add_argument("--concept-col", default="concept")
    return parser


def run_generator(
    args: argparse.Namespace,
    profile: MethodProfile,
    include_text_columns: bool,
    include_concept_intervention: bool,
) -> None:
    cohort_df = pd.read_csv(args.cohort_csv)
    concepts_map = _build_concepts_map(
        concepts_file=args.concepts_file,
        study_col=args.study_col,
        concept_col=args.concept_col,
    )

    artifacts = generate_family_artifacts(
        cohort_df=cohort_df,
        profile=profile,
        labels=list(args.labels),
        study_col=args.study_col,
        split_col=args.split_col,
        target_split=args.target_split,
        curve_steps=int(args.curve_steps),
        seed=int(args.seed),
        include_text_columns=include_text_columns,
        include_concept_intervention=include_concept_intervention,
        concepts_map=concepts_map,
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.meta_json.parent.mkdir(parents=True, exist_ok=True)
    artifacts.to_csv(args.output_csv, index=False)

    meta = {
        "method": profile.method_name,
        "generator_mode": "proxy",
        "cohort_csv": str(args.cohort_csv),
        "output_csv": str(args.output_csv),
        "study_col": args.study_col,
        "split_col": args.split_col,
        "target_split": args.target_split,
        "curve_steps": int(args.curve_steps),
        "seed": int(args.seed),
        "n_rows": int(artifacts.shape[0]),
        "labels_used": [label for label in args.labels if label in cohort_df.columns],
        "concepts_file": str(args.concepts_file) if args.concepts_file is not None else "",
        "include_text_columns": bool(include_text_columns),
        "include_concept_intervention": bool(include_concept_intervention),
        "warning": "Proxy generator output; replace with model-backed generation for final thesis claims.",
    }
    with args.meta_json.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)

