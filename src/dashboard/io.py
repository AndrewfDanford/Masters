from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def _existing(path: Path | None) -> bool:
    return bool(path is not None and path.exists())


def _resolve_preferred_or_latest(
    directory: Path,
    preferred_names: list[str],
    fallback_pattern: str,
) -> Path | None:
    for name in preferred_names:
        candidate = directory / name
        if candidate.exists():
            return candidate

    candidates = sorted(
        directory.glob(fallback_pattern),
        key=lambda value: value.stat().st_mtime if value.exists() else 0.0,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _resolve_preferred_or_latest_multi(
    directories: list[Path],
    preferred_names: list[str],
    fallback_pattern: str,
) -> Path | None:
    preferred_candidates: list[Path] = []
    for directory in directories:
        for name in preferred_names:
            candidate = directory / name
            if candidate.exists():
                preferred_candidates.append(candidate)

    if preferred_candidates:
        preferred_candidates = sorted(
            preferred_candidates,
            key=lambda value: value.stat().st_mtime if value.exists() else 0.0,
            reverse=True,
        )
        return preferred_candidates[0]

    candidates: list[Path] = []
    for directory in directories:
        candidates.extend(directory.glob(fallback_pattern))
    candidates = sorted(
        candidates,
        key=lambda value: value.stat().st_mtime if value.exists() else 0.0,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _preferred_with_fallback(primary: Path, fallback: Path) -> Path:
    return primary if primary.exists() else fallback


def parse_curve_string(value: object) -> np.ndarray:
    if isinstance(value, np.ndarray):
        array = value.astype(float).reshape(-1)
    elif isinstance(value, (list, tuple)):
        array = np.asarray(value, dtype=float).reshape(-1)
    elif isinstance(value, str):
        stripped = value.strip().strip("[]")
        if not stripped:
            return np.asarray([], dtype=float)
        parts = [part.strip() for part in stripped.replace(";", ",").split(",") if part.strip()]
        array = np.asarray([float(part) for part in parts], dtype=float)
    else:
        return np.asarray([], dtype=float)
    return array


def read_csv_optional(path: Path | None) -> pd.DataFrame | None:
    if not _existing(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def read_json_optional(path: Path | None) -> dict[str, object] | None:
    if not _existing(path):
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            return payload
        return {"value": payload}
    except Exception:
        return None


def format_timestamp(path: Path | None) -> str:
    if not _existing(path):
        return ""
    return pd.Timestamp(path.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M:%S")


def repo_commit(project_root: Path) -> str:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=project_root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return output.strip()
    except Exception:
        return "unknown"


def repo_branch(project_root: Path) -> str:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=project_root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return output.strip()
    except Exception:
        return "unknown"


def collect_artifact_paths(project_root: Path) -> dict[str, Path | None]:
    reports_root = project_root / "outputs" / "reports"
    smoke_root = project_root / "outputs" / "smoke"
    e1_root = reports_root / "e1"
    e1_cnn_root = reports_root / "e1_cnn"
    e23_root = reports_root / "e2_e3"
    e4_root = reports_root / "e4"
    e5_root = reports_root / "e5"
    e6_root = reports_root / "e6"
    e7_root = reports_root / "e7"
    e8_root = reports_root / "e8"
    smoke_e1_root = smoke_root / "e1"
    smoke_e23_root = smoke_root / "e2_e3"
    smoke_e4_root = smoke_root / "e4"
    smoke_e5_root = smoke_root / "e5"
    smoke_e6_root = smoke_root / "e6"
    smoke_e7_root = smoke_root / "e7"
    smoke_e8_root = smoke_root / "e8"

    paths: dict[str, Path | None] = {
        "e0_manifest": reports_root / "e0_cohort_manifest.csv",
        "e0_split_counts": reports_root / "e0_split_counts.csv",
        "e0_label_prevalence": reports_root / "e0_label_prevalence.csv",
        "e0_concept_coverage": reports_root / "e0_concept_coverage_summary.csv",
        "e1_metrics_summary": _preferred_with_fallback(
            e1_root / "e1_metrics_summary.csv",
            smoke_e1_root / "e1_metrics_summary.csv",
        ),
        "e1_metrics_by_label": _preferred_with_fallback(
            e1_root / "e1_metrics_by_label.csv",
            smoke_e1_root / "e1_metrics_by_label.csv",
        ),
        "e1_train_history": _preferred_with_fallback(
            e1_root / "e1_train_history.csv",
            smoke_e1_root / "e1_train_history.csv",
        ),
        "e1_model_card": _preferred_with_fallback(
            e1_root / "e1_model_card.json",
            smoke_e1_root / "e1_model_card.json",
        ),
        "e1_cnn_metrics_summary": e1_cnn_root / "e1_cnn_metrics_summary.csv",
        "e1_cnn_metrics_by_label": e1_cnn_root / "e1_cnn_metrics_by_label.csv",
        "e1_cnn_train_history": e1_cnn_root / "e1_cnn_train_history.csv",
        "e1_cnn_dataset_counts": e1_cnn_root / "e1_cnn_dataset_counts.csv",
        "e1_cnn_run_meta": e1_cnn_root / "e1_cnn_run_meta.json",
        "e23_artifacts": _preferred_with_fallback(
            e23_root / "e2e3_artifacts.csv",
            smoke_root / "synthetic_e23_artifacts.csv",
        ),
        "e23_generation_meta": e23_root / "e2e3_generation_meta.json",
        "e23_method_summary": _preferred_with_fallback(
            e23_root / "e2e3_saliency_method_summary.csv",
            smoke_e23_root / "e2e3_saliency_method_summary.csv",
        ),
        "e23_sample_scores": _preferred_with_fallback(
            e23_root / "e2e3_saliency_sample_scores.csv",
            smoke_e23_root / "e2e3_saliency_sample_scores.csv",
        ),
        "e23_pairwise": _preferred_with_fallback(
            e23_root / "e2e3_saliency_pairwise_nfi_deltas.csv",
            smoke_e23_root / "e2e3_saliency_pairwise_nfi_deltas.csv",
        ),
        "e4_artifacts": _preferred_with_fallback(
            e4_root / "e4_artifacts.csv",
            smoke_e4_root / "e4_artifacts.csv",
        ),
        "e4_meta": _preferred_with_fallback(
            e4_root / "e4_generation_meta.json",
            smoke_e4_root / "e4_generation_meta.json",
        ),
        "e5_artifacts": _preferred_with_fallback(
            e5_root / "e5_artifacts.csv",
            smoke_e5_root / "e5_artifacts.csv",
        ),
        "e5_meta": _preferred_with_fallback(
            e5_root / "e5_generation_meta.json",
            smoke_e5_root / "e5_generation_meta.json",
        ),
        "e6_artifacts": _preferred_with_fallback(
            e6_root / "e6_artifacts.csv",
            smoke_e6_root / "e6_artifacts.csv",
        ),
        "e6_meta": _preferred_with_fallback(
            e6_root / "e6_generation_meta.json",
            smoke_e6_root / "e6_generation_meta.json",
        ),
        "e7_input_all_methods": _preferred_with_fallback(
            e7_root / "e7_input_all_methods.csv",
            smoke_e7_root / "e7_input_all_methods.csv",
        ),
        "e7_input_meta": _preferred_with_fallback(
            e7_root / "e7_input_all_methods_meta.json",
            smoke_e7_root / "e7_input_all_methods_meta.json",
        ),
    }

    e7_dirs = [e7_root, smoke_e7_root]
    paths["e7_method_summary"] = _resolve_preferred_or_latest_multi(
        directories=e7_dirs,
        preferred_names=["e7_method_summary.csv"],
        fallback_pattern="*_method_summary.csv",
    )
    paths["e7_sample_scores"] = _resolve_preferred_or_latest_multi(
        directories=e7_dirs,
        preferred_names=["e7_sample_scores.csv"],
        fallback_pattern="*_sample_scores.csv",
    )
    paths["e7_pairwise"] = _resolve_preferred_or_latest_multi(
        directories=e7_dirs,
        preferred_names=["e7_pairwise_nfi_deltas.csv"],
        fallback_pattern="*_pairwise_nfi_deltas.csv",
    )

    e8_dirs = [e8_root, smoke_e8_root]
    paths["e8_method_summary"] = _resolve_preferred_or_latest_multi(
        directories=e8_dirs,
        preferred_names=["e8_randomization_method_summary.csv", "e8_smoke_method_summary.csv"],
        fallback_pattern="*_method_summary.csv",
    )
    paths["e8_run_scores"] = _resolve_preferred_or_latest_multi(
        directories=e8_dirs,
        preferred_names=["e8_randomization_run_scores.csv", "e8_smoke_run_scores.csv"],
        fallback_pattern="*_run_scores.csv",
    )
    paths["e8_sample_variability"] = _resolve_preferred_or_latest_multi(
        directories=e8_dirs,
        preferred_names=["e8_randomization_sample_variability.csv", "e8_smoke_sample_variability.csv"],
        fallback_pattern="*_sample_variability.csv",
    )
    paths["e8_meta"] = _resolve_preferred_or_latest_multi(
        directories=e8_dirs,
        preferred_names=["e8_randomization_meta.json", "e8_smoke_meta.json"],
        fallback_pattern="*_meta.json",
    )

    return paths


@dataclass(frozen=True)
class StageStatus:
    stage: str
    status: str
    completed_artifacts: int
    total_artifacts: int
    last_updated: str


def infer_stage_status(paths: dict[str, Path | None]) -> pd.DataFrame:
    stage_specs = [
        ("E0 Audit", ["e0_manifest", "e0_split_counts", "e0_label_prevalence"]),
        ("E1 Baseline", ["e1_metrics_summary", "e1_train_history"]),
        ("E1 CNN", ["e1_cnn_metrics_summary", "e1_cnn_train_history", "e1_cnn_run_meta"]),
        ("E2/E3 Generation", ["e23_artifacts"]),
        ("E2/E3 Scoring", ["e23_method_summary", "e23_sample_scores"]),
        ("E4 Concept", ["e4_artifacts"]),
        ("E7 Unified", ["e7_method_summary", "e7_sample_scores"]),
        ("E8 Randomization", ["e8_method_summary", "e8_run_scores"]),
    ]

    rows: list[StageStatus] = []
    for stage_name, keys in stage_specs:
        stage_paths = [paths.get(key) for key in keys]
        exists = [_existing(path) for path in stage_paths]
        completed = int(sum(exists))
        total = len(keys)

        if completed == total:
            status = "done"
        elif completed == 0:
            status = "not_started"
        else:
            status = "partial"

        existing_paths = [path for path in stage_paths if _existing(path)]
        if existing_paths:
            latest_path = max(existing_paths, key=lambda value: value.stat().st_mtime)
            last_updated = format_timestamp(latest_path)
        else:
            last_updated = ""

        rows.append(
            StageStatus(
                stage=stage_name,
                status=status,
                completed_artifacts=completed,
                total_artifacts=total,
                last_updated=last_updated,
            )
        )

    return pd.DataFrame([row.__dict__ for row in rows])
