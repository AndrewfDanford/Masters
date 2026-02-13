from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from src.explain.e4_concept_generate import PROFILE as E4_PROFILE
from src.explain.e5_text_generate import PROFILE as E5_PROFILE
from src.explain.e6_text_generate import PROFILE as E6_PROFILE
from src.explain.family_artifacts import generate_family_artifacts, run_generator, validate_artifact_frame


def _build_cohort() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "study_id": 1001,
                "split": "test",
                "Atelectasis": 1,
                "Cardiomegaly": 0,
                "Consolidation": 1,
                "Edema": 0,
                "Pleural Effusion": 1,
                "Pneumothorax": 0,
            },
            {
                "study_id": 1002,
                "split": "test",
                "Atelectasis": 0,
                "Cardiomegaly": 1,
                "Consolidation": 0,
                "Edema": 0,
                "Pleural Effusion": 0,
                "Pneumothorax": 1,
            },
            {
                "study_id": 1003,
                "split": "train",
                "Atelectasis": 1,
                "Cardiomegaly": 0,
                "Consolidation": 0,
                "Edema": 1,
                "Pleural Effusion": 0,
                "Pneumothorax": 0,
            },
        ]
    )


def test_generate_family_artifacts_e4_contract() -> None:
    cohort = _build_cohort()
    frame = generate_family_artifacts(
        cohort_df=cohort,
        profile=E4_PROFILE,
        labels=["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion", "Pneumothorax"],
        study_col="study_id",
        split_col="split",
        target_split="test",
        curve_steps=10,
        seed=13,
        include_text_columns=False,
        include_concept_intervention=True,
    )

    assert set(frame["method"].unique()) == {"concept_cbm"}
    assert "concept_intervention_effect" in frame.columns
    assert "text_contradiction" not in frame.columns
    assert frame.shape[0] == 2

    validate_artifact_frame(
        frame=frame,
        expected_method_name="concept_cbm",
        require_text_contradiction=False,
        curve_steps=10,
    )


def test_generate_family_artifacts_text_profiles_have_expected_contradiction_order() -> None:
    cohort = _build_cohort()
    e5_frame = generate_family_artifacts(
        cohort_df=cohort,
        profile=E5_PROFILE,
        labels=["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion", "Pneumothorax"],
        study_col="study_id",
        split_col="split",
        target_split="test",
        curve_steps=10,
        seed=17,
        include_text_columns=True,
        include_concept_intervention=False,
    )
    e6_frame = generate_family_artifacts(
        cohort_df=cohort,
        profile=E6_PROFILE,
        labels=["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion", "Pneumothorax"],
        study_col="study_id",
        split_col="split",
        target_split="test",
        curve_steps=10,
        seed=17,
        include_text_columns=True,
        include_concept_intervention=False,
    )

    assert "text_contradiction" in e5_frame.columns
    assert "finding_consistency" in e5_frame.columns
    assert e5_frame["text_contradiction"].mean() < e6_frame["text_contradiction"].mean()

    validate_artifact_frame(
        frame=e5_frame,
        expected_method_name="text_constrained",
        require_text_contradiction=True,
        curve_steps=10,
    )
    validate_artifact_frame(
        frame=e6_frame,
        expected_method_name="text_unconstrained",
        require_text_contradiction=True,
        curve_steps=10,
    )


def test_run_generator_writes_outputs_and_meta(tmp_path: Path) -> None:
    cohort = _build_cohort()
    cohort_csv = tmp_path / "cohort.csv"
    output_csv = tmp_path / "e5_artifacts.csv"
    meta_json = tmp_path / "e5_meta.json"
    cohort.to_csv(cohort_csv, index=False)

    args = SimpleNamespace(
        cohort_csv=cohort_csv,
        output_csv=output_csv,
        meta_json=meta_json,
        study_col="study_id",
        split_col="split",
        target_split="test",
        labels=["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion", "Pneumothorax"],
        curve_steps=10,
        seed=23,
        concepts_file=None,
        concept_col="concept",
    )

    run_generator(
        args=args,
        profile=E5_PROFILE,
        include_text_columns=True,
        include_concept_intervention=False,
    )

    assert output_csv.exists()
    assert meta_json.exists()
    written = pd.read_csv(output_csv)
    assert set(written["method"].unique()) == {"text_constrained"}

