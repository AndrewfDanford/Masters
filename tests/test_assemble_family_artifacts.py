from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.eval.assemble_family_artifacts import assemble_artifacts


def _artifact_frame(method: str, study_ids: list[int]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sid in study_ids:
        rows.append(
            {
                "method": method,
                "study_id": sid,
                "sanity_similarity": 0.22,
                "deletion_curve": "0.9,0.7,0.5,0.3,0.2",
                "insertion_curve": "0.2,0.4,0.6,0.8,0.9",
                "base_prediction": 0.81,
                "perturbed_prediction": 0.77,
                "nuisance_similarity": 0.74,
            }
        )
    return pd.DataFrame(rows)


def test_assemble_artifacts_combines_families(tmp_path: Path) -> None:
    e23_csv = tmp_path / "e23.csv"
    e4_csv = tmp_path / "e4.csv"
    output_csv = tmp_path / "assembled.csv"
    meta_json = tmp_path / "assembled_meta.json"

    _artifact_frame("gradcam", [101, 102]).to_csv(e23_csv, index=False)
    _artifact_frame("concept_cbm", [101, 102]).to_csv(e4_csv, index=False)

    combined, meta = assemble_artifacts(
        input_csvs=[e23_csv, e4_csv],
        output_csv=output_csv,
        meta_json=meta_json,
        expected_curve_len=5,
        strict_unique=True,
    )

    assert output_csv.exists()
    assert meta_json.exists()
    assert combined.shape[0] == 4
    assert set(combined["method"].unique()) == {"gradcam", "concept_cbm"}
    assert meta["num_methods"] == 2


def test_assemble_artifacts_rejects_duplicate_method_study_rows(tmp_path: Path) -> None:
    a_csv = tmp_path / "a.csv"
    b_csv = tmp_path / "b.csv"
    output_csv = tmp_path / "assembled.csv"
    meta_json = tmp_path / "assembled_meta.json"

    _artifact_frame("gradcam", [1001]).to_csv(a_csv, index=False)
    _artifact_frame("gradcam", [1001]).to_csv(b_csv, index=False)

    with pytest.raises(ValueError):
        assemble_artifacts(
            input_csvs=[a_csv, b_csv],
            output_csv=output_csv,
            meta_json=meta_json,
            expected_curve_len=5,
            strict_unique=True,
        )

