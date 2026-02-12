from pathlib import Path

import pandas as pd

from src.data.synthetic_smoke import METHOD_PROFILES, build_synthetic_bundle


def test_build_synthetic_bundle_outputs_expected_files(tmp_path: Path) -> None:
    outputs = build_synthetic_bundle(
        output_dir=tmp_path,
        num_train=4,
        num_val=2,
        num_test=2,
        image_size=64,
        seed=7,
        e8_runs=3,
    )

    cohort_csv = Path(outputs["cohort_csv"])
    e23_csv = Path(outputs["e23_artifact_csv"])
    e8_csv = Path(outputs["e8_randomization_csv"])
    image_root = Path(outputs["image_root"])

    assert cohort_csv.exists()
    assert e23_csv.exists()
    assert e8_csv.exists()
    assert image_root.exists()

    cohort = pd.read_csv(cohort_csv)
    assert cohort.shape[0] == 8
    assert {"subject_id", "study_id", "split", "path"}.issubset(cohort.columns)

    e23 = pd.read_csv(e23_csv)
    assert {"method", "study_id", "sanity_similarity", "deletion_curve", "insertion_curve"}.issubset(e23.columns)
    assert set(e23["method"]) == set(METHOD_PROFILES.keys())

    e8 = pd.read_csv(e8_csv)
    assert {"method", "study_id", "run_id", "sanity_similarity"}.issubset(e8.columns)
    assert e8["run_id"].nunique() == 3
