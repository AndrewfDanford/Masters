import pandas as pd
from argparse import Namespace

from src.data.audit import (
    apply_official_splits,
    assign_patient_splits,
    compute_label_prevalence,
    normalize_uncertain_labels,
    run_audit,
)


def test_assign_patient_splits_is_deterministic_and_subject_safe() -> None:
    frame = pd.DataFrame(
        {
            "subject_id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "study_id": [10, 11, 20, 21, 30, 31, 40, 41, 50, 51],
        }
    )

    out_a = assign_patient_splits(
        frame,
        subject_col="subject_id",
        seed=11,
        train_frac=0.6,
        val_frac=0.2,
    )
    out_b = assign_patient_splits(
        frame,
        subject_col="subject_id",
        seed=11,
        train_frac=0.6,
        val_frac=0.2,
    )

    assert out_a["split"].tolist() == out_b["split"].tolist()
    assert out_a.groupby("subject_id")["split"].nunique().max() == 1


def test_normalize_uncertain_labels_ignore_policy() -> None:
    frame = pd.DataFrame(
        {
            "Atelectasis": [1, 0, -1, None],
            "Edema": [0, -1, -1, 1],
        }
    )
    out = normalize_uncertain_labels(frame, findings=["Atelectasis", "Edema"], uncertain_policy="u_ignore")
    assert pd.isna(out.loc[2, "Atelectasis"])
    assert pd.isna(out.loc[1, "Edema"])
    assert out.loc[0, "Atelectasis"] == 1


def test_compute_label_prevalence_uses_known_denominator() -> None:
    frame = pd.DataFrame(
        {
            "split": ["train", "train", "train", "val"],
            "Atelectasis": [1, 0, None, 1],
            "Edema": [0, 1, None, None],
        }
    )

    prevalence = compute_label_prevalence(frame, findings=["Atelectasis", "Edema"], split_col="split")
    train_atelectasis = prevalence[
        (prevalence["split"] == "train") & (prevalence["finding"] == "Atelectasis")
    ].iloc[0]
    train_edema = prevalence[(prevalence["split"] == "train") & (prevalence["finding"] == "Edema")].iloc[0]

    assert train_atelectasis["positive"] == 1
    assert train_atelectasis["known"] == 2
    assert train_atelectasis["prevalence"] == 0.5

    assert train_edema["positive"] == 1
    assert train_edema["known"] == 2
    assert train_edema["prevalence"] == 0.5


def test_apply_official_splits_maps_validation_alias() -> None:
    cohort = pd.DataFrame({"study_id": [1, 2, 3], "subject_id": [10, 20, 30]})
    official = pd.DataFrame({"study_id": [1, 2, 3], "split": ["train", "validate", "test"]})

    out = apply_official_splits(cohort, official, study_col="study_id", split_col="split")
    mapped = out.sort_values("study_id")["split"].tolist()
    assert mapped == ["train", "val", "test"]


def test_run_audit_manifest_includes_dicom_and_path_when_available(tmp_path) -> None:
    metadata_path = tmp_path / "metadata.csv"
    labels_path = tmp_path / "labels.csv"
    out_dir = tmp_path / "out"

    pd.DataFrame(
        {
            "subject_id": [1001, 1002, 1003],
            "study_id": [2001, 2002, 2003],
            "dicom_id": ["a", "b", "c"],
            "path": [
                "files/p10/p1001/s2001/a.jpg",
                "files/p10/p1002/s2002/b.jpg",
                "files/p10/p1003/s2003/c.jpg",
            ],
            "ViewPosition": ["PA", "AP", "PA"],
        }
    ).to_csv(metadata_path, index=False)

    pd.DataFrame(
        {
            "study_id": [2001, 2002, 2003],
            "Edema": [1, 0, -1],
        }
    ).to_csv(labels_path, index=False)

    args = Namespace(
        metadata_csv=metadata_path,
        labels_csv=labels_path,
        output_dir=out_dir,
        official_split_csv=None,
        concepts_file=None,
        study_col="study_id",
        subject_col="subject_id",
        view_col="ViewPosition",
        split_col="split",
        train_frac=0.7,
        val_frac=0.1,
        seed=17,
        uncertain_policy="u_ignore",
        all_views=False,
        findings=["Edema"],
    )
    run_audit(args)

    manifest_df = pd.read_csv(out_dir / "e0_cohort_manifest.csv")
    assert "dicom_id" in manifest_df.columns
    assert "path" in manifest_df.columns
