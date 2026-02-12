from argparse import Namespace

import pandas as pd
import pytest

from src.models.e1_extract_features import build_mimic_jpg_relative_path, run_feature_extraction


def test_build_mimic_jpg_relative_path() -> None:
    rel = build_mimic_jpg_relative_path(subject_id=10000032, study_id=50414267, dicom_id="abc-def")
    assert str(rel) == "p10/p10000032/s50414267/abc-def.jpg"


def test_run_feature_extraction_writes_features(tmp_path) -> None:
    image_mod = pytest.importorskip("PIL.Image")

    image_root = tmp_path / "images"
    image_path = image_root / "files" / "p10" / "p10000032" / "s50414267" / "abc-def.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)

    image = image_mod.new("L", (64, 64), color=128)
    image.save(image_path)

    input_csv = tmp_path / "input.csv"
    output_csv = tmp_path / "features.csv"
    missing_csv = tmp_path / "missing.csv"

    pd.DataFrame(
        {
            "subject_id": [10000032],
            "study_id": [50414267],
            "dicom_id": ["abc-def"],
            "split": ["train"],
        }
    ).to_csv(input_csv, index=False)

    args = Namespace(
        input_csv=input_csv,
        output_csv=output_csv,
        image_root=image_root,
        missing_csv=missing_csv,
        study_col="study_id",
        subject_col="subject_id",
        dicom_col="dicom_id",
        split_col="split",
        image_path_col="path",
        resize_width=32,
        resize_height=32,
        hist_bins=8,
        limit=None,
        fail_on_missing=True,
    )

    run_feature_extraction(args)
    out_df = pd.read_csv(output_csv)
    assert out_df.shape[0] == 1
    assert "pixel_mean" in out_df.columns
    assert "hist_bin_00" in out_df.columns
