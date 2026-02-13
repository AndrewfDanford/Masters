from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


_CURVE_SPLIT_RE = re.compile(r"[,;|\s]+")
_CORE_COLUMNS = [
    "method",
    "study_id",
    "sanity_similarity",
    "deletion_curve",
    "insertion_curve",
    "base_prediction",
    "perturbed_prediction",
    "nuisance_similarity",
]
_TEXT_OPTIONAL_COLUMNS = ["text_contradiction", "finding_consistency", "concept_alignment", "concept_intervention_effect"]


def _parse_curve(value: object) -> np.ndarray:
    if isinstance(value, np.ndarray):
        curve = value.astype(float).reshape(-1)
    elif isinstance(value, (list, tuple)):
        curve = np.asarray(value, dtype=float).reshape(-1)
    elif isinstance(value, str):
        stripped = value.strip().strip("[]")
        if not stripped:
            return np.asarray([], dtype=float)
        tokens = [token for token in _CURVE_SPLIT_RE.split(stripped) if token]
        curve = np.asarray([float(token) for token in tokens], dtype=float)
    else:
        return np.asarray([], dtype=float)
    return curve


def _validate_frame(
    frame: pd.DataFrame,
    source: Path,
    expected_curve_len: int | None,
) -> None:
    missing = [column for column in _CORE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{source} missing required columns: {', '.join(missing)}")

    if frame.empty:
        raise ValueError(f"{source} is empty")

    required_numeric_columns = ["sanity_similarity", "base_prediction", "perturbed_prediction", "nuisance_similarity"]
    optional_numeric_columns = [
        column
        for column in ["text_contradiction", "finding_consistency", "concept_alignment", "concept_intervention_effect"]
        if column in frame.columns
    ]

    for column in required_numeric_columns:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.isna().any():
            raise ValueError(f"{source} has non-numeric values in required column {column}")
        if ((numeric < 0.0) | (numeric > 1.0)).any():
            raise ValueError(f"{source} has out-of-range [0,1] values in required column {column}")

    for column in optional_numeric_columns:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        invalid_non_numeric = numeric.isna() & frame[column].notna()
        if invalid_non_numeric.any():
            raise ValueError(f"{source} has non-numeric values in optional column {column}")
        if ((numeric.dropna() < 0.0) | (numeric.dropna() > 1.0)).any():
            raise ValueError(f"{source} has out-of-range [0,1] values in optional column {column}")

    for column in ["deletion_curve", "insertion_curve"]:
        for value in frame[column].tolist():
            curve = _parse_curve(value)
            if curve.size == 0:
                raise ValueError(f"{source} has empty curve values in column {column}")
            if expected_curve_len is not None and curve.size != expected_curve_len:
                raise ValueError(
                    f"{source} has curve length {curve.size} in {column}, expected {expected_curve_len}"
                )
            if ((curve < 0.0) | (curve > 1.0)).any():
                raise ValueError(f"{source} has out-of-range [0,1] values in column {column}")


def assemble_artifacts(
    input_csvs: list[Path],
    output_csv: Path,
    meta_json: Path,
    expected_curve_len: int | None = None,
    strict_unique: bool = True,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if not input_csvs:
        raise ValueError("at least one input CSV is required")

    frames: list[pd.DataFrame] = []
    source_rows: list[dict[str, object]] = []

    for path in input_csvs:
        frame = pd.read_csv(path)
        _validate_frame(frame=frame, source=path, expected_curve_len=expected_curve_len)
        frame = frame.copy()
        frame["__source_csv"] = str(path)
        frames.append(frame)
        source_rows.append(
            {
                "path": str(path),
                "num_rows": int(frame.shape[0]),
                "methods": sorted(frame["method"].astype(str).unique().tolist()),
            }
        )

    combined = pd.concat(frames, axis=0, ignore_index=True, sort=False)

    core_and_optionals = list(_CORE_COLUMNS)
    for column in _TEXT_OPTIONAL_COLUMNS:
        if column in combined.columns and column not in core_and_optionals:
            core_and_optionals.append(column)

    extra_columns = [
        column
        for column in combined.columns
        if column not in set(core_and_optionals + ["__source_csv"])
    ]
    ordered_columns = core_and_optionals + sorted(extra_columns) + ["__source_csv"]
    combined = combined[ordered_columns].copy()

    combined["study_id"] = combined["study_id"].map(lambda value: str(value).strip())
    combined["method"] = combined["method"].astype(str).str.strip()

    duplicates = combined.duplicated(subset=["method", "study_id"], keep=False)
    if strict_unique and duplicates.any():
        duplicate_rows = combined.loc[duplicates, ["method", "study_id", "__source_csv"]].head(20)
        raise ValueError(
            "duplicate (method, study_id) rows found across artifacts; "
            f"sample duplicates: {duplicate_rows.to_dict(orient='records')}"
        )

    if not strict_unique and duplicates.any():
        combined = combined.drop_duplicates(subset=["method", "study_id"], keep="first").reset_index(drop=True)

    combined = combined.sort_values(by=["method", "study_id"]).reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    meta_json.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_csv, index=False)

    method_counts = (
        combined.groupby("method")
        .size()
        .rename("num_rows")
        .reset_index()
        .sort_values(by="method")
        .to_dict(orient="records")
    )
    meta = {
        "output_csv": str(output_csv),
        "num_rows": int(combined.shape[0]),
        "num_methods": int(combined["method"].nunique()),
        "methods": sorted(combined["method"].astype(str).unique().tolist()),
        "curve_length_expected": expected_curve_len,
        "strict_unique": bool(strict_unique),
        "source_files": source_rows,
        "method_row_counts": method_counts,
    }
    with meta_json.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)

    return combined, meta


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Assemble E2/E3/E4/E5/E6 artifact CSVs into one unified E7/E8 input table.")
    parser.add_argument("--input-csv", type=Path, nargs="+", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--meta-json", type=Path, required=True)
    parser.add_argument("--expected-curve-len", type=int, default=None)
    parser.add_argument("--allow-duplicates", action="store_true")
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    assemble_artifacts(
        input_csvs=list(args.input_csv),
        output_csv=args.output_csv,
        meta_json=args.meta_json,
        expected_curve_len=args.expected_curve_len,
        strict_unique=not args.allow_duplicates,
    )


if __name__ == "__main__":
    main()
