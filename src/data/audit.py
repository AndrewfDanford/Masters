from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable

import pandas as pd

FRONTAL_VIEWS = {"AP", "PA"}
DEFAULT_FINDINGS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
    "Pneumothorax",
]


def _require_columns(df: pd.DataFrame, columns: Iterable[str], frame_name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"{frame_name} is missing required columns: {missing_text}")


def _canonical_split_name(raw_name: str) -> str:
    normalized = str(raw_name).strip().lower()
    aliases = {
        "train": "train",
        "training": "train",
        "val": "val",
        "valid": "val",
        "validate": "val",
        "validation": "val",
        "test": "test",
        "testing": "test",
    }
    return aliases.get(normalized, normalized)


def normalize_uncertain_labels(
    df: pd.DataFrame,
    findings: list[str],
    uncertain_policy: str,
) -> pd.DataFrame:
    if uncertain_policy not in {"u_ignore", "u_zero", "u_one"}:
        raise ValueError("uncertain_policy must be one of: u_ignore, u_zero, u_one")

    output = df.copy()
    _require_columns(output, findings, "labels frame")

    for finding in findings:
        series = pd.to_numeric(output[finding], errors="coerce")
        if uncertain_policy == "u_ignore":
            series = series.mask(series == -1)
        elif uncertain_policy == "u_zero":
            series = series.replace(-1, 0)
        else:
            series = series.replace(-1, 1)
        output[finding] = series

    return output


def merge_metadata_and_labels(
    metadata_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    study_col: str,
    subject_col: str,
    view_col: str,
    frontal_only: bool,
) -> pd.DataFrame:
    _require_columns(metadata_df, [subject_col, study_col], "metadata")
    _require_columns(labels_df, [study_col], "labels")

    metadata = metadata_df.copy()
    labels = labels_df.copy()

    if frontal_only:
        if view_col not in metadata.columns:
            raise ValueError(
                f"frontal filtering requested but view column '{view_col}' is missing from metadata"
            )
        metadata = metadata[metadata[view_col].astype(str).str.upper().isin(FRONTAL_VIEWS)].copy()

    merge_keys = [study_col]
    if subject_col in labels.columns:
        merge_keys = [subject_col, study_col]

    metadata = metadata.drop_duplicates(subset=merge_keys, keep="first")
    labels = labels.drop_duplicates(subset=merge_keys, keep="first")

    merged = metadata.merge(labels, on=merge_keys, how="inner")
    return merged


def assign_patient_splits(
    df: pd.DataFrame,
    subject_col: str,
    seed: int,
    train_frac: float,
    val_frac: float,
    split_col: str = "split",
) -> pd.DataFrame:
    if train_frac <= 0 or val_frac <= 0 or train_frac + val_frac >= 1:
        raise ValueError("train_frac and val_frac must be > 0 and sum to less than 1")
    if subject_col not in df.columns:
        raise ValueError(f"missing subject column: {subject_col}")

    output = df.copy()
    subjects = output[subject_col].dropna().drop_duplicates().tolist()
    rng = random.Random(seed)
    rng.shuffle(subjects)

    n_subjects = len(subjects)
    n_train = int(n_subjects * train_frac)
    n_val = int(n_subjects * val_frac)

    subject_to_split: dict[object, str] = {}
    for subject in subjects[:n_train]:
        subject_to_split[subject] = "train"
    for subject in subjects[n_train : n_train + n_val]:
        subject_to_split[subject] = "val"
    for subject in subjects[n_train + n_val :]:
        subject_to_split[subject] = "test"

    output[split_col] = output[subject_col].map(subject_to_split)
    output = output[output[split_col].notna()].copy()
    return output


def apply_official_splits(
    df: pd.DataFrame,
    split_df: pd.DataFrame,
    study_col: str,
    split_col: str,
) -> pd.DataFrame:
    _require_columns(split_df, [study_col, split_col], "official split frame")
    output = df.merge(
        split_df[[study_col, split_col]].drop_duplicates(subset=[study_col]),
        on=study_col,
        how="left",
    )
    output[split_col] = output[split_col].map(_canonical_split_name)
    output = output[output[split_col].isin({"train", "val", "test"})].copy()
    return output


def compute_split_counts(
    cohort_df: pd.DataFrame,
    subject_col: str,
    split_col: str,
) -> pd.DataFrame:
    _require_columns(cohort_df, [subject_col, split_col], "cohort")
    studies = cohort_df.groupby(split_col, dropna=False).size().rename("num_studies")
    patients = (
        cohort_df.groupby(split_col, dropna=False)[subject_col]
        .nunique(dropna=True)
        .rename("num_patients")
    )
    return (
        pd.concat([studies, patients], axis=1)
        .reset_index()
        .sort_values(by=split_col)
        .reset_index(drop=True)
    )


def compute_label_prevalence(
    cohort_df: pd.DataFrame,
    findings: list[str],
    split_col: str,
) -> pd.DataFrame:
    _require_columns(cohort_df, findings + [split_col], "cohort")
    rows: list[dict[str, object]] = []

    for split_name, split_frame in cohort_df.groupby(split_col):
        for finding in findings:
            known = int(split_frame[finding].notna().sum())
            positive = int((split_frame[finding] == 1).sum())
            prevalence = (positive / known) if known else None
            rows.append(
                {
                    "split": split_name,
                    "finding": finding,
                    "positive": positive,
                    "known": known,
                    "prevalence": prevalence,
                }
            )

    return pd.DataFrame(rows).sort_values(by=["split", "finding"]).reset_index(drop=True)


def _explode_concept_records(record: dict, study_col: str, concept_col: str) -> list[dict]:
    if study_col not in record:
        return []

    study_id = record[study_col]
    rows: list[dict] = []

    if concept_col in record:
        concept_value = record[concept_col]
        if isinstance(concept_value, list):
            rows.extend({study_col: study_id, concept_col: value} for value in concept_value)
        else:
            rows.append({study_col: study_id, concept_col: concept_value})
        return rows

    entities = record.get("entities")
    if isinstance(entities, dict):
        for entity in entities.values():
            label = entity.get("label")
            if label is not None:
                rows.append({study_col: study_id, concept_col: label})
    elif isinstance(entities, list):
        for entity in entities:
            label = entity.get("label")
            if label is not None:
                rows.append({study_col: study_id, concept_col: label})

    return rows


def load_concepts_table(
    concepts_path: Path,
    study_col: str,
    concept_col: str = "concept",
) -> pd.DataFrame:
    if concepts_path.suffix.lower() in {".csv", ".gz"}:
        concepts_df = pd.read_csv(concepts_path)
        _require_columns(concepts_df, [study_col, concept_col], "concept table")
        return concepts_df[[study_col, concept_col]].copy()

    if concepts_path.suffix.lower() in {".jsonl", ".json"}:
        rows: list[dict] = []
        with concepts_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                rows.extend(_explode_concept_records(record, study_col, concept_col))
        return pd.DataFrame(rows)

    raise ValueError("concepts file must be .csv/.gz or .json/.jsonl")


def compute_concept_coverage(
    cohort_df: pd.DataFrame,
    concepts_df: pd.DataFrame,
    study_col: str,
    concept_col: str = "concept",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _require_columns(cohort_df, [study_col], "cohort")
    _require_columns(concepts_df, [study_col, concept_col], "concept table")

    cohort_studies = cohort_df[[study_col]].drop_duplicates()
    merged = cohort_studies.merge(concepts_df[[study_col, concept_col]], on=study_col, how="left")

    total_studies = int(cohort_studies[study_col].nunique())
    studies_with_concepts = int(merged[merged[concept_col].notna()][study_col].nunique())
    unique_concepts = int(merged[concept_col].dropna().nunique())
    mean_concepts_per_study = float(
        merged[merged[concept_col].notna()]
        .groupby(study_col)[concept_col]
        .count()
        .mean()
        if studies_with_concepts > 0
        else 0.0
    )

    summary = pd.DataFrame(
        [
            {
                "total_studies": total_studies,
                "studies_with_concepts": studies_with_concepts,
                "coverage_rate": (studies_with_concepts / total_studies) if total_studies else None,
                "unique_concepts": unique_concepts,
                "mean_concepts_per_study": mean_concepts_per_study,
            }
        ]
    )

    concept_support = (
        merged[merged[concept_col].notna()]
        .groupby(concept_col)
        .size()
        .rename("count")
        .reset_index()
        .sort_values(by="count", ascending=False)
        .reset_index(drop=True)
    )
    return summary, concept_support


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run E0 cohort and prevalence audits.")
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--official-split-csv", type=Path, default=None)
    parser.add_argument("--concepts-file", type=Path, default=None)

    parser.add_argument("--study-col", default="study_id")
    parser.add_argument("--subject-col", default="subject_id")
    parser.add_argument("--view-col", default="ViewPosition")
    parser.add_argument("--split-col", default="split")

    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--uncertain-policy", choices=["u_ignore", "u_zero", "u_one"], default="u_ignore")
    parser.add_argument("--all-views", action="store_true")
    parser.add_argument("--findings", nargs="+", default=DEFAULT_FINDINGS)
    return parser


def run_audit(args: argparse.Namespace) -> None:
    metadata_df = pd.read_csv(args.metadata_csv)
    labels_df = pd.read_csv(args.labels_csv)

    cohort_df = merge_metadata_and_labels(
        metadata_df=metadata_df,
        labels_df=labels_df,
        study_col=args.study_col,
        subject_col=args.subject_col,
        view_col=args.view_col,
        frontal_only=not args.all_views,
    )
    cohort_df = normalize_uncertain_labels(
        cohort_df,
        findings=args.findings,
        uncertain_policy=args.uncertain_policy,
    )

    if args.official_split_csv is not None:
        split_df = pd.read_csv(args.official_split_csv)
        cohort_df = apply_official_splits(
            cohort_df,
            split_df=split_df,
            study_col=args.study_col,
            split_col=args.split_col,
        )
    else:
        cohort_df = assign_patient_splits(
            cohort_df,
            subject_col=args.subject_col,
            seed=args.seed,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            split_col=args.split_col,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_columns = [args.subject_col, args.study_col, args.split_col] + args.findings
    if args.view_col in cohort_df.columns:
        manifest_columns.append(args.view_col)
    for optional_col in ["dicom_id", "path"]:
        if optional_col in cohort_df.columns and optional_col not in manifest_columns:
            manifest_columns.append(optional_col)
    cohort_df[manifest_columns].to_csv(args.output_dir / "e0_cohort_manifest.csv", index=False)

    split_counts_df = compute_split_counts(
        cohort_df=cohort_df,
        subject_col=args.subject_col,
        split_col=args.split_col,
    )
    split_counts_df.to_csv(args.output_dir / "e0_split_counts.csv", index=False)

    prevalence_df = compute_label_prevalence(
        cohort_df=cohort_df,
        findings=args.findings,
        split_col=args.split_col,
    )
    prevalence_df.to_csv(args.output_dir / "e0_label_prevalence.csv", index=False)

    if args.concepts_file is not None:
        concepts_df = load_concepts_table(
            concepts_path=args.concepts_file,
            study_col=args.study_col,
        )
        summary_df, concept_support_df = compute_concept_coverage(
            cohort_df=cohort_df,
            concepts_df=concepts_df,
            study_col=args.study_col,
        )
        summary_df.to_csv(args.output_dir / "e0_concept_coverage_summary.csv", index=False)
        concept_support_df.to_csv(args.output_dir / "e0_concept_support.csv", index=False)


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    run_audit(args)


if __name__ == "__main__":
    main()
