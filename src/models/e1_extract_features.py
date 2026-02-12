from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from .backbone_features import extract_backbone_features
from .image_features import compute_image_features, load_grayscale_image


def _clean_identifier(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        return text[:-2]
    return text


def build_mimic_jpg_relative_path(subject_id: object, study_id: object, dicom_id: object) -> Path:
    subject = _clean_identifier(subject_id)
    study = _clean_identifier(study_id)
    dicom = _clean_identifier(dicom_id)

    if not subject or not study or not dicom:
        raise ValueError("subject_id, study_id, and dicom_id are all required to build MIMIC image path")

    if subject.startswith("p"):
        subject = subject[1:]
    if study.startswith("s"):
        study = study[1:]
    if dicom.endswith(".jpg"):
        dicom = dicom[:-4]

    subject_group = f"p{subject[:2]}"
    return Path(subject_group) / f"p{subject}" / f"s{study}" / f"{dicom}.jpg"


def _candidate_paths_from_row(
    row: pd.Series,
    image_root: Path | None,
    image_path_col: str,
    study_col: str,
    subject_col: str,
    dicom_col: str,
) -> list[Path]:
    candidates: list[Path] = []
    seen: set[str] = set()

    def add_candidate(path: Path) -> None:
        resolved_key = os.path.normpath(str(path))
        if resolved_key not in seen:
            seen.add(resolved_key)
            candidates.append(path)

    if image_path_col in row and pd.notna(row[image_path_col]):
        raw_path = str(row[image_path_col]).strip()
        if raw_path:
            base = Path(raw_path)
            if base.is_absolute():
                add_candidate(base)
            else:
                if image_root is not None:
                    add_candidate(image_root / raw_path)
                    if raw_path.startswith("files/"):
                        add_candidate(image_root / raw_path[len("files/") :])
                    else:
                        add_candidate(image_root / "files" / raw_path)
                add_candidate(base)

    if image_root is not None and all(column in row and pd.notna(row[column]) for column in [subject_col, study_col, dicom_col]):
        rel = build_mimic_jpg_relative_path(
            subject_id=row[subject_col],
            study_id=row[study_col],
            dicom_id=row[dicom_col],
        )
        add_candidate(image_root / rel)
        add_candidate(image_root / "files" / rel)

    return candidates


def resolve_image_path(
    row: pd.Series,
    image_root: Path | None,
    image_path_col: str,
    study_col: str,
    subject_col: str,
    dicom_col: str,
) -> Path | None:
    candidates = _candidate_paths_from_row(
        row=row,
        image_root=image_root,
        image_path_col=image_path_col,
        study_col=study_col,
        subject_col=subject_col,
        dicom_col=dicom_col,
    )
    if not candidates:
        return None
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def run_feature_extraction(args: argparse.Namespace) -> None:
    frame = pd.read_csv(args.input_csv)
    if args.study_col not in frame.columns:
        raise ValueError(f"missing required study column: {args.study_col}")

    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("limit must be positive when provided")
        frame = frame.head(args.limit).copy()

    image_root = Path(args.image_root) if args.image_root else None
    if image_root is not None and not image_root.exists():
        raise ValueError(f"image root does not exist: {image_root}")

    output_rows: list[dict[str, object]] = []
    missing_rows: list[dict[str, object]] = []
    valid_entries: list[dict[str, object]] = []

    for _, row in frame.iterrows():
        resolved_path = resolve_image_path(
            row=row,
            image_root=image_root,
            image_path_col=args.image_path_col,
            study_col=args.study_col,
            subject_col=args.subject_col,
            dicom_col=args.dicom_col,
        )
        if resolved_path is None or not resolved_path.exists():
            missing_rows.append(
                {
                    args.study_col: row.get(args.study_col),
                    args.subject_col: row.get(args.subject_col),
                    args.dicom_col: row.get(args.dicom_col),
                    "resolved_image_path": str(resolved_path) if resolved_path is not None else "",
                    "reason": "missing_image_file",
                }
            )
            if args.fail_on_missing:
                raise FileNotFoundError(
                    f"image not found for study_id={row.get(args.study_col)} "
                    f"(resolved candidate: {resolved_path})"
                )
            continue
        valid_entry: dict[str, object] = {
            args.study_col: row.get(args.study_col),
            "resolved_image_path": str(resolved_path),
            "_resolved_path": resolved_path,
        }
        for extra_col in [args.subject_col, args.dicom_col, args.split_col]:
            if extra_col in frame.columns:
                valid_entry[extra_col] = row.get(extra_col)
        valid_entries.append(valid_entry)

    if not valid_entries:
        raise ValueError("no valid image paths found; check input manifest and image_root")

    backbone_error_msg: str | None = None

    if args.extractor == "handcrafted":
        for entry in valid_entries:
            resolved_path = entry["_resolved_path"]
            try:
                image = load_grayscale_image(
                    path=resolved_path,
                    width=args.resize_width,
                    height=args.resize_height,
                )
                feature_map = compute_image_features(image=image, hist_bins=args.hist_bins)
            except Exception as exc:  # pragma: no cover - depends on image runtime issues
                missing_rows.append(
                    {
                        args.study_col: entry.get(args.study_col),
                        args.subject_col: entry.get(args.subject_col),
                        args.dicom_col: entry.get(args.dicom_col),
                        "resolved_image_path": str(resolved_path),
                        "reason": f"feature_extraction_error:{type(exc).__name__}",
                    }
                )
                if args.fail_on_missing:
                    raise
                continue

            output_row = {key: value for key, value in entry.items() if not key.startswith("_")}
            output_row["extractor"] = args.extractor
            output_row.update(feature_map)
            output_rows.append(output_row)
    else:
        try:
            feature_matrix = extract_backbone_features(
                image_paths=[entry["_resolved_path"] for entry in valid_entries],
                model_name=args.extractor,
                width=args.resize_width,
                height=args.resize_height,
                batch_size=args.batch_size,
                device=args.device,
                pretrained=args.pretrained_backbone,
                checkpoint_path=args.backbone_checkpoint,
            )
        except Exception as exc:
            backbone_error_msg = f"{type(exc).__name__}: {exc}"
            if args.fail_on_missing:
                raise
            for entry in valid_entries:
                missing_rows.append(
                    {
                        args.study_col: entry.get(args.study_col),
                        args.subject_col: entry.get(args.subject_col),
                        args.dicom_col: entry.get(args.dicom_col),
                        "resolved_image_path": str(entry.get("_resolved_path")),
                        "reason": "backbone_feature_extraction_error",
                    }
                )
            feature_matrix = None

        if feature_matrix is not None:
            if feature_matrix.shape[0] != len(valid_entries):
                raise ValueError(
                    "backbone feature matrix row count does not match resolved image count: "
                    f"{feature_matrix.shape[0]} vs {len(valid_entries)}"
                )

            for row_index, entry in enumerate(valid_entries):
                output_row = {key: value for key, value in entry.items() if not key.startswith("_")}
                output_row["extractor"] = args.extractor
                for feature_index, value in enumerate(feature_matrix[row_index]):
                    output_row[f"feat_{feature_index:04d}"] = float(value)
                output_rows.append(output_row)

    if not output_rows:
        if backbone_error_msg is not None:
            raise ValueError(
                "no features were extracted; backbone mode failed before writing output. "
                f"Root error: {backbone_error_msg}"
            )
        raise ValueError("no features were extracted; check image paths, extractor settings, and dependencies")

    output_df = pd.DataFrame(output_rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(args.output_csv, index=False)

    if args.missing_csv is not None:
        args.missing_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(missing_rows).to_csv(args.missing_csv, index=False)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract numeric image features for E1 baseline training from MIMIC-style image paths "
            "or explicit path column values."
        )
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--missing-csv", type=Path, default=None)

    parser.add_argument("--study-col", default="study_id")
    parser.add_argument("--subject-col", default="subject_id")
    parser.add_argument("--dicom-col", default="dicom_id")
    parser.add_argument("--split-col", default="split")
    parser.add_argument("--image-path-col", default="path")

    parser.add_argument("--resize-width", type=int, default=320)
    parser.add_argument("--resize-height", type=int, default=320)
    parser.add_argument("--hist-bins", type=int, default=16)
    parser.add_argument("--extractor", choices=["handcrafted", "resnet18", "densenet121"], default="handcrafted")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--pretrained-backbone", action="store_true")
    parser.add_argument("--backbone-checkpoint", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fail-on-missing", action="store_true")
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    run_feature_extraction(args)


if __name__ == "__main__":
    main()
