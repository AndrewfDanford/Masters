#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PYTHON_BIN=".venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

if [[ -z "${E1_IMAGE_INPUT_CSV:-}" ]]; then
  echo "Missing required env var: E1_IMAGE_INPUT_CSV"
  exit 1
fi

if [[ -z "${E1_IMAGE_ROOT:-}" ]]; then
  echo "Missing required env var: E1_IMAGE_ROOT"
  echo "Set this to the MIMIC-CXR-JPG root (directory that contains 'files/' or pXX folders)."
  exit 1
fi

OUTPUT_CSV="${E1_FEATURES_CSV:-outputs/reports/e1/e1_image_features.csv}"
MISSING_CSV="${E1_MISSING_CSV:-outputs/reports/e1/e1_missing_images.csv}"

cmd=(
  "${PYTHON_BIN}" -m src.models.e1_extract_features
  --input-csv "${E1_IMAGE_INPUT_CSV}"
  --output-csv "${OUTPUT_CSV}"
  --image-root "${E1_IMAGE_ROOT}"
  --missing-csv "${MISSING_CSV}"
  --study-col "${E1_STUDY_COL:-study_id}"
  --subject-col "${E1_SUBJECT_COL:-subject_id}"
  --dicom-col "${E1_DICOM_COL:-dicom_id}"
  --split-col "${E1_SPLIT_COL:-split}"
  --image-path-col "${E1_IMAGE_PATH_COL:-path}"
  --resize-width "${E1_RESIZE_WIDTH:-320}"
  --resize-height "${E1_RESIZE_HEIGHT:-320}"
  --hist-bins "${E1_HIST_BINS:-16}"
  --extractor "${E1_FEATURE_EXTRACTOR:-handcrafted}"
  --batch-size "${E1_BACKBONE_BATCH_SIZE:-32}"
  --device "${E1_BACKBONE_DEVICE:-cpu}"
)

if [[ "${E1_FAIL_ON_MISSING:-0}" == "1" ]]; then
  cmd+=(--fail-on-missing)
fi

if [[ -n "${E1_IMAGE_LIMIT:-}" ]]; then
  cmd+=(--limit "${E1_IMAGE_LIMIT}")
fi

if [[ "${E1_PRETRAINED_BACKBONE:-0}" == "1" ]]; then
  cmd+=(--pretrained-backbone)
fi

if [[ -n "${E1_BACKBONE_CHECKPOINT:-}" ]]; then
  cmd+=(--backbone-checkpoint "${E1_BACKBONE_CHECKPOINT}")
fi

"${cmd[@]}"
