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

if [[ -z "${E23_MANIFEST_CSV:-}" ]]; then
  echo "Missing required env var: E23_MANIFEST_CSV"
  exit 1
fi

if [[ -z "${E23_IMAGE_ROOT:-}" ]]; then
  echo "Missing required env var: E23_IMAGE_ROOT"
  exit 1
fi

if [[ -z "${E23_MODEL_CHECKPOINT:-}" ]]; then
  echo "Missing required env var: E23_MODEL_CHECKPOINT"
  echo "This should point to a trained CNN checkpoint compatible with E23_ARCH and E23_LABELS_CSV."
  exit 1
fi

OUTPUT_CSV="${E23_OUTPUT_CSV:-outputs/reports/e2_e3/e2e3_artifacts.csv}"
SKIPPED_CSV="${E23_SKIPPED_CSV:-outputs/reports/e2_e3/e2e3_skipped.csv}"
META_JSON="${E23_META_JSON:-outputs/reports/e2_e3/e2e3_generation_meta.json}"

cmd=(
  "${PYTHON_BIN}" -m src.explain.e2_e3_generate
  --manifest-csv "${E23_MANIFEST_CSV}"
  --image-root "${E23_IMAGE_ROOT}"
  --model-checkpoint "${E23_MODEL_CHECKPOINT}"
  --output-csv "${OUTPUT_CSV}"
  --skipped-csv "${SKIPPED_CSV}"
  --meta-json "${META_JSON}"
  --arch "${E23_ARCH:-resnet18}"
  --device "${E23_DEVICE:-cpu}"
  --input-size "${E23_INPUT_SIZE:-320}"
  --curve-steps "${E23_CURVE_STEPS:-10}"
  --target-split "${E23_TARGET_SPLIT:-test}"
  --study-col "${E23_STUDY_COL:-study_id}"
  --subject-col "${E23_SUBJECT_COL:-subject_id}"
  --dicom-col "${E23_DICOM_COL:-dicom_id}"
  --image-path-col "${E23_IMAGE_PATH_COL:-path}"
  --split-col "${E23_SPLIT_COL:-split}"
  --nuisance-brightness "${E23_NUISANCE_BRIGHTNESS:-0.05}"
  --nuisance-contrast "${E23_NUISANCE_CONTRAST:-1.05}"
  --sanity-mode "${E23_SANITY_MODE:-head}"
)

if [[ -n "${E23_LABELS_CSV:-}" ]]; then
  IFS=',' read -r -a labels <<< "${E23_LABELS_CSV}"
  for index in "${!labels[@]}"; do
    labels[$index]="$(echo "${labels[$index]}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
  done
  cmd+=(--labels "${labels[@]}")
fi

if [[ -n "${E23_METHODS_CSV:-}" ]]; then
  IFS=',' read -r -a methods <<< "${E23_METHODS_CSV}"
  for index in "${!methods[@]}"; do
    methods[$index]="$(echo "${methods[$index]}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
  done
  cmd+=(--methods "${methods[@]}")
fi

if [[ -n "${E23_MAX_SAMPLES:-}" ]]; then
  cmd+=(--max-samples "${E23_MAX_SAMPLES}")
fi

if [[ "${E23_PRETRAINED_BACKBONE:-0}" == "1" ]]; then
  cmd+=(--pretrained-backbone)
fi

if [[ -n "${E23_SAVE_CAM_DIR:-}" ]]; then
  cmd+=(--save-cam-dir "${E23_SAVE_CAM_DIR}")
fi

"${cmd[@]}"

