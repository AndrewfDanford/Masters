#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -z "${E1_COHORT_CSV:-}" ]]; then
  echo "Missing required env var: E1_COHORT_CSV"
  exit 1
fi

if [[ -z "${E1_IMAGE_ROOT:-}" ]]; then
  echo "Missing required env var: E1_IMAGE_ROOT"
  exit 1
fi

if [[ "${E1_RUN_CNN:-0}" == "1" ]]; then
  echo "[1/3] Running E1 CNN training..."
  bash scripts/run_e1_train_cnn.sh
  export E23_MODEL_CHECKPOINT="${E23_MODEL_CHECKPOINT:-${E1_CNN_CHECKPOINT:-outputs/models/e1_cnn_checkpoint.pt}}"
else
  echo "[1/3] Running E1 from images (feature extraction + baseline model)..."
  bash scripts/run_e1_from_images.sh
fi

if [[ -z "${E23_INPUT_CSV:-}" ]]; then
  if [[ -n "${E23_MODEL_CHECKPOINT:-}" ]]; then
    echo "[2/3] Generating E2/E3 artifacts from model checkpoint..."
    export E23_MANIFEST_CSV="${E23_MANIFEST_CSV:-${E1_COHORT_CSV}}"
    export E23_IMAGE_ROOT="${E23_IMAGE_ROOT:-${E1_IMAGE_ROOT}}"
    bash scripts/run_e2_e3_generate.sh
    export E23_INPUT_CSV="${E23_OUTPUT_CSV:-outputs/reports/e2_e3/e2e3_artifacts.csv}"
    echo "[2/3] Running E2/E3 saliency scoring..."
    bash scripts/run_e2_e3_saliency.sh
  else
    echo "[2/3] Skipping E2/E3 scoring: set E23_INPUT_CSV or E23_MODEL_CHECKPOINT."
  fi
else
  echo "[2/3] Running E2/E3 saliency scoring..."
  bash scripts/run_e2_e3_saliency.sh
fi

if [[ -z "${E7_INPUT_CSV:-}" ]]; then
  if [[ -n "${E23_INPUT_CSV:-}" ]]; then
    export E7_INPUT_CSV="${E23_INPUT_CSV}"
  else
    echo "[3/3] Skipping E7: set E7_INPUT_CSV (or E23_INPUT_CSV) to run unified benchmark."
    exit 0
  fi
fi

echo "[3/3] Running E7 unified benchmark..."
bash scripts/run_e7_unified.sh

echo "Pipeline completed."
