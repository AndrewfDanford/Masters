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

SMOKE_OUTPUT_DIR="${SMOKE_OUTPUT_DIR:-outputs/smoke}"
SMOKE_NUM_TRAIN="${SMOKE_NUM_TRAIN:-24}"
SMOKE_NUM_VAL="${SMOKE_NUM_VAL:-8}"
SMOKE_NUM_TEST="${SMOKE_NUM_TEST:-8}"
SMOKE_IMAGE_SIZE="${SMOKE_IMAGE_SIZE:-128}"
SMOKE_SEED="${SMOKE_SEED:-17}"
SMOKE_E8_RUNS="${SMOKE_E8_RUNS:-8}"

echo "[1/5] Generating synthetic smoke dataset and artifact tables..."
"${PYTHON_BIN}" -m src.data.synthetic_smoke \
  --output-dir "${SMOKE_OUTPUT_DIR}" \
  --num-train "${SMOKE_NUM_TRAIN}" \
  --num-val "${SMOKE_NUM_VAL}" \
  --num-test "${SMOKE_NUM_TEST}" \
  --image-size "${SMOKE_IMAGE_SIZE}" \
  --seed "${SMOKE_SEED}" \
  --e8-runs "${SMOKE_E8_RUNS}"

export E1_COHORT_CSV="${SMOKE_OUTPUT_DIR}/synthetic_cohort.csv"
export E1_IMAGE_INPUT_CSV="${E1_IMAGE_INPUT_CSV:-${E1_COHORT_CSV}}"
export E1_IMAGE_ROOT="${SMOKE_OUTPUT_DIR}/images"
export E1_FEATURES_CSV="${E1_FEATURES_CSV:-${SMOKE_OUTPUT_DIR}/e1_image_features.csv}"
export E1_MISSING_CSV="${E1_MISSING_CSV:-${SMOKE_OUTPUT_DIR}/e1_missing_images.csv}"
export E1_OUTPUT_DIR="${E1_OUTPUT_DIR:-${SMOKE_OUTPUT_DIR}/e1}"

echo "[2/5] Running E1 feature extraction + baseline training..."
bash scripts/run_e1_from_images.sh

export E23_INPUT_CSV="${SMOKE_OUTPUT_DIR}/synthetic_e23_artifacts.csv"
export E23_OUTPUT_DIR="${E23_OUTPUT_DIR:-${SMOKE_OUTPUT_DIR}/e2_e3}"

echo "[3/5] Running E2/E3 scoring on synthetic artifacts..."
bash scripts/run_e2_e3_saliency.sh

export E7_INPUT_CSV="${E23_INPUT_CSV}"
export E7_OUTPUT_DIR="${E7_OUTPUT_DIR:-${SMOKE_OUTPUT_DIR}/e7}"

echo "[4/5] Running E7 unified benchmark..."
bash scripts/run_e7_unified.sh

export E8_INPUT_CSV="${SMOKE_OUTPUT_DIR}/synthetic_e8_randomization.csv"
export E8_OUTPUT_DIR="${E8_OUTPUT_DIR:-${SMOKE_OUTPUT_DIR}/e8}"
export E8_OUTPUT_PREFIX="${E8_OUTPUT_PREFIX:-e8_smoke}"

echo "[5/5] Running E8 randomization benchmark..."
bash scripts/run_e8_randomization.sh

echo "Synthetic smoke run completed."
echo "Key outputs:"
echo "  - ${SMOKE_OUTPUT_DIR}/e1/e1_metrics_summary.csv"
echo "  - ${SMOKE_OUTPUT_DIR}/e2_e3/e2e3_saliency_method_summary.csv"
echo "  - ${SMOKE_OUTPUT_DIR}/e7/e7_method_summary.csv"
echo "  - ${SMOKE_OUTPUT_DIR}/e8/e8_smoke_method_summary.csv"
