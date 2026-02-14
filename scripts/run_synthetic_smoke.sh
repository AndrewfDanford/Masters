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
SMOKE_CURVE_STEPS="${SMOKE_CURVE_STEPS:-10}"
SMOKE_ENABLE_TEXT_PROXY="${SMOKE_ENABLE_TEXT_PROXY:-0}"

echo "[1/8] Generating synthetic smoke dataset and artifact tables..."
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

echo "[2/8] Running E1 feature extraction + baseline training..."
bash scripts/run_e1_from_images.sh

SMOKE_E23_SOURCE_CSV="${SMOKE_OUTPUT_DIR}/synthetic_e23_artifacts.csv"
SMOKE_E23_SALIENCY_CSV="${SMOKE_E23_SALIENCY_CSV:-${SMOKE_OUTPUT_DIR}/e2_e3/e2e3_artifacts_saliency_only.csv}"
export SMOKE_E23_SOURCE_CSV
export SMOKE_E23_SALIENCY_CSV
"${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path
import pandas as pd

source = Path(os.environ["SMOKE_E23_SOURCE_CSV"])
target = Path(os.environ["SMOKE_E23_SALIENCY_CSV"])
frame = pd.read_csv(source)
filtered = frame[frame["method"].astype(str).isin(["gradcam", "hirescam"])].copy()
target.parent.mkdir(parents=True, exist_ok=True)
filtered.to_csv(target, index=False)
PY

export E23_INPUT_CSV="${SMOKE_E23_SALIENCY_CSV}"
export E23_OUTPUT_DIR="${E23_OUTPUT_DIR:-${SMOKE_OUTPUT_DIR}/e2_e3}"
export E23_METHOD_REGEX="${E23_METHOD_REGEX:-^(?:gradcam|hirescam)$}"

echo "[3/8] Running E2/E3 scoring on synthetic artifacts..."
bash scripts/run_e2_e3_saliency.sh

export E4_COHORT_CSV="${SMOKE_OUTPUT_DIR}/synthetic_cohort.csv"
export E4_OUTPUT_CSV="${E4_OUTPUT_CSV:-${SMOKE_OUTPUT_DIR}/e4/e4_artifacts.csv}"
export E4_META_JSON="${E4_META_JSON:-${SMOKE_OUTPUT_DIR}/e4/e4_generation_meta.json}"
export E4_CURVE_STEPS="${E4_CURVE_STEPS:-${SMOKE_CURVE_STEPS}}"

echo "[4/8] Running E4 concept proxy generation..."
bash scripts/run_e4_concept.sh

if [[ "${SMOKE_ENABLE_TEXT_PROXY}" == "1" ]]; then
  export E5_COHORT_CSV="${SMOKE_OUTPUT_DIR}/synthetic_cohort.csv"
  export E5_OUTPUT_CSV="${E5_OUTPUT_CSV:-${SMOKE_OUTPUT_DIR}/e5/e5_artifacts.csv}"
  export E5_META_JSON="${E5_META_JSON:-${SMOKE_OUTPUT_DIR}/e5/e5_generation_meta.json}"
  export E5_CURVE_STEPS="${E5_CURVE_STEPS:-${SMOKE_CURVE_STEPS}}"

  echo "[5/8] Running E5 constrained-text proxy generation..."
  bash scripts/run_e5_text_constrained.sh

  export E6_COHORT_CSV="${SMOKE_OUTPUT_DIR}/synthetic_cohort.csv"
  export E6_OUTPUT_CSV="${E6_OUTPUT_CSV:-${SMOKE_OUTPUT_DIR}/e6/e6_artifacts.csv}"
  export E6_META_JSON="${E6_META_JSON:-${SMOKE_OUTPUT_DIR}/e6/e6_generation_meta.json}"
  export E6_CURVE_STEPS="${E6_CURVE_STEPS:-${SMOKE_CURVE_STEPS}}"

  echo "[6/8] Running E6 unconstrained-text proxy generation..."
  bash scripts/run_e6_text_unconstrained.sh
else
  echo "[5/8] Skipping E5 constrained-text proxy (SMOKE_ENABLE_TEXT_PROXY=${SMOKE_ENABLE_TEXT_PROXY})."
  echo "[6/8] Skipping E6 unconstrained-text proxy (SMOKE_ENABLE_TEXT_PROXY=${SMOKE_ENABLE_TEXT_PROXY})."
fi

export ASSEMBLE_E23_CSV="${SMOKE_E23_SALIENCY_CSV}"
export ASSEMBLE_E4_CSV="${E4_OUTPUT_CSV}"
export ASSEMBLE_OUTPUT_CSV="${ASSEMBLE_OUTPUT_CSV:-${SMOKE_OUTPUT_DIR}/e7/e7_input_all_methods.csv}"
export ASSEMBLE_META_JSON="${ASSEMBLE_META_JSON:-${SMOKE_OUTPUT_DIR}/e7/e7_input_all_methods_meta.json}"
unset ASSEMBLE_EXPECTED_CURVE_LEN

if [[ "${SMOKE_ENABLE_TEXT_PROXY}" == "1" ]]; then
  export ASSEMBLE_E5_CSV="${E5_OUTPUT_CSV}"
  export ASSEMBLE_E6_CSV="${E6_OUTPUT_CSV}"
  export ASSEMBLE_INPUT_CSVS="${SMOKE_E23_SALIENCY_CSV},${E4_OUTPUT_CSV},${E5_OUTPUT_CSV},${E6_OUTPUT_CSV}"
else
  unset ASSEMBLE_E5_CSV
  unset ASSEMBLE_E6_CSV
  export ASSEMBLE_INPUT_CSVS="${SMOKE_E23_SALIENCY_CSV},${E4_OUTPUT_CSV}"
fi

echo "[7/8] Assembling family artifacts and running E7 unified benchmark..."
bash scripts/run_assemble_family_artifacts.sh

export E7_INPUT_CSV="${ASSEMBLE_OUTPUT_CSV}"
export E7_OUTPUT_DIR="${E7_OUTPUT_DIR:-${SMOKE_OUTPUT_DIR}/e7}"
bash scripts/run_e7_unified.sh

export E8_INPUT_CSV="${SMOKE_OUTPUT_DIR}/synthetic_e8_randomization.csv"
export E8_OUTPUT_DIR="${E8_OUTPUT_DIR:-${SMOKE_OUTPUT_DIR}/e8}"
export E8_OUTPUT_PREFIX="${E8_OUTPUT_PREFIX:-e8_smoke}"

echo "[8/8] Running E8 randomization benchmark..."
bash scripts/run_e8_randomization.sh

echo "Synthetic smoke run completed."
echo "Key outputs:"
echo "  - ${SMOKE_OUTPUT_DIR}/e1/e1_metrics_summary.csv"
echo "  - ${SMOKE_OUTPUT_DIR}/e2_e3/e2e3_saliency_method_summary.csv"
echo "  - ${SMOKE_OUTPUT_DIR}/e4/e4_artifacts.csv"
echo "  - ${SMOKE_OUTPUT_DIR}/e5/e5_artifacts.csv (optional; SMOKE_ENABLE_TEXT_PROXY=1)"
echo "  - ${SMOKE_OUTPUT_DIR}/e6/e6_artifacts.csv (optional; SMOKE_ENABLE_TEXT_PROXY=1)"
echo "  - ${SMOKE_OUTPUT_DIR}/e7/e7_input_all_methods.csv"
echo "  - ${SMOKE_OUTPUT_DIR}/e7/e7_method_summary.csv"
echo "  - ${SMOKE_OUTPUT_DIR}/e8/e8_smoke_method_summary.csv"
