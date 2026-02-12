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

if [[ -z "${E23_INPUT_CSV:-}" ]]; then
  echo "Missing required env var: E23_INPUT_CSV"
  exit 1
fi

OUTPUT_DIR="${E23_OUTPUT_DIR:-outputs/reports/e2_e3}"
THRESHOLDS_JSON="${E23_THRESHOLDS_JSON:-configs/eval/faithfulness_thresholds.json}"
METHOD_REGEX="${E23_METHOD_REGEX:-}"

cmd=(
  "${PYTHON_BIN}" -m src.eval.unified_benchmark
  --input-csv "${E23_INPUT_CSV}"
  --output-dir "${OUTPUT_DIR}"
  --output-prefix "e2e3_saliency"
  --thresholds-json "${THRESHOLDS_JSON}"
)

if [[ -n "${METHOD_REGEX}" ]]; then
  cmd+=(--method-regex "${METHOD_REGEX}")
fi

if [[ -n "${E23_NUM_BOOTSTRAP:-}" ]]; then
  cmd+=(--num-bootstrap "${E23_NUM_BOOTSTRAP}")
fi

if [[ -n "${E23_ALPHA:-}" ]]; then
  cmd+=(--alpha "${E23_ALPHA}")
fi

if [[ -n "${E23_SEED:-}" ]]; then
  cmd+=(--seed "${E23_SEED}")
fi

"${cmd[@]}"

