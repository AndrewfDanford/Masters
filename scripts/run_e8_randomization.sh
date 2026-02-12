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

if [[ -z "${E8_INPUT_CSV:-}" ]]; then
  echo "Missing required env var: E8_INPUT_CSV"
  echo "Expected columns: method, study_id, sanity_similarity, and optional run_id."
  exit 1
fi

OUTPUT_DIR="${E8_OUTPUT_DIR:-outputs/reports/e8}"
OUTPUT_PREFIX="${E8_OUTPUT_PREFIX:-e8_randomization}"

cmd=(
  "${PYTHON_BIN}" -m src.eval.e8_randomization
  --input-csv "${E8_INPUT_CSV}"
  --output-dir "${OUTPUT_DIR}"
  --output-prefix "${OUTPUT_PREFIX}"
  --method-col "${E8_METHOD_COL:-method}"
  --sample-id-col "${E8_SAMPLE_ID_COL:-study_id}"
  --run-col "${E8_RUN_COL:-run_id}"
  --sanity-similarity-col "${E8_SANITY_SIMILARITY_COL:-sanity_similarity}"
  --min-sanity-score "${E8_MIN_SANITY_SCORE:-0.5}"
  --alpha "${E8_ALPHA:-0.05}"
  --num-bootstrap-runs "${E8_NUM_BOOTSTRAP_RUNS:-200}"
  --seed "${E8_SEED:-17}"
)

if [[ "${E8_DISABLE_BOOTSTRAP_PROXY:-0}" == "1" ]]; then
  cmd+=(--disable-bootstrap-proxy)
fi

"${cmd[@]}"
