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

if [[ -z "${E7_INPUT_CSV:-}" ]]; then
  echo "Missing required env var: E7_INPUT_CSV"
  exit 1
fi

OUTPUT_DIR="${E7_OUTPUT_DIR:-outputs/reports/e7}"
THRESHOLDS_JSON="${E7_THRESHOLDS_JSON:-configs/eval/faithfulness_thresholds.json}"
METHOD_REGEX="${E7_METHOD_REGEX:-}"
OUTPUT_PREFIX="${E7_OUTPUT_PREFIX:-e7}"

cmd=(
  "${PYTHON_BIN}" -m src.eval.unified_benchmark
  --input-csv "${E7_INPUT_CSV}"
  --output-dir "${OUTPUT_DIR}"
  --output-prefix "${OUTPUT_PREFIX}"
  --thresholds-json "${THRESHOLDS_JSON}"
)

if [[ -n "${METHOD_REGEX}" ]]; then
  cmd+=(--method-regex "${METHOD_REGEX}")
fi

if [[ -n "${E7_NUM_BOOTSTRAP:-}" ]]; then
  cmd+=(--num-bootstrap "${E7_NUM_BOOTSTRAP}")
fi

if [[ -n "${E7_ALPHA:-}" ]]; then
  cmd+=(--alpha "${E7_ALPHA}")
fi

if [[ -n "${E7_SEED:-}" ]]; then
  cmd+=(--seed "${E7_SEED}")
fi

if [[ -n "${E7_WEIGHTS:-}" ]]; then
  # Exactly three space-separated weights: sanity perturbation robustness
  # shellcheck disable=SC2206
  weights=( ${E7_WEIGHTS} )
  if [[ "${#weights[@]}" -ne 3 ]]; then
    echo "E7_WEIGHTS must contain exactly three values."
    exit 1
  fi
  cmd+=(--weights "${weights[@]}")
fi

"${cmd[@]}"

