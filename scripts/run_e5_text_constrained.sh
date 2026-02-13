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

if [[ -z "${E5_COHORT_CSV:-}" ]]; then
  echo "Missing required env var: E5_COHORT_CSV"
  echo "Expected: cohort manifest CSV with study_id/split and finding columns."
  exit 1
fi

OUTPUT_CSV="${E5_OUTPUT_CSV:-outputs/reports/e5/e5_artifacts.csv}"
META_JSON="${E5_META_JSON:-outputs/reports/e5/e5_generation_meta.json}"

cmd=(
  "${PYTHON_BIN}" -m src.explain.e5_text_generate
  --cohort-csv "${E5_COHORT_CSV}"
  --output-csv "${OUTPUT_CSV}"
  --meta-json "${META_JSON}"
  --study-col "${E5_STUDY_COL:-study_id}"
  --split-col "${E5_SPLIT_COL:-split}"
  --target-split "${E5_TARGET_SPLIT:-test}"
  --curve-steps "${E5_CURVE_STEPS:-10}"
  --seed "${E5_SEED:-17}"
  --concept-col "${E5_CONCEPT_COL:-concept}"
)

if [[ -n "${E5_LABELS_CSV:-}" ]]; then
  IFS=',' read -r -a labels <<< "${E5_LABELS_CSV}"
  for index in "${!labels[@]}"; do
    labels[$index]="$(echo "${labels[$index]}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
  done
  cmd+=(--labels "${labels[@]}")
fi

if [[ -n "${E5_CONCEPTS_FILE:-}" ]]; then
  cmd+=(--concepts-file "${E5_CONCEPTS_FILE}")
fi

"${cmd[@]}"

