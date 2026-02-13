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

if [[ -z "${E6_COHORT_CSV:-}" ]]; then
  echo "Missing required env var: E6_COHORT_CSV"
  echo "Expected: cohort manifest CSV with study_id/split and finding columns."
  exit 1
fi

OUTPUT_CSV="${E6_OUTPUT_CSV:-outputs/reports/e6/e6_artifacts.csv}"
META_JSON="${E6_META_JSON:-outputs/reports/e6/e6_generation_meta.json}"

cmd=(
  "${PYTHON_BIN}" -m src.explain.e6_text_generate
  --cohort-csv "${E6_COHORT_CSV}"
  --output-csv "${OUTPUT_CSV}"
  --meta-json "${META_JSON}"
  --study-col "${E6_STUDY_COL:-study_id}"
  --split-col "${E6_SPLIT_COL:-split}"
  --target-split "${E6_TARGET_SPLIT:-test}"
  --curve-steps "${E6_CURVE_STEPS:-10}"
  --seed "${E6_SEED:-17}"
  --concept-col "${E6_CONCEPT_COL:-concept}"
)

if [[ -n "${E6_LABELS_CSV:-}" ]]; then
  IFS=',' read -r -a labels <<< "${E6_LABELS_CSV}"
  for index in "${!labels[@]}"; do
    labels[$index]="$(echo "${labels[$index]}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
  done
  cmd+=(--labels "${labels[@]}")
fi

if [[ -n "${E6_CONCEPTS_FILE:-}" ]]; then
  cmd+=(--concepts-file "${E6_CONCEPTS_FILE}")
fi

"${cmd[@]}"

