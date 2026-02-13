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

if [[ -z "${E4_COHORT_CSV:-}" ]]; then
  echo "Missing required env var: E4_COHORT_CSV"
  echo "Expected: cohort manifest CSV with study_id/split and finding columns."
  exit 1
fi

OUTPUT_CSV="${E4_OUTPUT_CSV:-outputs/reports/e4/e4_artifacts.csv}"
META_JSON="${E4_META_JSON:-outputs/reports/e4/e4_generation_meta.json}"

cmd=(
  "${PYTHON_BIN}" -m src.explain.e4_concept_generate
  --cohort-csv "${E4_COHORT_CSV}"
  --output-csv "${OUTPUT_CSV}"
  --meta-json "${META_JSON}"
  --study-col "${E4_STUDY_COL:-study_id}"
  --split-col "${E4_SPLIT_COL:-split}"
  --target-split "${E4_TARGET_SPLIT:-test}"
  --curve-steps "${E4_CURVE_STEPS:-10}"
  --seed "${E4_SEED:-17}"
  --concept-col "${E4_CONCEPT_COL:-concept}"
)

if [[ -n "${E4_LABELS_CSV:-}" ]]; then
  IFS=',' read -r -a labels <<< "${E4_LABELS_CSV}"
  for index in "${!labels[@]}"; do
    labels[$index]="$(echo "${labels[$index]}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
  done
  cmd+=(--labels "${labels[@]}")
fi

if [[ -n "${E4_CONCEPTS_FILE:-}" ]]; then
  cmd+=(--concepts-file "${E4_CONCEPTS_FILE}")
fi

"${cmd[@]}"

