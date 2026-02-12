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

if [[ -z "${E1_COHORT_CSV:-}" ]]; then
  echo "Missing required env var: E1_COHORT_CSV"
  exit 1
fi

if [[ -z "${E1_FEATURES_CSV:-}" ]]; then
  echo "Missing required env var: E1_FEATURES_CSV"
  exit 1
fi

OUTPUT_DIR="${E1_OUTPUT_DIR:-outputs/reports/e1}"

cmd=(
  "${PYTHON_BIN}" -m src.models.e1_baseline
  --cohort-csv "${E1_COHORT_CSV}"
  --features-csv "${E1_FEATURES_CSV}"
  --output-dir "${OUTPUT_DIR}"
)

if [[ -n "${E1_STUDY_COL:-}" ]]; then
  cmd+=(--study-col "${E1_STUDY_COL}")
fi

if [[ -n "${E1_SPLIT_COL:-}" ]]; then
  cmd+=(--split-col "${E1_SPLIT_COL}")
fi

if [[ -n "${E1_LABELS_CSV:-}" ]]; then
  IFS=',' read -r -a labels <<< "${E1_LABELS_CSV}"
  for index in "${!labels[@]}"; do
    labels[$index]="$(echo "${labels[$index]}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
  done
  cmd+=(--labels "${labels[@]}")
elif [[ -n "${E1_LABELS:-}" ]]; then
  # Backward-compatible fallback; space-separated values.
  # shellcheck disable=SC2206
  labels=( ${E1_LABELS} )
  cmd+=(--labels "${labels[@]}")
fi

if [[ -n "${E1_FEATURE_COLUMNS:-}" ]]; then
  cmd+=(--feature-columns "${E1_FEATURE_COLUMNS}")
fi

if [[ -n "${E1_EPOCHS:-}" ]]; then
  cmd+=(--epochs "${E1_EPOCHS}")
fi

if [[ -n "${E1_LEARNING_RATE:-}" ]]; then
  cmd+=(--learning-rate "${E1_LEARNING_RATE}")
fi

if [[ -n "${E1_WEIGHT_DECAY:-}" ]]; then
  cmd+=(--weight-decay "${E1_WEIGHT_DECAY}")
fi

if [[ -n "${E1_SEED:-}" ]]; then
  cmd+=(--seed "${E1_SEED}")
fi

if [[ "${E1_NO_CLASS_BALANCE:-0}" == "1" ]]; then
  cmd+=(--no-class-balance)
fi

"${cmd[@]}"
