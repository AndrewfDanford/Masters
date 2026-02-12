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

export E1_IMAGE_INPUT_CSV="${E1_IMAGE_INPUT_CSV:-${E1_COHORT_CSV}}"
export E1_FEATURES_CSV="${E1_FEATURES_CSV:-outputs/reports/e1/e1_image_features.csv}"

bash scripts/run_e1_extract_features.sh
bash scripts/run_e1_baseline.sh

