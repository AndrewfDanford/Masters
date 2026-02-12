#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

require_file_var() {
  local var_name="$1"
  local hint="$2"
  local value="${!var_name:-}"
  if [[ -z "${value}" ]]; then
    echo "Missing required env var: ${var_name}"
    echo "${hint}"
    exit 1
  fi
  if [[ ! -f "${value}" ]]; then
    echo "File does not exist for ${var_name}: ${value}"
    exit 1
  fi
}

require_dir_var() {
  local var_name="$1"
  local hint="$2"
  local value="${!var_name:-}"
  if [[ -z "${value}" ]]; then
    echo "Missing required env var: ${var_name}"
    echo "${hint}"
    exit 1
  fi
  if [[ ! -d "${value}" ]]; then
    echo "Directory does not exist for ${var_name}: ${value}"
    exit 1
  fi
}

echo "[0/3] Validating required paths..."
require_file_var "MIMIC_METADATA_CSV" "Set to mimic-cxr-2.0.0-metadata.csv.gz"
require_file_var "MIMIC_LABELS_CSV" "Set to mimic-cxr-2.0.0-chexpert.csv.gz"
require_dir_var "E1_IMAGE_ROOT" "Set to the local MIMIC-CXR-JPG root directory."

if [[ -n "${MIMIC_OFFICIAL_SPLIT_CSV:-}" && ! -f "${MIMIC_OFFICIAL_SPLIT_CSV}" ]]; then
  echo "MIMIC_OFFICIAL_SPLIT_CSV was set but file does not exist: ${MIMIC_OFFICIAL_SPLIT_CSV}"
  exit 1
fi

echo "[1/3] Running E0 data audit..."
bash scripts/run_e0_data_audit.sh

export E1_COHORT_CSV="${E1_COHORT_CSV:-outputs/reports/e0_cohort_manifest.csv}"
if [[ ! -f "${E1_COHORT_CSV}" ]]; then
  echo "Expected cohort manifest missing: ${E1_COHORT_CSV}"
  echo "Check E0 logs and the MIMIC input CSV paths."
  exit 1
fi

export E1_RUN_CNN="${E1_RUN_CNN:-1}"
if [[ "${E1_RUN_CNN}" != "1" ]]; then
  echo "E1_RUN_CNN=${E1_RUN_CNN}. For thesis-default pipeline, set E1_RUN_CNN=1."
fi

echo "[2/3] Running E1 -> E2/E3 -> E7 pipeline..."
bash scripts/run_small_subset_pipeline.sh

echo "[3/3] Completed."
echo "Primary outputs:"
echo "  - outputs/reports/e0_cohort_manifest.csv"
echo "  - outputs/reports/e1_cnn/e1_cnn_metrics_summary.csv (if E1_RUN_CNN=1)"
echo "  - outputs/reports/e2_e3/e2e3_saliency_method_summary.csv"
echo "  - outputs/reports/e7/e7_method_summary.csv"
