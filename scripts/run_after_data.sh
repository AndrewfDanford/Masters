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

echo "[0/6] Validating required paths..."
require_file_var "MIMIC_METADATA_CSV" "Set to mimic-cxr-2.0.0-metadata.csv.gz"
require_file_var "MIMIC_LABELS_CSV" "Set to mimic-cxr-2.0.0-chexpert.csv.gz"
require_dir_var "E1_IMAGE_ROOT" "Set to the local MIMIC-CXR-JPG root directory."

if [[ -n "${MIMIC_OFFICIAL_SPLIT_CSV:-}" && ! -f "${MIMIC_OFFICIAL_SPLIT_CSV}" ]]; then
  echo "MIMIC_OFFICIAL_SPLIT_CSV was set but file does not exist: ${MIMIC_OFFICIAL_SPLIT_CSV}"
  exit 1
fi

echo "[1/6] Running E0 data audit..."
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

echo "[2/6] Running E1 -> E2/E3 pipeline..."
bash scripts/run_small_subset_pipeline.sh

export E4_COHORT_CSV="${E4_COHORT_CSV:-${E1_COHORT_CSV}}"
export E5_COHORT_CSV="${E5_COHORT_CSV:-${E1_COHORT_CSV}}"
export E6_COHORT_CSV="${E6_COHORT_CSV:-${E1_COHORT_CSV}}"

export E4_RUN="${E4_RUN:-1}"
export E5_RUN="${E5_RUN:-1}"
export E6_RUN="${E6_RUN:-1}"

echo "[3/6] Running E4/E5/E6 proxy generators (toggle with E4_RUN/E5_RUN/E6_RUN)..."
if [[ "${E4_RUN}" == "1" ]]; then
  bash scripts/run_e4_concept.sh
else
  echo "Skipping E4 (E4_RUN=${E4_RUN})."
fi

if [[ "${E5_RUN}" == "1" ]]; then
  bash scripts/run_e5_text_constrained.sh
else
  echo "Skipping E5 (E5_RUN=${E5_RUN})."
fi

if [[ "${E6_RUN}" == "1" ]]; then
  bash scripts/run_e6_text_unconstrained.sh
else
  echo "Skipping E6 (E6_RUN=${E6_RUN})."
fi

export ASSEMBLE_E23_CSV="${ASSEMBLE_E23_CSV:-${E23_INPUT_CSV:-outputs/reports/e2_e3/e2e3_artifacts.csv}}"
export ASSEMBLE_E4_CSV="${ASSEMBLE_E4_CSV:-outputs/reports/e4/e4_artifacts.csv}"
export ASSEMBLE_E5_CSV="${ASSEMBLE_E5_CSV:-outputs/reports/e5/e5_artifacts.csv}"
export ASSEMBLE_E6_CSV="${ASSEMBLE_E6_CSV:-outputs/reports/e6/e6_artifacts.csv}"
export ASSEMBLE_OUTPUT_CSV="${ASSEMBLE_OUTPUT_CSV:-outputs/reports/e7/e7_input_all_methods.csv}"
export ASSEMBLE_META_JSON="${ASSEMBLE_META_JSON:-outputs/reports/e7/e7_input_all_methods_meta.json}"

echo "[4/6] Assembling E2/E3/E4/E5/E6 artifacts for unified evaluation..."
bash scripts/run_assemble_family_artifacts.sh

export E7_RUN="${E7_RUN:-1}"
if [[ "${E7_RUN}" == "1" ]]; then
  export E7_INPUT_CSV="${E7_INPUT_CSV:-${ASSEMBLE_OUTPUT_CSV}}"
  echo "[5/6] Running E7 unified benchmark on assembled artifacts..."
  bash scripts/run_e7_unified.sh
else
  echo "[5/6] Skipping E7 (E7_RUN=${E7_RUN})."
fi

export E8_RUN="${E8_RUN:-1}"
if [[ "${E8_RUN}" == "1" ]]; then
  export E8_INPUT_CSV="${E8_INPUT_CSV:-${ASSEMBLE_OUTPUT_CSV}}"
  echo "[6/6] Running E8 randomization summary on assembled artifacts..."
  bash scripts/run_e8_randomization.sh
else
  echo "[6/6] Skipping E8 (E8_RUN=${E8_RUN})."
fi

echo "Completed."
echo "Primary outputs:"
echo "  - outputs/reports/e0_cohort_manifest.csv"
echo "  - outputs/reports/e1_cnn/e1_cnn_metrics_summary.csv (if E1_RUN_CNN=1)"
echo "  - outputs/reports/e2_e3/e2e3_saliency_method_summary.csv"
echo "  - outputs/reports/e4/e4_artifacts.csv (if E4_RUN=1)"
echo "  - outputs/reports/e5/e5_artifacts.csv (if E5_RUN=1)"
echo "  - outputs/reports/e6/e6_artifacts.csv (if E6_RUN=1)"
echo "  - outputs/reports/e7/e7_input_all_methods.csv"
echo "  - outputs/reports/e7/e7_method_summary.csv"
echo "  - outputs/reports/e8/e8_randomization_method_summary.csv"
