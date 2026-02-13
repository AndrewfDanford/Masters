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

OUTPUT_CSV="${ASSEMBLE_OUTPUT_CSV:-outputs/reports/e7/e7_input_all_methods.csv}"
META_JSON="${ASSEMBLE_META_JSON:-outputs/reports/e7/e7_input_all_methods_meta.json}"
EXPECTED_CURVE_LEN="${ASSEMBLE_EXPECTED_CURVE_LEN:-}"
STRICT_UNIQUE="${ASSEMBLE_STRICT_UNIQUE:-1}"

input_files=()

if [[ -n "${ASSEMBLE_INPUT_CSVS:-}" ]]; then
  IFS=',' read -r -a provided <<< "${ASSEMBLE_INPUT_CSVS}"
  for index in "${!provided[@]}"; do
    path="$(echo "${provided[$index]}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
    if [[ -n "${path}" ]]; then
      input_files+=("${path}")
    fi
  done
else
  default_candidates=(
    "${ASSEMBLE_E23_CSV:-outputs/reports/e2_e3/e2e3_artifacts.csv}"
    "${ASSEMBLE_E4_CSV:-outputs/reports/e4/e4_artifacts.csv}"
    "${ASSEMBLE_E5_CSV:-outputs/reports/e5/e5_artifacts.csv}"
    "${ASSEMBLE_E6_CSV:-outputs/reports/e6/e6_artifacts.csv}"
  )
  for path in "${default_candidates[@]}"; do
    if [[ -f "${path}" ]]; then
      input_files+=("${path}")
    fi
  done
fi

if [[ "${#input_files[@]}" -eq 0 ]]; then
  echo "No artifact files found for assembly."
  echo "Set ASSEMBLE_INPUT_CSVS or ensure default artifact paths exist."
  exit 1
fi

cmd=(
  "${PYTHON_BIN}" -m src.eval.assemble_family_artifacts
  --input-csv "${input_files[@]}"
  --output-csv "${OUTPUT_CSV}"
  --meta-json "${META_JSON}"
)

if [[ -n "${EXPECTED_CURVE_LEN}" ]]; then
  cmd+=(--expected-curve-len "${EXPECTED_CURVE_LEN}")
fi

if [[ "${STRICT_UNIQUE}" != "1" ]]; then
  cmd+=(--allow-duplicates)
fi

"${cmd[@]}"

