#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PYTHON_BIN=".venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

if [[ -z "${MIMIC_METADATA_CSV:-}" ]]; then
  echo "Missing required env var: MIMIC_METADATA_CSV"
  exit 1
fi

if [[ -z "${MIMIC_LABELS_CSV:-}" ]]; then
  echo "Missing required env var: MIMIC_LABELS_CSV"
  exit 1
fi

cmd=(
  "${PYTHON_BIN}" -m src.data.audit
  --metadata-csv "${MIMIC_METADATA_CSV}"
  --labels-csv "${MIMIC_LABELS_CSV}"
  --output-dir outputs/reports
  --uncertain-policy "${UNCERTAIN_POLICY:-u_ignore}"
  --seed "${AUDIT_SEED:-17}"
)

if [[ -n "${MIMIC_OFFICIAL_SPLIT_CSV:-}" ]]; then
  cmd+=(--official-split-csv "${MIMIC_OFFICIAL_SPLIT_CSV}")
fi

if [[ -n "${RADGRAPH_CONCEPTS_FILE:-}" ]]; then
  cmd+=(--concepts-file "${RADGRAPH_CONCEPTS_FILE}")
fi

if [[ "${ALL_VIEWS:-0}" == "1" ]]; then
  cmd+=(--all-views)
fi

"${cmd[@]}"
