#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

bash scripts/run_e2_e3_generate.sh

export E23_INPUT_CSV="${E23_INPUT_CSV:-${E23_OUTPUT_CSV:-outputs/reports/e2_e3/e2e3_artifacts.csv}}"
bash scripts/run_e2_e3_saliency.sh

