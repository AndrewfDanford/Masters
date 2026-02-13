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

PORT="${DASHBOARD_PORT:-8501}"
ADDRESS="${DASHBOARD_ADDRESS:-127.0.0.1}"

if ! "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import importlib.util
import sys
sys.exit(0 if importlib.util.find_spec("streamlit") is not None else 1)
PY
then
  echo "Missing Python dependency: streamlit"
  echo "Install project dependencies first:"
  echo "  bash scripts/setup_env.sh"
  echo "If network access is restricted, run:"
  echo "  .venv/bin/python -m pip install streamlit"
  exit 1
fi

"${PYTHON_BIN}" -m streamlit run dashboard/app.py \
  --server.port "${PORT}" \
  --server.address "${ADDRESS}"
