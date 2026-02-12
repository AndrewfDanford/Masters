#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_SETUP_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
INSTALL_TORCH="${INSTALL_TORCH:-1}"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Creating virtual environment at ${VENV_DIR}..."
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

echo "Installing pinned project dependencies..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if [[ "${INSTALL_TORCH}" == "1" ]]; then
  echo "Installing torch/torchvision..."
  python -m pip install torch torchvision
fi

echo "Verifying core imports..."
python - <<'PY'
import importlib
modules = ["numpy", "pandas", "PIL"]
for name in modules:
    importlib.import_module(name)
print("core_imports_ok")
try:
    importlib.import_module("torch")
    importlib.import_module("torchvision")
    print("torch_imports_ok")
except Exception:
    print("torch_imports_skipped_or_unavailable")
PY

echo "Environment setup complete."
