#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../thesis"
tectonic main.tex

echo "Built PDF: $(pwd)/main.pdf"
