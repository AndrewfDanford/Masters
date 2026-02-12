#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${PHYSIONET_USERNAME:-}" ]]; then
  echo "Missing required env var: PHYSIONET_USERNAME"
  echo "Example: export PHYSIONET_USERNAME=your_physionet_username"
  exit 1
fi

MIMIC_VER="${MIMIC_CXR_JPG_VERSION:-2.1.0}"
MSCXR_VER="${MS_CXR_VERSION:-1.1.0}"
RADGRAPH_VER="${RADGRAPH_VERSION:-1.0.0}"
OUT_ROOT="${DATA_ROOT:-data/raw}"

INCLUDE_RADGRAPH_FULL="${INCLUDE_RADGRAPH_FULL:-0}"   # 1 to download MIMIC-CXR_graphs.json
INCLUDE_MSCXR_IMAGES="${INCLUDE_MSCXR_IMAGES:-0}"     # 1 to download only images referenced by MS-CXR

mkdir -p "${OUT_ROOT}/mimic-cxr-jpg" "${OUT_ROOT}/ms-cxr" "${OUT_ROOT}/radgraph" "${OUT_ROOT}/manifests"

echo "Downloading MIMIC-CXR-JPG metadata/labels (no bulk images)..."
cat > "${OUT_ROOT}/manifests/mimic_metadata_urls.txt" <<EOF
https://physionet.org/files/mimic-cxr-jpg/${MIMIC_VER}/mimic-cxr-2.0.0-metadata.csv.gz
https://physionet.org/files/mimic-cxr-jpg/${MIMIC_VER}/mimic-cxr-2.0.0-split.csv.gz
https://physionet.org/files/mimic-cxr-jpg/${MIMIC_VER}/mimic-cxr-2.0.0-chexpert.csv.gz
https://physionet.org/files/mimic-cxr-jpg/${MIMIC_VER}/mimic-cxr-2.0.0-negbio.csv.gz
https://physionet.org/files/mimic-cxr-jpg/${MIMIC_VER}/mimic-cxr-2.1.0-test-set-labeled.csv
https://physionet.org/files/mimic-cxr-jpg/${MIMIC_VER}/IMAGE_FILENAMES
EOF

wget -N -c --user "${PHYSIONET_USERNAME}" --ask-password \
  -i "${OUT_ROOT}/manifests/mimic_metadata_urls.txt" \
  -P "${OUT_ROOT}/mimic-cxr-jpg"

echo "Downloading MS-CXR annotations..."
cat > "${OUT_ROOT}/manifests/mscxr_urls.txt" <<EOF
https://physionet.org/files/ms-cxr/${MSCXR_VER}/MS_CXR_Local_Alignment_v${MSCXR_VER}.json
https://physionet.org/files/ms-cxr/${MSCXR_VER}/MS_CXR_Local_Alignment_v${MSCXR_VER}.csv
https://physionet.org/files/ms-cxr/${MSCXR_VER}/convert_coco_json_to_csv.py
EOF

wget -N -c --user "${PHYSIONET_USERNAME}" --ask-password \
  -i "${OUT_ROOT}/manifests/mscxr_urls.txt" \
  -P "${OUT_ROOT}/ms-cxr"

echo "Downloading RadGraph core files..."
cat > "${OUT_ROOT}/manifests/radgraph_urls.txt" <<EOF
https://physionet.org/files/radgraph/${RADGRAPH_VER}/train.json
https://physionet.org/files/radgraph/${RADGRAPH_VER}/dev.json
https://physionet.org/files/radgraph/${RADGRAPH_VER}/test.json
EOF

if [[ "${INCLUDE_RADGRAPH_FULL}" == "1" ]]; then
  cat >> "${OUT_ROOT}/manifests/radgraph_urls.txt" <<EOF
https://physionet.org/files/radgraph/${RADGRAPH_VER}/MIMIC-CXR_graphs.json
https://physionet.org/files/radgraph/${RADGRAPH_VER}/CheXpert_graphs.json
EOF
fi

wget -N -c --user "${PHYSIONET_USERNAME}" --ask-password \
  -i "${OUT_ROOT}/manifests/radgraph_urls.txt" \
  -P "${OUT_ROOT}/radgraph"

if [[ "${INCLUDE_MSCXR_IMAGES}" == "1" ]]; then
  echo "Extracting image paths referenced by MS-CXR..."
  OUT_ROOT="${OUT_ROOT}" MIMIC_VER="${MIMIC_VER}" MSCXR_VER="${MSCXR_VER}" python3 - <<'PY'
import csv
import os
from pathlib import Path

out_root = Path(os.environ["OUT_ROOT"])
mimic_ver = os.environ["MIMIC_VER"]
mscxr_ver = os.environ["MSCXR_VER"]
csv_path = out_root / "ms-cxr" / f"MS_CXR_Local_Alignment_v{mscxr_ver}.csv"
out_path = out_root / "manifests" / "mscxr_image_urls.txt"
base_url = f"https://physionet.org/files/mimic-cxr-jpg/{mimic_ver}/"

if not csv_path.exists():
    raise SystemExit(f"Missing file: {csv_path}")

urls = set()
with csv_path.open("r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        raw = (row.get("path") or "").strip()
        if not raw:
            continue
        marker = "files/"
        idx = raw.find(marker)
        if idx < 0:
            continue
        rel = raw[idx:]
        urls.add(base_url + rel)

with out_path.open("w", encoding="utf-8") as f:
    for url in sorted(urls):
        f.write(url + "\n")
PY

  echo "Downloading only MS-CXR-referenced MIMIC images..."
  wget -N -c \
    --user "${PHYSIONET_USERNAME}" --ask-password \
    -i "${OUT_ROOT}/manifests/mscxr_image_urls.txt" \
    -P "${OUT_ROOT}/mimic-cxr-jpg"
fi

cat <<EOF
Download phase complete.
Files are under: ${OUT_ROOT}
Next:
  1) Set MIMIC metadata/labels env vars for E0.
  2) Run: bash scripts/run_e0_data_audit.sh
EOF
