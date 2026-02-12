#!/usr/bin/env bash
set -euo pipefail

MANIFEST="papers/open_access_manifest.csv"
OUT_DIR="papers/downloads"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "Missing manifest: ${MANIFEST}"
  exit 1
fi

mkdir -p "${OUT_DIR}"

tail -n +2 "${MANIFEST}" | while IFS=, read -r key pdf_url; do
  if [[ -z "${key}" || -z "${pdf_url}" ]]; then
    continue
  fi

  out_file="${OUT_DIR}/${key}.pdf"
  tmp_file="$(mktemp)"
  echo "Fetching ${key}..."

  if curl -L --fail --silent --show-error "${pdf_url}" -o "${tmp_file}"; then
    true
  elif curl -L --fail --silent --show-error \
    -A "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36" \
    -H "Accept: application/pdf,text/html;q=0.9,*/*;q=0.8" \
    "${pdf_url}" -o "${tmp_file}"; then
    true
  else
    echo "  failed: ${pdf_url}"
    rm -f "${tmp_file}"
    continue
  fi

  if head -c 4 "${tmp_file}" | grep -q "%PDF"; then
    mv "${tmp_file}" "${out_file}"
    echo "  saved: ${out_file}"
  else
    echo "  failed (not a direct PDF): ${pdf_url}"
    rm -f "${tmp_file}"
  fi
done

echo "Done. Check ${OUT_DIR} for downloaded files."
