# Paper Library

This folder is a reading and citation companion for the thesis sections.

## Files
- `paper_manifest.csv`: full paper list with landing pages and notes.
- `open_access_manifest.csv`: subset with direct PDF links expected to be publicly downloadable.
- `downloads/`: target folder for downloaded PDFs.

## Download open-access PDFs
From project root:

```bash
bash scripts/fetch_open_papers.sh
```

Notes:
- Some publisher-hosted papers are paywalled or institution-gated and are therefore not in `open_access_manifest.csv`.
- For gated papers, use your university library, DOI resolver, or Zotero connector from each landing page in `paper_manifest.csv`.
