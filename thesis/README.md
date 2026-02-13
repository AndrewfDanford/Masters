# Thesis LaTeX Template

## Files
- `main.tex`: thesis entrypoint
- `references.bib`: bibliography database
- `sections/*.tex`: chapter files

## Build
From project root:

```bash
cd thesis
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or with `latexmk` if available:

```bash
cd thesis
latexmk -pdf main.tex
```

Or use the project build script (recommended after setup):

```bash
cd /Users/andrew/Documents/New\ project
bash scripts/build_thesis.sh
```

## Format
Reflow thesis chapter source for consistent line wrapping (no content changes):

```bash
cd /Users/andrew/Documents/New\ project
bash scripts/format_thesis.sh
```

Optional width override:

```bash
THESIS_FMT_WIDTH=110 bash scripts/format_thesis.sh
```

## Customize first
- Update metadata commands in `main.tex` (name, advisor, institution, date).
- Replace placeholder chapter text with your thesis content.
