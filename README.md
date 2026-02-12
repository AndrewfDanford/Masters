# Clinically Interpretable XAI in Radiology

This repository is structured for a two-semester master's thesis on faithful, text-capable explainable AI for chest X-ray diagnosis.

## Current status
- Design and experiment scaffolding complete.
- E0 data audit pipeline implemented.
- Unified faithfulness evaluation scaffold implemented (sanity, deletion/insertion, nuisance robustness).
- E1 baseline training pipeline implemented (using precomputed feature tables).
- Unified artifact runners implemented for E2/E3 saliency scoring and E7 cross-method reporting.
- Next phase: explanation generation pipelines (`E2`-`E6`) and concept/text model runners.

## Key docs
- `docs/design_note.md`
- `docs/experiment_matrix.md`
- `docs/repo_structure.md`
- `docs/code_readiness.md`
- `data_specs/cohort_definition.md`
- `data_specs/label_mapping.md`
- `data_specs/concept_schema.md`

## Thesis build
Once Tectonic is installed, build with:

```bash
bash scripts/build_thesis.sh
```

Output:
- `thesis/main.pdf`

## Paper library helper
To organize papers discussed in this thesis:

```bash
bash scripts/fetch_open_papers.sh
```

See:
- `papers/paper_manifest.csv` for full citation landing links
- `papers/open_access_manifest.csv` for direct open-access PDF links

## E0 data audit (current implementation)
Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

## Data sources (official)
- MIMIC-CXR-JPG: https://physionet.org/content/mimic-cxr-jpg/2.1.0/
- MS-CXR: https://physionet.org/content/ms-cxr/1.1.0/
- RadGraph: https://physionet.org/content/radgraph/1.0.0/

Credentialed PhysioNet access is required for these resources.

Download metadata/annotations first (recommended):

```bash
export PHYSIONET_USERNAME=your_username
bash scripts/download_physionet_cxr_stack.sh
```

Optional flags:

```bash
export INCLUDE_RADGRAPH_FULL=1    # adds larger RadGraph graph files
export INCLUDE_MSCXR_IMAGES=1      # downloads only images referenced by MS-CXR
bash scripts/download_physionet_cxr_stack.sh
```

Run the cohort and prevalence audit:

```bash
export MIMIC_METADATA_CSV=/path/to/mimic-cxr-2.0.0-metadata.csv.gz
export MIMIC_LABELS_CSV=/path/to/mimic-cxr-2.0.0-chexpert.csv.gz
export MIMIC_OFFICIAL_SPLIT_CSV=/path/to/mimic-cxr-2.0.0-split.csv.gz   # optional
export RADGRAPH_CONCEPTS_FILE=/path/to/study_concepts.csv               # optional
bash scripts/run_e0_data_audit.sh
```

Outputs are written to:
- `outputs/reports/e0_cohort_manifest.csv`
- `outputs/reports/e0_split_counts.csv`
- `outputs/reports/e0_label_prevalence.csv`
- `outputs/reports/e0_concept_coverage_summary.csv` (if concepts provided)
- `outputs/reports/e0_concept_support.csv` (if concepts provided)

Run tests:

```bash
PYTHONPATH=. pytest -q
```

## Unified faithfulness scaffold (implemented)
Core evaluation modules:
- `src/eval/faithfulness.py`
- `src/eval/text_grounding.py`
- `src/eval/statistics.py`
- `src/eval/quality_gates.py`
- `src/eval/unified_benchmark.py`

Default quality-gate thresholds:
- `configs/eval/faithfulness_thresholds.json`

Run a synthetic demo (no dataset required):

```bash
bash scripts/run_faithfulness_demo.sh
```

## E1 baseline (implemented)
E1 trains a multi-label linear probe on cohort labels plus precomputed numeric features.

Expected files:
- Cohort CSV: include `study_id`, `split`, and label columns.
- Feature CSV: include `study_id` and numeric feature columns.

Run:

```bash
export E1_COHORT_CSV=/path/to/e0_cohort_manifest.csv
export E1_FEATURES_CSV=/path/to/study_features.csv
# Optional custom labels with commas:
# export E1_LABELS_CSV="Atelectasis,Cardiomegaly,Consolidation,Edema,Pleural Effusion,Pneumothorax"
bash scripts/run_e1_baseline.sh
```

Outputs:
- `outputs/reports/e1/e1_predictions_long.csv`
- `outputs/reports/e1/e1_metrics_by_label.csv`
- `outputs/reports/e1/e1_metrics_summary.csv`
- `outputs/reports/e1/e1_model.npz`
- `outputs/reports/e1/e1_model_card.json`

## E2/E3 saliency scoring runner (implemented)
Run unified scoring on Grad-CAM/HiResCAM artifact tables:

```bash
export E23_INPUT_CSV=configs/explain/e2e3_saliency_input_template.csv
bash scripts/run_e2_e3_saliency.sh
```

Outputs:
- `outputs/reports/e2_e3/e2e3_saliency_sample_scores.csv`
- `outputs/reports/e2_e3/e2e3_saliency_method_summary.csv`
- `outputs/reports/e2_e3/e2e3_saliency_pairwise_nfi_deltas.csv`

## E7 unified benchmark runner (implemented)
Run cross-family benchmark with one artifact schema:

```bash
export E7_INPUT_CSV=configs/eval/e7_input_template.csv
bash scripts/run_e7_unified.sh
```

Template schema:
- `configs/eval/e7_input_template.csv`
