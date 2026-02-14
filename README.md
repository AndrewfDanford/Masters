# Clinically Interpretable XAI in Radiology

This repository is structured for a two-semester master's thesis on faithful explainable AI for chest X-ray diagnosis, with core scope on saliency and concept methods.

## Current status
- Design and experiment scaffolding complete.
- E0 data audit pipeline implemented.
- Unified faithfulness evaluation scaffold implemented (sanity, deletion/insertion, nuisance robustness).
- E1 baseline training pipeline implemented (feature-table and end-to-end CNN variants).
- Unified artifact runners implemented for E2/E3 saliency scoring, E7 reporting, and E8 randomization checks.
- E4 proxy generator implemented for contract-complete pre-data execution in the concept track.
- Optional future-work text proxies (E5/E6) retained but disabled by default in the core pipeline.
- Next phase: replace proxy E4 with model-backed implementation and add stress-test pipeline (`E9`).

## Key docs
- `docs/design_note.md`
- `docs/experiment_matrix.md`
- `docs/repo_structure.md`
- `docs/code_readiness.md`
- `docs/handoff_quickstart.md`
- `docs/explanation_artifact_schemas.md`
- `data_specs/cohort_definition.md`
- `data_specs/label_mapping.md`
- `data_specs/concept_schema.md`

## Reproducible runtime
Native setup (recommended before first run):

```bash
bash scripts/setup_env.sh
```

Containerized setup:

```bash
docker compose build
docker compose run --rm thesis bash
```

## Monitoring dashboard
Launch the thesis pipeline dashboard:

```bash
bash scripts/run_dashboard.sh
```

Default URL:
- `http://127.0.0.1:8501`

The dashboard shows:
- stage-by-stage pipeline status (E0, E1, E2/E3, E4, E7, E8),
- data audit and run metadata,
- E1 training metrics,
- E2/E3 and E7 faithfulness summaries,
- E8 randomization summaries,
- per-study sample inspection for available artifacts.

The sidebar also includes run buttons so you can trigger:
- synthetic smoke pipeline,
- E4 proxy generator,
- family artifact assembly,
- E8 template demo run,
- full after-data pipeline.

## Quick start on a new machine
Use the handoff runbook:
- `docs/handoff_quickstart.md`

Fast copy-paste path (after environment setup):

```bash
export MIMIC_METADATA_CSV=/absolute/path/to/mimic-cxr-2.0.0-metadata.csv.gz
export MIMIC_LABELS_CSV=/absolute/path/to/mimic-cxr-2.0.0-chexpert.csv.gz
export MIMIC_OFFICIAL_SPLIT_CSV=/absolute/path/to/mimic-cxr-2.0.0-split.csv.gz
export E1_IMAGE_ROOT=/absolute/path/to/mimic-cxr-jpg
bash scripts/run_after_data.sh
```

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

Notes:
- `e0_cohort_manifest.csv` keeps `dicom_id` and `path` when present in metadata, so the same manifest can drive image-feature extraction.

One-command first real run (E0 -> E1 CNN -> E2/E3 -> E4 proxy -> E7 -> E8):

```bash
export MIMIC_METADATA_CSV=/path/to/mimic-cxr-2.0.0-metadata.csv.gz
export MIMIC_LABELS_CSV=/path/to/mimic-cxr-2.0.0-chexpert.csv.gz
export MIMIC_OFFICIAL_SPLIT_CSV=/path/to/mimic-cxr-2.0.0-split.csv.gz   # optional but recommended
export E1_IMAGE_ROOT=/path/to/mimic-cxr-jpg
# Optional future-work path:
# export E5_RUN=1
# export E6_RUN=1
bash scripts/run_after_data.sh
```

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

End-to-end synthetic smoke run (tiny generated images + artifacts):

```bash
bash scripts/run_synthetic_smoke.sh
```

Smoke outputs:
- `outputs/smoke/e1/e1_metrics_summary.csv`
- `outputs/smoke/e2_e3/e2e3_saliency_method_summary.csv`
- `outputs/smoke/e4/e4_artifacts.csv`
- `outputs/smoke/e5/e5_artifacts.csv` (optional; set `SMOKE_ENABLE_TEXT_PROXY=1`)
- `outputs/smoke/e6/e6_artifacts.csv` (optional; set `SMOKE_ENABLE_TEXT_PROXY=1`)
- `outputs/smoke/e7/e7_input_all_methods.csv`
- `outputs/smoke/e7/e7_method_summary.csv`
- `outputs/smoke/e8/e8_smoke_method_summary.csv`

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

## E1 image-feature extraction (implemented)
Extract image features directly from MIMIC JPGs:

```bash
export E1_IMAGE_INPUT_CSV=/path/to/e0_cohort_manifest.csv
export E1_IMAGE_ROOT=/path/to/mimic-cxr-jpg
bash scripts/run_e1_extract_features.sh
```

Output:
- `outputs/reports/e1/e1_image_features.csv`
- `outputs/reports/e1/e1_missing_images.csv`

Default extractor is `handcrafted` (no heavy ML dependency). Optional deep backbones:

```bash
export E1_FEATURE_EXTRACTOR=resnet18     # or densenet121
export E1_PRETRAINED_BACKBONE=1          # optional ImageNet init
export E1_BACKBONE_DEVICE=cpu            # or cuda
bash scripts/run_e1_extract_features.sh
```

Backbone mode requires `torch` and `torchvision` installed in your environment.

One-command E1 from images (extract + train):

```bash
export E1_COHORT_CSV=/path/to/e0_cohort_manifest.csv
export E1_IMAGE_ROOT=/path/to/mimic-cxr-jpg
bash scripts/run_e1_from_images.sh
```

## E1 end-to-end CNN training (implemented)
Train a multi-label chest X-ray classifier directly from image files and save a checkpoint for E2/E3 generation:

```bash
export E1_COHORT_CSV=/path/to/e0_cohort_manifest.csv
export E1_IMAGE_ROOT=/path/to/mimic-cxr-jpg
bash scripts/run_e1_train_cnn.sh
```

Optional knobs:

```bash
export E1_CNN_ARCH=resnet18              # or densenet121
export E1_CNN_PRETRAINED_BACKBONE=1
export E1_CNN_EPOCHS=8
export E1_CNN_DEVICE=cpu                 # or cuda
export E1_CNN_MAX_SAMPLES_PER_SPLIT=2000 # fast smoke run
bash scripts/run_e1_train_cnn.sh
```

Outputs:
- `outputs/models/e1_cnn_checkpoint.pt`
- `outputs/reports/e1_cnn/e1_cnn_metrics_by_label.csv`
- `outputs/reports/e1_cnn/e1_cnn_metrics_summary.csv`
- `outputs/reports/e1_cnn/e1_cnn_predictions_long.csv`
- `outputs/reports/e1_cnn/e1_cnn_dataset_counts.csv`

Small-subset orchestration (E1 -> optional E2/E3 -> optional E7):

```bash
export E1_COHORT_CSV=/path/to/e0_cohort_manifest.csv
export E1_IMAGE_ROOT=/path/to/mimic-cxr-jpg
# Optional:
# export E1_RUN_CNN=1                        # use scripts/run_e1_train_cnn.sh for step 1
# Optional:
# export E23_INPUT_CSV=/path/to/e2e3_artifacts.csv
# export E23_MODEL_CHECKPOINT=/path/to/cxr_model.pt   # enables E2/E3 generation inside pipeline
# export E7_INPUT_CSV=/path/to/unified_artifacts.csv
bash scripts/run_small_subset_pipeline.sh
```

When `E1_RUN_CNN=1`, the pipeline defaults `E23_MODEL_CHECKPOINT` to `outputs/models/e1_cnn_checkpoint.pt`.

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

## E2/E3 saliency artifact generation (implemented)
Generate Grad-CAM/HiResCAM artifact tables from a CNN checkpoint:

```bash
export E23_MANIFEST_CSV=/path/to/e0_cohort_manifest.csv
export E23_IMAGE_ROOT=/path/to/mimic-cxr-jpg
export E23_MODEL_CHECKPOINT=/path/to/cxr_model.pt
bash scripts/run_e2_e3_generate.sh
```

Then run scoring:

```bash
export E23_INPUT_CSV=outputs/reports/e2_e3/e2e3_artifacts.csv
bash scripts/run_e2_e3_saliency.sh
```

One-command generate+score:

```bash
bash scripts/run_e2_e3_from_model.sh
```

Generation template config:
- `configs/explain/e2e3_generation.json`

## E4 proxy artifact generator (core, implemented pre-data)
This runner generates contract-valid concept-family artifacts so E7/E8 can run end-to-end
before model-backed E4 is finished.

E4 concept-family proxy:

```bash
export E4_COHORT_CSV=outputs/reports/e0_cohort_manifest.csv
bash scripts/run_e4_concept.sh
```

Core proxy output:
- `outputs/reports/e4/e4_artifacts.csv`

The following text-family proxies are optional future-work paths (disabled by default in `run_after_data.sh`).

E5 constrained-text proxy (optional):

```bash
export E5_COHORT_CSV=outputs/reports/e0_cohort_manifest.csv
bash scripts/run_e5_text_constrained.sh
```

E6 unconstrained-text proxy (optional):

```bash
export E6_COHORT_CSV=outputs/reports/e0_cohort_manifest.csv
bash scripts/run_e6_text_unconstrained.sh
```

Proxy outputs:
- `outputs/reports/e4/e4_artifacts.csv`
- `outputs/reports/e5/e5_artifacts.csv`
- `outputs/reports/e6/e6_artifacts.csv`

Each proxy run also writes a `*_generation_meta.json` file with a warning that the artifacts are
for integration benchmarking only and should be replaced by model-backed generation for final claims.

## Assemble family artifacts for E7/E8 (implemented)
Merge available family artifacts into a strict, validated unified input table (core default: E2/E3 + E4):

```bash
bash scripts/run_assemble_family_artifacts.sh
```

Default output:
- `outputs/reports/e7/e7_input_all_methods.csv`
- `outputs/reports/e7/e7_input_all_methods_meta.json`

Optional explicit input list:

```bash
export ASSEMBLE_INPUT_CSVS="outputs/reports/e2_e3/e2e3_artifacts.csv,outputs/reports/e4/e4_artifacts.csv"
bash scripts/run_assemble_family_artifacts.sh
```

## E7 unified benchmark runner (implemented)
Run cross-family benchmark with one artifact schema:

```bash
export E7_INPUT_CSV=configs/eval/e7_input_template.csv
bash scripts/run_e7_unified.sh
```

Template schema:
- `configs/eval/e7_input_template.csv`

## E8 randomization sanity runner (implemented)
Run multi-run sanity sensitivity summary from artifact tables:

```bash
export E8_INPUT_CSV=configs/eval/e8_randomization_input_template.csv
bash scripts/run_e8_randomization.sh
```

Outputs:
- `outputs/reports/e8/e8_randomization_run_scores.csv`
- `outputs/reports/e8/e8_randomization_method_summary.csv`
- `outputs/reports/e8/e8_randomization_sample_variability.csv`

Template/config:
- `configs/eval/e8_randomization_input_template.csv`
- `configs/eval/e8_randomization.json`

## E4 core + optional E5/E6 schemas and templates
Artifact contracts and starter templates:

- `docs/explanation_artifact_schemas.md`
- `configs/explain/e4_concept_artifact_template.csv`
- `configs/explain/e5_text_constrained_artifact_template.csv`
- `configs/explain/e6_text_unconstrained_artifact_template.csv`
