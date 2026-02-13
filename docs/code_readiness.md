# Code Readiness vs Thesis Plan

## Summary
The repository is now "data-ready" through E8 with proxy generators for E4-E6:
- E0 (data audit) is implemented.
- E1 training is implemented for precomputed features, extracted image features, and end-to-end CNN training.
- Unified faithfulness evaluation is implemented and runnable from artifact tables.
- E4/E5/E6 proxy artifact generators are implemented with frozen contracts for E7/E8 compatibility.

## Status by experiment ID

| Experiment | Status | Notes |
|---|---|---|
| E0 | Implemented | Cohort manifest, split audit, prevalence, concept coverage. |
| E1 | Implemented | Baseline linear probe + MIMIC image-feature extraction + end-to-end CNN training/checkpoint export. |
| E2-E3 | Implemented (checkpoint-dependent) | Grad-CAM/HiResCAM generation + saliency scoring are implemented; requires trained CNN checkpoint. |
| E4 | Implemented (proxy generator) | Contract-valid concept-family artifacts generated; model-backed concept head remains future upgrade. |
| E5-E6 | Implemented (proxy generators) | Contract-valid constrained/unconstrained text artifacts generated; model-backed rationale generation remains future upgrade. |
| E7 | Implemented | Unified benchmark runner, CIs, quality gates, and pairwise method deltas. |
| E8 | Implemented (artifact-driven) | Multi-run randomization benchmark runner with pass-rate summaries and CIs. |
| E9 | Not implemented | Spurious-cue stress-test pipeline missing. |

## Implemented now (to match Sections 5-6)
- `src/eval/faithfulness.py`: sanity, deletion/insertion, nuisance robustness, NFI.
- `src/eval/text_grounding.py`: concept alignment, finding consistency, contradiction rate.
- `src/eval/statistics.py`: bootstrap CIs and paired bootstrap differences.
- `src/eval/quality_gates.py`: predefined threshold gates and pass/fail logic.
- `src/eval/unified_benchmark.py`: artifact ingestion, sample/method scoring, pairwise deltas.
- `src/models/e1_baseline.py`: baseline training/evaluation on precomputed feature tables.
- `src/models/e1_extract_features.py`: image feature extraction from MIMIC-style paths.
- `src/models/image_features.py`: deterministic handcrafted grayscale feature set.
- `src/models/backbone_features.py`: optional ResNet/DenseNet feature extraction (requires torch/torchvision).
- `src/models/e1_train_cnn.py`: end-to-end CNN training on image files with uncertain-label masking.
- `src/explain/cam.py`: Grad-CAM/HiResCAM core, perturbation helpers, randomization hooks.
- `src/explain/e2_e3_generate.py`: saliency artifact generation from model checkpoint and manifest.
- `src/explain/family_artifacts.py`: shared E4/E5/E6 proxy artifact generation + schema validation.
- `src/explain/e4_concept_generate.py`: E4 concept-family artifact generator.
- `src/explain/e5_text_generate.py`: E5 constrained-text artifact generator.
- `src/explain/e6_text_generate.py`: E6 unconstrained-text artifact generator.
- `src/eval/e8_randomization.py`: multi-run sanity randomization benchmark summaries with observed-run and bootstrap-proxy modes.
- `src/eval/assemble_family_artifacts.py`: strict assembly/validation of E2/E3/E4/E5/E6 artifacts for unified E7/E8 input.
- `src/data/synthetic_smoke.py`: synthetic image/manifest/artifact generation for data-free integration checks.
- `src/dashboard/io.py` + `dashboard/app.py`: artifact-driven monitoring dashboard for end-to-end pipeline visibility.
- `scripts/run_e4_concept.sh`: E4 concept artifact runner.
- `scripts/run_e5_text_constrained.sh`: E5 constrained-text artifact runner.
- `scripts/run_e6_text_unconstrained.sh`: E6 unconstrained-text artifact runner.
- `scripts/run_assemble_family_artifacts.sh`: artifact assembly runner for unified E7/E8 input.
- `scripts/run_synthetic_smoke.sh`: one-command synthetic integration smoke pipeline (E1 -> E2/E3 -> E4/E5/E6 -> E7 -> E8).
- `scripts/run_after_data.sh`: guarded one-command real-data pipeline runner including E4/E5/E6 proxy path.
- `scripts/run_dashboard.sh`: one-command dashboard launcher for live artifact monitoring.
- `scripts/setup_env.sh` + `Dockerfile`: reproducible runtime setup paths.
- `configs/eval/faithfulness_thresholds.json`: default thresholds.
- `configs/eval/e7_input_template.csv`: unified artifact schema example.
- `configs/explain/e2e3_saliency_input_template.csv`: E2/E3 artifact schema example.
- `configs/eval/e8_randomization_input_template.csv`: E8 input schema example.
- `docs/explanation_artifact_schemas.md`: frozen E4-E6 artifact contracts.
- Tests for all the above modules under `tests/`.

## Immediate next coding priorities
1. Replace E4 proxy generator with model-backed concept head + explicit intervention runner.
2. Replace E5/E6 proxy generators with model-backed rationale generation pipelines.
3. Add E9 stress-test data builder for spurious-cue sensitivity.
4. Add light integration tests for `run_e1_train_cnn.sh` + `run_e2_e3_generate.sh` with toy images/checkpoints.
5. Add optional external validation wiring after internal data runs stabilize.
