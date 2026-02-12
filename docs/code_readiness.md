# Code Readiness vs Thesis Plan

## Summary
The repository is close to "data-only blocked" for E0/E1/E2-E3/E7/E8:
- E0 (data audit) is implemented.
- E1 training is implemented for precomputed features, extracted image features, and end-to-end CNN training.
- Unified faithfulness evaluation is implemented and runnable from artifact tables.
- Concept and text explanation generation stages are still pending (with E4-E6 artifact schemas now fixed).

## Status by experiment ID

| Experiment | Status | Notes |
|---|---|---|
| E0 | Implemented | Cohort manifest, split audit, prevalence, concept coverage. |
| E1 | Implemented | Baseline linear probe + MIMIC image-feature extraction + end-to-end CNN training/checkpoint export. |
| E2-E3 | Implemented (checkpoint-dependent) | Grad-CAM/HiResCAM generation + saliency scoring are implemented; requires trained CNN checkpoint. |
| E4 | Not implemented (schema-ready) | Concept-head/CBM training and intervention runner missing; artifact schema/config finalized. |
| E5-E6 | Not implemented (schema-ready) | Constrained/less-constrained rationale generators missing; artifact schemas/configs finalized. |
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
- `src/eval/e8_randomization.py`: multi-run sanity randomization benchmark summaries with observed-run and bootstrap-proxy modes.
- `src/data/synthetic_smoke.py`: synthetic image/manifest/artifact generation for data-free integration checks.
- `scripts/run_synthetic_smoke.sh`: one-command synthetic integration smoke pipeline (E1 -> E2/E3 -> E7 -> E8).
- `scripts/run_after_data.sh`: guarded one-command real-data pipeline runner.
- `scripts/setup_env.sh` + `Dockerfile`: reproducible runtime setup paths.
- `configs/eval/faithfulness_thresholds.json`: default thresholds.
- `configs/eval/e7_input_template.csv`: unified artifact schema example.
- `configs/explain/e2e3_saliency_input_template.csv`: E2/E3 artifact schema example.
- `configs/eval/e8_randomization_input_template.csv`: E8 input schema example.
- `docs/explanation_artifact_schemas.md`: frozen E4-E6 artifact contracts.
- Tests for all the above modules under `tests/`.

## Immediate next coding priorities
1. Add concept intervention runner for `E4`.
2. Add constrained and less-constrained rationale generators for `E5-E6`.
3. Add E9 stress-test data builder for spurious-cue sensitivity.
4. Add light integration tests for `run_e1_train_cnn.sh` + `run_e2_e3_generate.sh` with toy images/checkpoints.
5. Add optional external validation wiring after internal data runs stabilize.
