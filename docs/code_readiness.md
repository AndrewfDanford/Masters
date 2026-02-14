# Code Readiness vs Thesis Plan

## Summary
The repository is now "data-ready" for the narrowed thesis core scope:
- E0 (data audit) is implemented.
- E1 training is implemented for precomputed features, extracted image features, and end-to-end CNN training.
- E2/E3 saliency artifact generation and scoring are implemented.
- E4 concept-family proxy artifacts are implemented to keep E7/E8 interfaces stable before model-backed E4.
- E7 unified faithfulness and E8 randomization are implemented.

Text-family runners (E5/E6) remain available as optional future-work paths but are not part of the core master's claim set.

## Status by experiment ID

| Experiment | Status | Notes |
|---|---|---|
| E0 | Implemented | Cohort manifest, split audit, prevalence, concept coverage. |
| E1 | Implemented | Baseline linear probe + MIMIC image-feature extraction + end-to-end CNN training/checkpoint export. |
| E2-E3 | Implemented (checkpoint-dependent) | Grad-CAM/HiResCAM generation + saliency scoring are implemented; requires trained CNN checkpoint. |
| E4 | Implemented (proxy generator) | Contract-valid concept-family artifacts generated; model-backed concept head remains future upgrade. |
| E5-E6 | Implemented (optional future-work proxies) | Constrained/unconstrained text artifacts remain available but are not in the default core pipeline. |
| E7 | Implemented | Unified benchmark runner, CIs, quality gates, and pairwise method deltas. |
| E8 | Implemented (artifact-driven) | Multi-run randomization benchmark runner with pass-rate summaries and CIs. |
| E9 | Not implemented | Spurious-cue stress-test pipeline missing (optional extension). |

## Implemented now (core-relevant)
- `src/eval/faithfulness.py`: sanity, deletion/insertion, nuisance robustness, NFI.
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
- `src/explain/e4_concept_generate.py`: E4 concept-family artifact generator.
- `src/eval/e8_randomization.py`: multi-run sanity randomization benchmark summaries with observed-run and bootstrap-proxy modes.
- `src/eval/assemble_family_artifacts.py`: strict assembly/validation for unified E7/E8 input.
- `src/data/synthetic_smoke.py`: synthetic image/manifest/artifact generation for data-free integration checks.
- `src/dashboard/io.py` + `dashboard/app.py`: artifact-driven monitoring dashboard aligned to core scope.
- `scripts/run_e4_concept.sh`: E4 concept artifact runner.
- `scripts/run_assemble_family_artifacts.sh`: artifact assembly runner for unified E7/E8 input.
- `scripts/run_synthetic_smoke.sh`: synthetic integration smoke pipeline (default core path skips E5/E6).
- `scripts/run_after_data.sh`: real-data pipeline runner (defaults to E5/E6 disabled).

## Optional future-work components (kept in repo)
- `src/explain/e5_text_generate.py`
- `src/explain/e6_text_generate.py`
- `scripts/run_e5_text_constrained.sh`
- `scripts/run_e6_text_unconstrained.sh`
- `src/eval/text_grounding.py`
- `docs/explanation_artifact_schemas.md`

## Immediate next coding priorities
1. Replace E4 proxy generator with model-backed concept head + explicit intervention runner.
2. Add light integration tests for `run_e1_train_cnn.sh` + `run_e2_e3_generate.sh` with toy images/checkpoints.
3. Add E9 stress-test data builder for spurious-cue sensitivity (optional extension).
4. Add optional external validation wiring after internal data runs stabilize.
5. Revisit text-family E5/E6 as a separate follow-on project.
