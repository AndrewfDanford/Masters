# Code Readiness vs Thesis Plan

## Summary
The repository is close to "data-only blocked" for E0/E1/E2-E3/E7:
- E0 (data audit) is implemented.
- E1 baseline training is implemented for precomputed features.
- Unified faithfulness evaluation is implemented and runnable from artifact tables.
- Explanation generation stages (producing Grad-CAM/HiResCAM, concept, and text artifacts) are still pending.

## Status by experiment ID

| Experiment | Status | Notes |
|---|---|---|
| E0 | Implemented | Cohort manifest, split audit, prevalence, concept coverage. |
| E1 | Implemented | Baseline multi-label linear probe on precomputed features + metrics export. |
| E2-E3 | Partially implemented | Saliency evaluation runner is implemented; saliency extraction generation is pending. |
| E4 | Not implemented | Concept-head/CBM training and intervention runner missing. |
| E5-E6 | Not implemented | Constrained/less-constrained rationale generators missing. |
| E7 | Implemented | Unified benchmark runner, CIs, quality gates, and pairwise method deltas. |
| E8 | Partially implemented | Sanity scoring utilities available; randomized model loop missing. |
| E9 | Not implemented | Spurious-cue stress-test pipeline missing. |

## Implemented now (to match Sections 5-6)
- `src/eval/faithfulness.py`: sanity, deletion/insertion, nuisance robustness, NFI.
- `src/eval/text_grounding.py`: concept alignment, finding consistency, contradiction rate.
- `src/eval/statistics.py`: bootstrap CIs and paired bootstrap differences.
- `src/eval/quality_gates.py`: predefined threshold gates and pass/fail logic.
- `src/eval/unified_benchmark.py`: artifact ingestion, sample/method scoring, pairwise deltas.
- `src/models/e1_baseline.py`: baseline training/evaluation on precomputed feature tables.
- `configs/eval/faithfulness_thresholds.json`: default thresholds.
- `configs/eval/e7_input_template.csv`: unified artifact schema example.
- `configs/explain/e2e3_saliency_input_template.csv`: E2/E3 artifact schema example.
- Tests for all the above modules under `tests/`.

## Immediate next coding priorities
1. Add feature extraction from image backbones to feed E1 directly from CXR images.
2. Add Grad-CAM and HiResCAM generation pipeline to populate E2/E3 artifact tables.
3. Add concept intervention runner for `E4`.
4. Add text rationale + grounding extractor for `E5-E6`.
5. Add E8 randomized sanity-run driver and E9 stress-test data builder.
