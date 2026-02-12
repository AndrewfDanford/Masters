# Explanation Artifact Schemas (E4-E6)

This document freezes the artifact table contracts for concept and text families before full model implementation.

## Core contract for E7 compatibility

Every family must output a CSV containing at least:

- `method`
- `study_id`
- `sanity_similarity`
- `deletion_curve`
- `insertion_curve`
- `base_prediction`
- `perturbed_prediction`
- `nuisance_similarity`

For text methods (`E5`, `E6`), also include:

- `text_contradiction`

## Curves format

- `deletion_curve` and `insertion_curve` are comma-separated numeric strings.
- Curve values must be normalized to `[0, 1]`.
- Curves should be generated on the same fraction grid per run (for fair method comparison).

## Family-specific recommended fields

### E4 (Concept / CBM)

- Required by schema: core contract fields.
- Recommended extras:
  - `concept_alignment`
  - `concept_intervention_effect`

Template and config:

- `configs/explain/e4_concept_artifact_template.csv`
- `configs/explain/e4_concept_pipeline.json`

### E5 (Constrained text rationale)

- Required by schema: core contract fields + `text_contradiction`.
- Recommended extras:
  - `concept_alignment`
  - `finding_consistency`

Template and config:

- `configs/explain/e5_text_constrained_artifact_template.csv`
- `configs/explain/e5_text_constrained_pipeline.json`

### E6 (Less-constrained text rationale)

- Required by schema: core contract fields + `text_contradiction`.
- Recommended extras:
  - `concept_alignment`
  - `finding_consistency`

Template and config:

- `configs/explain/e6_text_unconstrained_artifact_template.csv`
- `configs/explain/e6_text_unconstrained_pipeline.json`

## Method names (canonical)

Use these `method` values to avoid mismatches downstream:

- `concept_cbm` (E4)
- `text_constrained` (E5)
- `text_unconstrained` (E6)
