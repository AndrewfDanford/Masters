# Minimal Repository Structure (for implementation phase)

```
/Users/andrew/Documents/New project/
  docs/
    design_note.md
    experiment_matrix.md
    repo_structure.md
  configs/
    data/
    model/
    explain/
    eval/
  data_specs/
    cohort_definition.md
    label_mapping.md
    concept_schema.md
  src/
    data/
    models/
    explain/
    eval/
    utils/
  scripts/
    run_e0_data_audit.sh
    run_e1_baseline.sh
    run_e2_e3_saliency.sh
    run_e7_unified.sh
  tests/
    test_explain_metrics.py
    test_concept_interventions.py
    test_text_faithfulness_checks.py
  outputs/
    reports/
    figures/
```

## Layout rationale
- `data_specs/` stores frozen protocol decisions to avoid hidden drift.
- `configs/` makes experiments reproducible and comparable.
- `src/explain/` and `src/eval/` are separated to avoid coupling explanation generation with evaluation logic.
- `tests/` focuses on explanation metrics and intervention logic, not only model training.
- `outputs/reports/` is the canonical target for intermediate benchmark summaries.
