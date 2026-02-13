# Minimal Repository Structure (for implementation phase)

```
/Users/andrew/Documents/New project/
  Dockerfile
  docker-compose.yml
  dashboard/
    app.py
  docs/
    design_note.md
    experiment_matrix.md
    repo_structure.md
    handoff_quickstart.md
    explanation_artifact_schemas.md
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
    dashboard/
    models/
    explain/
    eval/
    utils/
  scripts/
    setup_env.sh
    run_after_data.sh
    run_e0_data_audit.sh
    run_e1_extract_features.sh
    run_e1_baseline.sh
    run_e1_from_images.sh
    run_e1_train_cnn.sh
    run_e2_e3_generate.sh
    run_e2_e3_from_model.sh
    run_e2_e3_saliency.sh
    run_e4_concept.sh
    run_e5_text_constrained.sh
    run_e6_text_unconstrained.sh
    run_assemble_family_artifacts.sh
    run_e8_randomization.sh
    run_dashboard.sh
    run_synthetic_smoke.sh
    run_e7_unified.sh
    run_small_subset_pipeline.sh
  tests/
    test_faithfulness.py
    test_e1_train_cnn.py
    test_e2_e3_generate.py
    test_family_artifacts.py
    test_assemble_family_artifacts.py
    test_e8_randomization.py
    test_dashboard_io.py
    test_synthetic_smoke.py
    test_unified_benchmark.py
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
