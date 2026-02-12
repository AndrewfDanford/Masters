# Cohort Definition (Draft)

## Inclusion
- Chest X-ray studies from MIMIC-CXR-JPG.
- Frontal views only.
- Valid patient ID and study ID metadata.

## Exclusion
- Missing or unreadable image files.
- Studies without required label fields.
- Non-frontal views.

## Initial target findings
- Atelectasis
- Cardiomegaly
- Consolidation
- Edema
- Pleural Effusion
- Pneumothorax

## Split policy
- Patient-level split: train/validation/test.
- No patient appears in more than one split.

## Pending decisions
- Exact prevalence thresholds for finding inclusion.
- Whether to keep uncertain labels or map to negative/excluded.
