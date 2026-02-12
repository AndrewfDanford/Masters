# Label Mapping (Draft)

## Primary labels
- Atelectasis
- Cardiomegaly
- Consolidation
- Edema
- Pleural Effusion
- Pneumothorax

## Mapping notes
- Start with CheXpert-style labels from MIMIC-CXR metadata.
- Keep mapping rules explicit and versioned.
- Define handling for uncertain labels in a single global policy.

## Pending decisions
- Uncertain label policy (`U -> 0`, `U -> 1`, or exclude).
- Class weighting vs resampling strategy.
- Any additional labels for robustness analysis.
