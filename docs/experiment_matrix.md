# Experiment Matrix

## Locked decisions
- Primary gap: unified faithfulness benchmark (incremental).
- Secondary axis: safety-oriented proxy utility without mandatory radiologist participation.
- Dataset stack: MIMIC-CXR + MS-CXR + RadGraph.
- Findings subset (initial): Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion, Pneumothorax.
- Explanation families: saliency, concept, text.
- Text comparison: constrained (primary) vs less-constrained (baseline).

## Core experiment grid

| ID | Model / Explanation | Data Split | Primary Outputs | Key Metrics | Purpose |
|---|---|---|---|---|---|
| E0 | Data audit pipeline | Train/val/test candidate splits | Prevalence, label quality, concept coverage | Label prevalence, missingness, concept frequency | Finalize reproducible cohort definition |
| E1 | Baseline CXR classifier (no explanation training objective) | Patient-level split | Multi-label predictions | AUROC, AUPRC, ECE/Brier | Establish diagnostic anchor |
| E2 | E1 + Grad-CAM | MS-CXR test subset | Heatmaps | Pointing/localization + deletion/insertion curves | Saliency baseline 1 |
| E3 | E1 + HiResCAM | MS-CXR test subset | Heatmaps | Same as E2 + robustness under nuisance perturbations | Saliency baseline 2 |
| E4 | Concept-head / CBM-style model | Same split as E1 | Label predictions + concept activations | Label metrics + concept F1 + concept intervention effect | Concept-family benchmark |
| E5 | Constrained text rationale model | Same split as E1 | Short grounded rationale | Factual consistency, contradiction rate, perturbation-linked faithfulness | Primary text track |
| E6 | Less-constrained text rationale model | Same split as E1 | Short rationale | Same as E5 | Text comparator |
| E7 | Unified perturbation benchmark across E2-E6 | Standardized perturbation suite | Method-wise faithfulness curves | Cross-family faithfulness score + paired CI tests | Main thesis claim |
| E8 | Randomization sanity checks for E2-E6 | Controlled randomization runs | Sensitivity signatures | Correlation drop / sanity-test pass rates | Validate non-spurious explanation behavior |
| E9 | Spurious-cue stress test | Curated confound-heavy subset | Error + explanation patterns | Error detection proxy, inconsistency flags | Secondary Gap 4 proxy without clinicians |

## Required ablations

| ID | Ablation | Compared Systems | Why it matters |
|---|---|---|---|
| A1 | Concept supervision fraction (25/50/100%) | E4, E5 | Tests dependence on concept label density |
| A2 | Constraint strength in text generation | E5 vs E6 + intermediate settings | Quantifies faithfulness-fluency tradeoff |
| A3 | Evidence granularity (coarse vs fine perturbation masks) | E2-E6 | Tests robustness of faithfulness conclusions |
| A4 | Backbone sensitivity | E1 with one alternate backbone | Ensures findings are not architecture-specific |

## Reporting standard
- Report all performance and faithfulness metrics with bootstrap confidence intervals.
- Separate in-domain test and stress-test results.
- Use identical cohort IDs across methods for paired comparisons.
- Predefine primary endpoint: cross-family faithfulness score from E7.

## External validation decision
- Keep external validation as a stretch goal for semester 2.
- If included, use a mapped CXR external set and report as supplemental only, not primary claim support.

## Exit criteria for thesis-level completion
1. All core experiments E1-E9 executed with reproducible configs.
2. At least two explanation families pass sanity checks while showing measurable differences in faithfulness.
3. Primary claim supported by statistically stable cross-family comparison from E7.
