# Design Note: Unified Faithfulness Benchmark for CXR Explanations

## Problem framing
Current radiology XAI work often reports explanation quality within one method family (for example, saliency only), with heterogeneous datasets and metrics. This prevents fair cross-family comparison between saliency, concept-based, and text-based explanations.

## Thesis objective
Build a unified chest X-ray (CXR) benchmark that compares explanation families under the same:
- prediction task,
- perturbation protocol,
- faithfulness criteria,
- and reporting standard.

## Hypothesis
Under a unified faithfulness protocol, explanation methods with explicit concept grounding will show stronger faithfulness than purely post-hoc explanations at similar diagnostic performance.

## Scope
- Modality: CXR only.
- Primary data stack: MIMIC-CXR + MS-CXR + RadGraph.
- Explanation families: saliency, concept, text.
- Text strategy: constrained and less-constrained variants.
- Non-goal: direct bedside deployment claims.

## Data plan
- Use patient-level splits to avoid leakage.
- Restrict to frontal views.
- Start with 6 findings for robust localization and prevalence:
  - Atelectasis
  - Cardiomegaly
  - Consolidation
  - Edema
  - Pleural Effusion
  - Pneumothorax

## Model and explanation families
1. Saliency family
- Baseline classifier with Grad-CAM and HiResCAM explanations.

2. Concept family
- Concept-head or concept-bottleneck style model with report-derived concepts from RadGraph.

3. Text family
- Constrained rationale generator (primary text track).
- Less-constrained short rationale generator (comparison track).

## Unified faithfulness protocol
Apply the same stress tests across families:
1. Randomization sanity checks.
2. Deletion and insertion perturbation tests.
3. Robustness tests under nuisance perturbations where predictions should remain stable.

Core requirement: explanation behavior should track model behavior under controlled interventions.

## Evaluation metrics
- Diagnostic: AUROC, AUPRC, calibration (ECE/Brier).
- Saliency: localization overlap/pointing and perturbation curves.
- Concept: concept prediction metrics and intervention effect.
- Text: factual consistency with structured findings, contradiction rate, perturbation-linked faithfulness.
- Cross-family: paired comparisons with confidence intervals.

## Expected failure modes
- Explanations appear plausible but are weakly linked to model internals.
- Text rationales become generic and non-discriminative.
- Concept vocabulary incompleteness limits intervention validity.
- Dataset artifacts induce shortcut learning.

## What this work will not attempt
- It will not claim causal medical mechanism discovery.
- It will not require radiologist reader studies in the master’s phase.
- It will not prioritize long free-form report generation as the primary target.

## Two-semester execution
- Semester 1: data curation, baseline model, saliency benchmark, first faithfulness report.
- Semester 2: concept + text tracks, unified comparison, ablations, thesis write-up.
