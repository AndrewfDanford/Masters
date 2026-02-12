"""Evaluation utilities for unified faithfulness benchmarking."""

from .faithfulness import (
    StabilityResult,
    deletion_insertion_score,
    normalized_faithfulness_index,
    nuisance_robustness_score,
    prediction_stability_mask,
    saliency_similarity,
    sanity_score_from_similarities,
    topk_overlap_similarity,
)
from .quality_gates import (
    FaithfulnessScores,
    FaithfulnessThresholds,
    evaluate_quality_gates,
    load_thresholds,
)
from .unified_benchmark import (
    BenchmarkOutputs,
    build_sample_scores,
    pairwise_nfi_deltas,
    run_unified_benchmark,
    summarize_methods,
)
