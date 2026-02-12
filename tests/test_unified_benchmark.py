import pandas as pd

from src.eval.quality_gates import FaithfulnessThresholds
from src.eval.unified_benchmark import run_unified_benchmark


def test_run_unified_benchmark_ranks_higher_quality_method_first() -> None:
    frame = pd.DataFrame(
        [
            {
                "method": "hirescam",
                "study_id": sid,
                "sanity_similarity": 0.10,
                "deletion_curve": "1.0,0.6,0.35,0.22,0.14",
                "insertion_curve": "0.18,0.42,0.66,0.83,0.94",
                "base_prediction": 0.86,
                "perturbed_prediction": 0.84,
                "nuisance_similarity": 0.82,
            }
            for sid in [101, 102, 103, 104]
        ]
        + [
            {
                "method": "gradcam",
                "study_id": sid,
                "sanity_similarity": 0.35,
                "deletion_curve": "1.0,0.84,0.70,0.58,0.49",
                "insertion_curve": "0.21,0.32,0.43,0.51,0.62",
                "base_prediction": 0.86,
                "perturbed_prediction": 0.79,
                "nuisance_similarity": 0.58,
            }
            for sid in [101, 102, 103, 104]
        ]
    )

    thresholds = FaithfulnessThresholds(
        sanity_min=0.4,
        perturbation_min=0.45,
        robustness_min=0.5,
        nfi_min=0.5,
    )
    outputs = run_unified_benchmark(
        frame=frame,
        thresholds=thresholds,
        num_bootstrap=300,
        seed=11,
    )

    top_method = outputs.method_summary.iloc[0]["method"]
    assert top_method == "hirescam"
    assert not outputs.pairwise_nfi_deltas.empty

