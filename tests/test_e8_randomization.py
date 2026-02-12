import pandas as pd

from src.eval.e8_randomization import run_randomization_benchmark


def test_run_randomization_benchmark_with_observed_runs() -> None:
    frame = pd.DataFrame(
        [
            {"method": "gradcam", "study_id": 1, "run_id": 1, "sanity_similarity": 0.40},
            {"method": "gradcam", "study_id": 2, "run_id": 1, "sanity_similarity": 0.35},
            {"method": "gradcam", "study_id": 1, "run_id": 2, "sanity_similarity": 0.45},
            {"method": "gradcam", "study_id": 2, "run_id": 2, "sanity_similarity": 0.30},
            {"method": "hirescam", "study_id": 1, "run_id": 1, "sanity_similarity": 0.20},
            {"method": "hirescam", "study_id": 2, "run_id": 1, "sanity_similarity": 0.25},
            {"method": "hirescam", "study_id": 1, "run_id": 2, "sanity_similarity": 0.22},
            {"method": "hirescam", "study_id": 2, "run_id": 2, "sanity_similarity": 0.28},
        ]
    )

    run_scores, method_summary, sample_variability, meta = run_randomization_benchmark(
        input_frame=frame,
        min_sanity_score=0.6,
        alpha=0.1,
    )

    assert meta["mode"] == "observed_runs"
    assert run_scores.shape[0] == 4
    assert set(method_summary["method"]) == {"gradcam", "hirescam"}
    assert not sample_variability.empty


def test_run_randomization_benchmark_with_bootstrap_proxy() -> None:
    frame = pd.DataFrame(
        [
            {"method": "gradcam", "study_id": 1, "sanity_similarity": 0.40},
            {"method": "gradcam", "study_id": 2, "sanity_similarity": 0.35},
            {"method": "gradcam", "study_id": 3, "sanity_similarity": 0.30},
            {"method": "hirescam", "study_id": 1, "sanity_similarity": 0.20},
            {"method": "hirescam", "study_id": 2, "sanity_similarity": 0.25},
            {"method": "hirescam", "study_id": 3, "sanity_similarity": 0.30},
        ]
    )

    run_scores, method_summary, sample_variability, meta = run_randomization_benchmark(
        input_frame=frame,
        min_sanity_score=0.5,
        num_bootstrap_runs=25,
        seed=13,
    )

    assert meta["mode"] == "bootstrap_proxy"
    assert run_scores.shape[0] == 50
    assert set(method_summary["num_runs"]) == {25}
    assert sample_variability.empty
