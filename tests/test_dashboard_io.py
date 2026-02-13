from pathlib import Path

from src.dashboard.io import collect_artifact_paths, infer_stage_status, parse_curve_string


def test_parse_curve_string_handles_common_formats() -> None:
    values_a = parse_curve_string("1.0,0.7,0.3")
    values_b = parse_curve_string("[1.0; 0.7; 0.3]")

    assert values_a.tolist() == [1.0, 0.7, 0.3]
    assert values_b.tolist() == [1.0, 0.7, 0.3]


def test_collect_artifact_paths_picks_latest_e7_files(tmp_path: Path) -> None:
    e7_dir = tmp_path / "outputs" / "reports" / "e7"
    e7_dir.mkdir(parents=True, exist_ok=True)

    older = e7_dir / "aaa_method_summary.csv"
    newer = e7_dir / "zzz_method_summary.csv"
    older.write_text("method,nfi\nm1,0.5\n", encoding="utf-8")
    newer.write_text("method,nfi\nm1,0.6\n", encoding="utf-8")

    paths = collect_artifact_paths(tmp_path)
    assert paths["e7_method_summary"] is not None
    assert paths["e7_method_summary"].name == "zzz_method_summary.csv"


def test_collect_artifact_paths_falls_back_to_smoke_outputs(tmp_path: Path) -> None:
    smoke_e1 = tmp_path / "outputs" / "smoke" / "e1"
    smoke_e1.mkdir(parents=True, exist_ok=True)
    (smoke_e1 / "e1_metrics_summary.csv").write_text("split,macro_auroc\ntest,0.7\n", encoding="utf-8")

    paths = collect_artifact_paths(tmp_path)
    assert paths["e1_metrics_summary"] is not None
    assert "outputs/smoke/e1/e1_metrics_summary.csv" in str(paths["e1_metrics_summary"])


def test_infer_stage_status_marks_done_when_required_files_exist(tmp_path: Path) -> None:
    e0_dir = tmp_path / "outputs" / "reports"
    e0_dir.mkdir(parents=True, exist_ok=True)
    (e0_dir / "e0_cohort_manifest.csv").write_text("study_id,split\n1,train\n", encoding="utf-8")
    (e0_dir / "e0_split_counts.csv").write_text("split,num_studies\ntrain,1\n", encoding="utf-8")
    (e0_dir / "e0_label_prevalence.csv").write_text("split,finding,positive,known\ntrain,Edema,1,1\n", encoding="utf-8")

    paths = collect_artifact_paths(tmp_path)
    stage_df = infer_stage_status(paths)
    e0_row = stage_df[stage_df["stage"] == "E0 Audit"].iloc[0]

    assert e0_row["status"] == "done"
    assert int(e0_row["completed_artifacts"]) == 3


def test_infer_stage_status_includes_e4_e5_e6_stages(tmp_path: Path) -> None:
    reports = tmp_path / "outputs" / "reports"
    (reports / "e4").mkdir(parents=True, exist_ok=True)
    (reports / "e5").mkdir(parents=True, exist_ok=True)
    (reports / "e6").mkdir(parents=True, exist_ok=True)
    (reports / "e4" / "e4_artifacts.csv").write_text("method,study_id,sanity_similarity,deletion_curve,insertion_curve,base_prediction,perturbed_prediction,nuisance_similarity\nconcept_cbm,1,0.2,\"0.9,0.7,0.5\",\"0.2,0.4,0.6\",0.8,0.75,0.7\n", encoding="utf-8")
    (reports / "e5" / "e5_artifacts.csv").write_text("method,study_id,sanity_similarity,deletion_curve,insertion_curve,base_prediction,perturbed_prediction,nuisance_similarity,text_contradiction\ntext_constrained,1,0.3,\"0.9,0.7,0.5\",\"0.2,0.4,0.6\",0.8,0.75,0.7,0.1\n", encoding="utf-8")
    (reports / "e6" / "e6_artifacts.csv").write_text("method,study_id,sanity_similarity,deletion_curve,insertion_curve,base_prediction,perturbed_prediction,nuisance_similarity,text_contradiction\ntext_unconstrained,1,0.5,\"0.9,0.7,0.5\",\"0.2,0.4,0.6\",0.8,0.75,0.7,0.2\n", encoding="utf-8")

    stage_df = infer_stage_status(collect_artifact_paths(tmp_path))
    assert "E4 Concept" in stage_df["stage"].tolist()
    assert "E5 Text Constrained" in stage_df["stage"].tolist()
    assert "E6 Text Unconstrained" in stage_df["stage"].tolist()
