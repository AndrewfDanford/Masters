from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .faithfulness import sanity_score_from_similarities


def _clip01(values: np.ndarray) -> np.ndarray:
    return np.clip(values.astype(float), 0.0, 1.0)


def _require_columns(frame: pd.DataFrame, columns: list[str], frame_name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{frame_name} missing required columns: {', '.join(missing)}")


def _quantile_ci(values: np.ndarray, alpha: float) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    return (
        float(np.quantile(values, alpha / 2.0)),
        float(np.quantile(values, 1.0 - alpha / 2.0)),
    )


def build_run_scores_from_long(
    frame: pd.DataFrame,
    method_col: str,
    sample_id_col: str,
    run_col: str,
    sanity_similarity_col: str,
    min_sanity_score: float,
) -> pd.DataFrame:
    _require_columns(frame, [method_col, sample_id_col, run_col, sanity_similarity_col], "input frame")

    data = frame.copy()
    data[sanity_similarity_col] = _clip01(pd.to_numeric(data[sanity_similarity_col], errors="coerce").to_numpy())
    if data[sanity_similarity_col].isna().any():
        raise ValueError("input frame contains NaN sanity similarities")

    rows: list[dict[str, object]] = []
    grouped = data.groupby([method_col, run_col], dropna=False)
    for (method_name, run_id), group in grouped:
        similarities = group[sanity_similarity_col].to_numpy(dtype=float)
        sanity_score = float(sanity_score_from_similarities(similarities))
        mean_similarity = float(np.mean(similarities))
        rows.append(
            {
                "method": method_name,
                "run_id": str(run_id),
                "num_samples": int(group[sample_id_col].nunique(dropna=True)),
                "mean_sanity_similarity": mean_similarity,
                "sanity_score": sanity_score,
                "sanity_pass": bool(sanity_score >= min_sanity_score),
            }
        )

    return pd.DataFrame(rows).sort_values(by=["method", "run_id"]).reset_index(drop=True)


def build_bootstrap_proxy_run_scores(
    frame: pd.DataFrame,
    method_col: str,
    sample_id_col: str,
    sanity_similarity_col: str,
    min_sanity_score: float,
    num_bootstrap_runs: int,
    seed: int,
) -> pd.DataFrame:
    if num_bootstrap_runs <= 0:
        raise ValueError("num_bootstrap_runs must be positive")

    _require_columns(frame, [method_col, sample_id_col, sanity_similarity_col], "input frame")

    data = frame.copy()
    data[sanity_similarity_col] = _clip01(pd.to_numeric(data[sanity_similarity_col], errors="coerce").to_numpy())
    if data[sanity_similarity_col].isna().any():
        raise ValueError("input frame contains NaN sanity similarities")

    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []

    for method_name, group in data.groupby(method_col):
        similarities = group[sanity_similarity_col].to_numpy(dtype=float)
        sample_count = int(group[sample_id_col].nunique(dropna=True))
        n = similarities.size
        for run_index in range(num_bootstrap_runs):
            sampled = similarities[rng.integers(0, n, size=n)]
            sanity_score = float(sanity_score_from_similarities(sampled))
            mean_similarity = float(np.mean(sampled))
            rows.append(
                {
                    "method": method_name,
                    "run_id": f"bootstrap_{run_index + 1}",
                    "num_samples": sample_count,
                    "mean_sanity_similarity": mean_similarity,
                    "sanity_score": sanity_score,
                    "sanity_pass": bool(sanity_score >= min_sanity_score),
                }
            )

    return pd.DataFrame(rows).sort_values(by=["method", "run_id"]).reset_index(drop=True)


def summarize_run_scores(
    run_scores: pd.DataFrame,
    min_sanity_score: float,
    alpha: float,
) -> pd.DataFrame:
    if run_scores.empty:
        raise ValueError("run_scores cannot be empty")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be between 0 and 1")

    rows: list[dict[str, object]] = []
    for method_name, group in run_scores.groupby("method"):
        sanity_scores = group["sanity_score"].to_numpy(dtype=float)
        mean_similarities = group["mean_sanity_similarity"].to_numpy(dtype=float)
        score_ci_low, score_ci_high = _quantile_ci(sanity_scores, alpha=alpha)
        pass_rate = float(np.mean(group["sanity_pass"].astype(float).to_numpy()))

        rows.append(
            {
                "method": method_name,
                "num_runs": int(group.shape[0]),
                "min_sanity_score_threshold": float(min_sanity_score),
                "pass_rate": pass_rate,
                "num_pass_runs": int(group["sanity_pass"].sum()),
                "mean_sanity_score": float(np.mean(sanity_scores)),
                "sanity_score_ci_low": score_ci_low,
                "sanity_score_ci_high": score_ci_high,
                "std_sanity_score": float(np.std(sanity_scores)),
                "mean_sanity_similarity": float(np.mean(mean_similarities)),
                "std_sanity_similarity": float(np.std(mean_similarities)),
            }
        )

    return pd.DataFrame(rows).sort_values(by="method").reset_index(drop=True)


def summarize_sample_variability_from_long(
    frame: pd.DataFrame,
    method_col: str,
    sample_id_col: str,
    run_col: str,
    sanity_similarity_col: str,
) -> pd.DataFrame:
    _require_columns(frame, [method_col, sample_id_col, run_col, sanity_similarity_col], "input frame")

    data = frame.copy()
    data[sanity_similarity_col] = _clip01(pd.to_numeric(data[sanity_similarity_col], errors="coerce").to_numpy())
    if data[sanity_similarity_col].isna().any():
        raise ValueError("input frame contains NaN sanity similarities")

    rows: list[dict[str, object]] = []
    grouped = data.groupby([method_col, sample_id_col], dropna=False)
    for (method_name, sample_id), group in grouped:
        sims = group[sanity_similarity_col].to_numpy(dtype=float)
        rows.append(
            {
                "method": method_name,
                "sample_id": sample_id,
                "num_runs": int(group[run_col].nunique(dropna=True)),
                "mean_sanity_similarity": float(np.mean(sims)),
                "std_sanity_similarity": float(np.std(sims)),
                "min_sanity_similarity": float(np.min(sims)),
                "max_sanity_similarity": float(np.max(sims)),
                "mean_sanity_score": float(sanity_score_from_similarities(sims)),
            }
        )

    return pd.DataFrame(rows).sort_values(by=["method", "sample_id"]).reset_index(drop=True)


def run_randomization_benchmark(
    input_frame: pd.DataFrame,
    method_col: str = "method",
    sample_id_col: str = "study_id",
    run_col: str = "run_id",
    sanity_similarity_col: str = "sanity_similarity",
    min_sanity_score: float = 0.5,
    alpha: float = 0.05,
    num_bootstrap_runs: int = 200,
    seed: int = 17,
    allow_bootstrap_proxy: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    if not (0.0 <= min_sanity_score <= 1.0):
        raise ValueError("min_sanity_score must be between 0 and 1")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be between 0 and 1")

    _require_columns(input_frame, [method_col, sample_id_col, sanity_similarity_col], "input frame")
    run_column_present = run_col in input_frame.columns and input_frame[run_col].notna().any()

    if run_column_present:
        mode = "observed_runs"
        run_scores = build_run_scores_from_long(
            frame=input_frame,
            method_col=method_col,
            sample_id_col=sample_id_col,
            run_col=run_col,
            sanity_similarity_col=sanity_similarity_col,
            min_sanity_score=min_sanity_score,
        )
        sample_variability = summarize_sample_variability_from_long(
            frame=input_frame,
            method_col=method_col,
            sample_id_col=sample_id_col,
            run_col=run_col,
            sanity_similarity_col=sanity_similarity_col,
        )
    elif allow_bootstrap_proxy:
        mode = "bootstrap_proxy"
        run_scores = build_bootstrap_proxy_run_scores(
            frame=input_frame,
            method_col=method_col,
            sample_id_col=sample_id_col,
            sanity_similarity_col=sanity_similarity_col,
            min_sanity_score=min_sanity_score,
            num_bootstrap_runs=num_bootstrap_runs,
            seed=seed,
        )
        sample_variability = pd.DataFrame(
            columns=[
                "method",
                "sample_id",
                "num_runs",
                "mean_sanity_similarity",
                "std_sanity_similarity",
                "min_sanity_similarity",
                "max_sanity_similarity",
                "mean_sanity_score",
            ]
        )
    else:
        raise ValueError(
            "run_col is missing or empty. Provide observed randomization runs or enable bootstrap proxy mode."
        )

    method_summary = summarize_run_scores(
        run_scores=run_scores,
        min_sanity_score=min_sanity_score,
        alpha=alpha,
    )

    meta = {
        "mode": mode,
        "method_col": method_col,
        "sample_id_col": sample_id_col,
        "run_col": run_col,
        "sanity_similarity_col": sanity_similarity_col,
        "min_sanity_score": float(min_sanity_score),
        "alpha": float(alpha),
        "num_bootstrap_runs": int(num_bootstrap_runs),
        "seed": int(seed),
        "num_input_rows": int(input_frame.shape[0]),
        "num_methods": int(method_summary.shape[0]),
    }
    return run_scores, method_summary, sample_variability, meta


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run E8 randomization sanity benchmark from artifact tables.")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", default="e8_randomization")
    parser.add_argument("--method-col", default="method")
    parser.add_argument("--sample-id-col", default="study_id")
    parser.add_argument("--run-col", default="run_id")
    parser.add_argument("--sanity-similarity-col", default="sanity_similarity")
    parser.add_argument("--min-sanity-score", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--num-bootstrap-runs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--disable-bootstrap-proxy", action="store_true")
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    frame = pd.read_csv(args.input_csv)

    run_scores, method_summary, sample_variability, meta = run_randomization_benchmark(
        input_frame=frame,
        method_col=args.method_col,
        sample_id_col=args.sample_id_col,
        run_col=args.run_col,
        sanity_similarity_col=args.sanity_similarity_col,
        min_sanity_score=args.min_sanity_score,
        alpha=args.alpha,
        num_bootstrap_runs=args.num_bootstrap_runs,
        seed=args.seed,
        allow_bootstrap_proxy=not args.disable_bootstrap_proxy,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_scores.to_csv(args.output_dir / f"{args.output_prefix}_run_scores.csv", index=False)
    method_summary.to_csv(args.output_dir / f"{args.output_prefix}_method_summary.csv", index=False)
    sample_variability.to_csv(args.output_dir / f"{args.output_prefix}_sample_variability.csv", index=False)
    with (args.output_dir / f"{args.output_prefix}_meta.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)


if __name__ == "__main__":
    main()
