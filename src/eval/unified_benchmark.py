from __future__ import annotations

import argparse
import itertools
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .faithfulness import (
    deletion_insertion_score,
    normalized_faithfulness_index,
    nuisance_robustness_score,
    prediction_stability_mask,
)
from .quality_gates import FaithfulnessScores, FaithfulnessThresholds, evaluate_quality_gates, load_thresholds
from .statistics import paired_bootstrap_mean_diff_ci

_CURVE_SPLIT_RE = re.compile(r"[,;|\s]+")


def _clip01(values: np.ndarray) -> np.ndarray:
    return np.clip(values.astype(float), 0.0, 1.0)


def _to_numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        raise ValueError(f"missing required column: {column}")
    return pd.to_numeric(frame[column], errors="coerce")


def _parse_curve(value: object) -> np.ndarray:
    if isinstance(value, np.ndarray):
        curve = value.astype(float).reshape(-1)
    elif isinstance(value, (list, tuple)):
        curve = np.asarray(value, dtype=float).reshape(-1)
    elif isinstance(value, str):
        stripped = value.strip().strip("[]")
        tokens = [token for token in _CURVE_SPLIT_RE.split(stripped) if token]
        curve = np.asarray([float(token) for token in tokens], dtype=float)
    else:
        raise ValueError(f"unsupported curve value type: {type(value)}")

    if curve.size == 0:
        raise ValueError("curve cannot be empty")
    return curve


@dataclass(frozen=True)
class BenchmarkOutputs:
    sample_scores: pd.DataFrame
    method_summary: pd.DataFrame
    pairwise_nfi_deltas: pd.DataFrame


def build_sample_scores(
    frame: pd.DataFrame,
    method_col: str = "method",
    sample_id_col: str = "study_id",
    base_pred_col: str = "base_prediction",
    perturbed_pred_col: str = "perturbed_prediction",
    nuisance_similarity_col: str = "nuisance_similarity",
    sanity_score_col: str | None = "sanity_score",
    sanity_similarity_col: str | None = "sanity_similarity",
    perturbation_score_col: str | None = "perturbation_score",
    deletion_curve_col: str | None = "deletion_curve",
    insertion_curve_col: str | None = "insertion_curve",
    text_contradiction_col: str | None = "text_contradiction",
    stability_abs_tolerance: float = 0.05,
    weights: tuple[float, float, float] | None = None,
) -> pd.DataFrame:
    for required_column in [method_col, sample_id_col, base_pred_col, perturbed_pred_col, nuisance_similarity_col]:
        if required_column not in frame.columns:
            raise ValueError(f"missing required column: {required_column}")

    output = frame.copy()

    if sanity_score_col and sanity_score_col in output.columns:
        output["sanity_score"] = _clip01(pd.to_numeric(output[sanity_score_col], errors="coerce").to_numpy())
    elif sanity_similarity_col and sanity_similarity_col in output.columns:
        similarities = _clip01(pd.to_numeric(output[sanity_similarity_col], errors="coerce").to_numpy())
        output["sanity_score"] = 1.0 - similarities
    else:
        raise ValueError("expected either sanity_score or sanity_similarity column")

    if perturbation_score_col and perturbation_score_col in output.columns:
        output["perturbation_score"] = _clip01(
            pd.to_numeric(output[perturbation_score_col], errors="coerce").to_numpy()
        )
    elif deletion_curve_col and insertion_curve_col and deletion_curve_col in output.columns and insertion_curve_col in output.columns:
        perturbation_values: list[float] = []
        for _, row in output.iterrows():
            deletion_curve = _parse_curve(row[deletion_curve_col])
            insertion_curve = _parse_curve(row[insertion_curve_col])
            perturbation_values.append(deletion_insertion_score(deletion_curve, insertion_curve))
        output["perturbation_score"] = perturbation_values
    else:
        raise ValueError("expected perturbation_score column or deletion/insertion curves")

    base_predictions = _to_numeric_series(output, base_pred_col).to_numpy()
    perturbed_predictions = _to_numeric_series(output, perturbed_pred_col).to_numpy()
    output["is_stable"] = prediction_stability_mask(
        base_predictions=base_predictions,
        perturbed_predictions=perturbed_predictions,
        abs_tolerance=stability_abs_tolerance,
    )

    output["nuisance_similarity"] = _clip01(_to_numeric_series(output, nuisance_similarity_col).to_numpy())
    output["sample_nfi"] = np.array(
        [
            normalized_faithfulness_index(
                sanity_score=float(sanity),
                perturbation_score=float(perturbation),
                stability_score=float(robustness),
                weights=weights,
            )
            for sanity, perturbation, robustness in zip(
                output["sanity_score"].to_numpy(),
                output["perturbation_score"].to_numpy(),
                output["nuisance_similarity"].to_numpy(),
                strict=True,
            )
        ],
        dtype=float,
    )

    if text_contradiction_col and text_contradiction_col in output.columns:
        output["text_contradiction"] = _clip01(
            pd.to_numeric(output[text_contradiction_col], errors="coerce").to_numpy()
        )

    numeric_required = ["sanity_score", "perturbation_score", "nuisance_similarity", "sample_nfi"]
    if output[numeric_required].isna().any().any():
        raise ValueError("NaN values found after deriving sample scores; check artifact input columns")

    return output


def _ci(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    low = float(np.quantile(values, alpha / 2.0))
    high = float(np.quantile(values, 1.0 - alpha / 2.0))
    return low, high


def _bootstrap_method_distribution(
    sanity_scores: np.ndarray,
    perturbation_scores: np.ndarray,
    nuisance_similarities: np.ndarray,
    stable_mask: np.ndarray,
    num_resamples: int,
    seed: int,
    weights: tuple[float, float, float] | None,
) -> np.ndarray:
    n = sanity_scores.size
    if n == 0:
        return np.empty((0, 4), dtype=float)

    rng = np.random.default_rng(seed)
    dist = np.empty((num_resamples, 4), dtype=float)
    for index in range(num_resamples):
        sample_idx = rng.integers(0, n, size=n)
        sanity_mean = float(np.mean(sanity_scores[sample_idx]))
        perturbation_mean = float(np.mean(perturbation_scores[sample_idx]))
        robust = nuisance_robustness_score(
            explanation_similarities=nuisance_similarities[sample_idx],
            stable_prediction_mask=stable_mask[sample_idx],
        ).score
        nfi = normalized_faithfulness_index(
            sanity_score=sanity_mean,
            perturbation_score=perturbation_mean,
            stability_score=robust,
            weights=weights,
        )
        dist[index] = [sanity_mean, perturbation_mean, robust, nfi]

    return dist


def summarize_methods(
    sample_scores: pd.DataFrame,
    thresholds: FaithfulnessThresholds,
    method_col: str = "method",
    num_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 17,
    weights: tuple[float, float, float] | None = None,
) -> pd.DataFrame:
    if num_bootstrap <= 0:
        raise ValueError("num_bootstrap must be positive")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be between 0 and 1")

    rows: list[dict[str, object]] = []

    for method_name, method_df in sample_scores.groupby(method_col):
        sanity_scores = method_df["sanity_score"].to_numpy(dtype=float)
        perturb_scores = method_df["perturbation_score"].to_numpy(dtype=float)
        nuisance_scores = method_df["nuisance_similarity"].to_numpy(dtype=float)
        stable_mask = method_df["is_stable"].to_numpy(dtype=bool)

        sanity = float(np.mean(sanity_scores))
        perturbation = float(np.mean(perturb_scores))
        robustness_result = nuisance_robustness_score(
            explanation_similarities=nuisance_scores,
            stable_prediction_mask=stable_mask,
        )
        nfi = normalized_faithfulness_index(
            sanity_score=sanity,
            perturbation_score=perturbation,
            stability_score=robustness_result.score,
            weights=weights,
        )

        text_contradiction_value: float | None = None
        if "text_contradiction" in method_df.columns and method_df["text_contradiction"].notna().any():
            text_contradiction_value = float(method_df["text_contradiction"].dropna().mean())

        gates = evaluate_quality_gates(
            scores=FaithfulnessScores(
                sanity=sanity,
                perturbation=perturbation,
                robustness=robustness_result.score,
                nfi=nfi,
                text_contradiction=text_contradiction_value,
            ),
            thresholds=thresholds,
        )

        bootstrap_distribution = _bootstrap_method_distribution(
            sanity_scores=sanity_scores,
            perturbation_scores=perturb_scores,
            nuisance_similarities=nuisance_scores,
            stable_mask=stable_mask,
            num_resamples=num_bootstrap,
            seed=seed,
            weights=weights,
        )
        sanity_ci = _ci(bootstrap_distribution[:, 0], alpha=alpha) if bootstrap_distribution.size else (float("nan"), float("nan"))
        perturb_ci = _ci(bootstrap_distribution[:, 1], alpha=alpha) if bootstrap_distribution.size else (float("nan"), float("nan"))
        robust_ci = _ci(bootstrap_distribution[:, 2], alpha=alpha) if bootstrap_distribution.size else (float("nan"), float("nan"))
        nfi_ci = _ci(bootstrap_distribution[:, 3], alpha=alpha) if bootstrap_distribution.size else (float("nan"), float("nan"))

        row = {
            method_col: method_name,
            "n_samples": int(method_df.shape[0]),
            "num_stable": int(robustness_result.num_stable),
            "stable_fraction": float(robustness_result.stable_fraction),
            "sanity": sanity,
            "sanity_ci_low": sanity_ci[0],
            "sanity_ci_high": sanity_ci[1],
            "perturbation": perturbation,
            "perturbation_ci_low": perturb_ci[0],
            "perturbation_ci_high": perturb_ci[1],
            "robustness": float(robustness_result.score),
            "robustness_ci_low": robust_ci[0],
            "robustness_ci_high": robust_ci[1],
            "nfi": nfi,
            "nfi_ci_low": nfi_ci[0],
            "nfi_ci_high": nfi_ci[1],
        }
        if text_contradiction_value is not None:
            row["text_contradiction"] = text_contradiction_value

        row.update(gates)
        rows.append(row)

    return pd.DataFrame(rows).sort_values(by="nfi", ascending=False).reset_index(drop=True)


def pairwise_nfi_deltas(
    sample_scores: pd.DataFrame,
    method_col: str = "method",
    sample_id_col: str = "study_id",
    num_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 17,
) -> pd.DataFrame:
    pivot = sample_scores.pivot_table(index=sample_id_col, columns=method_col, values="sample_nfi", aggfunc="mean")
    methods = list(pivot.columns)
    rows: list[dict[str, object]] = []

    for method_a, method_b in itertools.combinations(methods, 2):
        paired = pivot[[method_a, method_b]].dropna()
        if paired.empty:
            continue

        mean_diff, ci_low, ci_high = paired_bootstrap_mean_diff_ci(
            paired[method_a].to_numpy(dtype=float),
            paired[method_b].to_numpy(dtype=float),
            num_resamples=num_resamples,
            alpha=alpha,
            seed=seed,
        )
        rows.append(
            {
                "method_a": method_a,
                "method_b": method_b,
                "num_paired_samples": int(paired.shape[0]),
                "mean_nfi_diff_a_minus_b": mean_diff,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["method_a", "method_b", "num_paired_samples", "mean_nfi_diff_a_minus_b", "ci_low", "ci_high"])

    return pd.DataFrame(rows).sort_values(by="mean_nfi_diff_a_minus_b", ascending=False).reset_index(drop=True)


def run_unified_benchmark(
    frame: pd.DataFrame,
    thresholds: FaithfulnessThresholds,
    method_col: str = "method",
    sample_id_col: str = "study_id",
    base_pred_col: str = "base_prediction",
    perturbed_pred_col: str = "perturbed_prediction",
    nuisance_similarity_col: str = "nuisance_similarity",
    sanity_score_col: str | None = "sanity_score",
    sanity_similarity_col: str | None = "sanity_similarity",
    perturbation_score_col: str | None = "perturbation_score",
    deletion_curve_col: str | None = "deletion_curve",
    insertion_curve_col: str | None = "insertion_curve",
    text_contradiction_col: str | None = "text_contradiction",
    stability_abs_tolerance: float = 0.05,
    num_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 17,
    weights: tuple[float, float, float] | None = None,
) -> BenchmarkOutputs:
    sample_scores = build_sample_scores(
        frame=frame,
        method_col=method_col,
        sample_id_col=sample_id_col,
        base_pred_col=base_pred_col,
        perturbed_pred_col=perturbed_pred_col,
        nuisance_similarity_col=nuisance_similarity_col,
        sanity_score_col=sanity_score_col,
        sanity_similarity_col=sanity_similarity_col,
        perturbation_score_col=perturbation_score_col,
        deletion_curve_col=deletion_curve_col,
        insertion_curve_col=insertion_curve_col,
        text_contradiction_col=text_contradiction_col,
        stability_abs_tolerance=stability_abs_tolerance,
        weights=weights,
    )

    method_summary = summarize_methods(
        sample_scores=sample_scores,
        thresholds=thresholds,
        method_col=method_col,
        num_bootstrap=num_bootstrap,
        alpha=alpha,
        seed=seed,
        weights=weights,
    )
    pairwise = pairwise_nfi_deltas(
        sample_scores=sample_scores,
        method_col=method_col,
        sample_id_col=sample_id_col,
        num_resamples=num_bootstrap,
        alpha=alpha,
        seed=seed,
    )
    return BenchmarkOutputs(sample_scores=sample_scores, method_summary=method_summary, pairwise_nfi_deltas=pairwise)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run unified E7-style faithfulness benchmark over method artifacts.")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", default="e7")
    parser.add_argument("--thresholds-json", type=Path, default=Path("configs/eval/faithfulness_thresholds.json"))

    parser.add_argument("--method-col", default="method")
    parser.add_argument("--sample-id-col", default="study_id")
    parser.add_argument("--base-pred-col", default="base_prediction")
    parser.add_argument("--perturbed-pred-col", default="perturbed_prediction")
    parser.add_argument("--nuisance-similarity-col", default="nuisance_similarity")
    parser.add_argument("--sanity-score-col", default="sanity_score")
    parser.add_argument("--sanity-similarity-col", default="sanity_similarity")
    parser.add_argument("--perturbation-score-col", default="perturbation_score")
    parser.add_argument("--deletion-curve-col", default="deletion_curve")
    parser.add_argument("--insertion-curve-col", default="insertion_curve")
    parser.add_argument("--text-contradiction-col", default="text_contradiction")
    parser.add_argument("--stability-abs-tolerance", type=float, default=0.05)
    parser.add_argument("--weights", nargs=3, type=float, default=None)
    parser.add_argument("--method-regex", default="")
    parser.add_argument("--num-bootstrap", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=17)
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    frame = pd.read_csv(args.input_csv)
    if args.method_regex:
        method_series = frame[args.method_col].astype(str)
        frame = frame[method_series.str.contains(args.method_regex, regex=True, na=False)].copy()
        if frame.empty:
            raise ValueError("method-regex removed all rows")

    thresholds = load_thresholds(args.thresholds_json)
    outputs = run_unified_benchmark(
        frame=frame,
        thresholds=thresholds,
        method_col=args.method_col,
        sample_id_col=args.sample_id_col,
        base_pred_col=args.base_pred_col,
        perturbed_pred_col=args.perturbed_pred_col,
        nuisance_similarity_col=args.nuisance_similarity_col,
        sanity_score_col=args.sanity_score_col,
        sanity_similarity_col=args.sanity_similarity_col,
        perturbation_score_col=args.perturbation_score_col,
        deletion_curve_col=args.deletion_curve_col,
        insertion_curve_col=args.insertion_curve_col,
        text_contradiction_col=args.text_contradiction_col,
        stability_abs_tolerance=args.stability_abs_tolerance,
        num_bootstrap=args.num_bootstrap,
        alpha=args.alpha,
        seed=args.seed,
        weights=tuple(args.weights) if args.weights is not None else None,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs.sample_scores.to_csv(args.output_dir / f"{args.output_prefix}_sample_scores.csv", index=False)
    outputs.method_summary.to_csv(args.output_dir / f"{args.output_prefix}_method_summary.csv", index=False)
    outputs.pairwise_nfi_deltas.to_csv(args.output_dir / f"{args.output_prefix}_pairwise_nfi_deltas.csv", index=False)


if __name__ == "__main__":
    main()
