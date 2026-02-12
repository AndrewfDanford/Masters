from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.audit import DEFAULT_FINDINGS
from src.utils.metrics import (
    binary_auprc,
    binary_auroc,
    binary_brier,
    expected_calibration_error,
)


def _require_columns(df: pd.DataFrame, columns: list[str], frame_name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{frame_name} is missing required columns: {', '.join(missing)}")


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _label_matrix(df: pd.DataFrame, labels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    _require_columns(df, labels, "cohort")
    raw = np.column_stack([pd.to_numeric(df[label], errors="coerce").to_numpy() for label in labels])

    # Treat uncertainty marker -1 as unknown.
    raw[raw == -1] = np.nan
    mask = ~np.isnan(raw)
    targets = np.where(mask, (raw > 0).astype(float), 0.0)
    return targets.astype(float), mask


@dataclass(frozen=True)
class LinearProbeModel:
    weights: np.ndarray
    bias: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray
    labels: list[str]
    feature_columns: list[str]

    def transform(self, features: np.ndarray) -> np.ndarray:
        return (features - self.feature_mean) / self.feature_std

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        x = self.transform(features)
        return _sigmoid(x @ self.weights + self.bias)


def train_multilabel_linear_probe(
    features: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
    labels: list[str],
    feature_columns: list[str],
    epochs: int = 300,
    learning_rate: float = 0.05,
    weight_decay: float = 1e-4,
    seed: int = 17,
    class_balance: bool = True,
) -> tuple[LinearProbeModel, list[dict[str, float]]]:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")

    x = np.asarray(features, dtype=float)
    y = np.asarray(targets, dtype=float)
    m = np.asarray(mask, dtype=bool)

    if x.ndim != 2:
        raise ValueError("features must be a 2D array")
    if y.shape != m.shape:
        raise ValueError("targets and mask must have the same shape")
    if x.shape[0] != y.shape[0]:
        raise ValueError("features and targets must have matching number of rows")
    if y.shape[1] != len(labels):
        raise ValueError("label count does not match target columns")
    if x.shape[1] != len(feature_columns):
        raise ValueError("feature column count does not match feature width")

    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    x_norm = (x - mean) / std

    rng = np.random.default_rng(seed)
    num_features = x_norm.shape[1]
    num_labels = y.shape[1]
    weights = rng.normal(0.0, 0.01, size=(num_features, num_labels))
    bias = np.zeros(num_labels, dtype=float)

    if class_balance:
        pos_counts = np.sum((y == 1.0) & m, axis=0).astype(float)
        neg_counts = np.sum((y == 0.0) & m, axis=0).astype(float)
        pos_weight = np.where(pos_counts > 0, neg_counts / np.maximum(pos_counts, 1.0), 1.0)
    else:
        pos_weight = np.ones(num_labels, dtype=float)

    history: list[dict[str, float]] = []
    eps = 1e-8
    for epoch in range(1, epochs + 1):
        logits = x_norm @ weights + bias
        probs = _sigmoid(logits)

        grad_w = np.zeros_like(weights)
        grad_b = np.zeros_like(bias)
        per_label_loss: list[float] = []

        for index in range(num_labels):
            active = m[:, index]
            if not np.any(active):
                continue

            x_label = x_norm[active]
            y_label = y[active, index]
            p_label = probs[active, index]
            sample_weight = np.where(y_label == 1.0, pos_weight[index], 1.0)
            error = (p_label - y_label) * sample_weight

            grad_w[:, index] = (x_label.T @ error) / active.sum() + (weight_decay * weights[:, index])
            grad_b[index] = float(np.mean(error))

            loss_value = -np.mean(
                sample_weight * (y_label * np.log(p_label + eps) + (1.0 - y_label) * np.log(1.0 - p_label + eps))
            )
            per_label_loss.append(float(loss_value))

        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b

        mean_loss = float(np.mean(per_label_loss)) if per_label_loss else float("nan")
        history.append({"epoch": float(epoch), "loss": mean_loss})

    model = LinearProbeModel(
        weights=weights,
        bias=bias,
        feature_mean=mean,
        feature_std=std,
        labels=list(labels),
        feature_columns=list(feature_columns),
    )
    return model, history


def _safe_metric(metric_fn, y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        value = metric_fn(y_true, y_prob)
    except ValueError:
        return float("nan")
    return float(value)


def evaluate_predictions_by_split(
    split: np.ndarray,
    labels: list[str],
    targets: np.ndarray,
    mask: np.ndarray,
    probs: np.ndarray,
    ece_bins: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    split_values = pd.Series(split).astype(str)
    for split_name in sorted(split_values.unique()):
        split_index = split_values == split_name
        split_rows: list[dict[str, object]] = []

        for label_index, label in enumerate(labels):
            known = split_index.to_numpy() & mask[:, label_index]
            n_known = int(np.sum(known))
            y_true = targets[known, label_index]
            y_prob = probs[known, label_index]
            n_positive = int(np.sum(y_true == 1.0))

            if n_known == 0:
                metric_row = {
                    "split": split_name,
                    "label": label,
                    "num_known": 0,
                    "num_positive": 0,
                    "auroc": float("nan"),
                    "auprc": float("nan"),
                    "brier": float("nan"),
                    "ece": float("nan"),
                }
            else:
                metric_row = {
                    "split": split_name,
                    "label": label,
                    "num_known": n_known,
                    "num_positive": n_positive,
                    "auroc": _safe_metric(binary_auroc, y_true, y_prob),
                    "auprc": _safe_metric(binary_auprc, y_true, y_prob),
                    "brier": _safe_metric(binary_brier, y_true, y_prob),
                    "ece": _safe_metric(
                        lambda a, b: expected_calibration_error(a, b, num_bins=ece_bins),
                        y_true,
                        y_prob,
                    ),
                }

            split_rows.append(metric_row)
            rows.append(metric_row)

        split_df = pd.DataFrame(split_rows)
        summary_rows.append(
            {
                "split": split_name,
                "num_labels": int(split_df["label"].nunique()),
                "macro_auroc": float(split_df["auroc"].mean(skipna=True)),
                "macro_auprc": float(split_df["auprc"].mean(skipna=True)),
                "macro_brier": float(split_df["brier"].mean(skipna=True)),
                "macro_ece": float(split_df["ece"].mean(skipna=True)),
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(summary_rows)


def run_e1_baseline(args: argparse.Namespace) -> None:
    cohort_df = pd.read_csv(args.cohort_csv)
    feature_df = pd.read_csv(args.features_csv)

    _require_columns(cohort_df, [args.study_col, args.split_col] + args.labels, "cohort")
    _require_columns(feature_df, [args.study_col], "features")

    feature_columns = args.feature_columns
    if feature_columns:
        selected_features = [column.strip() for column in feature_columns.split(",") if column.strip()]
    else:
        selected_features = [
            column
            for column in feature_df.columns
            if column != args.study_col and pd.api.types.is_numeric_dtype(feature_df[column])
        ]

    if not selected_features:
        raise ValueError("no numeric feature columns were selected")

    merged = cohort_df[[args.study_col, args.split_col] + args.labels].merge(
        feature_df[[args.study_col] + selected_features],
        on=args.study_col,
        how="inner",
    )
    if merged.empty:
        raise ValueError("no rows remained after merging cohort and features")

    x_all = merged[selected_features].to_numpy(dtype=float)
    y_all, label_mask_all = _label_matrix(merged, args.labels)
    split_values = merged[args.split_col].astype(str).to_numpy()
    train_mask = split_values == "train"
    if int(np.sum(train_mask)) == 0:
        raise ValueError("no train rows found; expected split column to include 'train'")

    model, history = train_multilabel_linear_probe(
        features=x_all[train_mask],
        targets=y_all[train_mask],
        mask=label_mask_all[train_mask],
        labels=args.labels,
        feature_columns=selected_features,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        class_balance=not args.no_class_balance,
    )

    prob_all = model.predict_proba(x_all)
    metrics_df, summary_df = evaluate_predictions_by_split(
        split=split_values,
        labels=args.labels,
        targets=y_all,
        mask=label_mask_all,
        probs=prob_all,
        ece_bins=args.ece_bins,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    history_df = pd.DataFrame(history)
    history_df.to_csv(args.output_dir / "e1_train_history.csv", index=False)

    predictions_long_rows: list[dict[str, object]] = []
    for row_index, row in merged.reset_index(drop=True).iterrows():
        for label_index, label in enumerate(args.labels):
            known = bool(label_mask_all[row_index, label_index])
            target = float(y_all[row_index, label_index]) if known else float("nan")
            predictions_long_rows.append(
                {
                    args.study_col: row[args.study_col],
                    args.split_col: row[args.split_col],
                    "label": label,
                    "known": known,
                    "target": target,
                    "probability": float(prob_all[row_index, label_index]),
                }
            )
    pd.DataFrame(predictions_long_rows).to_csv(args.output_dir / "e1_predictions_long.csv", index=False)

    metrics_df.to_csv(args.output_dir / "e1_metrics_by_label.csv", index=False)
    summary_df.to_csv(args.output_dir / "e1_metrics_summary.csv", index=False)

    np.savez(
        args.output_dir / "e1_model.npz",
        weights=model.weights,
        bias=model.bias,
        feature_mean=model.feature_mean,
        feature_std=model.feature_std,
    )

    model_card = {
        "study_col": args.study_col,
        "split_col": args.split_col,
        "labels": args.labels,
        "feature_columns": selected_features,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "class_balance": not args.no_class_balance,
        "num_rows_merged": int(merged.shape[0]),
        "num_train_rows": int(np.sum(train_mask)),
    }
    with (args.output_dir / "e1_model_card.json").open("w", encoding="utf-8") as handle:
        json.dump(model_card, handle, indent=2)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "E1 baseline training on cohort labels + precomputed feature table. "
            "Designed for fast iteration while data access is pending."
        )
    )
    parser.add_argument("--cohort-csv", type=Path, required=True)
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument("--study-col", default="study_id")
    parser.add_argument("--split-col", default="split")
    parser.add_argument("--labels", nargs="+", default=DEFAULT_FINDINGS)
    parser.add_argument(
        "--feature-columns",
        default="",
        help="Comma-separated numeric feature columns. Default: all numeric columns except study id.",
    )

    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--ece-bins", type=int, default=10)
    parser.add_argument("--no-class-balance", action="store_true")
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    run_e1_baseline(args)


if __name__ == "__main__":
    main()

