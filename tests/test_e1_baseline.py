import numpy as np

from src.models.e1_baseline import evaluate_predictions_by_split, train_multilabel_linear_probe


def test_train_multilabel_linear_probe_learns_signal() -> None:
    rng = np.random.default_rng(7)
    n = 240
    x = rng.normal(0.0, 1.0, size=(n, 4))

    logits = 1.8 * x[:, 0] - 1.4 * x[:, 1] + 0.3 * rng.normal(size=n)
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (probs > 0.5).astype(float).reshape(-1, 1)
    mask = np.ones_like(y, dtype=bool)

    labels = ["Atelectasis"]
    feature_columns = [f"f{i}" for i in range(x.shape[1])]

    train_idx = np.arange(n) < 180
    model, _history = train_multilabel_linear_probe(
        features=x[train_idx],
        targets=y[train_idx],
        mask=mask[train_idx],
        labels=labels,
        feature_columns=feature_columns,
        epochs=180,
        learning_rate=0.08,
        seed=13,
    )

    test_probs = model.predict_proba(x[~train_idx]).reshape(-1)
    test_truth = y[~train_idx].reshape(-1)
    accuracy = np.mean((test_probs >= 0.5).astype(float) == test_truth)
    assert accuracy > 0.8


def test_evaluate_predictions_by_split_outputs_expected_columns() -> None:
    split = np.array(["train", "train", "val", "test"])
    labels = ["Edema"]
    targets = np.array([[0.0], [1.0], [1.0], [0.0]])
    mask = np.array([[True], [True], [True], [True]])
    probs = np.array([[0.1], [0.9], [0.8], [0.2]])

    by_label, summary = evaluate_predictions_by_split(split, labels, targets, mask, probs)
    assert {"split", "label", "auroc", "auprc", "brier", "ece"}.issubset(by_label.columns)
    assert {"split", "macro_auroc", "macro_auprc", "macro_brier", "macro_ece"}.issubset(summary.columns)

