import numpy as np
import pandas as pd

from src.explain.e2_e3_generate import choose_target_index


def test_choose_target_index_prefers_positive_manifest_label() -> None:
    row = pd.Series({"Edema": 1, "Consolidation": 0})
    labels = ["Edema", "Consolidation"]
    probs = np.array([0.2, 0.9], dtype=float)
    index, label = choose_target_index(row=row, labels=labels, probabilities=probs)
    assert index == 0
    assert label == "Edema"


def test_choose_target_index_falls_back_to_probability_argmax() -> None:
    row = pd.Series({"Edema": 0, "Consolidation": 0})
    labels = ["Edema", "Consolidation"]
    probs = np.array([0.25, 0.65], dtype=float)
    index, label = choose_target_index(row=row, labels=labels, probabilities=probs)
    assert index == 1
    assert label == "Consolidation"

