import numpy as np
import pandas as pd
import pytest

from src.models.e1_train_cnn import (
    _canonical_split_name,
    _normalize_state_dict_keys,
    build_targets_from_row,
    compute_pos_weight,
    label_value_to_target_mask,
)


def test_label_value_to_target_mask_uncertain_policies() -> None:
    assert label_value_to_target_mask(-1, "u_ignore") == (0.0, 0.0)
    assert label_value_to_target_mask(-1, "u_zero") == (0.0, 1.0)
    assert label_value_to_target_mask(-1, "u_one") == (1.0, 1.0)
    assert label_value_to_target_mask(1, "u_ignore") == (1.0, 1.0)
    assert label_value_to_target_mask(0, "u_ignore") == (0.0, 1.0)


def test_label_value_to_target_mask_rejects_bad_policy() -> None:
    with pytest.raises(ValueError, match="uncertain_policy"):
        label_value_to_target_mask(1, "invalid-policy")


def test_build_targets_from_row() -> None:
    row = pd.Series({"Edema": 1, "Consolidation": -1, "Atelectasis": np.nan})
    targets, mask = build_targets_from_row(row, labels=["Edema", "Consolidation", "Atelectasis"], uncertain_policy="u_ignore")
    assert targets.tolist() == [1.0, 0.0, 0.0]
    assert mask.tolist() == [1.0, 0.0, 0.0]


def test_compute_pos_weight_shape_and_values() -> None:
    targets = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )
    mask = np.ones_like(targets, dtype=np.float32)
    weights = compute_pos_weight(targets, mask)
    assert weights.shape == (2,)
    assert np.all(weights > 0.0)


def test_compute_pos_weight_uses_mask_and_handles_no_positives() -> None:
    targets = np.array(
        [
            [1.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )
    mask = np.array(
        [
            [1.0, 1.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    weights = compute_pos_weight(targets, mask)
    assert np.allclose(weights, np.array([1.0, 1.0], dtype=np.float32))


def test_normalize_state_dict_keys_removes_module_prefix() -> None:
    state = {"module.layer.weight": 1, "layer.bias": 2}
    normalized = _normalize_state_dict_keys(state)
    assert "module.layer.weight" not in normalized
    assert normalized["layer.weight"] == 1
    assert normalized["layer.bias"] == 2


def test_canonical_split_name_maps_validate_alias() -> None:
    assert _canonical_split_name("validate") == "val"
    assert _canonical_split_name("validation") == "val"
    assert _canonical_split_name("train") == "train"
