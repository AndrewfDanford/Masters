import numpy as np

from src.explain.cam import top_fraction_mask


def test_top_fraction_mask_counts() -> None:
    values = np.array([[0.1, 0.9], [0.4, 0.2]], dtype=float)
    mask_zero = top_fraction_mask(values, fraction=0.0)
    mask_half = top_fraction_mask(values, fraction=0.5)
    mask_full = top_fraction_mask(values, fraction=1.0)

    assert int(mask_zero.sum()) == 0
    assert int(mask_half.sum()) == 2
    assert int(mask_full.sum()) == values.size

