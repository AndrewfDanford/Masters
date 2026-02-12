import numpy as np

from src.models.image_features import compute_image_features


def test_compute_image_features_has_expected_statistics() -> None:
    image = np.linspace(0.0, 1.0, num=64 * 64, dtype=np.float32).reshape(64, 64)
    features = compute_image_features(image, hist_bins=8)

    required = {
        "pixel_mean",
        "pixel_std",
        "pixel_p50",
        "gradient_mag_mean",
        "center_mean",
        "border_mean",
        "center_border_ratio",
    }
    assert required.issubset(features.keys())

    hist_values = [features[f"hist_bin_{i:02d}"] for i in range(8)]
    assert np.isclose(sum(hist_values), 1.0, atol=1e-6)
    assert 0.0 <= features["pixel_mean"] <= 1.0

