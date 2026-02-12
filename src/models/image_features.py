from __future__ import annotations

from pathlib import Path

import numpy as np


def _require_pillow():
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - exercised in runtime environments without Pillow
        raise RuntimeError("Pillow is required for image feature extraction. Install with: pip install Pillow") from exc
    return Image


def load_grayscale_image(path: Path, width: int = 320, height: int = 320) -> np.ndarray:
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive integers")

    image_cls = _require_pillow()
    with image_cls.open(path) as image:
        image = image.convert("L")
        image = image.resize((width, height), resample=image_cls.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
    return np.clip(array, 0.0, 1.0)


def _safe_ratio(numerator: float, denominator: float, fallback: float = 0.0) -> float:
    if denominator == 0:
        return fallback
    return float(numerator / denominator)


def compute_image_features(image: np.ndarray, hist_bins: int = 16) -> dict[str, float]:
    if hist_bins < 4:
        raise ValueError("hist_bins must be at least 4")

    image_array = np.asarray(image, dtype=np.float32)
    if image_array.ndim != 2:
        raise ValueError("image must be a 2D grayscale array")
    if image_array.size == 0:
        raise ValueError("image cannot be empty")

    features: dict[str, float] = {
        "pixel_mean": float(np.mean(image_array)),
        "pixel_std": float(np.std(image_array)),
        "pixel_min": float(np.min(image_array)),
        "pixel_max": float(np.max(image_array)),
        "pixel_p10": float(np.quantile(image_array, 0.10)),
        "pixel_p25": float(np.quantile(image_array, 0.25)),
        "pixel_p50": float(np.quantile(image_array, 0.50)),
        "pixel_p75": float(np.quantile(image_array, 0.75)),
        "pixel_p90": float(np.quantile(image_array, 0.90)),
    }

    grad_y, grad_x = np.gradient(image_array)
    grad_mag = np.sqrt((grad_x ** 2) + (grad_y ** 2))
    features.update(
        {
            "gradient_x_abs_mean": float(np.mean(np.abs(grad_x))),
            "gradient_y_abs_mean": float(np.mean(np.abs(grad_y))),
            "gradient_mag_mean": float(np.mean(grad_mag)),
            "gradient_mag_std": float(np.std(grad_mag)),
        }
    )

    h, w = image_array.shape
    border_h = max(1, int(round(0.15 * h)))
    border_w = max(1, int(round(0.15 * w)))
    center = image_array[border_h : h - border_h, border_w : w - border_w]
    if center.size == 0:
        center = image_array

    border_mask = np.ones_like(image_array, dtype=bool)
    border_mask[border_h : h - border_h, border_w : w - border_w] = False
    border_values = image_array[border_mask]
    if border_values.size == 0:
        border_values = image_array

    center_mean = float(np.mean(center))
    border_mean = float(np.mean(border_values))
    features.update(
        {
            "center_mean": center_mean,
            "border_mean": border_mean,
            "center_border_ratio": _safe_ratio(center_mean, border_mean, fallback=1.0),
        }
    )

    hist, _bins = np.histogram(image_array, bins=hist_bins, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    hist = hist / np.maximum(hist.sum(), 1.0)
    for index, value in enumerate(hist):
        features[f"hist_bin_{index:02d}"] = float(value)

    return features

