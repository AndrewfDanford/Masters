from __future__ import annotations

from copy import deepcopy
from typing import Callable

import numpy as np


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "Grad-CAM/HiResCAM requires torch. Install torch before running saliency generation."
        ) from exc
    return torch, nn, F


def top_fraction_mask(values: np.ndarray, fraction: float) -> np.ndarray:
    if not (0.0 <= fraction <= 1.0):
        raise ValueError("fraction must be in [0, 1]")
    flat = np.asarray(values, dtype=float).reshape(-1)
    count = flat.size
    if count == 0:
        raise ValueError("values cannot be empty")

    k = int(round(fraction * count))
    mask = np.zeros(count, dtype=bool)
    if k <= 0:
        return mask.reshape(values.shape)
    if k >= count:
        mask[:] = True
        return mask.reshape(values.shape)

    top_idx = np.argpartition(flat, -k)[-k:]
    mask[top_idx] = True
    return mask.reshape(values.shape)


def _normalize_cam_map(cam_map):
    torch, _nn, _F = _require_torch()
    cam = cam_map
    cam = cam - cam.amin(dim=(1, 2), keepdim=True)
    denom = cam.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
    cam = cam / denom
    return torch.nan_to_num(cam, nan=0.0, posinf=0.0, neginf=0.0)


class CAMExtractor:
    def __init__(self, model, target_layer):
        torch, _nn, _F = _require_torch()
        self._torch = torch
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._forward_handle = self.target_layer.register_forward_hook(self._on_forward)
        self._backward_handle = self.target_layer.register_full_backward_hook(self._on_backward)

    def _on_forward(self, _module, _args, output):
        self.activations = output.detach()

    def _on_backward(self, _module, _grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def close(self) -> None:
        self._forward_handle.remove()
        self._backward_handle.remove()

    def generate(self, inputs, class_index: int, method: str):
        if method not in {"gradcam", "hirescam"}:
            raise ValueError("method must be gradcam or hirescam")
        if inputs.ndim != 4:
            raise ValueError("inputs must be BCHW")

        torch = self._torch
        self.model.zero_grad(set_to_none=True)
        logits = self.model(inputs)
        if logits.ndim != 2:
            raise ValueError("model output must be [batch, classes]")
        if class_index < 0 or class_index >= logits.shape[1]:
            raise ValueError(f"class_index {class_index} is out of bounds for {logits.shape[1]} classes")

        score = logits[:, class_index].sum()
        score.backward(retain_graph=False)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("missing activations/gradients; target layer hook did not fire")

        acts = self.activations
        grads = self.gradients
        if method == "gradcam":
            weights = grads.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * acts).sum(dim=1))
        else:
            cam = torch.relu((grads * acts).sum(dim=1))

        cam = torch.nn.functional.interpolate(
            cam.unsqueeze(1),
            size=inputs.shape[2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        cam = _normalize_cam_map(cam)
        return cam


def apply_deletion(raw_image, cam_map: np.ndarray, fraction: float, baseline: float = 0.0):
    torch, _nn, _F = _require_torch()
    mask = top_fraction_mask(cam_map, fraction=fraction)
    out = raw_image.clone()
    out[:, mask] = baseline
    return out


def apply_insertion(raw_image, cam_map: np.ndarray, fraction: float, baseline: float = 0.0):
    torch, _nn, _F = _require_torch()
    mask = top_fraction_mask(cam_map, fraction=fraction)
    out = torch.full_like(raw_image, fill_value=float(baseline))
    out[:, mask] = raw_image[:, mask]
    return out


def deletion_insertion_curves(
    raw_image,
    cam_map: np.ndarray,
    fractions: np.ndarray,
    probability_fn: Callable,
):
    deletion: list[float] = []
    insertion: list[float] = []
    for fraction in fractions:
        deletion_image = apply_deletion(raw_image, cam_map=cam_map, fraction=float(fraction))
        insertion_image = apply_insertion(raw_image, cam_map=cam_map, fraction=float(fraction))
        deletion.append(float(probability_fn(deletion_image)))
        insertion.append(float(probability_fn(insertion_image)))
    return np.asarray(deletion, dtype=float), np.asarray(insertion, dtype=float)


def nuisance_perturbation(raw_image, brightness_delta: float = 0.05, contrast_scale: float = 1.05):
    mean = raw_image.mean(dim=(1, 2), keepdim=True)
    adjusted = (raw_image - mean) * contrast_scale + mean + brightness_delta
    return adjusted.clamp(0.0, 1.0)


def randomized_model_copy(model, mode: str, head_attr: str):
    torch, nn, _F = _require_torch()
    if mode not in {"none", "head", "all"}:
        raise ValueError("mode must be none, head, or all")
    randomized = deepcopy(model)
    if mode == "none":
        randomized.eval()
        return randomized

    def reset_module(module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

    if mode == "all":
        randomized.apply(reset_module)
    else:
        head_module = getattr(randomized, head_attr, None)
        if head_module is None:
            raise ValueError(f"head attribute '{head_attr}' not found on model")
        head_module.apply(reset_module)

    randomized.eval()
    return randomized

