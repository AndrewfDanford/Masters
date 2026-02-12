from __future__ import annotations

from pathlib import Path

import numpy as np


def _require_torch_stack():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from PIL import Image
        from torchvision import models, transforms
    except ImportError as exc:  # pragma: no cover - depends on runtime environment
        raise RuntimeError(
            "Backbone feature extraction requires torch, torchvision, and Pillow. "
            "Install these before using --extractor resnet18/densenet121."
        ) from exc

    return torch, nn, F, Image, models, transforms


def _normalize_state_dict_keys(state_dict: dict[str, object]) -> dict[str, object]:
    normalized: dict[str, object] = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            normalized[key[len("module.") :]] = value
        else:
            normalized[key] = value
    return normalized


def _build_backbone_model(
    model_name: str,
    pretrained: bool,
    checkpoint_path: Path | None,
    device: str,
):
    torch, nn, F, _image_mod, models, _transforms = _require_torch_stack()

    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        base_model = models.resnet18(weights=weights)
        feature_model = nn.Sequential(*list(base_model.children())[:-1])
    elif model_name == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        base_model = models.densenet121(weights=weights)

        class _DenseNetFeatureExtractor(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.features = model.features

            def forward(self, x):
                x = self.features(x)
                x = F.relu(x, inplace=False)
                x = F.adaptive_avg_pool2d(x, (1, 1))
                return x

        feature_model = _DenseNetFeatureExtractor(base_model)
    else:
        raise ValueError("model_name must be one of: resnet18, densenet121")

    if checkpoint_path is not None:
        checkpoint_obj = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(checkpoint_obj, dict) and "state_dict" in checkpoint_obj and isinstance(
            checkpoint_obj["state_dict"], dict
        ):
            state_dict = checkpoint_obj["state_dict"]
        elif isinstance(checkpoint_obj, dict):
            state_dict = checkpoint_obj
        else:
            raise ValueError("unsupported checkpoint format; expected state dict or {'state_dict': ...}")
        base_model.load_state_dict(_normalize_state_dict_keys(state_dict), strict=False)

    resolved_device = device
    if resolved_device.startswith("cuda") and not torch.cuda.is_available():
        resolved_device = "cpu"

    feature_model = feature_model.to(resolved_device)
    feature_model.eval()
    return feature_model, resolved_device


def extract_backbone_features(
    image_paths: list[Path],
    model_name: str = "resnet18",
    width: int = 320,
    height: int = 320,
    batch_size: int = 32,
    device: str = "cpu",
    pretrained: bool = False,
    checkpoint_path: Path | None = None,
) -> np.ndarray:
    if not image_paths:
        raise ValueError("image_paths cannot be empty")
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    torch, _nn, _F, Image, _models, transforms = _require_torch_stack()
    feature_model, resolved_device = _build_backbone_model(
        model_name=model_name,
        pretrained=pretrained,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    transform = transforms.Compose(
        [
            transforms.Resize((height, width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    all_features: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start : start + batch_size]
            tensors = []
            for path in batch_paths:
                with Image.open(path) as image:
                    image = image.convert("RGB")
                    tensors.append(transform(image))

            batch_tensor = torch.stack(tensors, dim=0).to(resolved_device)
            batch_features = feature_model(batch_tensor).reshape(batch_tensor.shape[0], -1)
            all_features.append(batch_features.cpu().numpy().astype(np.float32))

    return np.concatenate(all_features, axis=0)

