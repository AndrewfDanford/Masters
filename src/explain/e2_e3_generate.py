from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.audit import DEFAULT_FINDINGS
from src.eval.faithfulness import saliency_similarity
from src.models.e1_extract_features import resolve_image_path

from .cam import CAMExtractor, deletion_insertion_curves, nuisance_perturbation, randomized_model_copy


def _require_torchvision_stack():
    try:
        import torch
        import torch.nn as nn
        from PIL import Image
        from torchvision import models, transforms
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "E2/E3 saliency generation requires torch, torchvision, and Pillow."
        ) from exc
    return torch, nn, Image, models, transforms


def _clean_identifier(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        return text[:-2]
    return text


def _normalize_state_dict_keys(state_dict: dict[str, object]) -> dict[str, object]:
    normalized: dict[str, object] = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            normalized[key[len("module.") :]] = value
        else:
            normalized[key] = value
    return normalized


def choose_target_index(
    row: pd.Series,
    labels: list[str],
    probabilities: np.ndarray,
) -> tuple[int, str]:
    positives: list[int] = []
    for index, label in enumerate(labels):
        if label not in row:
            continue
        value = pd.to_numeric(pd.Series([row[label]]), errors="coerce").iloc[0]
        if pd.notna(value) and float(value) > 0:
            positives.append(index)

    if positives:
        idx = positives[0]
        return idx, labels[idx]

    idx = int(np.argmax(np.asarray(probabilities, dtype=float)))
    return idx, labels[idx]


def _build_model(
    arch: str,
    num_classes: int,
    checkpoint_path: Path,
    device: str,
    pretrained_backbone: bool,
):
    torch, nn, _Image, models, _transforms = _require_torchvision_stack()

    if arch == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained_backbone else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        target_layer = model.layer4[-1]
        head_attr = "fc"
    elif arch == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained_backbone else None
        model = models.densenet121(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
        target_layer = model.features[-1]
        head_attr = "classifier"
    else:
        raise ValueError("arch must be one of: resnet18, densenet121")

    checkpoint_obj = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint_obj, dict) and "state_dict" in checkpoint_obj and isinstance(checkpoint_obj["state_dict"], dict):
        state_dict = checkpoint_obj["state_dict"]
    elif isinstance(checkpoint_obj, dict):
        state_dict = checkpoint_obj
    else:
        raise ValueError("unsupported checkpoint format")

    model.load_state_dict(_normalize_state_dict_keys(state_dict), strict=False)
    model.to(device)
    model.eval()
    return model, target_layer, head_attr


def _predict_probs(model, normalized_batch):
    torch, _nn, _Image, _models, _transforms = _require_torchvision_stack()
    with torch.no_grad():
        logits = model(normalized_batch)
        probs = torch.sigmoid(logits)
    return probs


def run_generation(args: argparse.Namespace) -> None:
    torch, _nn, Image, _models, transforms = _require_torchvision_stack()

    frame = pd.read_csv(args.manifest_csv)
    required_cols = [args.study_col, args.subject_col, args.dicom_col]
    for col in required_cols:
        if col not in frame.columns:
            raise ValueError(f"manifest is missing required column: {col}")

    labels = list(args.labels)
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("max-samples must be positive")
        frame = frame.head(args.max_samples).copy()

    if args.split_col in frame.columns and args.target_split:
        frame = frame[frame[args.split_col].astype(str) == args.target_split].copy()

    if frame.empty:
        raise ValueError("no rows to process after split/max-sample filtering")

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    model, target_layer, head_attr = _build_model(
        arch=args.arch,
        num_classes=len(labels),
        checkpoint_path=args.model_checkpoint,
        device=device,
        pretrained_backbone=args.pretrained_backbone,
    )

    random_model = randomized_model_copy(model, mode=args.sanity_mode, head_attr=head_attr).to(device)

    to_tensor = transforms.Compose(
        [
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor(),
        ]
    )
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    fractions = np.linspace(0.0, 1.0, args.curve_steps + 1)

    rows: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []

    cam_extractors = {
        "gradcam": CAMExtractor(model=model, target_layer=target_layer),
        "hirescam": CAMExtractor(model=model, target_layer=target_layer),
    }
    if args.arch == "resnet18":
        random_target_layer = random_model.layer4[-1]
    elif args.arch == "densenet121":
        random_target_layer = random_model.features[-1]
    else:  # pragma: no cover - guarded by argparse choices
        raise ValueError(f"unsupported architecture: {args.arch}")

    random_extractors = {
        "gradcam": CAMExtractor(model=random_model, target_layer=random_target_layer),
        "hirescam": CAMExtractor(model=random_model, target_layer=random_target_layer),
    }

    try:
        for _, row in frame.iterrows():
            resolved_path = resolve_image_path(
                row=row,
                image_root=args.image_root,
                image_path_col=args.image_path_col,
                study_col=args.study_col,
                subject_col=args.subject_col,
                dicom_col=args.dicom_col,
            )
            if resolved_path is None or not resolved_path.exists():
                skipped.append(
                    {
                        args.study_col: row.get(args.study_col),
                        args.subject_col: row.get(args.subject_col),
                        args.dicom_col: row.get(args.dicom_col),
                        "reason": "missing_image_file",
                        "resolved_path": str(resolved_path) if resolved_path is not None else "",
                    }
                )
                continue

            with Image.open(resolved_path) as image:
                raw = to_tensor(image.convert("RGB")).to(device)

            normalized = normalize(raw).unsqueeze(0)
            probs = _predict_probs(model, normalized)[0].detach().cpu().numpy()
            target_index, target_label = choose_target_index(row=row, labels=labels, probabilities=probs)
            base_prediction = float(probs[target_index])

            perturbed_raw = nuisance_perturbation(
                raw,
                brightness_delta=args.nuisance_brightness,
                contrast_scale=args.nuisance_contrast,
            )
            perturbed_prediction = float(
                _predict_probs(model, normalize(perturbed_raw).unsqueeze(0))[0, target_index].detach().cpu().item()
            )

            for method in args.methods:
                extractor = cam_extractors[method]
                random_extractor = random_extractors[method]

                cam_map = extractor.generate(normalized, class_index=target_index, method=method)[0].detach().cpu().numpy()
                perturbed_cam_map = (
                    extractor.generate(normalize(perturbed_raw).unsqueeze(0), class_index=target_index, method=method)[0]
                    .detach()
                    .cpu()
                    .numpy()
                )

                nuisance_similarity = float(saliency_similarity(cam_map, perturbed_cam_map))
                random_cam = (
                    random_extractor.generate(normalized, class_index=target_index, method=method)[0]
                    .detach()
                    .cpu()
                    .numpy()
                )
                sanity_similarity = float(saliency_similarity(cam_map, random_cam))

                def probability_from_raw(candidate_raw):
                    return _predict_probs(model, normalize(candidate_raw).unsqueeze(0))[0, target_index].detach().cpu().item()

                deletion_curve, insertion_curve = deletion_insertion_curves(
                    raw_image=raw,
                    cam_map=cam_map,
                    fractions=fractions,
                    probability_fn=probability_from_raw,
                )

                out_row: dict[str, object] = {
                    "method": method,
                    args.study_col: row.get(args.study_col),
                    args.subject_col: row.get(args.subject_col),
                    args.dicom_col: row.get(args.dicom_col),
                    "target_label": target_label,
                    "target_index": target_index,
                    "base_prediction": base_prediction,
                    "perturbed_prediction": perturbed_prediction,
                    "nuisance_similarity": nuisance_similarity,
                    "sanity_similarity": sanity_similarity,
                    "deletion_curve": ",".join(f"{float(v):.6f}" for v in deletion_curve),
                    "insertion_curve": ",".join(f"{float(v):.6f}" for v in insertion_curve),
                    "resolved_image_path": str(resolved_path),
                    "extractor_arch": args.arch,
                    "sanity_mode": args.sanity_mode,
                }
                if args.split_col in frame.columns:
                    out_row[args.split_col] = row.get(args.split_col)
                rows.append(out_row)

                if args.save_cam_dir is not None:
                    args.save_cam_dir.mkdir(parents=True, exist_ok=True)
                    study = _clean_identifier(row.get(args.study_col))
                    stem = f"{study}_{target_label}_{method}"
                    np.save(args.save_cam_dir / f"{stem}_cam.npy", cam_map)
                    np.save(args.save_cam_dir / f"{stem}_cam_perturbed.npy", perturbed_cam_map)
                    np.save(args.save_cam_dir / f"{stem}_cam_random.npy", random_cam)
    finally:
        for extractor in cam_extractors.values():
            extractor.close()
        for extractor in random_extractors.values():
            extractor.close()

    if not rows:
        raise ValueError("no saliency artifacts were generated")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_csv, index=False)

    if args.skipped_csv is not None:
        args.skipped_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(skipped).to_csv(args.skipped_csv, index=False)

    run_meta = {
        "manifest_csv": str(args.manifest_csv),
        "image_root": str(args.image_root),
        "model_checkpoint": str(args.model_checkpoint),
        "arch": args.arch,
        "labels": labels,
        "methods": args.methods,
        "curve_steps": args.curve_steps,
        "nuisance_brightness": args.nuisance_brightness,
        "nuisance_contrast": args.nuisance_contrast,
        "sanity_mode": args.sanity_mode,
        "num_rows_written": len(rows),
        "num_rows_skipped": len(skipped),
    }
    if args.meta_json is not None:
        args.meta_json.parent.mkdir(parents=True, exist_ok=True)
        with args.meta_json.open("w", encoding="utf-8") as handle:
            json.dump(run_meta, handle, indent=2)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate E2/E3 saliency artifacts from a CNN checkpoint.")
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--model-checkpoint", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--skipped-csv", type=Path, default=None)
    parser.add_argument("--meta-json", type=Path, default=None)
    parser.add_argument("--save-cam-dir", type=Path, default=None)

    parser.add_argument("--arch", choices=["resnet18", "densenet121"], default="resnet18")
    parser.add_argument("--pretrained-backbone", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--input-size", type=int, default=320)
    parser.add_argument("--curve-steps", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--target-split", default="test")

    parser.add_argument("--study-col", default="study_id")
    parser.add_argument("--subject-col", default="subject_id")
    parser.add_argument("--dicom-col", default="dicom_id")
    parser.add_argument("--image-path-col", default="path")
    parser.add_argument("--split-col", default="split")
    parser.add_argument("--labels", nargs="+", default=DEFAULT_FINDINGS)
    parser.add_argument("--methods", nargs="+", choices=["gradcam", "hirescam"], default=["gradcam", "hirescam"])
    parser.add_argument("--nuisance-brightness", type=float, default=0.05)
    parser.add_argument("--nuisance-contrast", type=float, default=1.05)
    parser.add_argument("--sanity-mode", choices=["none", "head", "all"], default="head")
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    run_generation(args)


if __name__ == "__main__":
    main()
