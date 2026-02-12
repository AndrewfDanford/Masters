from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.audit import DEFAULT_FINDINGS
from src.models.e1_baseline import evaluate_predictions_by_split
from src.models.e1_extract_features import resolve_image_path


def _require_torchvision_stack():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from PIL import Image
        from torch.utils.data import DataLoader, Dataset
        from torchvision import models, transforms
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "E1 CNN training requires torch, torchvision, and Pillow."
        ) from exc
    return torch, nn, F, Image, DataLoader, Dataset, models, transforms


def _normalize_state_dict_keys(state_dict: dict[str, object]) -> dict[str, object]:
    normalized: dict[str, object] = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            normalized[key[len("module.") :]] = value
        else:
            normalized[key] = value
    return normalized


def _canonical_split_name(raw_name: object) -> str:
    normalized = str(raw_name).strip().lower()
    aliases = {
        "train": "train",
        "training": "train",
        "val": "val",
        "valid": "val",
        "validate": "val",
        "validation": "val",
        "test": "test",
        "testing": "test",
    }
    return aliases.get(normalized, normalized)


def label_value_to_target_mask(raw_value: object, uncertain_policy: str) -> tuple[float, float]:
    if uncertain_policy not in {"u_ignore", "u_zero", "u_one"}:
        raise ValueError("uncertain_policy must be one of: u_ignore, u_zero, u_one")

    numeric = pd.to_numeric(pd.Series([raw_value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return 0.0, 0.0

    numeric_value = float(numeric)
    if numeric_value == -1.0:
        if uncertain_policy == "u_ignore":
            return 0.0, 0.0
        if uncertain_policy == "u_zero":
            return 0.0, 1.0
        return 1.0, 1.0
    return (1.0 if numeric_value > 0 else 0.0), 1.0


def build_targets_from_row(row: pd.Series, labels: list[str], uncertain_policy: str) -> tuple[np.ndarray, np.ndarray]:
    targets: list[float] = []
    mask: list[float] = []
    for label in labels:
        target, known = label_value_to_target_mask(row.get(label), uncertain_policy=uncertain_policy)
        targets.append(target)
        mask.append(known)
    return np.asarray(targets, dtype=np.float32), np.asarray(mask, dtype=np.float32)


def compute_pos_weight(targets: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if targets.shape != mask.shape:
        raise ValueError("targets and mask must have the same shape")
    if targets.ndim != 2:
        raise ValueError("targets and mask must be 2D")

    known = mask > 0
    pos = np.sum((targets > 0) & known, axis=0).astype(float)
    neg = np.sum((targets <= 0) & known, axis=0).astype(float)
    return np.where(pos > 0, neg / np.maximum(pos, 1.0), 1.0).astype(np.float32)


def _build_model(
    arch: str,
    num_classes: int,
    pretrained_backbone: bool,
    checkpoint_path: Path | None,
):
    torch, nn, _F, _Image, _DataLoader, _Dataset, models, _transforms = _require_torchvision_stack()

    if arch == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained_backbone else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        head_attr = "fc"
    elif arch == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained_backbone else None
        model = models.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        head_attr = "classifier"
    else:
        raise ValueError("arch must be one of: resnet18, densenet121")

    if checkpoint_path is not None:
        checkpoint_obj = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(checkpoint_obj, dict) and "state_dict" in checkpoint_obj and isinstance(
            checkpoint_obj["state_dict"], dict
        ):
            state_dict = checkpoint_obj["state_dict"]
        elif isinstance(checkpoint_obj, dict):
            state_dict = checkpoint_obj
        else:
            raise ValueError("unsupported checkpoint format for init checkpoint")
        model.load_state_dict(_normalize_state_dict_keys(state_dict), strict=False)

    return model, head_attr


def _set_backbone_trainable(model, arch: str, trainable: bool) -> None:
    if arch == "resnet18":
        for name, param in model.named_parameters():
            if name.startswith("fc."):
                continue
            param.requires_grad = trainable
    elif arch == "densenet121":
        for name, param in model.named_parameters():
            if name.startswith("classifier."):
                continue
            param.requires_grad = trainable
    else:  # pragma: no cover - guarded by argparse
        raise ValueError(f"unsupported arch for freeze policy: {arch}")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch, _nn, _F, _Image, _DataLoader, _Dataset, _models, _transforms = _require_torchvision_stack()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ManifestImageDataset:
    def __init__(
        self,
        frame: pd.DataFrame,
        labels: list[str],
        uncertain_policy: str,
        image_root: Path,
        study_col: str,
        subject_col: str,
        dicom_col: str,
        image_path_col: str,
        split_col: str,
        transform,
    ) -> None:
        self.labels = labels
        self.study_col = study_col
        self.subject_col = subject_col
        self.dicom_col = dicom_col
        self.split_col = split_col
        self.transform = transform
        self.records: list[dict[str, object]] = []
        self.skipped: list[dict[str, object]] = []

        for _, row in frame.iterrows():
            resolved = resolve_image_path(
                row=row,
                image_root=image_root,
                image_path_col=image_path_col,
                study_col=study_col,
                subject_col=subject_col,
                dicom_col=dicom_col,
            )
            if resolved is None or not resolved.exists():
                self.skipped.append(
                    {
                        study_col: row.get(study_col),
                        subject_col: row.get(subject_col),
                        dicom_col: row.get(dicom_col),
                        "split": row.get(split_col),
                        "reason": "missing_image_file",
                        "resolved_image_path": str(resolved) if resolved is not None else "",
                    }
                )
                continue

            targets, mask = build_targets_from_row(row, labels=labels, uncertain_policy=uncertain_policy)
            self.records.append(
                {
                    study_col: row.get(study_col),
                    subject_col: row.get(subject_col),
                    dicom_col: row.get(dicom_col),
                    split_col: row.get(split_col),
                    "resolved_image_path": str(resolved),
                    "targets": targets,
                    "mask": mask,
                }
            )

        torch, _nn, _F, Image, _DataLoader, Dataset, _models, _transforms = _require_torchvision_stack()

        class _WrappedDataset(Dataset):
            def __init__(self, parent):
                self.parent = parent

            def __len__(self):
                return len(self.parent.records)

            def __getitem__(self, index):
                record = self.parent.records[index]
                with Image.open(Path(record["resolved_image_path"])) as image:
                    image = image.convert("RGB")
                    image_tensor = self.parent.transform(image)
                target_tensor = torch.tensor(record["targets"], dtype=torch.float32)
                mask_tensor = torch.tensor(record["mask"], dtype=torch.float32)
                return (
                    image_tensor,
                    target_tensor,
                    mask_tensor,
                    str(record[self.parent.study_col]),
                    str(record[self.parent.subject_col]),
                    str(record[self.parent.split_col]),
                    str(record["resolved_image_path"]),
                )

        self.dataset = _WrappedDataset(self)

    def __len__(self) -> int:
        return len(self.records)

    def target_mask_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.records:
            return np.zeros((0, len(self.labels)), dtype=np.float32), np.zeros((0, len(self.labels)), dtype=np.float32)
        targets = np.stack([record["targets"] for record in self.records], axis=0).astype(np.float32)
        mask = np.stack([record["mask"] for record in self.records], axis=0).astype(np.float32)
        return targets, mask


def _build_transforms(input_size: int, train_augment: bool):
    _torch, _nn, _F, _Image, _DataLoader, _Dataset, _models, transforms = _require_torchvision_stack()
    base = [
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    if not train_augment:
        transform = transforms.Compose(base)
        return transform, transform

    train_transform = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_transform = transforms.Compose(base)
    return train_transform, eval_transform


def _collect_predictions(model, dataset: ManifestImageDataset, batch_size: int, num_workers: int, device: str):
    torch, _nn, _F, _Image, DataLoader, _Dataset, _models, _transforms = _require_torchvision_stack()
    loader = DataLoader(
        dataset.dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.startswith("cuda")),
    )
    probs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    meta_rows: list[dict[str, object]] = []

    model.eval()
    with torch.no_grad():
        for images, target_tensor, mask_tensor, study_ids, subject_ids, splits, image_paths in loader:
            images = images.to(device)
            logits = model(images)
            batch_probs = torch.sigmoid(logits).cpu().numpy()
            probs.append(batch_probs)
            targets.append(target_tensor.cpu().numpy())
            masks.append(mask_tensor.cpu().numpy())
            for idx in range(len(study_ids)):
                meta_rows.append(
                    {
                        "study_id": study_ids[idx],
                        "subject_id": subject_ids[idx],
                        "split": splits[idx],
                        "resolved_image_path": image_paths[idx],
                    }
                )

    if not probs:
        return np.zeros((0, len(dataset.labels)), dtype=np.float32), np.zeros((0, len(dataset.labels)), dtype=np.float32), np.zeros((0, len(dataset.labels)), dtype=np.float32), []

    return (
        np.vstack(probs).astype(np.float32),
        np.vstack(targets).astype(np.float32),
        np.vstack(masks).astype(np.float32),
        meta_rows,
    )


def run_train(args: argparse.Namespace) -> None:
    torch, _nn, F, _Image, DataLoader, _Dataset, _models, _transforms = _require_torchvision_stack()
    _seed_everything(args.seed)

    cohort_df = pd.read_csv(args.cohort_csv)
    required_cols = [args.study_col, args.subject_col, args.split_col] + args.labels
    missing = [col for col in required_cols if col not in cohort_df.columns]
    if missing:
        raise ValueError(f"cohort CSV missing required columns: {', '.join(missing)}")
    cohort_df = cohort_df.copy()
    cohort_df[args.split_col] = cohort_df[args.split_col].map(_canonical_split_name)

    image_root = Path(args.image_root)
    if not image_root.exists():
        raise ValueError(f"image root does not exist: {image_root}")

    if args.max_samples_per_split is not None and args.max_samples_per_split <= 0:
        raise ValueError("max_samples_per_split must be positive")

    labels = list(args.labels)
    train_transform, eval_transform = _build_transforms(args.input_size, train_augment=args.train_augment)

    split_frames: dict[str, pd.DataFrame] = {}
    for split_name in ["train", "val", "test"]:
        split_frame = cohort_df[cohort_df[args.split_col].astype(str) == split_name].copy()
        if args.max_samples_per_split is not None:
            split_frame = split_frame.head(args.max_samples_per_split).copy()
        split_frames[split_name] = split_frame

    datasets: dict[str, ManifestImageDataset] = {
        "train": ManifestImageDataset(
            frame=split_frames["train"],
            labels=labels,
            uncertain_policy=args.uncertain_policy,
            image_root=image_root,
            study_col=args.study_col,
            subject_col=args.subject_col,
            dicom_col=args.dicom_col,
            image_path_col=args.image_path_col,
            split_col=args.split_col,
            transform=train_transform,
        ),
        "val": ManifestImageDataset(
            frame=split_frames["val"],
            labels=labels,
            uncertain_policy=args.uncertain_policy,
            image_root=image_root,
            study_col=args.study_col,
            subject_col=args.subject_col,
            dicom_col=args.dicom_col,
            image_path_col=args.image_path_col,
            split_col=args.split_col,
            transform=eval_transform,
        ),
        "test": ManifestImageDataset(
            frame=split_frames["test"],
            labels=labels,
            uncertain_policy=args.uncertain_policy,
            image_root=image_root,
            study_col=args.study_col,
            subject_col=args.subject_col,
            dicom_col=args.dicom_col,
            image_path_col=args.image_path_col,
            split_col=args.split_col,
            transform=eval_transform,
        ),
    }

    if len(datasets["train"]) == 0:
        raise ValueError("train split has no usable images")

    model, _head_attr = _build_model(
        arch=args.arch,
        num_classes=len(labels),
        pretrained_backbone=args.pretrained_backbone,
        checkpoint_path=args.init_checkpoint,
    )

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    model = model.to(device)

    train_loader = DataLoader(
        datasets["train"].dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.startswith("cuda")),
    )

    train_targets, train_masks = datasets["train"].target_mask_arrays()
    if args.no_class_balance:
        pos_weight_np = np.ones(len(labels), dtype=np.float32)
    else:
        pos_weight_np = compute_pos_weight(train_targets, train_masks)
    pos_weight = torch.tensor(pos_weight_np, dtype=torch.float32, device=device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    history_rows: list[dict[str, object]] = []
    for epoch in range(1, args.epochs + 1):
        backbone_trainable = epoch > args.freeze_backbone_epochs
        _set_backbone_trainable(model, arch=args.arch, trainable=backbone_trainable)
        model.train()

        running_loss = 0.0
        running_weight = 0.0
        num_steps = 0

        for images, target_tensor, mask_tensor, _study_ids, _subject_ids, _splits, _paths in train_loader:
            images = images.to(device, non_blocking=True)
            targets = target_tensor.to(device, non_blocking=True)
            masks = mask_tensor.to(device, non_blocking=True)

            logits = model(images)
            loss_matrix = F.binary_cross_entropy_with_logits(
                logits,
                targets,
                reduction="none",
                pos_weight=pos_weight,
            )
            valid_weight = masks.sum().clamp_min(1.0)
            loss = (loss_matrix * masks).sum() / valid_weight

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * float(valid_weight.item())
            running_weight += float(valid_weight.item())
            num_steps += 1

        epoch_loss = running_loss / max(running_weight, 1.0)
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": epoch_loss,
                "train_steps": num_steps,
                "backbone_trainable": backbone_trainable,
            }
        )

    probs_parts: list[np.ndarray] = []
    targets_parts: list[np.ndarray] = []
    masks_parts: list[np.ndarray] = []
    split_parts: list[np.ndarray] = []
    prediction_rows: list[dict[str, object]] = []

    skipped_rows: list[dict[str, object]] = []
    for split_name, dataset in datasets.items():
        skipped_rows.extend(dataset.skipped)
        probs, targets, masks, meta_rows = _collect_predictions(
            model=model,
            dataset=dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )
        if probs.shape[0] == 0:
            continue
        probs_parts.append(probs)
        targets_parts.append(targets)
        masks_parts.append(masks)
        split_parts.append(np.asarray([split_name] * probs.shape[0], dtype=object))

        for row_idx, meta in enumerate(meta_rows):
            for label_idx, label in enumerate(labels):
                known = bool(masks[row_idx, label_idx] > 0)
                prediction_rows.append(
                    {
                        args.study_col: meta["study_id"],
                        args.subject_col: meta["subject_id"],
                        args.split_col: split_name,
                        "label": label,
                        "known": known,
                        "target": float(targets[row_idx, label_idx]) if known else float("nan"),
                        "probability": float(probs[row_idx, label_idx]),
                        "resolved_image_path": meta["resolved_image_path"],
                    }
                )

    if not probs_parts:
        raise ValueError("no predictions collected; verify dataset rows and image availability")

    probs_all = np.vstack(probs_parts)
    targets_all = np.vstack(targets_parts)
    masks_all = np.vstack(masks_parts)
    splits_all = np.concatenate(split_parts, axis=0)

    metrics_by_label_df, metrics_summary_df = evaluate_predictions_by_split(
        split=splits_all,
        labels=labels,
        targets=targets_all,
        mask=(masks_all > 0),
        probs=probs_all,
        ece_bins=args.ece_bins,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_out = args.checkpoint_out or (args.output_dir / "e1_cnn_checkpoint.pt")
    checkpoint_out.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "labels": labels,
            "arch": args.arch,
            "input_size": args.input_size,
            "uncertain_policy": args.uncertain_policy,
        },
        checkpoint_out,
    )

    pd.DataFrame(history_rows).to_csv(args.output_dir / "e1_cnn_train_history.csv", index=False)
    metrics_by_label_df.to_csv(args.output_dir / "e1_cnn_metrics_by_label.csv", index=False)
    metrics_summary_df.to_csv(args.output_dir / "e1_cnn_metrics_summary.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(args.output_dir / "e1_cnn_predictions_long.csv", index=False)
    pd.DataFrame(skipped_rows).to_csv(args.output_dir / "e1_cnn_skipped_images.csv", index=False)

    dataset_count_rows = []
    for split_name, dataset in datasets.items():
        dataset_count_rows.append(
            {
                "split": split_name,
                "num_requested_rows": int(split_frames[split_name].shape[0]),
                "num_usable_images": int(len(dataset)),
                "num_skipped_images": int(len(dataset.skipped)),
            }
        )
    pd.DataFrame(dataset_count_rows).to_csv(args.output_dir / "e1_cnn_dataset_counts.csv", index=False)

    run_meta = {
        "cohort_csv": str(args.cohort_csv),
        "image_root": str(args.image_root),
        "output_dir": str(args.output_dir),
        "checkpoint_out": str(checkpoint_out),
        "arch": args.arch,
        "labels": labels,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "input_size": args.input_size,
        "uncertain_policy": args.uncertain_policy,
        "freeze_backbone_epochs": args.freeze_backbone_epochs,
        "class_balance": not args.no_class_balance,
        "num_train_usable": int(len(datasets["train"])),
        "num_val_usable": int(len(datasets["val"])),
        "num_test_usable": int(len(datasets["test"])),
    }
    with (args.output_dir / "e1_cnn_run_meta.json").open("w", encoding="utf-8") as handle:
        json.dump(run_meta, handle, indent=2)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train E1 baseline CNN directly from image files.")
    parser.add_argument("--cohort-csv", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-out", type=Path, default=None)
    parser.add_argument("--init-checkpoint", type=Path, default=None)

    parser.add_argument("--study-col", default="study_id")
    parser.add_argument("--subject-col", default="subject_id")
    parser.add_argument("--dicom-col", default="dicom_id")
    parser.add_argument("--split-col", default="split")
    parser.add_argument("--image-path-col", default="path")
    parser.add_argument("--labels", nargs="+", default=DEFAULT_FINDINGS)

    parser.add_argument("--uncertain-policy", choices=["u_ignore", "u_zero", "u_one"], default="u_ignore")
    parser.add_argument("--arch", choices=["resnet18", "densenet121"], default="resnet18")
    parser.add_argument("--pretrained-backbone", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--input-size", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--ece-bins", type=int, default=10)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=0)
    parser.add_argument("--max-samples-per-split", type=int, default=None)
    parser.add_argument("--train-augment", action="store_true")
    parser.add_argument("--no-class-balance", action="store_true")
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    run_train(args)


if __name__ == "__main__":
    main()
