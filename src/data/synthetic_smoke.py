from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .audit import DEFAULT_FINDINGS


METHOD_PROFILES: dict[str, dict[str, float | None]] = {
    "gradcam": {"sanity_mu": 0.38, "nuisance_mu": 0.76, "quality": 0.62, "text_contradiction": None},
    "hirescam": {"sanity_mu": 0.28, "nuisance_mu": 0.82, "quality": 0.68, "text_contradiction": None},
    "concept_cbm": {"sanity_mu": 0.22, "nuisance_mu": 0.80, "quality": 0.70, "text_contradiction": None},
    "text_constrained": {"sanity_mu": 0.30, "nuisance_mu": 0.74, "quality": 0.61, "text_contradiction": 0.12},
    "text_unconstrained": {"sanity_mu": 0.48, "nuisance_mu": 0.66, "quality": 0.52, "text_contradiction": 0.26},
}


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _format_curve(values: np.ndarray) -> str:
    return ",".join(f"{float(v):.6f}" for v in values.reshape(-1))


def _monotonic_nonincreasing(values: np.ndarray) -> np.ndarray:
    out = values.copy()
    for i in range(1, out.size):
        out[i] = min(out[i - 1], out[i])
    return out


def _monotonic_nondecreasing(values: np.ndarray) -> np.ndarray:
    out = values.copy()
    for i in range(1, out.size):
        out[i] = max(out[i - 1], out[i])
    return out


def _build_curves(base_prediction: float, quality: float, rng: np.random.Generator) -> tuple[str, str]:
    x = np.linspace(0.0, 1.0, 5)
    deletion = base_prediction * (1.0 - quality * x) + rng.normal(0.0, 0.02, size=x.shape[0])
    insertion = (base_prediction * 0.22) + (quality * x * 0.7) + rng.normal(0.0, 0.02, size=x.shape[0])
    deletion = _monotonic_nonincreasing(np.clip(deletion, 0.0, 1.0))
    insertion = _monotonic_nondecreasing(np.clip(insertion, 0.0, 1.0))
    return _format_curve(deletion), _format_curve(insertion)


def _draw_synthetic_image(image_size: int, label_map: dict[str, float], rng: np.random.Generator) -> np.ndarray:
    image = rng.normal(loc=0.40, scale=0.08, size=(image_size, image_size))
    image = np.clip(image, 0.0, 1.0)

    # Mild vignetting to mimic radiography intensity falloff.
    yy, xx = np.mgrid[0:image_size, 0:image_size]
    center = (image_size - 1) / 2.0
    distance = np.sqrt((xx - center) ** 2 + (yy - center) ** 2)
    image -= 0.07 * (distance / (np.sqrt(2.0) * center))

    if label_map.get("Edema", 0.0) > 0:
        image[image_size // 3 : (2 * image_size) // 3, image_size // 3 : (2 * image_size) // 3] += 0.12
    if label_map.get("Consolidation", 0.0) > 0:
        image[(2 * image_size) // 3 :, image_size // 4 : (3 * image_size) // 4] += 0.15
    if label_map.get("Pleural Effusion", 0.0) > 0:
        image[(3 * image_size) // 4 :, :] += 0.10
    if label_map.get("Pneumothorax", 0.0) > 0:
        image[: image_size // 3, (2 * image_size) // 3 :] -= 0.12
    if label_map.get("Cardiomegaly", 0.0) > 0:
        image[image_size // 3 : (2 * image_size) // 3, image_size // 2 - image_size // 10 : image_size // 2 + image_size // 10] += 0.10
    if label_map.get("Atelectasis", 0.0) > 0:
        image[image_size // 2 :, : image_size // 3] += 0.08

    image = np.clip(image, 0.0, 1.0)
    rgb = np.stack([image, image, image], axis=-1)
    return (rgb * 255.0).astype(np.uint8)


def build_synthetic_bundle(
    output_dir: Path,
    num_train: int,
    num_val: int,
    num_test: int,
    image_size: int,
    seed: int,
    e8_runs: int,
    labels: list[str] | None = None,
) -> dict[str, str]:
    if num_train <= 0 or num_val <= 0 or num_test <= 0:
        raise ValueError("num_train, num_val, and num_test must all be positive")
    if image_size < 32:
        raise ValueError("image_size must be at least 32")
    if e8_runs <= 0:
        raise ValueError("e8_runs must be positive")

    labels = list(labels or DEFAULT_FINDINGS)
    rng = np.random.default_rng(seed)

    image_root = output_dir / "images"
    image_root.mkdir(parents=True, exist_ok=True)

    split_sequence = (["train"] * num_train) + (["val"] * num_val) + (["test"] * num_test)
    cohort_rows: list[dict[str, object]] = []

    for index, split_name in enumerate(split_sequence):
        subject_id = 100000 + (index // 2)
        study_id = 500000 + index
        dicom_id = f"synthetic_{index:06d}"
        relative_path = Path(split_name) / f"{study_id}_{dicom_id}.png"
        absolute_path = image_root / relative_path
        absolute_path.parent.mkdir(parents=True, exist_ok=True)

        label_map: dict[str, float] = {}
        for label in labels:
            label_state = rng.choice([0.0, 1.0, -1.0], p=[0.56, 0.34, 0.10])
            label_map[label] = float(label_state)

        image = _draw_synthetic_image(image_size=image_size, label_map=label_map, rng=rng)
        Image.fromarray(image, mode="RGB").save(absolute_path)

        row = {
            "subject_id": subject_id,
            "study_id": study_id,
            "dicom_id": dicom_id,
            "split": split_name,
            "path": str(relative_path),
        }
        row.update(label_map)
        cohort_rows.append(row)

    cohort_df = pd.DataFrame(cohort_rows)

    artifact_rows: list[dict[str, object]] = []
    for _, sample in cohort_df[cohort_df["split"] == "test"].iterrows():
        positive_count = int(sum(float(sample[label]) > 0 for label in labels))
        base_prediction = _clip01(0.40 + (0.10 * positive_count) + rng.normal(0.0, 0.03))

        for method_name, profile in METHOD_PROFILES.items():
            sanity_mu = float(profile["sanity_mu"])
            nuisance_mu = float(profile["nuisance_mu"])
            quality = float(profile["quality"])

            sanity_similarity = _clip01(sanity_mu + rng.normal(0.0, 0.03))
            nuisance_similarity = _clip01(nuisance_mu + rng.normal(0.0, 0.04))
            perturbed_prediction = _clip01(base_prediction + rng.normal(0.0, 0.03))
            deletion_curve, insertion_curve = _build_curves(
                base_prediction=base_prediction,
                quality=quality,
                rng=rng,
            )
            contradiction_base = profile["text_contradiction"]
            text_contradiction = (
                _clip01(float(contradiction_base) + rng.normal(0.0, 0.03))
                if contradiction_base is not None
                else np.nan
            )

            artifact_rows.append(
                {
                    "method": method_name,
                    "study_id": sample["study_id"],
                    "sanity_similarity": sanity_similarity,
                    "deletion_curve": deletion_curve,
                    "insertion_curve": insertion_curve,
                    "base_prediction": base_prediction,
                    "perturbed_prediction": perturbed_prediction,
                    "nuisance_similarity": nuisance_similarity,
                    "text_contradiction": text_contradiction,
                }
            )

    artifact_df = pd.DataFrame(artifact_rows)

    e8_rows: list[dict[str, object]] = []
    for _, artifact in artifact_df.iterrows():
        method_name = str(artifact["method"])
        sanity_mu = float(METHOD_PROFILES[method_name]["sanity_mu"])
        for run_id in range(1, e8_runs + 1):
            e8_rows.append(
                {
                    "method": method_name,
                    "study_id": artifact["study_id"],
                    "run_id": run_id,
                    "sanity_similarity": _clip01(sanity_mu + rng.normal(0.0, 0.05)),
                }
            )
    e8_df = pd.DataFrame(e8_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    cohort_csv = output_dir / "synthetic_cohort.csv"
    artifact_csv = output_dir / "synthetic_e23_artifacts.csv"
    e8_csv = output_dir / "synthetic_e8_randomization.csv"
    manifest_json = output_dir / "synthetic_manifest.json"

    cohort_df.to_csv(cohort_csv, index=False)
    artifact_df.to_csv(artifact_csv, index=False)
    e8_df.to_csv(e8_csv, index=False)

    manifest = {
        "image_root": str(image_root),
        "cohort_csv": str(cohort_csv),
        "e23_artifact_csv": str(artifact_csv),
        "e8_randomization_csv": str(e8_csv),
        "num_train": int(num_train),
        "num_val": int(num_val),
        "num_test": int(num_test),
        "image_size": int(image_size),
        "seed": int(seed),
        "e8_runs": int(e8_runs),
        "labels": labels,
    }
    with manifest_json.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    return {
        "image_root": str(image_root),
        "cohort_csv": str(cohort_csv),
        "e23_artifact_csv": str(artifact_csv),
        "e8_randomization_csv": str(e8_csv),
        "manifest_json": str(manifest_json),
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate synthetic artifacts for smoke-testing the thesis pipeline.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-train", type=int, default=24)
    parser.add_argument("--num-val", type=int, default=8)
    parser.add_argument("--num-test", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--e8-runs", type=int, default=8)
    parser.add_argument("--labels", nargs="+", default=DEFAULT_FINDINGS)
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    outputs = build_synthetic_bundle(
        output_dir=args.output_dir,
        num_train=args.num_train,
        num_val=args.num_val,
        num_test=args.num_test,
        image_size=args.image_size,
        seed=args.seed,
        e8_runs=args.e8_runs,
        labels=list(args.labels),
    )
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
