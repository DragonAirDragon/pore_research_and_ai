"""Prepare train/val/test datasets from manually annotated pore images."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class AnnotatedSample:
    sample_id: str
    sample_dir: Path
    original_path: Path
    mask_path: Path
    distance_map_path: Path | None


def load_grayscale(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return image


def compute_distance_map(mask: np.ndarray) -> np.ndarray:
    pore_mask = (mask > 127).astype(np.uint8) * 255
    return cv2.distanceTransform(pore_mask, cv2.DIST_L2, 5).astype(np.uint8)


def discover_samples(input_dir: Path) -> list[AnnotatedSample]:
    samples: list[AnnotatedSample] = []

    for sample_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        original_path = sample_dir / "original.png"
        mask_path = sample_dir / "mask.png"
        distance_map_path = sample_dir / "distance_map.png"

        if not original_path.exists() or not mask_path.exists():
            continue

        samples.append(
            AnnotatedSample(
                sample_id=sample_dir.name,
                sample_dir=sample_dir,
                original_path=original_path,
                mask_path=mask_path,
                distance_map_path=distance_map_path if distance_map_path.exists() else None,
            )
        )

    return samples


def split_samples(
    samples: list[AnnotatedSample],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[AnnotatedSample]]:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)

    ratios = {"train": train_ratio, "val": val_ratio, "test": test_ratio}
    non_zero_splits = [name for name, ratio in ratios.items() if ratio > 0]
    if total < len(non_zero_splits):
        raise ValueError(
            f"Not enough samples ({total}) to create non-empty splits for {', '.join(non_zero_splits)}"
        )

    raw_counts = {name: total * ratio for name, ratio in ratios.items()}
    counts = {name: int(raw_counts[name]) for name in ratios}

    for name in non_zero_splits:
        if counts[name] == 0:
            counts[name] = 1

    while sum(counts.values()) > total:
        reducible = [name for name in counts if counts[name] > (1 if ratios[name] > 0 else 0)]
        if not reducible:
            break
        name_to_reduce = max(reducible, key=lambda name: counts[name] - raw_counts[name])
        counts[name_to_reduce] -= 1

    while sum(counts.values()) < total:
        name_to_increase = max(ratios, key=lambda name: raw_counts[name] - counts[name])
        counts[name_to_increase] += 1

    train_end = counts["train"]
    val_end = train_end + counts["val"]

    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def copy_sample(output_dir: Path, sample_name: str, original: np.ndarray, mask: np.ndarray, distance_map: np.ndarray) -> None:
    sample_dir = output_dir / sample_name
    sample_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(sample_dir / "original.png"), original)
    cv2.imwrite(str(sample_dir / "mask.png"), mask)
    cv2.imwrite(str(sample_dir / "distance_map.png"), distance_map)


def geometric_transforms() -> list[tuple[str, callable]]:
    return [
        ("rot0", lambda x: x),
        ("rot90", lambda x: cv2.rotate(x, cv2.ROTATE_90_CLOCKWISE)),
        ("rot180", lambda x: cv2.rotate(x, cv2.ROTATE_180)),
        ("rot270", lambda x: cv2.rotate(x, cv2.ROTATE_90_COUNTERCLOCKWISE)),
        ("flipH", lambda x: cv2.flip(x, 1)),
        ("flipV", lambda x: cv2.flip(x, 0)),
    ]


def intensity_transforms() -> list[tuple[str, callable]]:
    def add_noise(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            noise = np.random.normal(0, 12, image.shape)
        else:
            noise = np.random.normal(0, 12, image.shape)
        noisy = image.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def adjust_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]).astype(np.uint8)
        return cv2.LUT(image, table)

    return [
        ("orig", lambda x: x),
        ("bright", lambda x: cv2.convertScaleAbs(x, alpha=1.0, beta=20)),
        ("dark", lambda x: cv2.convertScaleAbs(x, alpha=1.0, beta=-20)),
        ("contrast_hi", lambda x: cv2.convertScaleAbs(x, alpha=1.2, beta=0)),
        ("contrast_lo", lambda x: cv2.convertScaleAbs(x, alpha=0.8, beta=0)),
        ("blur", lambda x: cv2.GaussianBlur(x, (3, 3), 0)),
        ("noise", add_noise),
        ("gamma", lambda x: adjust_gamma(x, gamma=1.4)),
    ]


def augment_and_save_train_sample(sample: AnnotatedSample, train_dir: Path) -> list[str]:
    saved_names: list[str] = []
    original = load_grayscale(sample.original_path)
    mask = load_grayscale(sample.mask_path)
    distance_map = load_grayscale(sample.distance_map_path) if sample.distance_map_path else compute_distance_map(mask)

    for geo_name, geo_fn in geometric_transforms():
        geo_original = geo_fn(original)
        geo_mask = geo_fn(mask)
        geo_distance = geo_fn(distance_map)

        for intensity_name, intensity_fn in intensity_transforms():
            augmented_original = intensity_fn(geo_original)
            sample_name = f"{sample.sample_id}_{geo_name}_{intensity_name}"
            copy_sample(train_dir, sample_name, augmented_original, geo_mask, geo_distance)
            saved_names.append(sample_name)

    return saved_names


def save_eval_sample(sample: AnnotatedSample, split_dir: Path) -> str:
    original = load_grayscale(sample.original_path)
    mask = load_grayscale(sample.mask_path)
    distance_map = load_grayscale(sample.distance_map_path) if sample.distance_map_path else compute_distance_map(mask)
    copy_sample(split_dir, sample.sample_id, original, mask, distance_map)
    return sample.sample_id


def create_robustness_test(sample: AnnotatedSample, split_dir: Path) -> list[str]:
    saved_names: list[str] = []
    original = load_grayscale(sample.original_path)
    mask = load_grayscale(sample.mask_path)
    distance_map = load_grayscale(sample.distance_map_path) if sample.distance_map_path else compute_distance_map(mask)

    for intensity_name, intensity_fn in intensity_transforms():
        sample_name = f"{sample.sample_id}_{intensity_name}"
        transformed_original = intensity_fn(original)
        copy_sample(split_dir, sample_name, transformed_original, mask, distance_map)
        saved_names.append(sample_name)

    return saved_names


def prepare_dataset(
    input_dir: Path,
    output_dir: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    create_test_robustness_split: bool,
) -> dict:
    samples = discover_samples(input_dir)
    if not samples:
        raise RuntimeError(f"No annotated samples found in {input_dir}")

    splits = split_samples(samples, train_ratio, val_ratio, test_ratio, seed)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "source_dir": str(input_dir),
        "seed": seed,
        "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "source_samples": {split_name: [sample.sample_id for sample in split_samples_] for split_name, split_samples_ in splits.items()},
        "generated": {},
    }

    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    test_dir = output_dir / "test"

    metadata["generated"]["train"] = []
    for sample in splits["train"]:
        metadata["generated"]["train"].extend(augment_and_save_train_sample(sample, train_dir))

    metadata["generated"]["val"] = [save_eval_sample(sample, val_dir) for sample in splits["val"]]
    metadata["generated"]["test"] = [save_eval_sample(sample, test_dir) for sample in splits["test"]]

    if create_test_robustness_split:
        robustness_dir = output_dir / "test_robust"
        metadata["generated"]["test_robust"] = []
        for sample in splits["test"]:
            metadata["generated"]["test_robust"].extend(create_robustness_test(sample, robustness_dir))

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2, ensure_ascii=False)

    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare datasets from manually annotated pore images")
    parser.add_argument("--input", type=str, default="RealPoresImages/dataset_manual", help="Directory with manually annotated samples")
    parser.add_argument("--output", type=str, default="dataset_manual_prepared", help="Output dataset directory")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--with-test-robustness",
        action="store_true",
        help="Create an additional held-out stress-test split with photometric perturbations",
    )

    args = parser.parse_args()

    metadata = prepare_dataset(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        create_test_robustness_split=args.with_test_robustness,
    )

    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"Prepared dataset saved to {args.output}")


if __name__ == "__main__":
    main()