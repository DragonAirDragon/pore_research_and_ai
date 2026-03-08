import json
from pathlib import Path

import cv2
import numpy as np

from models.regression.train import load_split_samples
from scripts.data_prep.prepare_manual_dataset import prepare_dataset


def create_sample(sample_dir: Path, radius: int) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    original = np.full((64, 64), 180, dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)
    cv2.circle(mask, (32, 32), radius, 255, -1)
    original[mask > 0] = 60

    cv2.imwrite(str(sample_dir / "original.png"), original)
    cv2.imwrite(str(sample_dir / "mask.png"), mask)


def test_prepare_manual_dataset_creates_split_structure(tmp_path: Path):
    input_dir = tmp_path / "input"
    for idx, radius in enumerate((5, 7, 9, 11), start=1):
        create_sample(input_dir / str(idx), radius)

    output_dir = tmp_path / "prepared"
    metadata = prepare_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        train_ratio=0.5,
        val_ratio=0.25,
        test_ratio=0.25,
        seed=123,
        create_test_robustness_split=True,
    )

    assert (output_dir / "train").exists()
    assert (output_dir / "val").exists()
    assert (output_dir / "test").exists()
    assert (output_dir / "test_robust").exists()

    assert len(metadata["source_samples"]["train"]) == 2
    assert len(metadata["source_samples"]["val"]) == 1
    assert len(metadata["source_samples"]["test"]) == 1
    assert len(metadata["generated"]["train"]) == 96
    assert len(metadata["generated"]["val"]) == 1
    assert len(metadata["generated"]["test"]) == 1
    assert len(metadata["generated"]["test_robust"]) == 8

    saved_metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert saved_metadata["generated"]["test"] == metadata["generated"]["test"]


def test_load_split_samples_reads_annotator_style_dataset(tmp_path: Path):
    dataset_dir = tmp_path / "dataset"
    create_sample(dataset_dir / "train" / "sample_a", 6)
    create_sample(dataset_dir / "val" / "sample_b", 8)
    create_sample(dataset_dir / "test" / "sample_c", 10)

    for split, sample_name in (("train", "sample_a"), ("val", "sample_b"), ("test", "sample_c")):
        mask = cv2.imread(str(dataset_dir / split / sample_name / "mask.png"), cv2.IMREAD_GRAYSCALE)
        distance_map = cv2.distanceTransform((mask > 127).astype(np.uint8) * 255, cv2.DIST_L2, 5).astype(np.uint8)
        cv2.imwrite(str(dataset_dir / split / sample_name / "distance_map.png"), distance_map)

    train_samples = load_split_samples(dataset_dir, "train")
    val_samples = load_split_samples(dataset_dir, "val")
    test_samples = load_split_samples(dataset_dir, "test")

    assert len(train_samples) == 1
    assert len(val_samples) == 1
    assert len(test_samples) == 1
    assert train_samples[0][0].name == "original.png"
    assert train_samples[0][1].name == "distance_map.png"