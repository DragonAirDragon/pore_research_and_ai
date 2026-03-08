"""Training script for Regression UNet (distance map prediction)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from models.regression.model import RegressionUNet


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class RegressionDataset(Dataset):
    """Dataset for distance-map regression from either metadata or annotator-style folders."""

    def __init__(self, samples: list[tuple[Path, Path]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, distance_path = self.samples[idx]
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        distance = cv2.imread(str(distance_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        if distance is None:
            raise FileNotFoundError(f"Could not load distance map: {distance_path}")

        image_tensor = torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0)
        distance_tensor = torch.from_numpy(distance.astype(np.float32)).unsqueeze(0)
        return image_tensor, distance_tensor


def load_samples_from_metadata(dataset_dir: Path, split: str) -> list[tuple[Path, Path]]:
    metadata_path = dataset_dir / "metadata.json"
    with open(metadata_path, "r", encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)

    split_files = metadata.get("files", {}).get(split, [])
    samples: list[tuple[Path, Path]] = []
    for item in split_files:
        image_name = item.get("noisy") or item.get("original")
        distance_name = item.get("distance") or item.get("distance_map")

        if not image_name or not distance_name:
            continue

        noisy_dir = dataset_dir / split / "noisy"
        if noisy_dir.exists():
            image_path = noisy_dir / image_name
            distance_path = dataset_dir / split / "distance" / distance_name
        else:
            sample_id = item["id"]
            image_path = dataset_dir / split / sample_id / image_name
            distance_path = dataset_dir / split / sample_id / distance_name
        samples.append((image_path, distance_path))

    return samples


def load_samples_from_annotator_split(dataset_dir: Path, split: str) -> list[tuple[Path, Path]]:
    split_dir = dataset_dir / split
    samples: list[tuple[Path, Path]] = []
    if not split_dir.exists():
        return samples

    for sample_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
        image_path = sample_dir / "original.png"
        distance_path = sample_dir / "distance_map.png"
        if image_path.exists() and distance_path.exists():
            samples.append((image_path, distance_path))

    return samples


def load_split_samples(dataset_dir: Path, split: str) -> list[tuple[Path, Path]]:
    metadata_path = dataset_dir / "metadata.json"
    if metadata_path.exists():
        samples = load_samples_from_metadata(dataset_dir, split)
        if samples:
            return samples

    samples = load_samples_from_annotator_split(dataset_dir, split)
    if samples:
        return samples

    raise RuntimeError(f"Could not resolve split '{split}' in dataset {dataset_dir}")


def load_checkpoint_state_dict(checkpoint_path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def evaluate_model(model: nn.Module, loader: DataLoader | None, criterion: nn.Module, device: torch.device) -> float:
    if loader is None:
        return float("nan")

    model.eval()
    losses = []
    with torch.no_grad():
        for images, distances in loader:
            images = images.to(device)
            distances = distances.to(device)
            predictions = model(images)
            loss = criterion(predictions, distances)
            losses.append(loss.item())

    if not losses:
        return float("nan")
    return float(np.mean(losses))


def plot_history(history: dict[str, list[float]], output_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="val")
    if history.get("test_loss") and any(not np.isnan(value) for value in history["test_loss"]):
        plt.plot(history["test_loss"], label="test")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Regression training history")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "training_history.png", dpi=150)
    plt.close()


def train(args: argparse.Namespace) -> None:
    dataset_dir = Path(args.dataset)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_samples = load_split_samples(dataset_dir, "train")
    val_samples = load_split_samples(dataset_dir, "val")
    test_samples = load_split_samples(dataset_dir, "test") if (dataset_dir / "test").exists() else []

    print(f"Dataset: {dataset_dir}")
    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")
    print(f"Test samples: {len(test_samples)}")

    pin_memory = args.pin_memory and get_device().type == "cuda"
    train_loader = DataLoader(
        RegressionDataset(train_samples),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        RegressionDataset(val_samples),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = None
    if test_samples:
        test_loader = DataLoader(
            RegressionDataset(test_samples),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )

    device = get_device()
    model = RegressionUNet().to(device)
    if args.init_model:
        init_model_path = Path(args.init_model)
        model.load_state_dict(load_checkpoint_state_dict(init_model_path, device))
        print(f"Initialized model from {init_model_path}")
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = GradScaler("cuda") if device.type == "cuda" and args.use_amp else None

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "test_loss": []}
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        batch_losses = []
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

        for images, distances in progress:
            images = images.to(device)
            distances = distances.to(device)

            optimizer.zero_grad()
            if scaler is not None:
                with autocast("cuda"):
                    predictions = model(images)
                    loss = criterion(predictions, distances)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                predictions = model(images)
                loss = criterion(predictions, distances)
                loss.backward()
                optimizer.step()

            batch_losses.append(loss.item())
            progress.set_postfix(train_loss=loss.item())

        train_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
        val_loss = evaluate_model(model, val_loader, criterion, device)
        test_loss = evaluate_model(model, test_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["test_loss"].append(test_loss)

        if np.isnan(test_loss):
            print(f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}")
        else:
            print(f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}, test={test_loss:.4f}")

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "test_loss": test_loss,
            "history": history,
            "dataset": str(dataset_dir),
        }
        torch.save(checkpoint, checkpoint_dir / "last_model.pth")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, checkpoint_dir / "best_model.pth")
            print("Saved new best model")

    with open(checkpoint_dir / "history.json", "w", encoding="utf-8") as history_file:
        json.dump(history, history_file, indent=2)

    plot_history(history, checkpoint_dir)
    print(f"Training complete. Best val loss: {best_val_loss:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the regression UNet on split datasets")
    parser.add_argument("--dataset", type=str, default="dataset_regression", help="Path to dataset root")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="artifacts/checkpoints/regression/synthetic",
        help="Directory to save checkpoints",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--init-model", type=str, default=None, help="Optional checkpoint to warm-start from")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--use-amp", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
