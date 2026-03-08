"""Evaluate segmentation or regression pore models on manually annotated real images."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy import ndimage


ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from models.regression.model import RegressionUNet
from models.segmentation.train import UNetWithAttention


def discover_samples(dataset_dir: Path) -> list[dict[str, Path]]:
    """Collect annotated samples saved by the GUI annotator."""
    samples: list[dict[str, Path]] = []

    for sample_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
        original_path = sample_dir / "original.png"
        mask_path = sample_dir / "mask.png"
        distance_path = sample_dir / "distance_map.png"

        if not original_path.exists() or not mask_path.exists():
            continue

        samples.append(
            {
                "id": sample_dir.name,
                "dir": sample_dir,
                "original": original_path,
                "mask": mask_path,
                "distance": distance_path,
            }
        )

    return samples


def load_grayscale_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    return image


def load_binary_mask(mask_path: Path) -> np.ndarray:
    mask = load_grayscale_image(mask_path)
    return (mask > 127).astype(np.uint8)


def mask_to_distance_map(mask: np.ndarray) -> np.ndarray:
    return cv2.distanceTransform((mask * 255).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)


def load_distance_map(distance_path: Path, mask: np.ndarray) -> np.ndarray:
    if distance_path.exists():
        return load_grayscale_image(distance_path).astype(np.float32)
    return mask_to_distance_map(mask)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_segmentation_model(checkpoint_path: Path, device: torch.device) -> UNetWithAttention:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = UNetWithAttention(in_channels=1, out_channels=1, init_features=32)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def load_regression_model(checkpoint_path: Path, device: torch.device) -> RegressionUNet:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = RegressionUNet()
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def predict_segmentation(model: UNetWithAttention, image: np.ndarray, device: torch.device) -> np.ndarray:
    image_tensor = torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.sigmoid(logits)
    return probabilities.squeeze().cpu().numpy()


def predict_regression(model: RegressionUNet, image: np.ndarray, device: torch.device) -> np.ndarray:
    image_tensor = torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(image_tensor)
    return prediction.squeeze().cpu().numpy()


def compute_binary_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict[str, float]:
    pred_bool = pred_mask.astype(bool)
    gt_bool = gt_mask.astype(bool)

    tp = float(np.logical_and(pred_bool, gt_bool).sum())
    fp = float(np.logical_and(pred_bool, np.logical_not(gt_bool)).sum())
    fn = float(np.logical_and(np.logical_not(pred_bool), gt_bool).sum())
    tn = float(np.logical_and(np.logical_not(pred_bool), np.logical_not(gt_bool)).sum())

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    dice = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-6)
    iou = tp / (tp + fp + fn + 1e-6)
    specificity = tn / (tn + fp + 1e-6)
    porosity_pred = float(pred_bool.mean())
    porosity_gt = float(gt_bool.mean())

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "porosity_pred": porosity_pred,
        "porosity_gt": porosity_gt,
        "porosity_error": abs(porosity_pred - porosity_gt),
    }


def detect_pores_from_distance_map(
    dist_map: np.ndarray,
    min_peak_value: float,
    min_peak_distance: int,
) -> list[dict[str, float]]:
    """Detect pore centers and radii from a distance map using local maxima."""
    if min_peak_distance < 1:
        raise ValueError("min_peak_distance must be >= 1")

    local_max = ndimage.maximum_filter(dist_map, size=min_peak_distance)
    peaks = (dist_map == local_max) & (dist_map >= min_peak_value)
    labels, num_labels = ndimage.label(peaks)

    pores: list[dict[str, float]] = []
    for label_index in range(1, num_labels + 1):
        coords = np.argwhere(labels == label_index)
        if coords.size == 0:
            continue

        peak_values = dist_map[labels == label_index]
        best_index = int(np.argmax(peak_values))
        y_coord, x_coord = coords[best_index]
        pores.append(
            {
                "x": float(x_coord),
                "y": float(y_coord),
                "radius": float(dist_map[y_coord, x_coord]),
            }
        )

    return pores


def detect_pores_from_mask(mask: np.ndarray, min_peak_distance: int, min_peak_value: float) -> list[dict[str, float]]:
    dist_map = mask_to_distance_map(mask)
    return detect_pores_from_distance_map(dist_map, min_peak_value=min_peak_value, min_peak_distance=min_peak_distance)


def match_pores(
    predicted: list[dict[str, float]],
    target: list[dict[str, float]],
    tolerance_px: float,
) -> dict[str, float]:
    remaining_targets = list(target)
    true_positive = 0
    radius_errors: list[float] = []
    center_errors: list[float] = []

    for pred in predicted:
        best_index = -1
        best_distance = float("inf")

        for index, gt in enumerate(remaining_targets):
            distance = math.hypot(pred["x"] - gt["x"], pred["y"] - gt["y"])
            if distance < best_distance:
                best_distance = distance
                best_index = index

        if best_index >= 0 and best_distance <= tolerance_px:
            true_positive += 1
            matched_gt = remaining_targets.pop(best_index)
            center_errors.append(best_distance)
            radius_errors.append(abs(pred["radius"] - matched_gt["radius"]))

    false_positive = max(0, len(predicted) - true_positive)
    false_negative = max(0, len(target) - true_positive)
    precision = true_positive / (true_positive + false_positive + 1e-6)
    recall = true_positive / (true_positive + false_negative + 1e-6)
    f1 = (2.0 * precision * recall) / (precision + recall + 1e-6)

    return {
        "true_positive": float(true_positive),
        "false_positive": float(false_positive),
        "false_negative": float(false_negative),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "count_pred": float(len(predicted)),
        "count_gt": float(len(target)),
        "count_error": abs(float(len(predicted) - len(target))),
        "center_mae": float(np.mean(center_errors)) if center_errors else float("nan"),
        "radius_mae": float(np.mean(radius_errors)) if radius_errors else float("nan"),
    }


def nanmean(values: list[float]) -> float:
    filtered = [value for value in values if not math.isnan(value)]
    if not filtered:
        return float("nan")
    return float(np.mean(filtered))


def aggregate_metrics(metrics_list: list[dict[str, float]]) -> dict[str, float]:
    aggregated: dict[str, float] = {}
    if not metrics_list:
        return aggregated

    for key in metrics_list[0]:
        values = [float(item[key]) for item in metrics_list]
        aggregated[key] = nanmean(values)

    aggregated["num_samples"] = float(len(metrics_list))
    return aggregated


def choose_best_threshold(
    predictions: list[np.ndarray],
    targets: list[np.ndarray],
    candidate_thresholds: list[float],
) -> tuple[float, dict[str, float]]:
    best_threshold = candidate_thresholds[0]
    best_metrics: dict[str, float] | None = None
    best_score = -1.0

    for threshold in candidate_thresholds:
        per_sample = []
        for prediction, target in zip(predictions, targets, strict=True):
            per_sample.append(compute_binary_metrics((prediction >= threshold).astype(np.uint8), target))

        aggregated = aggregate_metrics(per_sample)
        if aggregated.get("dice", -1.0) > best_score:
            best_score = aggregated["dice"]
            best_threshold = threshold
            best_metrics = aggregated

    if best_metrics is None:
        raise RuntimeError("Could not choose a threshold")

    return best_threshold, best_metrics


def evaluate_segmentation(
    model: UNetWithAttention,
    samples: list[dict[str, Path]],
    device: torch.device,
    tolerance_px: float,
    min_peak_distance: int,
    min_peak_value: float,
    thresholds: list[float],
    fixed_threshold: float | None = None,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    probability_maps: list[np.ndarray] = []
    masks: list[np.ndarray] = []

    for sample in samples:
        image = load_grayscale_image(sample["original"])
        gt_mask = load_binary_mask(sample["mask"])
        probability_maps.append(predict_segmentation(model, image, device))
        masks.append(gt_mask)

    if fixed_threshold is None:
        best_threshold, threshold_metrics = choose_best_threshold(probability_maps, masks, thresholds)
    else:
        best_threshold = fixed_threshold
        threshold_metrics = aggregate_metrics(
            [compute_binary_metrics((prediction >= best_threshold).astype(np.uint8), target) for prediction, target in zip(probability_maps, masks, strict=True)]
        )

    per_sample_results = []
    for sample, probability_map, gt_mask in zip(samples, probability_maps, masks, strict=True):
        pred_mask = (probability_map >= best_threshold).astype(np.uint8)
        pixel_metrics = compute_binary_metrics(pred_mask, gt_mask)
        pred_pores = detect_pores_from_mask(pred_mask, min_peak_distance=min_peak_distance, min_peak_value=min_peak_value)
        gt_pores = detect_pores_from_mask(gt_mask, min_peak_distance=min_peak_distance, min_peak_value=min_peak_value)
        object_metrics = match_pores(pred_pores, gt_pores, tolerance_px=tolerance_px)
        per_sample_results.append({"sample_id": sample["id"], "threshold": best_threshold, **pixel_metrics, **object_metrics})

    summary = aggregate_metrics([{key: value for key, value in result.items() if key != "sample_id"} for result in per_sample_results])
    summary["selected_threshold"] = best_threshold
    summary["selected_threshold_dice"] = threshold_metrics["dice"]
    return summary, per_sample_results


def evaluate_regression(
    model: RegressionUNet,
    samples: list[dict[str, Path]],
    device: torch.device,
    tolerance_px: float,
    min_peak_distance: int,
    min_peak_value: float,
    mask_thresholds: list[float],
    fixed_threshold: float | None = None,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    predicted_distance_maps: list[np.ndarray] = []
    gt_distance_maps: list[np.ndarray] = []
    gt_masks: list[np.ndarray] = []

    for sample in samples:
        image = load_grayscale_image(sample["original"])
        gt_mask = load_binary_mask(sample["mask"])
        gt_distance = load_distance_map(sample["distance"], gt_mask)

        predicted_distance_maps.append(predict_regression(model, image, device))
        gt_distance_maps.append(gt_distance)
        gt_masks.append(gt_mask)

    predicted_masks = [np.clip(prediction, a_min=0.0, a_max=None) for prediction in predicted_distance_maps]
    if fixed_threshold is None:
        best_mask_threshold, _ = choose_best_threshold(predicted_masks, gt_masks, mask_thresholds)
    else:
        best_mask_threshold = fixed_threshold

    per_sample_results = []
    for sample, pred_distance, gt_distance, gt_mask in zip(
        samples,
        predicted_distance_maps,
        gt_distance_maps,
        gt_masks,
        strict=True,
    ):
        pred_mask = (pred_distance >= best_mask_threshold).astype(np.uint8)
        pixel_metrics = compute_binary_metrics(pred_mask, gt_mask)
        distance_error = pred_distance - gt_distance
        pred_pores = detect_pores_from_distance_map(
            np.clip(pred_distance, a_min=0.0, a_max=None),
            min_peak_value=min_peak_value,
            min_peak_distance=min_peak_distance,
        )
        gt_pores = detect_pores_from_distance_map(
            gt_distance,
            min_peak_value=min_peak_value,
            min_peak_distance=min_peak_distance,
        )
        object_metrics = match_pores(pred_pores, gt_pores, tolerance_px=tolerance_px)
        per_sample_results.append(
            {
                "sample_id": sample["id"],
                "mask_threshold": best_mask_threshold,
                "distance_mae": float(np.mean(np.abs(distance_error))),
                "distance_rmse": float(np.sqrt(np.mean(distance_error ** 2))),
                **pixel_metrics,
                **object_metrics,
            }
        )

    summary = aggregate_metrics([{key: value for key, value in result.items() if key != "sample_id"} for result in per_sample_results])
    summary["selected_mask_threshold"] = best_mask_threshold
    return summary, per_sample_results


def save_results(output_dir: Path, summary: dict[str, float], per_sample: list[dict[str, float]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "summary.json", "w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2, ensure_ascii=False)

    if per_sample:
        fieldnames = list(per_sample[0].keys())
        with open(output_dir / "per_sample_metrics.csv", "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_sample)


def parse_thresholds(raw_value: str) -> list[float]:
    return [float(value.strip()) for value in raw_value.split(",") if value.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate pore models on manually annotated real images")
    parser.add_argument("--task", choices=["segmentation", "regression"], required=True)
    parser.add_argument("--model", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument(
        "--dataset",
        type=str,
        default="RealPoresImages/dataset_manual",
        help="Path to the manually annotated dataset root",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/evaluations/real_eval",
        help="Directory for evaluation outputs",
    )
    parser.add_argument("--tolerance-px", type=float, default=10.0, help="Max center distance for pore matching")
    parser.add_argument("--min-peak-distance", type=int, default=10, help="Window size for local maxima search")
    parser.add_argument("--min-peak-value", type=float, default=2.0, help="Minimum peak value in distance pixels")
    parser.add_argument(
        "--thresholds",
        type=str,
        default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
        help="Candidate thresholds for segmentation probability or regression mask extraction",
    )
    parser.add_argument(
        "--fixed-threshold",
        type=float,
        default=None,
        help="Use a fixed threshold instead of selecting the best one on the evaluated dataset",
    )

    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    output_dir = Path(args.output)
    checkpoint_path = Path(args.model)
    thresholds = parse_thresholds(args.thresholds)

    samples = discover_samples(dataset_dir)
    if not samples:
        raise RuntimeError(f"No annotated samples found in {dataset_dir}")

    device = get_device()

    if args.task == "segmentation":
        model = load_segmentation_model(checkpoint_path, device)
        summary, per_sample = evaluate_segmentation(
            model=model,
            samples=samples,
            device=device,
            tolerance_px=args.tolerance_px,
            min_peak_distance=args.min_peak_distance,
            min_peak_value=args.min_peak_value,
            thresholds=thresholds,
            fixed_threshold=args.fixed_threshold,
        )
    else:
        model = load_regression_model(checkpoint_path, device)
        summary, per_sample = evaluate_regression(
            model=model,
            samples=samples,
            device=device,
            tolerance_px=args.tolerance_px,
            min_peak_distance=args.min_peak_distance,
            min_peak_value=args.min_peak_value,
            mask_thresholds=thresholds,
            fixed_threshold=args.fixed_threshold,
        )

    summary["task"] = args.task
    summary["dataset"] = str(dataset_dir)
    summary["model"] = str(checkpoint_path)

    save_results(output_dir, summary, per_sample)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved results to {output_dir}")


if __name__ == "__main__":
    main()