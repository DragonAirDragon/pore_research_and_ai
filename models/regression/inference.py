"""Inference script for Regression UNet."""

import cv2
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
from scipy import ndimage
import sys
from pathlib import Path

# Добавляем корневую директорию в PYTHONPATH
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from models.regression.model import RegressionUNet

def predict_pores(image_path, model_path, output_path=None, threshold=2.0):
    """Predicts pores using the regression model."""
    
    # 1. Load Model
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    model = RegressionUNet().to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    
    # 2. Load Image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")
        
    # Preprocess
    img_float = img.astype(np.float32) / 255.0
    input_tensor = torch.from_numpy(img_float).unsqueeze(0).unsqueeze(0).to(device)
    
    # 3. Predict Distance Map
    with torch.no_grad():
        pred_dist = model(input_tensor)
        
    # Convert to numpy
    dist_map = pred_dist.squeeze().cpu().numpy()
    
    # 4. Find Peaks (Local Maxima)
    # Filter size depends on minimum pore distance. 
    # Since we predict radius directly, we can be smarter, but let's stick to local max.
    local_max = ndimage.maximum_filter(dist_map, size=10)
    peaks = (dist_map == local_max) & (dist_map > threshold)
    
    labeled_peaks, num_peaks = ndimage.label(peaks)
    objs = ndimage.find_objects(labeled_peaks)
    
    pores = []
    for i, slice_obj in enumerate(objs):
        # Center of the peak
        y_slice, x_slice = slice_obj
        cy, cx = (y_slice.start + y_slice.stop - 1) / 2, (x_slice.start + x_slice.stop - 1) / 2
        
        # Radius is the value at the peak
        # We can take the max value in the slice
        radius = dist_map[y_slice, x_slice].max()
        
        pores.append({
            "id": i + 1,
            "x": cx,
            "y": cy,
            "radius": float(radius)
        })
        
    print(f"Detected {len(pores)} pores.")
    
    # 5. Visualize
    vis_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    for pore in pores:
        cx, cy = int(pore['x']), int(pore['y'])
        r = int(pore['radius'])
        
        # Draw Circle (Green)
        cv2.circle(vis_img, (cx, cy), r, (0, 255, 0), 1)
        # Draw Center (Red)
        cv2.circle(vis_img, (cx, cy), 2, (0, 0, 255), -1)
        # Text
        cv2.putText(vis_img, f"{r:.1f}", (cx, cy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        
    # Show Distance Map (Heatmap)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("Input Image")
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title("Predicted Distance Map")
    plt.imshow(dist_map, cmap='jet')
    plt.colorbar()
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title("Detected Pores")
    plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    if output_path:
        plt.savefig(output_path)
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input image path")
    parser.add_argument("--model", type=str, default="checkpoints_regression/best_model.pth", help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="inference_result.png", help="Output visualization path")
    parser.add_argument("--threshold", type=float, default=2.0, help="Minimum radius threshold")
    
    args = parser.parse_args()
    
    predict_pores(args.input, args.model, args.output, args.threshold)
