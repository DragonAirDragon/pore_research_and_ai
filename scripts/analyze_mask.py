import cv2
import numpy as np
import argparse
import csv
import os
from pathlib import Path

def analyze_pore_mask(image_path, output_dir, pixel_scale=1.0, invert=False, watershed_radius=15):
    """
    Analyzes a binary mask of pores.
    
    Args:
        image_path (str): Path to the input mask image.
        output_dir (str): Directory to save results.
        pixel_scale (float): Scale factor (microns per pixel, etc.). Default is 1.0 (pixels).
    """
    # 1. Load Image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")

    # 2. Preprocess (Thresholding)
    if invert:
        # If background is white and pores are black
        _, binary_mask = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    else:
        # If background is black and pores are white
        _, binary_mask = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # 3. Watershed Segmentation
    # Use Distance Transform to find sure foreground (centers of pores)
    dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
    
    # Find local maxima to use as markers (better for variable sized pores)
    # We use a minimum distance for peaks to avoid over-segmentation
    # min_distance of ~20 pixels is a reasonable default for these pore sizes
    from scipy import ndimage
    local_max = ndimage.maximum_filter(dist_transform, size=watershed_radius)
    
    # Identify the peaks: must be equal to local max AND above a noise threshold
    # We use a low absolute threshold (e.g. 2) instead of relative to max() 
    # to ensure we don't miss small pores when large pores are present.
    peaks = (dist_transform == local_max) & (dist_transform > 2)
    
    # Label the markers
    markers, num_peaks = ndimage.label(peaks)
    markers = markers.astype(np.int32)
    
    # Find sure background (dilate the pores a bit to be sure we are in background)
    kernel = np.ones((3,3), np.uint8)
    sure_bg = cv2.dilate(binary_mask, kernel, iterations=3)
    
    # Unknown region is the area between sure background and sure foreground
    # Note: sure_bg is the dilated FOREGROUND mask. The "unknown" is sure_bg - sure_fg (peaks).
    # But for watershed, we want the UNKNOWN region to be 0.
    # The background (outside sure_bg) should be marked as 1.
    # The peaks (inside sure_fg) should be marked > 1.
    
    # Let's adjust markers:
    # Currently markers has 0 for background (everything not a peak) and 1..N for peaks.
    # We want 0 for UNKNOWN.
    # We want 1 for BACKGROUND.
    
    # Create a new marker array
    watershed_markers = np.zeros_like(markers, dtype=np.int32)
    
    # Mark sure background as 1. 
    # sure_bg is the dilated pores (255). So background is where sure_bg == 0.
    watershed_markers[sure_bg == 0] = 1
    
    # Mark peaks as 2..N+1
    # markers has peaks as 1..N
    watershed_markers[markers > 0] = markers[markers > 0] + 1
    
    # The rest (where sure_bg==255 AND markers==0) is 0 (UNKNOWN).
    # This is automatically handled by initialization to 0.
    
    # Apply Watershed
    img_color = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
    watershed_markers = cv2.watershed(img_color, watershed_markers)
    
    # Extract Metrics from Markers
    pore_data = []
    total_pore_area = 0
    
    unique_labels = np.unique(watershed_markers)
    
    for label in unique_labels:
        if label <= 1: # -1 is boundary, 1 is background
            continue
            
        # Create a mask for this pore
        pore_mask = np.zeros_like(binary_mask)
        pore_mask[watershed_markers == label] = 255
        
        # Calculate stats
        # Fix: Use countNonZero for area to avoid 255 factor error
        area = cv2.countNonZero(pore_mask)
        
        if area == 0:
            continue
            
        # Using moments for centroid
        M = cv2.moments(pore_mask)
        if M["m00"] != 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
        else:
            # Fallback if moment is 0 (shouldn't happen if area > 0 but safe)
            cx, cy = 0, 0
        
        # Bounding box
        x, y, w, h = cv2.boundingRect(pore_mask)
        
        total_pore_area += area
        
        pore_data.append({
            'pore_id': int(label),
            'area_pixels': area,
            'area_scaled': area * (pixel_scale ** 2),
            'centroid_x': cx,
            'centroid_y': cy,
            'bbox_x': x,
            'bbox_y': y,
            'bbox_w': w,
            'bbox_h': h
        })
    
    # Update num_labels for reporting
    num_labels = len(pore_data) + 1 # +1 for background logic consistency


    # 5. Global Statistics
    total_image_area = img.shape[0] * img.shape[1]
    porosity_percentage = (total_pore_area / total_image_area) * 100.0
    
    print(f"Analysis Complete:")
    print(f"  Total Pores: {len(pore_data)}")
    print(f"  Total Porosity: {porosity_percentage:.2f}%")

    # 6. Visualization
    # Convert to BGR to draw colored overlays
    vis_img = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
    
    for pore in pore_data:
        x, y, w, h = pore['bbox_x'], pore['bbox_y'], pore['bbox_w'], pore['bbox_h']
        cx, cy = int(pore['centroid_x']), int(pore['centroid_y'])
        
        # Draw bounding box (Green)
        cv2.rectangle(vis_img, (x, y), (x+w, y+h), (0, 255, 0), 1)
        # Draw centroid (Red)
        cv2.circle(vis_img, (cx, cy), 2, (0, 0, 255), -1)
        # Put ID
        cv2.putText(vis_img, str(pore['pore_id']), (x, y-2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

    # 7. Save Results
    os.makedirs(output_dir, exist_ok=True)
    base_name = Path(image_path).stem
    
    # Save CSV
    csv_path = os.path.join(output_dir, f"{base_name}_results.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['pore_id', 'area_pixels', 'area_scaled', 'centroid_x', 'centroid_y', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h'])
        writer.writeheader()
        writer.writerows(pore_data)
        
    # Save Summary Text
    summary_path = os.path.join(output_dir, f"{base_name}_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Image: {image_path}\n")
        f.write(f"Total Pores: {len(pore_data)}\n")
        f.write(f"Total Porosity: {porosity_percentage:.4f}%\n")

    # Save Visualization
    vis_path = os.path.join(output_dir, f"{base_name}_overlay.png")
    cv2.imwrite(vis_path, vis_img)
    
    print(f"Results saved to {output_dir}")

def create_dummy_mask(path):
    """Creates a simple dummy mask for testing."""
    # 100x100 black image
    img = np.zeros((100, 100), dtype=np.uint8)
    
    # Draw a white square (pore 1)
    # Area = 10*10 = 100 pixels
    cv2.rectangle(img, (10, 10), (20, 20), 255, -1)
    
    # Draw a white circle (pore 2)
    # Radius = 5. Area ~= pi * 5^2 = 78.5 -> approx 79 pixels (discrete)
    cv2.circle(img, (60, 60), 5, 255, -1)
    
    cv2.imwrite(path, img)
    print(f"Created dummy mask at {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze pore masks.")
    parser.add_argument("--input", type=str, help="Path to input mask image")
    parser.add_argument("--output", type=str, default="results", help="Directory to save results")
    parser.add_argument("--scale", type=float, default=1.0, help="Pixel scale factor")
    parser.add_argument("--invert", action="store_true", help="Invert mask (use if pores are black on white background)")
    parser.add_argument("--watershed-radius", type=int, default=15, help="Radius for watershed peak detection (default: 15)")
    parser.add_argument("--test", action="store_true", help="Run on a generated dummy mask")
    
    args = parser.parse_args()
    
    if args.test:
        dummy_path = "dummy_mask.png"
        create_dummy_mask(dummy_path)
        analyze_pore_mask(dummy_path, args.output, args.scale, invert=False, watershed_radius=args.watershed_radius) # Dummy is black bg, white pores
        # Cleanup dummy file if desired, or keep it for inspection
        # os.remove(dummy_path) 
    elif args.input:
        analyze_pore_mask(args.input, args.output, args.scale, invert=args.invert, watershed_radius=args.watershed_radius)
    else:
        print("Please provide --input or use --test")
