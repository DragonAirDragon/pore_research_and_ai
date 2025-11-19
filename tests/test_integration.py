import os
import sys
import json
import cv2
import numpy as np
import csv
from pathlib import Path

# Add current directory to path to import src
sys.path.append(os.getcwd())

from src.pore_generator import PoreGenerator
from analyze_mask import analyze_pore_mask

class InstrumentedPoreGenerator(PoreGenerator):
    """
    A subclass of PoreGenerator that tracks the exact location and area of placed pores.
    """
    def __init__(self, config):
        super().__init__(config)
        self.placed_pores = []

    def _place_on_image(self, image, occupied_mask, canvas, x, y):
        """
        Overrides the placement method to record ground truth data.
        """
        # Call the original method to actually place the pore
        super()._place_on_image(image, occupied_mask, canvas, x, y)
        
        # Calculate exact area of the pore (black pixels in canvas)
        # Note: canvas has white background (255) and black pore (0)
        pore_area = np.sum(canvas == 0)
        
        # Store ground truth
        # x, y are the center coordinates on the image
        self.placed_pores.append({
            'x': x,
            'y': y,
            'area': pore_area
        })

def run_integration_test():
    print("Starting Integration Test...")
    
    # 1. Setup Config
    # Use a simple config with fewer pores for easier debugging
    test_config = {
        "image_settings": {"width": 512, "height": 512},
        "pore_settings": {
            "medium_pores": {
                "count_range": [5, 10],
                "radius_mean_relative": 0.05,
                "min_distance_relative": 0.05,
                "stretch_enabled": True,
                "stretch_factor_range": [1.0, 1.2],
                "rotation_enabled": True
            }
        }
    }
    
    # 2. Generate Image with Ground Truth
    generator = InstrumentedPoreGenerator(test_config)
    image = generator.generate_image()
    
    test_image_path = "integration_test_mask.png"
    cv2.imwrite(test_image_path, image)
    print(f"Generated test image: {test_image_path}")
    print(f"Ground Truth: {len(generator.placed_pores)} pores placed.")
    
    # 3. Run Analysis
    output_dir = "integration_results"
    # Note: Generator produces White Background (255) / Black Pores (0)
    # So we MUST use invert=True
    analyze_pore_mask(test_image_path, output_dir, invert=True)
    
    # 4. Compare Results
    csv_path = os.path.join(output_dir, "integration_test_mask_results.csv")
    
    detected_pores = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            detected_pores.append({
                'x': float(row['centroid_x']),
                'y': float(row['centroid_y']),
                'area': float(row['area_pixels'])
            })
            
    print(f"Analysis: {len(detected_pores)} pores detected.")
    
    # Basic Validation
    if len(detected_pores) != len(generator.placed_pores):
        print(f"❌ Mismatch in pore count! Expected {len(generator.placed_pores)}, got {len(detected_pores)}")
    else:
        print(f"✅ Pore count matches: {len(detected_pores)}")
        
    # Match pores by distance
    matched_count = 0
    total_area_diff = 0
    
    # Copy lists to avoid modifying originals during matching
    gt_pores = list(generator.placed_pores)
    dt_pores = list(detected_pores)
    
    for gt in gt_pores:
        # Find closest detected pore
        best_match = None
        min_dist = float('inf')
        best_idx = -1
        
        for i, dt in enumerate(dt_pores):
            dist = np.sqrt((gt['x'] - dt['x'])**2 + (gt['y'] - dt['y'])**2)
            if dist < min_dist:
                min_dist = dist
                best_match = dt
                best_idx = i
        
        # Threshold for matching (e.g., 10 pixels)
        if min_dist < 10:
            matched_count += 1
            area_diff = abs(gt['area'] - best_match['area'])
            total_area_diff += area_diff
            # Remove matched pore to avoid double counting
            dt_pores.pop(best_idx)
            print(f"  Matched Pore: GT({gt['x']:.1f}, {gt['y']:.1f}) <-> DT({best_match['x']:.1f}, {best_match['y']:.1f}) | Dist: {min_dist:.2f} | Area Diff: {area_diff:.1f}")
        else:
            print(f"  ❌ Unmatched GT Pore at ({gt['x']:.1f}, {gt['y']:.1f})")

    if matched_count == len(generator.placed_pores):
        print(f"✅ All pores matched successfully.")
        avg_area_diff = total_area_diff / matched_count
        print(f"  Average Area Difference: {avg_area_diff:.2f} pixels")
        if avg_area_diff < 50: # Arbitrary threshold
             print("✅ Area calculation is accurate.")
        else:
             print("⚠️ Area difference is high.")
    else:
        print(f"❌ Only {matched_count}/{len(generator.placed_pores)} pores matched.")

class OverlappingPoreGenerator(InstrumentedPoreGenerator):
    """
    Generator that allows pores to overlap.
    """
    def _can_place_pore(self, pore_canvas, occupied_mask, x, y, canvas_center, min_distance):
        # Always allow placement to force overlaps
        return True

def run_overlapping_test():
    print("\n" + "="*50)
    print("Starting Overlapping Pores Test (Watershed Verification)...")
    print("="*50)
    
    # 1. Setup Config
    # High density to ensure overlaps
    test_config = {
        "image_settings": {"width": 512, "height": 512},
        "pore_settings": {
            "medium_pores": {
                "count_range": [15, 25], # Higher count
                "radius_mean_relative": 0.08, # Larger pores
                "min_distance_relative": 0, # No distance constraint
                "stretch_enabled": False,
                "rotation_enabled": False
            }
        }
    }
    
    # 2. Generate Image with Overlaps
    generator = OverlappingPoreGenerator(test_config)
    image = generator.generate_image()
    
    test_image_path = "overlapping_test_mask.png"
    cv2.imwrite(test_image_path, image)
    print(f"Generated overlapping test image: {test_image_path}")
    print(f"Ground Truth: {len(generator.placed_pores)} pores placed.")
    
    # 3. Run Analysis
    output_dir = "overlapping_results"
    analyze_pore_mask(test_image_path, output_dir, invert=True)
    
    # 4. Compare Results
    csv_path = os.path.join(output_dir, "overlapping_test_mask_results.csv")
    
    detected_pores = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            detected_pores.append({
                'x': float(row['centroid_x']),
                'y': float(row['centroid_y']),
                'area': float(row['area_pixels'])
            })
            
    print(f"Analysis: {len(detected_pores)} pores detected.")
    
    # Validation
    diff = abs(len(detected_pores) - len(generator.placed_pores))
    print(f"Difference in count: {diff}")
    
    if diff <= 2: # Allow small margin of error for complex overlaps
        print(f"✅ Watershed successfully separated most pores!")
    else:
        print(f"⚠️ Watershed might have missed some overlaps or over-segmented.")
        
    # We don't check area strictly here because overlaps reduce total area
    
if __name__ == "__main__":
    run_integration_test()
    run_overlapping_test()
