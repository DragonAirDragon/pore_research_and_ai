import os
import sys
import json
import cv2
import numpy as np
import csv
from pathlib import Path

# Add current directory to path to import src
sys.path.append(os.getcwd())

from src.config_loader import ConfigLoader
from test_integration import InstrumentedPoreGenerator
from analyze_mask import analyze_pore_mask

def tune_watershed_radius():
    print("\n" + "="*50)
    print("Starting Parameter Tuning (Watershed Radius)...")
    print("="*50)
    
    # 1. Load Real Config
    config_path = "config.json"
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return

    loader = ConfigLoader(config_path)
    real_config = loader.load()
    
    # Ensure we generate only 1 image for testing
    real_config["image_settings"]["total_images"] = 1
    
    # 2. Generate Image with Ground Truth
    # We use InstrumentedPoreGenerator but initialized with the REAL config
    generator = InstrumentedPoreGenerator(real_config)
    image = generator.generate_image()
    
    test_image_path = "tuning_test_mask.png"
    cv2.imwrite(test_image_path, image)
    print(f"Generated tuning mask using {config_path}")
    print(f"Ground Truth: {len(generator.placed_pores)} pores placed.")
    
    # 3. Iterate Parameters
    radii_to_test = [5, 10, 15, 20, 25, 30]
    best_radius = -1
    min_diff = float('inf')
    
    results = []
    
    output_dir = "tuning_results"
    
    print(f"\nTesting radii: {radii_to_test}")
    print(f"{'Radius':<10} | {'Detected':<10} | {'Diff':<10}")
    print("-" * 36)
    
    for radius in radii_to_test:
        # Run Analysis
        # Note: Generator produces White Background (255) / Black Pores (0) -> Invert=True
        analyze_pore_mask(test_image_path, output_dir, invert=True, watershed_radius=radius)
        
        # Read Results
        csv_path = os.path.join(output_dir, "tuning_test_mask_results.csv")
        detected_count = 0
        with open(csv_path, 'r') as f:
            # Subtract header
            detected_count = sum(1 for row in f) - 1
            
        diff = abs(detected_count - len(generator.placed_pores))
        results.append((radius, detected_count, diff))
        
        print(f"{radius:<10} | {detected_count:<10} | {diff:<10}")
        
        if diff < min_diff:
            min_diff = diff
            best_radius = radius
            
    print("-" * 36)
    print(f"✅ Best Watershed Radius: {best_radius} (Diff: {min_diff})")
    
    # 4. Verification of Best Result
    if min_diff == 0:
        print("Perfect match found!")
    else:
        print(f"Closest match found. Error: {min_diff} pores.")

if __name__ == "__main__":
    tune_watershed_radius()
