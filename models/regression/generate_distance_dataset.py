import os
import cv2
import numpy as np
import json
import shutil
from pathlib import Path
from tqdm import tqdm
import sys

# Добавляем корневую директорию в PYTHONPATH
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from scripts.generate_images import PoreImageGenerator

def generate_distance_dataset(output_dir="dataset_regression", num_train=500, num_val=100):
    """Generates dataset with Distance Maps as targets."""
    
    print(f"Generating Regression Dataset in {output_dir}...")
    
    output_path = Path(output_dir)
    if output_path.exists():
        shutil.rmtree(output_path)
    
    for split in ["train", "val"]:
        (output_path / split / "noisy").mkdir(parents=True, exist_ok=True)
        (output_path / split / "distance").mkdir(parents=True, exist_ok=True) # Target is distance map
        
    generator = PoreImageGenerator("config.json")
    
    metadata = {"files": {"train": [], "val": []}}
    
    splits = [("train", num_train), ("val", num_val)]
    
    for split_name, count in splits:
        print(f"Generating {split_name} set ({count} images)...")
        
        for i in tqdm(range(count)):
            # 1. Generate Image Pair
            # We need to access the internal generator to get the mask directly
            # or just use the file saving mechanism and read it back.
            # Let's use the public API but intercept the mask.
            
            # Actually, PoreImageGenerator saves files. Let's use that for simplicity
            # but we need to process the "clean" image into a distance map.
            
            # Generate raw data
            # Use ceramic texture rendering for realistic appearance
            clean_mask = generator.pore_generator.generate_image()
            noisy_img = generator.image_processor.apply_ceramic_texture(clean_mask)
            
            # 2. Compute Distance Map
            # clean_mask is 0 (black) for pores? No, wait.
            # In main.py: pores are drawn with color=0 (black) on white (255).
            # Distance Transform needs binary image where object is 1 (white).
            
            # Invert: Pores become 255 (White), Background 0 (Black)
            _, binary_pore = cv2.threshold(clean_mask, 127, 255, cv2.THRESH_BINARY_INV)
            
            # Compute Distance Transform
            # DIST_L2 = Euclidean distance
            # MaskSize = 5 (precise)
            dist_map = cv2.distanceTransform(binary_pore, cv2.DIST_L2, 5)
            
            # Normalize? 
            # For regression, it's better to keep absolute values (pixels) so the network learns radius directly.
            # However, neural networks like normalized inputs/outputs (0-1).
            # If we keep it absolute, we might need a linear activation and careful learning rate.
            # Let's keep it absolute for now (easier interpretation) but we will normalize in the Dataset loader if needed.
            # Actually, saving as float image (TIFF or EXR) is best, but PNG is 8-bit/16-bit.
            # 8-bit PNG (0-255) is fine if max radius < 255 pixels.
            # Our pores are usually radius 5-30. So 8-bit is perfectly fine.
            
            # Save Distance Map as 8-bit image
            # We don't normalize to 0-1 here, we just save the pixel values.
            # If radius is 20, pixel value is 20.
            dist_map_8u = dist_map.astype(np.uint8)
            
            # 3. Save Files
            file_id = f"{split_name}_{i:04d}"
            noisy_filename = f"{file_id}_noisy.png"
            dist_filename = f"{file_id}_dist.png"
            
            cv2.imwrite(str(output_path / split_name / "noisy" / noisy_filename), noisy_img)
            cv2.imwrite(str(output_path / split_name / "distance" / dist_filename), dist_map_8u)
            
            metadata["files"][split_name].append({
                "id": file_id,
                "noisy": noisy_filename,
                "distance": dist_filename
            })
            
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
        
    print("Dataset generation complete!")

if __name__ == "__main__":
    generate_distance_dataset()
