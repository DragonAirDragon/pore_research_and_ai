import cv2
import numpy as np
import argparse
from pathlib import Path
import shutil
from tqdm import tqdm

def augment_data(input_dir, output_dir, num_variations=50):
    """
    Augments data from input_dir and saves to output_dir.
    Assumes input_dir structure:
    input_dir/
      image_name/
        original.png
        mask.png
        distance_map.png
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all subdirectories (annotated images)
    subdirs = [d for d in input_path.iterdir() if d.is_dir()]
    
    if not subdirs:
        print(f"No annotated images found in {input_dir}")
        return

    print(f"Found {len(subdirs)} annotated images. Starting augmentation...")
    
    count = 0
    
    for subdir in subdirs:
        img_path = subdir / "original.png"
        mask_path = subdir / "mask.png"
        dist_path = subdir / "distance_map.png"
        
        if not (img_path.exists() and mask_path.exists() and dist_path.exists()):
            print(f"Skipping {subdir}: Missing files")
            continue
            
        img = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        dist = cv2.imread(str(dist_path), cv2.IMREAD_GRAYSCALE)
        
        base_name = subdir.name
        
        # 1. Save original
        cv2.imwrite(str(output_path / f"{base_name}_orig.png"), img)
        cv2.imwrite(str(output_path / f"{base_name}_orig_mask.png"), mask)
        cv2.imwrite(str(output_path / f"{base_name}_orig_dist.png"), dist)
        
        # 2. Geometric Augmentations (Rotations/Flips)
        geo_transforms = []
        geo_transforms.append(("rot0", lambda x: x))
        geo_transforms.append(("rot90", lambda x: cv2.rotate(x, cv2.ROTATE_90_CLOCKWISE)))
        geo_transforms.append(("rot180", lambda x: cv2.rotate(x, cv2.ROTATE_180)))
        geo_transforms.append(("rot270", lambda x: cv2.rotate(x, cv2.ROTATE_90_COUNTERCLOCKWISE)))
        geo_transforms.append(("flipH", lambda x: cv2.flip(x, 1)))
        geo_transforms.append(("flipV", lambda x: cv2.flip(x, 0)))
        geo_transforms.append(("flipH_rot90", lambda x: cv2.rotate(cv2.flip(x, 1), cv2.ROTATE_90_CLOCKWISE)))
        geo_transforms.append(("flipV_rot90", lambda x: cv2.rotate(cv2.flip(x, 0), cv2.ROTATE_90_CLOCKWISE)))
        
        # 3. Intensity Augmentations (Apply only to Image, not Mask/Dist)
        def add_noise(img):
            row, col, ch = img.shape
            mean = 0
            var = 0.1
            sigma = var**0.5
            gauss = np.random.normal(mean, sigma, (row, col, ch))
            gauss = gauss.reshape(row, col, ch)
            noisy = img + gauss * 50
            return np.clip(noisy, 0, 255).astype(np.uint8)

        def adjust_gamma(image, gamma=1.0):
            invGamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            return cv2.LUT(image, table)

        int_transforms = []
        int_transforms.append(("orig", lambda x: x))
        int_transforms.append(("bright", lambda x: cv2.convertScaleAbs(x, alpha=1, beta=30)))
        int_transforms.append(("dark", lambda x: cv2.convertScaleAbs(x, alpha=1, beta=-30)))
        int_transforms.append(("high_contrast", lambda x: cv2.convertScaleAbs(x, alpha=1.3, beta=0)))
        int_transforms.append(("low_contrast", lambda x: cv2.convertScaleAbs(x, alpha=0.7, beta=0)))
        int_transforms.append(("noise", add_noise))
        int_transforms.append(("blur", lambda x: cv2.GaussianBlur(x, (3, 3), 0)))
        int_transforms.append(("gamma", lambda x: adjust_gamma(x, gamma=1.5)))

        # Apply combinations
        # Total variations = 8 (Geo) * 8 (Int) = 64 per image
        
        for geo_name, geo_func in geo_transforms:
            # Apply geometric transform to all
            geo_img = geo_func(img)
            geo_mask = geo_func(mask)
            geo_dist = geo_func(dist)
            
            for int_name, int_func in int_transforms:
                # Apply intensity transform ONLY to image
                final_img = int_func(geo_img)
                
                # Save
                suffix = f"{geo_name}_{int_name}"
                cv2.imwrite(str(output_path / f"{base_name}_{suffix}.png"), final_img)
                cv2.imwrite(str(output_path / f"{base_name}_{suffix}_mask.png"), geo_mask)
                cv2.imwrite(str(output_path / f"{base_name}_{suffix}_dist.png"), geo_dist)
                count += 1
            
        print(f"Processed {base_name}: Generated {len(geo_transforms) * len(int_transforms)} variations.")

    print(f"Augmentation complete. Total images: {count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment Annotated Data")
    parser.add_argument("--input", default="dataset_manual", help="Input directory with annotated folders")
    parser.add_argument("--output", default="dataset_augmented", help="Output directory for augmented data")
    
    args = parser.parse_args()
    augment_data(args.input, args.output)
