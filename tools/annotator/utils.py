import cv2
import numpy as np
from pathlib import Path

def load_image(path):
    """Loads an image from path in Grayscale."""
    # Use cv2.imdecode to handle unicode paths correctly on Windows
    stream = open(path, "rb")
    bytes = bytearray(stream.read())
    numpyarray = np.asarray(bytes, dtype=np.uint8)
    img = cv2.imdecode(numpyarray, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise ValueError(f"Could not load image: {path}")
        
    return img

def generate_distance_map(mask):
    """
    Generates a distance map from a binary mask.
    Mask should be uint8, where pores are 255 and background is 0.
    """
    # Distance Transform needs binary image where object is 1 (white).
    # Our mask is already 255 for pores.
    dist_map = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    
    # Save as 8-bit for visualization/training (if radius < 255)
    dist_map_8u = dist_map.astype(np.uint8)
    return dist_map_8u

def save_results(original_image_gray, mask, output_dir, base_name):
    """
    Saves the original image, mask, and distance map.
    original_image_gray: numpy array (Grayscale)
    mask: numpy array (uint8, 0-255)
    output_dir: Path object
    base_name: string (filename without extension)
    """
    save_path = output_dir / base_name
    save_path.mkdir(parents=True, exist_ok=True)
    
    cv2.imwrite(str(save_path / "original.png"), original_image_gray)
    cv2.imwrite(str(save_path / "mask.png"), mask)
    
    dist_map = generate_distance_map(mask)
    cv2.imwrite(str(save_path / "distance_map.png"), dist_map)
    
    return save_path
