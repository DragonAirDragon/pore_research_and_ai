import cv2
import numpy as np
import argparse
from pathlib import Path
import shutil

class PoreAnnotator:
    def __init__(self, image_path, output_dir="dataset_manual"):
        self.image_path = Path(image_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load image
        self.original_image = cv2.imread(str(self.image_path))
        if self.original_image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Resize for easier viewing if too large (optional, but good for 4k images)
        # For now, we keep original size to ensure accuracy, but we could add scaling.
        
        self.display_image = self.original_image.copy()
        self.mask = np.zeros(self.original_image.shape[:2], dtype=np.uint8)
        
        # State
        self.pores = [] # List of (center_x, center_y, radius)
        self.current_center = None
        self.current_radius = 0
        self.drawing = False # True if we are currently dragging/setting radius
        
        # Window
        self.window_name = "Pore Annotator"
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if not self.drawing:
                # First click: Set center
                self.current_center = (x, y)
                self.drawing = True
            else:
                # Second click: Confirm radius
                radius = int(np.linalg.norm(np.array(self.current_center) - np.array((x, y))))
                if radius > 0:
                    self.pores.append((*self.current_center, radius))
                    self.update_mask()
                self.drawing = False
                self.current_center = None
                self.current_radius = 0
                
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing and self.current_center:
                self.current_radius = int(np.linalg.norm(np.array(self.current_center) - np.array((x, y))))
                
        self.update_display()

    def update_mask(self):
        # Redraw mask from scratch based on pores list
        self.mask.fill(0)
        for (cx, cy, r) in self.pores:
            cv2.circle(self.mask, (cx, cy), r, (255), -1)

    def update_display(self):
        self.display_image = self.original_image.copy()
        
        # Draw existing pores
        for (cx, cy, r) in self.pores:
            cv2.circle(self.display_image, (cx, cy), r, (0, 255, 0), 2) # Green circles
            cv2.circle(self.display_image, (cx, cy), 2, (0, 0, 255), -1) # Red center
            
        # Draw current being dragged
        if self.drawing and self.current_center:
            cv2.circle(self.display_image, self.current_center, self.current_radius, (255, 0, 0), 2) # Blue active circle
            
        cv2.imshow(self.window_name, self.display_image)

    def save_results(self):
        # 1. Save Original (copy)
        base_name = self.image_path.stem
        save_dir = self.output_dir / base_name
        save_dir.mkdir(exist_ok=True)
        
        cv2.imwrite(str(save_dir / "original.png"), self.original_image)
        
        # 2. Save Mask
        cv2.imwrite(str(save_dir / "mask.png"), self.mask)
        
        # 3. Generate and Save Distance Map
        # Distance Transform needs binary image where object is 1 (white).
        # Our mask is already 255 for pores.
        dist_map = cv2.distanceTransform(self.mask, cv2.DIST_L2, 5)
        
        # Save as 8-bit for visualization/training (if radius < 255)
        dist_map_8u = dist_map.astype(np.uint8)
        cv2.imwrite(str(save_dir / "distance_map.png"), dist_map_8u)
        
        # Save raw float distance map just in case (optional, maybe .exr or .npy)
        # np.save(str(save_dir / "distance_map.npy"), dist_map)
        
        print(f"Saved results to {save_dir}")
        print(f" - Mask: mask.png")
        print(f" - Distance Map: distance_map.png")

    def run(self):
        print("Controls:")
        print("  Left Click: Start drawing circle (Center)")
        print("  Move Mouse: Adjust radius")
        print("  Left Click again: Confirm circle")
        print("  z: Undo last pore")
        print("  s: Save results")
        print("  q: Quit")
        
        self.update_display()
        
        while True:
            key = cv2.waitKey(20) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_results()
                # Visual feedback
                cv2.putText(self.display_image, "SAVED!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.imshow(self.window_name, self.display_image)
                cv2.waitKey(500)
            elif key == ord('z'):
                if self.pores:
                    self.pores.pop()
                    self.update_mask()
                    self.update_display()
                    
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manual Pore Annotator")
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument("--output", default="dataset_manual", help="Output directory")
    
    args = parser.parse_args()
    
    if not Path(args.image_path).exists():
        print(f"Error: File {args.image_path} not found.")
    else:
        annotator = PoreAnnotator(args.image_path, args.output)
        annotator.run()
