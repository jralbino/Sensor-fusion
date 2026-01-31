import cv2
import sys
from pathlib import Path

from config.utils.path_manager import path_manager # Import the consolidated path manager

from Vision.src.lanes.lane_detector import LaneDetector

def test_lane_detection():
    # Load a test image
    images_dir = path_manager.get("bdd_images_val")
    img_paths = list(images_dir.glob("*.jpg"))[:5] # Test with 5 images
    
    if not img_paths:
        print("No images found for testing.")
        return

    detector = LaneDetector(debug=True)
    output_dir = path_manager.get("vision_lanes_test_output", create=True)

    print("🛣️ Testing lane detection...")
    
    for p in img_paths:
        img = cv2.imread(str(p))
        if img is None: continue
        
        # Detection
        result = detector.detect(img)
        
        # Save
        save_path = output_dir / p.name
        cv2.imwrite(str(save_path), result)
        print(f"   Saved: {save_path}")
    
    print("✅ Test finished. Check the output folder")

if __name__ == "__main__":
    test_lane_detection()