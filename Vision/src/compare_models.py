import cv2
import os
from pathlib import Path

from config.utils.path_manager import path_manager # Import the consolidated path manager
from Vision.src.detectors.object_detector import ObjectDetector

def run_comparison(image_folder, output_folder):
    # Define the models to compare
    # 'n' is nano (fast), 's' small, 'm' medium, 'l' large, 'x' extra large
    models_to_test_info = [
        ("YOLO11-L", path_manager.get_model_detail("yolo11l_path")),
        ("RTDETR-L", path_manager.get_model_detail("rtdetr_l_path"))
    ]
    
    detectors = [ObjectDetector(model_path=path) for name, path in models_to_test_info]
    
    # Get some images
    img_paths = list(Path(image_folder).glob("*.jpg"))[:3] # Test with only 3
    
    if not img_paths:
        print(f"No images found in {image_folder}")
        return

    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    for img_p in img_paths:
        print(f"\nProcessing: {img_p.name}")
        original_img = cv2.imread(str(img_p))
        
        for i, detector in enumerate(detectors):
            model_name_display = models_to_test_info[i][0] # Use the display name
            
            _, result_img, info = detector.detect(original_img)
            
            print(f" -> Model: {model_name_display} | Latency: {info['inference_time_ms']:.1f}ms | Objects: {info['total_objects']}")
            
            # Save result
            save_name = output_path / f"{img_p.stem}_{model_name_display.replace(' ', '_')}.jpg"
            cv2.imwrite(str(save_name), result_img)

if __name__ == "__main__":
    # ADJUST THE PATH TO WHERE YOUR IMAGES ARE LOCATED (TEST OR VAL)
    imgs_dir = path_manager.get("bdd_images_val") 
    out_dir = path_manager.get("vision_comparison_output", create=True)
    
    run_comparison(str(imgs_dir), str(out_dir))