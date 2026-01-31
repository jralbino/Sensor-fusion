import json
import os
from pathlib import Path
from tqdm import tqdm

from config.utils.path_manager import path_manager # Import the consolidated path manager

def convert_bdd_folder_to_coco(json_folder, output_labels_dir):
    """
    Converts a FOLDER of individual JSONs (alternative BDD format)
    to COCO-compatible YOLO txt format.
    """
    
    # BDD -> COCO Mapping
    BDD_TO_COCO_MAP = {
        "pedestrian": 0, "rider": 0, "bicycle": 1, "car": 2,
        "motorcycle": 3, "bus": 5, "train": 6, "truck": 7, "traffic light": 9
    }
    
    source_path = Path(json_folder)
    out_path = Path(output_labels_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    json_files = list(source_path.glob("*.json"))
    print(f"📂 Found {len(json_files)} JSON files in {source_path}")
    
    # Fixed BDD dimensions
    IMG_W = 1280
    IMG_H = 720
    
    converted_count = 0
    
    for json_file in tqdm(json_files):
        with open(json_file, 'r') as f:
            item = json.load(f)
            
          
        labels = []
        if 'labels' in item:
            labels = item['labels']
        elif 'frames' in item and len(item['frames']) > 0:
            # BDD tracking or attribute structure
            labels = item['frames'][0].get('objects', [])
        
        # The txt filename will be the same as the json but .txt
        txt_filename = out_path / (json_file.stem + ".txt")
        
        with open(txt_filename, 'w') as out_f:
            has_valid_labels = False
            for label in labels:
                # Key adjustment based on format (category vs label)
                category = label.get('category', label.get('label'))
                
                if category in BDD_TO_COCO_MAP:
                    coco_id = BDD_TO_COCO_MAP[category]
                    
                    # Coordinates
                    box = label.get('box2d')
                    if not box: continue
                    
                    x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
                    
                    # Normalization
                    xc = ((x1 + x2) / 2) / IMG_W
                    yc = ((y1 + y2) / 2) / IMG_H
                    w = (x2 - x1) / IMG_W
                    h = (y2 - y1) / IMG_H
                    
                    out_f.write(f"{coco_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
                    has_valid_labels = True
            
            if has_valid_labels:
                converted_count += 1

    print(f"✅ Conversion finished. Valid labels generated: {converted_count}")

if __name__ == "__main__":
    # ADJUST THE PATH TO WHERE YOUR JSONS ARE LOCATED
    INPUT_FOLDER = path_manager.get("bdd_labels")
    # Output labels
    OUTPUT_DIR = path_manager.get("bdd_labels")
    
    convert_bdd_folder_to_coco(str(INPUT_FOLDER), str(OUTPUT_DIR))