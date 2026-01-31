import os
import shutil
from pathlib import Path
from tqdm import tqdm

from config.utils.path_manager import path_manager # Import the consolidated path manager

def filter_dataset():
    # Paths (Adjust to your server)
    base_dir = path_manager.get("bdd100k")
    source_imgs = path_manager.get("bdd_images_train")
    source_lbls = path_manager.get("bdd_labels")
    
    # New "Balanced" Dataset
    target_dir = path_manager.get_data_detail("bdd_balanced_train_base", create=True) # Use path_manager.get_data_detail
    (target_dir / "images").mkdir(parents=True, exist_ok=True)
    (target_dir / "labels").mkdir(parents=True, exist_ok=True)

    # Class IDs that urgently need improvement (Based on your YAML)
    # 0: person, 9: traffic light, 11: stop sign
    PRIORITY_CLASSES = [0, 9, 11] 

    print("🕵️ Filtering dataset to prioritize Persons and Signs...")
    
    count = 0
    for label_file in tqdm(list(source_lbls.glob("*.txt"))):
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        has_priority = False
        for line in lines:
            cls_id = int(line.split()[0])
            if cls_id in PRIORITY_CLASSES:
                has_priority = True
                break
        
        # If the image has what we are looking for, we save it
        if has_priority:
            shutil.copy(source_imgs / (label_file.stem + ".jpg"), target_dir / "images")
            shutil.copy(label_file, target_dir / "labels")
            count += 1

    print(f"✅ New dataset created with {count} images rich in pedestrians/signs.")
    print(f"📁 Location: {target_dir}")

if __name__ == "__main__":
    filter_dataset()