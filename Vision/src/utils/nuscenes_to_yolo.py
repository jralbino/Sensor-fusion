import os
import shutil
import yaml
import numpy as np
from tqdm import tqdm
from pathlib import Path
import sys

# Import the consolidated path manager
from config.utils.path_manager import path_manager

# Intentamos importar nuscenes (el usuario debe instalarlo: pip install nuscenes-devkit)
try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.geometry_utils import view_points
except ImportError:
    print("❌ Error: Necesitas instalar el devkit de NuScenes.")
    print("👉 Ejecuta: pip install nuscenes-devkit")
    sys.exit(1) # Use sys.exit(1) for error exit

def convert_nuscenes_to_yolo(nusc_root, output_root):
    nusc = NuScenes(version='v1.0-mini', dataroot=nusc_root, verbose=True)
    
    # --- Critical Mapping: NuScenes -> COCO/BDD IDs ---
    # We align classes so that 'car' is 2, 'person' is 0, etc.
    # This allows pre-trained models to work without retraining.
    CLASS_MAPPING = {
        'human.pedestrian.adult': 0,        # person
        'human.pedestrian.child': 0,
        'human.pedestrian.construction_worker': 0,
        'human.pedestrian.police_officer': 0,
        'vehicle.bicycle': 1,               # bicycle
        'vehicle.car': 2,                   # car
        'vehicle.motorcycle': 3,            # motorcycle
        'vehicle.bus.bendy': 5,             # bus
        'vehicle.bus.rigid': 5,
        'vehicle.truck': 7,                 # truck
        # Add others if your models support them
    }
    
    # Create folders
    out_img = Path(output_root) / "images" / "val"
    out_lbl = Path(output_root) / "labels" / "val"
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lbl, exist_ok=True)
    
    print(f"🚀 Converting NuScenes to YOLO in: {output_root}")
    
    for sample in tqdm(nusc.sample):
        # We use only the front camera (CAM_FRONT) for the benchmark
        cam_token = sample['data']['CAM_FRONT']
        cam_data = nusc.get('sample_data', cam_token)
        
        # 1. Copy Image
        src_img_path = os.path.join(nusc_root, cam_data['filename'])
        dst_img_name = f"{sample['token']}.jpg"
        dst_img_path = out_img / dst_img_name
        
        if not dst_img_path.exists():
            shutil.copy(src_img_path, dst_img_path)
            
        # 2. Generate Labels (3D -> 2D Projection)
        _, boxes, camera_intrinsic = nusc.get_sample_data(cam_token)
        labels = []
        
        for box in boxes:
            # Filter classes we are not interested in
            if box.name not in CLASS_MAPPING:
                continue
                
            cls_id = CLASS_MAPPING[box.name]
            
            # Project to image plane
            corners_3d = box.corners()
            corners_img = view_points(corners_3d, camera_intrinsic, normalize=True)[:2, :]
            
            # Get 2D Bounding Box
            x_min, x_max = np.min(corners_img[0]), np.max(corners_img[0])
            y_min, y_max = np.min(corners_img[1]), np.max(corners_img[1])
            
            # Normalize for YOLO (0-1)
            W, H = 1600, 900 # NuScenes resolution
            
            # --- CORRECTION: CLIP COORDINATES ---
            # This ensures they are never negative or larger than the image
            x_min = max(0, min(W, x_min))
            x_max = max(0, min(W, x_max))
            y_min = max(0, min(H, y_min))
            y_max = max(0, min(H, y_max))

            # Check if the box is still valid after clipping
            # (e.g., if clipping resulted in a width of 0, ignore it)
            if (x_max - x_min) < 1 or (y_max - y_min) < 1:
                continue

            # --- YOLO CALCULATION ---
            # Now it's safe to normalize
            x_center = ((x_min + x_max) / 2) / W
            y_center = ((y_min + y_max) / 2) / H
            w_norm = (x_max - x_min) / W
            h_norm = (y_max - y_min) / H
            
            # Final safety check for floating point errors (optional but recommended)
            x_center = min(max(0.0, x_center), 1.0)
            y_center = min(max(0.0, y_center), 1.0)
            w_norm = min(max(0.0, w_norm), 1.0)
            h_norm = min(max(0.0, h_norm), 1.0)

            labels.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
            
        # Save txt
        with open(out_lbl / f"{sample['token']}.txt", 'w') as f:
            f.write('\n'.join(labels))

if __name__ == "__main__":
    # Adjust these paths if necessary
    NUSC_ROOT = str(path_manager.get_data_detail("nuscenes_base"))
    OUTPUT_YOLO = str(path_manager.get("fusion_yolo_output"))
    convert_nuscenes_to_yolo(NUSC_ROOT, OUTPUT_YOLO)