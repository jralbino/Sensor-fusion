import sys
import cv2
import numpy as np
from pathlib import Path

from config.utils.path_manager import path_manager # Import the consolidated path manager

from Lidar.src.lidar_utils import DataLoader
from Lidar.src.lidar_models import ModelManager
from Lidar.src.lidar_vis import LidarVisualizer

def main():
    print("🚀 Starting diagnosis: 5 FRAME SEQUENCE...")
    
    # 1. Initialize
    data_loader = DataLoader()
    model_mgr = ModelManager(base_dir=path_manager.BASE_DIR) # Use path_manager.BASE_DIR
    visualizer = LidarVisualizer()
    
    # Load models
    model_mgr.load_model('pointpillars')
    model_mgr.load_model('centerpoint')
    
    # Get first token
    scene = data_loader.nusc.scene[0]
    token = scene['first_sample_token']
    
    # Output directory
    output_dir = path_manager.get("lidar_debug_sequence_output", create=True) # Use path_manager
    
    # 2. 5-Frame Loop
    for i in range(5):
        print(f"\n📸 Processing Frame {i+1}/5 [Token: {token}]")
        calib = data_loader.get_sample_data(token)
        
        # --- A. PointPillars ---
        dets_pp = model_mgr.predict('pp', calib['lidar_path'])
        print(f"   🔹 PointPillars: {len(dets_pp)} detections")
        
        img_pp = visualizer.project_lidar_to_cam(
            calib['cam_path'], calib['lidar_path'], dets_pp, calib,
            title=f"Frame {i+1}: PointPillars"
        )
        cv2.imwrite(str(output_dir / f"frame_{i+1}_pointpillars.jpg"), img_pp)
        
        # --- B. CenterPoint ---
        dets_cp = model_mgr.predict('cp', calib['lidar_path'])
        print(f"   🔹 CenterPoint: {len(dets_cp)} detections")
        
        img_cp = visualizer.project_lidar_to_cam(
            calib['cam_path'], calib['lidar_path'], dets_cp, calib,
            title=f"Frame {i+1}: CenterPoint"
        )
        cv2.imwrite(str(output_dir / f"frame_{i+1}_centerpoint.jpg"), img_cp)
        
        # --- DEBUG: Print coordinates of the most confident detection ---
        # This will help us see if the Z-coordinate or the size make sense
        if dets_cp:
            # Sort by score
            top_det = sorted(dets_cp, key=lambda x: x['score'], reverse=True)[0]
            print(f"   🔍 Top Box CenterPoint: Pos={np.round(top_det['box'][:3], 2)} | Dim={np.round(top_det['box'][3:6], 2)}")

        # Advance to the next frame
        sample_record = data_loader.nusc.get('sample', token)
        if sample_record['next']:
            token = sample_record['next']
        else:
            print("⚠️ End of scene reached before 5 frames.")
            break

    print(f"\n✅ Process completed. Check the folder: {output_dir}")

if __name__ == "__main__":
    main()