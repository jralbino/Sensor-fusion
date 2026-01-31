import sys
import cv2
from pathlib import Path

from config.utils.path_manager import path_manager # Import the consolidated path manager

from Lidar.src.lidar_utils import DataLoader
from Lidar.src.lidar_models import ModelManager
from Lidar.src.lidar_vis import LidarVisualizer

def main():
    print("🚀 Starting COMPARATIVE test (PointPillars vs CenterPoint)...")
    
    # 1. Initialize Modules
    data_loader = DataLoader()
    model_mgr = ModelManager(base_dir=path_manager.BASE_DIR) # Use path_manager.BASE_DIR
    visualizer = LidarVisualizer()
    
    # Ensure debug output directory exists
    debug_output_dir = path_manager.get("lidar_debug_output", create=True)
    
    # 2. Load BOTH Models
    model_mgr.load_model('pointpillars')
    model_mgr.load_model('centerpoint')
    
    # 3. Get Data from the First Frame
    scene = data_loader.nusc.scene[0]
    token = scene['first_sample_token']
    calib_data = data_loader.get_sample_data(token)
    
    print(f"📸 Processing frame: {token}")
    
    # --- MODEL 1: POINTPILLARS ---
    print("\n--- 1. Running PointPillars ---")
    dets_pp = model_mgr.predict('pp', calib_data['lidar_path'])
    print(f"✅ Detections: {len(dets_pp)}")
    
    img_pp = visualizer.project_lidar_to_cam(
        calib_data['cam_path'],
        calib_data['lidar_path'],
        dets_pp,
        calib_data,
        title="PointPillars (Undistorted)"
    )
    cv2.imwrite(str(debug_output_dir / "debug_pointpillars.jpg"), img_pp)
    print("💾 Saved: debug_pointpillars.jpg")

    # --- MODEL 2: CENTERPOINT ---
    print("\n--- 2. Running CenterPoint ---")
    dets_cp = model_mgr.predict('cp', calib_data['lidar_path'])
    print(f"✅ Detections: {len(dets_cp)}")
    
    img_cp = visualizer.project_lidar_to_cam(
        calib_data['cam_path'],
        calib_data['lidar_path'],
        dets_cp,
        calib_data,
        title="CenterPoint (Undistorted)"
    )
    cv2.imwrite(str(debug_output_dir / "debug_centerpoint.jpg"), img_cp)
    print("💾 Saved: debug_centerpoint.jpg")

    print("\n🏁 Test finished. Compare the two generated images.")

if __name__ == "__main__":
    main()