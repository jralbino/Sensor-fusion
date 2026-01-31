import os
from pathlib import Path
from config.utils.path_manager import path_manager
from nuscenes.nuscenes import NuScenes

# Initialize nuScenes (adjust dataroot according to your configuration)
nusc = NuScenes(version='v1.0-mini', dataroot=str(path_manager.get('nuscenes')), verbose=True)

def demo_lidar_projection(scene_index=0):
    """
    Projects LiDAR onto the camera and saves the image in Fusion/runs/test/,
    creating the folder automatically if it doesn't exist.
    """
    
    # --- 1. PATH CONFIGURATION ---
    # Get the Fusion module root from path_manager
    FUSION_ROOT = path_manager.get("fusion")
    
    # Define the destination folder using path_manager
    OUTPUT_DIR = path_manager.get("fusion_runs_test", create=True)
    
    print(f"Output directory verified: {OUTPUT_DIR}")

    # --- 2. NUSCENES LOGIC ---
    my_scene = nusc.scene[scene_index]
    first_sample_token = my_scene['first_sample_token']
    my_sample = nusc.get('sample', first_sample_token)

    # Define the final file name
    output_path = OUTPUT_DIR / f"fusion_sample_{first_sample_token}.jpg"

    print(f"Rendering projection for token: {first_sample_token}")
    
    nusc.render_pointcloud_in_image(
        my_sample['token'],
        pointsensor_channel='LIDAR_TOP',
        camera_channel='CAM_FRONT',
        render_intensity=True,
        show_lidarseg=False,
        out_path=str(output_path)  # Convert Path to string for the library
    )
    
    print(f"✅ Image saved successfully in:\n{output_path}")

if __name__ == "__main__":
    demo_lidar_projection()