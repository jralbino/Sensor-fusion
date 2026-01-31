import sys
import json
import cv2
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

from config.utils.path_manager import path_manager # Import the consolidated path manager

from Lidar.src.lidar_utils import DataLoader
from Lidar.src.lidar_models import ModelManager
from Lidar.src.lidar_vis import LidarVisualizer

# --- MODEL CONFIGURATION ---
AVAILABLE_MODELS = {
    'pointpillars': 'pp',
    'centerpoint': 'cp'
}

def parse_args():
    parser = argparse.ArgumentParser(description="LiDAR Fusion Benchmark Pipeline")
    parser.add_argument('--frames', type=int, default=20, help="Number of frames to process")
    parser.add_argument('--skip_infer', action='store_true', help="Skip inference and use existing JSONs")
    return parser.parse_args()

def run_inference(data_loader, model_mgr, frames_limit, output_dir):
    """Phase 1: Run models and save JSONs."""
    print(f"\n🧠 --- PHASE 1: INFERENCE ({frames_limit} frames) ---")
    
    scene = data_loader.nusc.scene[0]
    first_token = scene['first_sample_token']
    
    # Process each model sequentially to save VRAM
    for model_name, model_key in AVAILABLE_MODELS.items():
        print(f"\n🚀 Processing model: {model_name.upper()}...")
        
        # Load model into memory
        try:
            model_mgr.load_model(model_name)
        except Exception as e:
            print(f"❌ Error loading {model_name}: {e}")
            continue

        results = []
        token = first_token
        
        for i in tqdm(range(frames_limit), desc=f"Inferring {model_name}"):
            # Get data
            try:
                calib = data_loader.get_sample_data(token)
            except:
                break # End of scene
            
            # Predict
            detections = model_mgr.predict(model_key, calib['lidar_path'])
            
            # Save result
            results.append({
                "token": token,
                "timestamp": i, # Dummy timestamp for order
                "detections": detections
            })
            
            # Next token
            sample = data_loader.nusc.get('sample', token)
            if not sample['next']: break
            token = sample['next']
            
        # Save JSON
        json_path = output_dir / f"detections_{model_name}.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"✅ Results saved to: {json_path}")
        
        # Clean model from memory (optional, depends on your GPU)
        model_mgr.models = {} 

def generate_videos(data_loader, output_dir, limit=50):
    """Phase 2: Generate combined videos (Cam + BEV) from JSONs."""
    print(f"\n🎥 --- PHASE 2: VIDEO GENERATION ---")
    
    viz = LidarVisualizer()
    
    for model_name in AVAILABLE_MODELS.keys():
        json_path = output_dir / f"detections_{model_name}.json"
        if not json_path.exists():
            print(f"⚠️ JSON not found for {model_name}, skipping video.")
            continue
            
        print(f"🎬 Rendering video for: {model_name}...")
        
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        # Configure Video Writer
        # Cam: 1600x900. BEV: we will make it 900x900 to match the height
        # Total: 2500x900
        video_path = str(output_dir / f"video_{model_name}_combined.mp4")
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 5.0, (2500, 900))
        
        for frame_data in tqdm(data[:limit], desc="Rendering"):
            token = frame_data['token']
            dets = frame_data['detections']
            calib = data_loader.get_sample_data(token)
            
            # 1. Render Front Camera (1600 x 900)
            img_cam = viz.render_camera(
                calib['cam_path'], 
                calib['lidar_path'], 
                dets, 
                calib, 
                title=f"{model_name.upper()} - Cam Projection"
            )
            
            # 2. Render BEV (Resized to 900x900 to fit)
            img_bev_raw = viz.render_bev(
                calib['lidar_path'], 
                dets, 
                title="Eagle Eye View"
            )
            img_bev = cv2.resize(img_bev_raw, (900, 900))
            
            # 3. Concatenate (Side by Side)
            combined = np.hstack((img_cam, img_bev))
            
            out.write(combined)
            
        out.release()
        print(f"🎉 Video ready: {video_path}")

def main():
    args = parse_args()
    
    # Directories
    output_dir = path_manager.get("lidar_final_benchmark_output", create=True)
    
    # Initialize common components
    data_loader = DataLoader() # Load NuScenes only once
    model_mgr = ModelManager(base_dir=path_manager.BASE_DIR) # Use path_manager.BASE_DIR
    
    # PHASE 1: Inference
    if not args.skip_infer:
        run_inference(data_loader, model_mgr, args.frames, output_dir)
    else:
        print("⏩ Skipping inference, using existing JSONs...")
        
    # PHASE 2: Video
    generate_videos(data_loader, output_dir)
    
    print("\n✅ Pipeline completed successfully!")
    print(f"📂 Files in: {output_dir}")

if __name__ == "__main__":
    main()