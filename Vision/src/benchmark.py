from ultralytics import YOLO, RTDETR
import sys
import os
from pathlib import Path
from config.utils.path_manager import path_manager


class ModelBenchmark:
    def __init__(self, data_yaml):
        self.data_yaml = data_yaml

    def run(self, models_config):
        print("\n" + "="*40)
        print("--- PHASE 3: FORMAL BENCHMARK (mAP) ---")
        print("="*40)

        results = {}
        
        output_project = path_manager.get("output_vision")

        for name, path in models_config:
            print(f"\n📊 Evaluating Model: {name}")
            print(f"   File: {path}")
            
            try:
                if "rtdetr" in path.stem:
                    model = RTDETR(path)
                else:
                    model = YOLO(path)
                
                # --- DEFINITIVE PATH CORRECTION ---
                metrics = model.val(
                    data=self.data_yaml, 
                    split='val', 
                    device='cuda', 
                    verbose=False,
                    plots=False,
                    
                    # 1. 'project': Absolute/secure base path (avoids ./runs)
                    project=str(output_project),  
                    
                    # 2. 'name': Subfolder (e.g., benchmark_YOLO11-L)
                    name=f'benchmark_{name}',
                    
                    # 3. 'exist_ok': Prevents creating benchmark_YOLO11-L2, L3, etc.
                    exist_ok=True 
                )
                
                map50_95 = metrics.box.map
                map50 = metrics.box.map50
                precision = metrics.box.mp
                recall = metrics.box.mr
                
                print(f"   ✅ Results {name}:")
                print(f"   👉 mAP@50-95: {map50_95:.4f}")
                print(f"   👉 mAP@50:    {map50:.4f}")
                print(f"   👉 Precision: {precision:.4f}")
                print(f"   👉 Recall:    {recall:.4f}")
                
                results[name] = {
                    "mAP": map50_95,
                    "mAP50": map50,
                    "Precision": precision,
                    "Recall": recall
                }
                
            except Exception as e:
                print(f"❌ Error evaluating {name}: {e}")

        return results