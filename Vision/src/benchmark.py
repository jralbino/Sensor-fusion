from ultralytics import YOLO, RTDETR
import sys
import os
from pathlib import Path
from utils.paths import PathManager


class ModelBenchmark:
    def __init__(self, data_yaml):
        self.data_yaml = data_yaml

    def run(self, models_config):
        print("\n" + "="*40)
        print("--- FASE 3: BENCHMARK FORMAL (mAP) ---")
        print("="*40)

        results = {}
        
        output_project = PathManager.get_data_path("output_vision")

        for name, path in models_config:
            print(f"\n📊 Evaluando Modelo: {name}")
            print(f"   Archivo: {path}")
            
            try:
                if "rtdetr" in path.stem:
                    model = RTDETR(path)
                else:
                    model = YOLO(path)
                
                # --- CORRECCIÓN DEFINITIVA DE RUTAS ---
                metrics = model.val(
                    data=self.data_yaml, 
                    split='val', 
                    device='cuda', 
                    verbose=False,
                    plots=False,
                    
                    # 1. 'project': Ruta base absoluta/segura (evita ./runs)
                    project=str(output_project),  
                    
                    # 2. 'name': Subcarpeta (ej: benchmark_YOLO11-L)
                    name=f'benchmark_{name}',
                    
                    # 3. 'exist_ok': Evita crear benchmark_YOLO11-L2, L3, etc.
                    exist_ok=True 
                )
                
                map50_95 = metrics.box.map
                map50 = metrics.box.map50
                precision = metrics.box.mp
                recall = metrics.box.mr
                
                print(f"   ✅ Resultados {name}:")
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
                print(f"❌ Error evaluando {name}: {e}")

        return results