from ultralytics import YOLO, RTDETR
import sys
from pathlib import Path

class ModelBenchmark:
    def __init__(self, data_yaml):
        self.data_yaml = data_yaml

    def run(self, models_config):
        print("\n" + "="*40)
        print("--- FASE 3: BENCHMARK FORMAL (mAP) ---")
        print("="*40)

        results = {}

        for name, path in models_config:
            print(f"\n📊 Evaluando Modelo: {name}")
            print(f"   Archivo: {path}")
            
            try:
                if "rtdetr" in path.lower():
                    model = RTDETR(path)
                else:
                    model = YOLO(path)
                
                # --- CORRECCIÓN AQUÍ ---
                # Redirigir la salida del benchmark a 'output/runs/benchmark'
                metrics = model.val(
                    data=self.data_yaml, 
                    split='val', 
                    device='cuda', 
                    verbose=False,
                    plots=False,
                    project='output/runs',  # <--- Carpeta base correcta
                    name=f'benchmark_{name}' # <--- Subcarpeta organizada por modelo
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