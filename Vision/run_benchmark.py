import json
import os
from ultralytics import RTDETR, YOLO
from pathlib import Path
from datetime import datetime

def run_benchmark():
    # 1. Definir Modelos a Evaluar
    # Nombre visual : Ruta del archivo
    models_to_test = {
        "YOLO11-L (Coco)": "Vision/models/yolo11l.pt",
        "RTDETR-L (Coco)": "Vision/models/rtdetr-l.pt",
        "RTDETR-BDD (Finetuned)": "Vision/models/rtdetr-bdd-best.pt"
    }

    # 2. Definir Datasets
    # Nombre : Ruta del YAML
    datasets = {
        "BDD100K": "Vision/config/bdd_det_train.yaml", # Usamos el mismo yaml de train que apunta a val
        # "NuScenes": "Vision/config/nuscenes.yaml"      # Descomentar si tienes los datos listos
    }

    results_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "datasets": {}
    }

    print("🚀 Iniciando Benchmark Masivo...")

    for dataset_name, yaml_path in datasets.items():
        print(f"\n📂 Dataset: {dataset_name} ({yaml_path})")
        results_data["datasets"][dataset_name] = []

        for model_name, model_path in models_to_test.items():
            print(f"   👉 Evaluando: {model_name}...")
            
            try:
                # Cargar modelo
                if "rtdetr" in model_path.lower():
                    model = RTDETR(model_path)
                else:
                    model = YOLO(model_path)

                # Ejecutar validación
                metrics = model.val(
                    data=yaml_path,
                    split='val',
                    device=0, # Usa GPU
                    verbose=False,
                    plots=False
                )

                # Extraer métricas clave
                entry = {
                    "model": model_name,
                    "mAP50-95": round(metrics.box.map, 4),
                    "mAP50": round(metrics.box.map50, 4),
                    "Precision": round(metrics.box.mp, 4),
                    "Recall": round(metrics.box.mr, 4),
                    "Inference_Time_ms": round(metrics.speed['inference'], 2)
                }
                
                # Desglose por clases (para ver el problema de 'Person')
                # metrics.box.maps es un array con el mAP50-95 de cada clase
                class_map = {}
                for i, cname in enumerate(metrics.names.values()):
                    if i < len(metrics.box.maps):
                        class_map[cname] = round(metrics.box.maps[i], 4)
                entry["per_class"] = class_map

                results_data["datasets"][dataset_name].append(entry)
                print(f"      ✅ mAP: {entry['mAP50-95']} | Speed: {entry['Inference_Time_ms']}ms")

            except Exception as e:
                print(f"      ❌ Error: {e}")

    # 3. Guardar Resultados para la App
    output_file = "Vision/data/benchmark_results.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results_data, f, indent=4)
    
    print(f"\n💾 Resultados guardados en {output_file}")

if __name__ == "__main__":
    run_benchmark()