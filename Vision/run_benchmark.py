import json
import os
import sys
from pathlib import Path
from datetime import datetime

# --- BLOQUE DE CORRECCIÓN DE PATH ---
# Obtenemos la ruta absoluta del archivo actual y subimos 2 niveles (Vision -> Raíz)
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # Raíz del proyecto
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # Añadimos la raíz al path de Python
# ------------------------------------

from ultralytics import RTDETR, YOLO
from config.global_config import USER_SETTINGS, DATA_PATHS, MODEL_PATHS

def run_benchmark():
    # 1. Definir Modelos a Evaluar
    models_to_test = {
        # Convertimos a string por seguridad, aunque YOLO acepta Path
        "YOLO11-L (Coco)": str(MODEL_PATHS["yolo11l"]),
        "YOLO11-X (Coco)": str(MODEL_PATHS["yolo11x"]),  
        "RTDETR-L (Coco)": str(MODEL_PATHS["rtdetr_l"]),
        "RTDETR-BDD (Finetuned)": str(MODEL_PATHS["rtdetr_bdd"])
    }

    # 2. Definir Datasets
    datasets = {
        # Usamos rutas relativas seguras o absolutas desde DATA_PATHS si es necesario
        "BDD100K": str(Path("Vision/config/bdd_det_train.yaml").absolute()), 
        "NuScenes": "Vision/config/nuscenes.yaml"
    }

    results_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "datasets": {}
    }

    output_project = DATA_PATHS["output_vision"]

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
                    device='cuda', # Usa GPU
                    verbose=False,
                    plots=False,
                    # 1. 'project': Ruta base absoluta/segura (evita ./runs)
                    project=str(output_project),  
                    
                    # 2. 'name': Subcarpeta (ej: benchmark_YOLO11-L)
                    name=f'benchmark_{model_name}',
                    
                    # 3. 'exist_ok': Evita crear benchmark_YOLO11-L2, L3, etc.
                    exist_ok=True 
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
                
                # Desglose por clases
                class_map = {}
                for i, cname in enumerate(metrics.names.values()):
                    if i < len(metrics.box.maps):
                        class_map[cname] = round(metrics.box.maps[i], 4)
                entry["per_class"] = class_map

                results_data["datasets"][dataset_name].append(entry)
                print(f"      ✅ mAP: {entry['mAP50-95']} | Speed: {entry['Inference_Time_ms']}ms")

            except Exception as e:
                print(f"      ❌ Error: {e}")

    # 3. Guardar Resultados usando la ruta centralizada
    output_file = DATA_PATHS["output_vision"] / "data/benchmark_results.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results_data, f, indent=4)
    
    print(f"\n💾 Resultados guardados en {output_file}")

if __name__ == "__main__":
    run_benchmark()