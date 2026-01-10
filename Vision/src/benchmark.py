from ultralytics import YOLO, RTDETR
import sys

class ModelBenchmark:
    def __init__(self, data_yaml):
        """
        Inicializa el benchmark con la configuración del dataset.
        Args:
            data_yaml (str): Ruta al archivo .yaml del dataset (ej: coco_format.yaml)
        """
        self.data_yaml = data_yaml

    def run(self, models_config):
        """
        Ejecuta la validación oficial (mAP) para una lista de modelos.
        
        Args:
            models_config (list): Lista de tuplas [("Nombre", "ruta.pt"), ...]
        """
        print("\n" + "="*40)
        print("--- FASE 3: BENCHMARK FORMAL (mAP) ---")
        print("="*40)

        results = {}

        for name, path in models_config:
            print(f"\n📊 Evaluando Modelo: {name}")
            print(f"   Archivo: {path}")
            
            try:
                # Cargar el modelo adecuado
                if "rtdetr" in path.lower():
                    model = RTDETR(path)
                else:
                    model = YOLO(path)
                
                # Ejecutar validación (Benchmark)
                # imgsz=640 es estándar, conf=0.001 es estándar para mAP (no filtrar mucho)
                metrics = model.val(
                    data=self.data_yaml, 
                    split='val', 
                    device='cuda', 
                    verbose=False,
                    plots=False
                )
                
                # Extraer métricas clave
                map50_95 = metrics.box.map    # mAP@50-95
                map50 = metrics.box.map50     # mAP@50
                precision = metrics.box.mp    # Mean Precision
                recall = metrics.box.mr       # Mean Recall
                
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