import json
import cv2
import time
from pathlib import Path
from tqdm import tqdm
from detectors.object_detector import ObjectDetector

class BatchPredictor:
    def __init__(self, images_dir, output_dir):
        """
        Inicializa el gestor de predicciones por lotes.
        
        Args:
            images_dir (str): Ruta a la carpeta de imágenes originales.
            output_dir (str): Ruta donde se guardarán los JSONs resultantes.
        """
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_inference(self, model_name, model_path, conf=0.5, iou=0.45, limit=None):
        """
        Ejecuta la inferencia sobre el dataset y guarda el JSON.
        
        Args:
            model_name (str): Nombre identificador (ej: "YOLO11-X").
            model_path (str): Ruta al archivo .pt (ej: "models/yolo11x.pt").
            limit (int, opcional): Número máximo de imágenes a procesar (para pruebas rápidas).
        
        Returns:
            str: Ruta del archivo JSON generado.
        """
        # 1. Definir nombre de salida
        # Limpiamos el nombre del modelo para usarlo en el archivo
        safe_name = model_name.lower().replace(" ", "").replace("-", "")
        json_filename = f"{safe_name}_conf{int(conf*100)}.json"
        output_json_path = self.output_dir / json_filename

        # Si ya existe y no queremos sobreescribir, podríamos retornar aquí.
        # Por ahora, sobreescribimos siempre para tener datos frescos.
        
        print(f"\n🚀 Iniciando Inferencia: {model_name}")
        print(f"   Modelo: {model_path}")
        print(f"   Output: {output_json_path}")

        # 2. Cargar Detector
        # Manejo de error si el modelo no existe
        if not Path(model_path).exists():
            print(f"❌ Error: El modelo no existe en: {model_path}")
            return None

        detector = ObjectDetector(model_path=model_path, conf=conf, iou=iou)
        
        # 3. Listar imágenes
        image_paths = sorted(list(self.images_dir.glob("*.jpg")))
        
        # APLICAR LIMITE (Para pruebas rápidas)
        if limit:
            print(f"⚠️  Modo prueba: Procesando solo las primeras {limit} imágenes.")
            image_paths = image_paths[:limit]
        else:
            print(f"📊 Procesando dataset completo ({len(image_paths)} imágenes).")
        
        params = detector.get_parameters()
        for key, value in params.items():
            if isinstance(value, Path):
                params[key] = str(value)

        experiment_data = {
            "meta": detector.get_parameters(),
            "results": []
        }

        # 4. Loop de inferencia
        start_global = time.time()
        
        for img_p in tqdm(image_paths, desc=f"Inferencias {model_name}"):
            img = cv2.imread(str(img_p))
            if img is None: continue
            
            detections, _, stats = detector.detect(img)
            
            experiment_data["results"].append({
                "image_name": img_p.name,
                "inference_ms": stats["inference_time_ms"],
                "detections": detections
            })

        total_time = time.time() - start_global
        print(f"✅ Finalizado en {total_time:.2f}s. Guardando JSON...")

        # 5. Guardar
        with open(output_json_path, 'w') as f:
            json.dump(experiment_data, f, indent=4, default=str)
            
        return str(output_json_path)