import json
import cv2
import os
from pathlib import Path
from tqdm import tqdm
from detectors.object_detector import ObjectDetector

def run_batch_inference(image_folder, output_json_path, model_path, conf=0.25, iou=0.45):
    """
    Corre inferencia en un folder y guarda un JSON estandarizado.
    """
    # 1. Instanciar detector con parámetros específicos
    detector = ObjectDetector(model_path=model_path, conf=conf, iou=iou)
    
    # 2. Preparar estructura de salida
    experiment_data = {
        "meta": detector.get_parameters(),
        "results": []
    }
    
    # 3. Obtener imágenes
    image_paths = list(Path(image_folder).glob("*.jpg"))
    #########################################
    # Limitamos a 50 para prueba rápida 
    ########################################
    image_paths = image_paths[:50] 
    
    print(f"Procesando {len(image_paths)} imágenes...")
    
    for img_p in tqdm(image_paths):
        # Leer imagen
        img = cv2.imread(str(img_p))
        if img is None:
            continue
            
        # Detectar
        detections, _, stats = detector.detect(img)
        
        # Guardar resultado por imagen
        image_result = {
            "image_name": img_p.name,
            "inference_ms": stats["inference_time_ms"],
            "detections": detections
        }
        
        experiment_data["results"].append(image_result)
        
    # 4. Guardar JSON
    out_path = Path(output_json_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, 'w') as f:
        json.dump(experiment_data, f, indent=4)
        
    print(f"Resultados guardados en: {output_json_path}")

if __name__ == "__main__":
    # CONFIGURACIÓN DEL EXPERIMENTO
    
    # Ruta de imágenes 
    IMAGES_DIR = "data/raw/bdd100k/images/100k/val" 
    
    # 1. Prueba con YOLO (v8/11) 
    run_batch_inference(
        image_folder=IMAGES_DIR,
        output_json_path="output/predictions/yolo11x_conf50.json",
        model_path='yolo11x.pt',
        conf=0.50,  # Solo detecciones muy seguras
        iou=0.45
    )
    
    # 2. Prueba con RT-DETR (Transformer)
    run_batch_inference(
         image_folder=IMAGES_DIR,
         output_json_path="output/predictions/rtdetr_l_conf50.json",
         model_path='rtdetr-l.pt',
         conf=0.50,
         iou=0.45
     )