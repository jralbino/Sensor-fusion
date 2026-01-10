import json
import os
from pathlib import Path
from tqdm import tqdm

def convert_bdd_to_yolo(json_path, images_dir, output_labels_dir):
    """
    Convierte las anotaciones de BDD100K (JSON) al formato estándar de YOLO (TXT).
    Maneja la normalización de coordenadas.
    """
    
    # Mapeo de clases de BDD100K a índices para YOLO
    # BDD tiene 10 clases para detección
    bdd_classes = {
        "pedestrian": 0, "rider": 1, "car": 2, "truck": 3,
        "bus": 4, "train": 5, "motorcycle": 6, "bicycle": 7,
        "traffic light": 8, "traffic sign": 9
    }
    
    # Crear carpeta de salida
    out_path = Path(output_labels_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Cargando {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    print(f"Convirtiendo etiquetas para {len(data)} imágenes...")
    
    for item in tqdm(data):
        img_name = item['name']
        
        # Verificar si la imagen existe en nuestra carpeta (por si usas el mini dataset)
        # Si usas el dataset completo, esto ayuda a saltar etiquetas de imágenes que no tienes
        image_path = Path(images_dir) / img_name
        if not image_path.exists():
            continue
            
        # Dimensiones de BDD100K (usualmente 1280x720)
        # Es mejor leerlas del JSON si están, o asumir standard si se conoce.
        # BDD json no siempre trae w/h directo en el root, pero sabemos que son 1280x720
        img_w = 1280
        img_h = 720
        
        txt_filename = out_path / (Path(img_name).stem + ".txt")
        
        with open(txt_filename, 'w') as out_f:
            if 'labels' not in item:
                continue
                
            for label in item['labels']:
                if label['category'] not in bdd_classes:
                    continue
                
                class_id = bdd_classes[label['category']]
                box = label['box2d']
                
                # Conversión a formato YOLO: x_center, y_center, width, height (NORMALIZADOS 0-1)
                x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
                
                # Calcular centro y ancho/alto
                w = x2 - x1
                h = y2 - y1
                x_center = x1 + (w / 2)
                y_center = y1 + (h / 2)
                
                # Normalizar dividiendo por dimensiones de imagen
                x_center /= img_w
                y_center /= img_h
                w /= img_w
                h /= img_h
                
                # Escribir línea: class_id x_c y_c w h
                # Usamos 6 decimales para precisión
                out_f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

if __name__ == "__main__":
    # CONFIGURACIÓN
    # Asegúrate de apuntar a tu carpeta de validación o a tu mini-dataset
    JSON_FILE = "data/raw/bdd100k/labels/det_20/det_val.json" 
    IMAGES_DIR = "data/raw/bdd100k/images/100k/val"
    OUTPUT_DIR = "data/raw/bdd100k/labels/val_yolo_format"
    
    convert_bdd_to_yolo(JSON_FILE, images_dir=IMAGES_DIR, output_labels_dir=OUTPUT_DIR)