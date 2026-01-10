import json
import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURACIÓN DE CLASES COCO (Para modelos pre-entrenados) ---
# Mapeo oficial de COCO (80 clases). Definimos las relevantes para tráfico.
COCO_CLASSES = {
    0: "Person",
    1: "Bicycle",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
    9: "Traffic Light",
    11: "Stop Sign"
}

# Colores BGR para visualización
COLORS = {
    0: (0, 0, 255),      # Person -> Rojo
    1: (0, 165, 255),    # Bicycle -> Naranja
    2: (255, 0, 0),      # Car -> Azul
    3: (0, 255, 255),    # Motorcycle -> Amarillo
    5: (255, 0, 255),    # Bus -> Magenta
    7: (255, 255, 0),    # Truck -> Cyan
    9: (0, 255, 0),      # Traffic Light -> Verde
    11: (200, 200, 200)  # Stop Sign -> Gris
}

def load_predictions(json_path):
    """
    Carga el JSON y retorna un diccionario de búsqueda y un set de imágenes encontradas.
    """
    print(f"Cargando resultados de: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Mapa hash: "imagen1.jpg" -> {detecciones...}
    lookup = {item['image_name']: item for item in data['results']}
    
    # Set de nombres de imágenes presentes en este JSON
    image_names = set(lookup.keys())
    
    meta = data.get('meta', {})
    return lookup, image_names, meta

def draw_detections(img, result_entry, model_name):
    """
    Dibuja cajas y etiquetas sobre la imagen.
    """
    canvas = img.copy()
    
    # Si no hay entrada para esta imagen en este modelo específico
    if result_entry is None:
        cv2.putText(canvas, f"{model_name} (No Data)", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return canvas

    # Dibujar cada detección
    for det in result_entry['detections']:
        cls_id = det['class_id']
        conf = det['confidence']
        bbox = det['bbox']
        
        # Obtener nombre y color
        class_name = COCO_CLASSES.get(cls_id, str(cls_id))
        color = COLORS.get(cls_id, (128, 128, 128)) # Gris si no es relevante
        
        # Solo dibujamos si es una clase de tráfico relevante (o si quieres ver todo, comenta el if)
        if cls_id in COCO_CLASSES:
            x1, y1, x2, y2 = map(int, bbox)
            
            # Caja
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            
            # Etiqueta
            label_text = f"{class_name} {conf:.2f}"
            
            # Fondo pequeño para el texto (legibilidad)
            (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(canvas, (x1, y1 - 20), (x1 + text_w, y1), color, -1)
            cv2.putText(canvas, label_text, (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    # Info General del Modelo (Esquina superior izquierda)
    latency = result_entry.get('inference_ms', 0)
    info_text = f"{model_name} | {latency}ms"
    
    cv2.rectangle(canvas, (0, 0), (300, 40), (0, 0, 0), -1)
    cv2.putText(canvas, info_text, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return canvas

def create_comparison_video(images_dir, json_files, output_dir, fps=5):
    """
    Genera video solo con las imágenes presentes en los JSONs.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    images_path_root = Path(images_dir)
    
    model_data = []
    all_detected_images = set()

    # 1. Cargar JSONs y recolectar nombres de imágenes
    for name, path in json_files:
        if not os.path.exists(path):
            print(f"Advertencia: No se encontró el archivo {path}")
            continue
            
        lookup, img_names, meta = load_predictions(path)
        model_data.append({
            "name": name,
            "lookup": lookup,
            "meta": meta
        })
        # Agregamos al set maestro de imágenes a procesar (Unión de todos los JSONs)
        all_detected_images.update(img_names)

    # 2. Ordenar imágenes alfabéticamente para mantener secuencia temporal
    sorted_image_names = sorted(list(all_detected_images))
    
    if not sorted_image_names:
        print("No se encontraron imágenes en los JSONs proporcionados.")
        return

    print(f"Imágenes totales a procesar (encontradas en JSONs): {len(sorted_image_names)}")

    # 3. Preparar Video Writer (leemos la primera imagen válida para dimensiones)
    first_img_path = images_path_root / sorted_image_names[0]
    if not first_img_path.exists():
        print(f"Error crítico: La imagen {first_img_path} listada en el JSON no existe en {images_dir}")
        return
        
    first_img = cv2.imread(str(first_img_path))
    h, w, _ = first_img.shape
    
    # Ancho total = ancho imagen * cantidad de modelos
    sbs_w = w * len(model_data)
    sbs_out_path = output_path / "yoloxVSrtdetr_l.mp4"
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(sbs_out_path), fourcc, fps, (sbs_w, h))
    
    # 4. Loop Principal
    for img_name in tqdm(sorted_image_names):
        full_img_path = images_path_root / img_name
        
        # Verificar existencia física
        if not full_img_path.exists():
            continue
            
        original_img = cv2.imread(str(full_img_path))
        if original_img is None: continue
            
        frames_row = []
        
        # Generar frame para cada modelo
        for m in model_data:
            result_entry = m["lookup"].get(img_name)
            frame_painted = draw_detections(original_img, result_entry, m["name"])
            frames_row.append(frame_painted)
            
        # Unir horizontalmente
        sbs_frame = np.hstack(frames_row)
        writer.write(sbs_frame)
        
    writer.release()
    print(f"Video generado exitosamente en: {sbs_out_path}")

if __name__ == "__main__":
    # --- RUTAS DE ENTRADA ---
    
    # Carpeta donde están las imágenes originales .jpg
    IMAGES_DIR = "data/raw/bdd100k/images/100k/val"
    
    # Archivos JSON generados anteriormente
    JSONS_TO_COMPARE = [
        ("YOLO11-X", "output/predictions/yolo11x_conf50.json"),
        ("RTDETR-L", "output/predictions/rtdetr_l_conf50.json") # Descomentar si tienes el segundo
    ]
    
    OUTPUT_FOLDER = "output/videos"
    
    create_comparison_video(IMAGES_DIR, JSONS_TO_COMPARE, OUTPUT_FOLDER, fps=5)