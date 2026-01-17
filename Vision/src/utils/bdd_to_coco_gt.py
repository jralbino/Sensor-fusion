import json
import os
from pathlib import Path
from tqdm import tqdm

def convert_bdd_folder_to_coco(json_folder, output_labels_dir):
    """
    Convierte una CARPETA de JSONs individuales (formato alternativo BDD)
    a formato YOLO txt compatible con COCO.
    """
    
    # Mapeo BDD -> COCO
    BDD_TO_COCO_MAP = {
        "pedestrian": 0, "rider": 0, "bicycle": 1, "car": 2,
        "motorcycle": 3, "bus": 5, "train": 6, "truck": 7, "traffic light": 9
    }
    
    source_path = Path(json_folder)
    out_path = Path(output_labels_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    json_files = list(source_path.glob("*.json"))
    print(f"📂 Encontrados {len(json_files)} archivos JSON en {source_path}")
    
    # Dimensiones fijas BDD
    IMG_W = 1280
    IMG_H = 720
    
    converted_count = 0
    
    for json_file in tqdm(json_files):
        with open(json_file, 'r') as f:
            item = json.load(f)
            
          
        labels = []
        if 'labels' in item:
            labels = item['labels']
        elif 'frames' in item and len(item['frames']) > 0:
            # Estructura BDD tracking o atributos
            labels = item['frames'][0].get('objects', [])
        
        # El nombre del archivo txt será el mismo que el json pero .txt
        txt_filename = out_path / (json_file.stem + ".txt")
        
        with open(txt_filename, 'w') as out_f:
            has_valid_labels = False
            for label in labels:
                # Ajuste de claves según formato (category vs label)
                category = label.get('category', label.get('label'))
                
                if category in BDD_TO_COCO_MAP:
                    coco_id = BDD_TO_COCO_MAP[category]
                    
                    # Coordenadas
                    box = label.get('box2d')
                    if not box: continue
                    
                    x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
                    
                    # Normalización
                    xc = ((x1 + x2) / 2) / IMG_W
                    yc = ((y1 + y2) / 2) / IMG_H
                    w = (x2 - x1) / IMG_W
                    h = (y2 - y1) / IMG_H
                    
                    out_f.write(f"{coco_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
                    has_valid_labels = True
            
            if has_valid_labels:
                converted_count += 1

    print(f"✅ Conversión terminada. Labels válidos generados: {converted_count}")

if __name__ == "__main__":
    # Carpeta JSONs
    INPUT_FOLDER = "Vision/data/raw/bdd100k/labels/100k/train" 
    # labels 
    OUTPUT_DIR = "Vision/data/raw/bdd100k/labels/100k/train"
    
    convert_bdd_folder_to_coco(INPUT_FOLDER, OUTPUT_DIR)