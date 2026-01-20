import os
import shutil
from pathlib import Path
from tqdm import tqdm

def filter_dataset():
    # Rutas (Ajusta a tu servidor)
    base_dir = Path("Vision/data/raw/bdd100k")
    source_imgs = base_dir / "images/100k/train"
    source_lbls = base_dir / "labels/100k/train"
    
    # Nuevo Dataset "Balanceado"
    target_dir = base_dir / "balanced_train"
    (target_dir / "images").mkdir(parents=True, exist_ok=True)
    (target_dir / "labels").mkdir(parents=True, exist_ok=True)

    # IDs de clases que nos urgen mejorar (Basado en tu YAML)
    # 0: person, 9: traffic light, 11: stop sign
    PRIORITY_CLASSES = [0, 9, 11] 

    print("🕵️ Filtrando dataset para priorizar Personas y Señales...")
    
    count = 0
    for label_file in tqdm(list(source_lbls.glob("*.txt"))):
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        has_priority = False
        for line in lines:
            cls_id = int(line.split()[0])
            if cls_id in PRIORITY_CLASSES:
                has_priority = True
                break
        
        # Si la imagen tiene lo que buscamos, la guardamos
        if has_priority:
            shutil.copy(source_imgs / (label_file.stem + ".jpg"), target_dir / "images")
            shutil.copy(label_file, target_dir / "labels")
            count += 1

    print(f"✅ Nuevo dataset creado con {count} imágenes ricas en peatones/señales.")
    print(f"📁 Ubicación: {target_dir}")

if __name__ == "__main__":
    filter_dataset()