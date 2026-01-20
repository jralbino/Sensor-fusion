import os
import shutil
import yaml
import numpy as np
from tqdm import tqdm
from pathlib import Path

# Intentamos importar nuscenes (el usuario debe instalarlo: pip install nuscenes-devkit)
try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.geometry_utils import view_points
except ImportError:
    print("❌ Error: Necesitas instalar el devkit de NuScenes.")
    print("👉 Ejecuta: pip install nuscenes-devkit")
    exit()

def convert_nuscenes_to_yolo(nusc_root, output_root):
    nusc = NuScenes(version='v1.0-mini', dataroot=nusc_root, verbose=True)
    
    # --- MAPEO CRÍTICO: NuScenes -> COCO/BDD IDs ---
    # Alineamos las clases para que 'car' sea 2, 'person' sea 0, etc.
    # Esto permite que tus modelos pre-entrenados funcionen sin re-entrenar.
    CLASS_MAPPING = {
        'human.pedestrian.adult': 0,        # person
        'human.pedestrian.child': 0,
        'human.pedestrian.construction_worker': 0,
        'human.pedestrian.police_officer': 0,
        'vehicle.bicycle': 1,               # bicycle
        'vehicle.car': 2,                   # car
        'vehicle.motorcycle': 3,            # motorcycle
        'vehicle.bus.bendy': 5,             # bus
        'vehicle.bus.rigid': 5,
        'vehicle.truck': 7,                 # truck
        # Agregamos otros si tus modelos los soportan
    }
    
    # Crear carpetas
    out_img = Path(output_root) / "images" / "val"
    out_lbl = Path(output_root) / "labels" / "val"
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lbl, exist_ok=True)
    
    print(f"🚀 Convirtiendo NuScenes a YOLO en: {output_root}")
    
    for sample in tqdm(nusc.sample):
        # Usamos solo la cámara frontal (CAM_FRONT) para el benchmark
        cam_token = sample['data']['CAM_FRONT']
        cam_data = nusc.get('sample_data', cam_token)
        
        # 1. Copiar Imagen
        src_img_path = os.path.join(nusc_root, cam_data['filename'])
        dst_img_name = f"{sample['token']}.jpg"
        dst_img_path = out_img / dst_img_name
        
        if not dst_img_path.exists():
            shutil.copy(src_img_path, dst_img_path)
            
        # 2. Generar Etiquetas (Proyección 3D -> 2D)
        _, boxes, camera_intrinsic = nusc.get_sample_data(cam_token)
        labels = []
        
        for box in boxes:
            # Filtrar clases que no nos interesan
            if box.name not in CLASS_MAPPING:
                continue
                
            cls_id = CLASS_MAPPING[box.name]
            
            # Proyectar al plano de la imagen
            corners_3d = box.corners()
            corners_img = view_points(corners_3d, camera_intrinsic, normalize=True)[:2, :]
            
            # Obtener Bounding Box 2D
            x_min, x_max = np.min(corners_img[0]), np.max(corners_img[0])
            y_min, y_max = np.min(corners_img[1]), np.max(corners_img[1])
            
            # Normalizar para YOLO (0-1)
            W, H = 1600, 900 # Resolución NuScenes
            
            # Validar que esté dentro de la imagen
            if x_max < 0 or x_min > W or y_max < 0 or y_min > H:
                continue
                
            x_center = ((x_min + x_max) / 2) / W
            y_center = ((y_min + y_max) / 2) / H
            w_norm = (x_max - x_min) / W
            h_norm = (y_max - y_min) / H
            
            labels.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
            
        # Guardar txt
        with open(out_lbl / f"{sample['token']}.txt", 'w') as f:
            f.write('\n'.join(labels))

if __name__ == "__main__":
    # Ajusta estas rutas si es necesario
    NUSC_ROOT = "/home/jolux/sensor-fusion-bdd/Sensor-fusion/Fusion/data/sets/nuscenes"
    OUTPUT_YOLO = "/home/jolux/sensor-fusion-bdd/Sensor-fusion/Fusion/data/yolo_format"
    convert_nuscenes_to_yolo(NUSC_ROOT, OUTPUT_YOLO)