import sys
import numpy as np
from pathlib import Path

from lidar_utils import setup_mocks

# Configurar mocks antes de importar mmdet3d
setup_mocks()

from mmdet3d.utils import register_all_modules
from detectors.pointpillars import PointPillarsDetector
from detectors.centerpoint import CenterPointDetector

try: register_all_modules(init_default_scope=False)
except: pass

class ModelManager:
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.ckpt_dir = self.base_dir / "checkpoints"
        self.configs_dir = self.base_dir / "configs"
        self.models = {}

    def load_model(self, model_name):
        if model_name == 'pointpillars':
            # print("🏗️ Cargando PointPillars...")
            cfg = self.configs_dir / "pointpillars/pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d.py"
            ckpt = self.ckpt_dir / "pointpillars_nus.pth"
            self.models['pp'] = PointPillarsDetector(str(cfg), str(ckpt))
            
        elif model_name == 'centerpoint':
            # print("🎯 Cargando CenterPoint...")
            cfg = self.configs_dir / "centerpoint/centerpoint_voxel0075_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py"
            ckpt = self.ckpt_dir / "centerpoint_nus.pth"
            self.models['cp'] = CenterPointDetector(str(cfg), str(ckpt))

    def predict(self, model_key, lidar_path):
        """Ejecuta inferencia y limpia la salida."""
        model = self.models.get(model_key)
        if not model: raise ValueError(f"Modelo {model_key} no cargado")
        
        raw_dets = model.detect(lidar_path)
        
        # Normalizar salida a lista de dicts
        clean_results = []
        for d in raw_dets:
            # Soporte para diferentes formatos de salida de mmdet
            box = d['box_3d'] if 'box_3d' in d else d['boxes_3d']
            score = float(d['score']) if 'score' in d else float(d['scores_3d'])
            label = int(d['label']) if 'label' in d else int(d['labels_3d'])
            
            # Convertir Tensores/Listas a Numpy para manipulación matemática
            if hasattr(box, 'numpy'):
                box = box.numpy()
            if isinstance(box, list):
                box = np.array(box)

            # --- CORRECCIÓN DE HEADING PARA CENTERPOINT ---
            # CenterPoint a veces tiene el Yaw invertido respecto a NuScenes/OpenCV
            if model_key == 'cp':
                 box[6] = -box[6] 

            clean_results.append({
                "box": box.tolist(), # <--- CORRECCIÓN: Convertir numpy a lista para JSON
                "score": score, 
                "label": label
            })
        return clean_results