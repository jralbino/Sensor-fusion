import sys
import numpy as np
from pathlib import Path

# The setup_mocks is imported here and should remain.
from .lidar_utils import setup_mocks
setup_mocks()

# Import the consolidated path manager
from config.utils.path_manager import path_manager

from mmdet3d.utils import register_all_modules
from mmdet3d.apis import init_model, inference_detector

try: register_all_modules(init_default_scope=False)
except: pass

class ModelManager:
    def __init__(self): # Removed base_dir parameter
        self.base_dir = path_manager.BASE_DIR # Use path_manager.BASE_DIR
        self.ckpt_dir = path_manager.get("lidar_checkpoints") # Use path_manager
        self.configs_dir = path_manager.get("lidar_configs") # Use path_manager
        self.models = {}

    def _check_files(self, cfg_path, ckpt_path):
        missing = []
        if not Path(cfg_path).exists(): missing.append(f"Config: {cfg_path}")
        if not Path(ckpt_path).exists(): missing.append(f"Weights: {ckpt_path}")
        if missing:
            raise FileNotFoundError(f"❌ Missing files:\n" + "\n".join(missing))

    def load_model(self, model_name):
        key = model_name.lower().replace("-", "").replace(" ", "").replace("_", "")
        cfg, ckpt, m_key = None, None, None

        if 'pointpillars' in key or key == 'pp':
            cfg = path_manager.get_model_detail("pointpillars_cfg") # Use path_manager
            ckpt = path_manager.get_model_detail("pointpillars_ckpt") # Use path_manager
            m_key = 'pp'
            
        elif 'centerpoint' in key or key == 'cp':
            cfg = path_manager.get_model_detail("centerpoint_cfg") # Use path_manager
            ckpt = path_manager.get_model_detail("centerpoint_ckpt") # Use path_manager
            m_key = 'cp'
            
        else:
            raise ValueError(f"Model '{model_name}' not recognized.")

        self._check_files(cfg, ckpt)
        self.models[m_key] = init_model(str(cfg), str(ckpt), device='cuda:0')

    def predict(self, model_key, lidar_path):
        model = self.models.get(model_key)
        if not model: 
            raise ValueError(f"Model {model_key} not loaded. Call load_model first.")
        
        result, data = inference_detector(model, lidar_path)
        
        pred_instances = result.pred_instances_3d
        bboxes_3d = pred_instances.bboxes_3d.tensor.cpu().numpy()
        scores = pred_instances.scores_3d.cpu().numpy()
        labels = pred_instances.labels_3d.cpu().numpy()
        
        clean_results = []
        for box, score, label in zip(bboxes_3d, scores, labels):
            if model_key == 'cp':
                 box[6] = -box[6] 
            
            clean_results.append({
                "box": box.tolist(), 
                "score": float(score), 
                "label": int(label),
                "model": model_key
            })
        return clean_results