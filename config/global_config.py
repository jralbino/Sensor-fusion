# config/global_config.py
from pathlib import Path

# Raíz del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent

# Rutas de Datos
DATA_PATHS = {
    "bdd100k": BASE_DIR / "Vision/data/raw/bdd100k",
    "nuscenes": BASE_DIR / "Fusion/data/sets/nuscenes",
    "output_vision": BASE_DIR / "Vision/output",
    "output_lidar": BASE_DIR / "Lidar/runs"
}

# Rutas de Modelos
MODEL_PATHS = {
    "yolo11l": BASE_DIR / "Vision/models/yolo11l.pt",
    "yolo11x": BASE_DIR / "Vision/models/yolo11x.pt",
    "rtdetr_l": BASE_DIR / "Vision/models/rtdetr-l.pt",
    "rtdetr_bdd": BASE_DIR / "Vision/models/rtdetr-bdd-best.pt",
    "ufld": BASE_DIR / "Vision/models/tusimple_18.pth",
    "polylanenet": BASE_DIR / "Vision/models/model_2305.pt"
}

# Parámetros de Usuario (Configurables)
USER_SETTINGS = {
    "video_limit": 50,  # Límite de imágenes solicitado para videos
    "nuscenes_version": "v1.0-mini",
    "default_conf": 0.5
}