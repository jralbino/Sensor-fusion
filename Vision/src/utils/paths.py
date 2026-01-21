# Vision/src/utils/paths.py
import sys
from pathlib import Path

# Añadir la raíz al path para importar la config global
ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from config.global_config import DATA_PATHS, MODEL_PATHS, BASE_DIR

class PathManager:
    BASE_DIR = BASE_DIR

    @staticmethod
    def get_data_path(key):
        return DATA_PATHS.get(key)

    @staticmethod
    def get_model_path(key):
        return MODEL_PATHS.get(key)

    @staticmethod
    def ensure_dir(path):
        Path(path).mkdir(parents=True, exist_ok=True)
        return path