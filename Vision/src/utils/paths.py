import os
from pathlib import Path

class PathManager:
    # Definimos la raíz del proyecto (Vision/)
    # Vision/src/utils/paths.py -> .parent (utils) -> .parent (src) -> .parent (Vision)
    BASE_DIR = Path(__file__).resolve().parent.parent.parent

    @staticmethod
    def get_path(*args):
        """Construye una ruta absoluta dentro del proyecto Vision."""
        return PathManager.BASE_DIR.joinpath(*args)

    @staticmethod
    def ensure_dir(path):
        """Crea el directorio si no existe."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    # Rutas estandarizadas (usando la variable de clase BASE_DIR)
    MODELS = BASE_DIR / "models"
    DATA = BASE_DIR / "data"
    OUTPUT = BASE_DIR / "output"
    CONFIG = BASE_DIR / "config"