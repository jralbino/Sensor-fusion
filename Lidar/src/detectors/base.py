from abc import ABC, abstractmethod

class BaseLidarDetector(ABC):
    def __init__(self, config_path, checkpoint_path, device='cuda:0'):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.model = self.load_model()

    @abstractmethod
    def load_model(self):
        """Inicializa el modelo usando la librería backend."""
        pass

    @abstractmethod
    def detect(self, pcd_path):
        """
        Args:
            pcd_path (str): Ruta al archivo .bin del LiDAR.
        Returns:
            list[dict]: Lista estandarizada de detecciones.
            [{'box_3d': [x,y,z,dx,dy,dz,rot], 'score': float, 'label': int}, ...]
        """
        pass