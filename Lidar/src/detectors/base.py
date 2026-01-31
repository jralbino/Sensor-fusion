from abc import ABC, abstractmethod

class BaseLidarDetector(ABC):
    def __init__(self, config_path, checkpoint_path, device='cuda:0'):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.model = self.load_model()

    @abstractmethod
    def load_model(self):
        """Initializes the model using the backend library."""
        pass

    @abstractmethod
    def detect(self, pcd_path):
        """
        Args:
            pcd_path (str): Path to the LiDAR .bin file.
        Returns:
            list[dict]: Standardized list of detections.
            [{'box_3d': [x,y,z,dx,dy,dz,rot], 'score': float, 'label': int}, ...]
        """
        pass