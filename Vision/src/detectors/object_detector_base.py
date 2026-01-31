from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
import numpy as np

class ObjectDetectorBase(ABC):
    """
    Base class for object detectors.
    """
    @abstractmethod
    def detect(self, image: np.ndarray, classes: List[int] = None) -> Tuple[List[Dict], np.ndarray, Dict]:
        """
        Detects objects in an image.

        Args:
            image (np.ndarray): The input image.
            classes (List[int], optional): A list of class IDs to detect. If None, all classes are detected. Defaults to None.

        Returns:
            Tuple[List[Dict], np.ndarray, Dict]: A tuple containing:
                - A list of detections. Each detection is a dictionary with the following keys:
                    - "class_id": The class ID of the detected object.
                    - "class_name": The class name of the detected object.
                    - "confidence": The confidence of the detection.
                    - "bbox": The bounding box of the detected object, in the format [x1, y1, x2, y2].
                - The image with the detections plotted.
                - A dictionary with statistics about the inference, such as the inference time.
        """
        pass
