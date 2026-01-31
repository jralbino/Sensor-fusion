from .object_detector import ObjectDetector
from .object_detector_base import ObjectDetectorBase

def get_object_detector(model_name: str, model_path: str, **kwargs) -> ObjectDetectorBase:
    """
    Factory function for object detectors.
    """
    if model_name.lower() in ["yolo", "rtdetr"]:
        return ObjectDetector(model_path=model_path, **kwargs)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
