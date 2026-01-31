# -*- coding: utf-8 -*-
"""
Factory for creating lane detectors.
"""

from .yolop_detector import YOLOPDetector
from .ufld_detector import UFLDDetector
from .polylanenet_detector import PolyLaneNetDetector
from .deeplab_detector import DeepLabDetector
from .segformer_detector import SegFormerDetector
from config.utils.path_manager import path_manager

class LaneDetectorFactory:
    @staticmethod
    def create_detector(name: str):
        """
        Creates a lane detector instance based on the given name.

        Args:
            name (str): The name of the detector.

        Returns:
            An instance of a lane detector.
        """
        try:
            if "YOLOP" in name:
                return YOLOPDetector()
            elif "UFLD" in name:
                model_path = path_manager.get_model("ufld", check_exists=True)
                return UFLDDetector(model_path=str(model_path))
            elif "PolyLaneNet" in name:
                model_path = path_manager.get_model("polylanenet", check_exists=True)
                return PolyLaneNetDetector(model_path=str(model_path))
            elif "DeepLab" in name:
                return DeepLabDetector()
            elif "SegFormer" in name:
                return SegFormerDetector()
            else:
                raise ValueError(f"Unknown lane detector: {name}")
        except FileNotFoundError as e:
            # Re-raise as a more specific error for the UI
            raise RuntimeError(f"Model file not found for {name}: {e}")
        except Exception as e:
            # General exceptions
            raise RuntimeError(f"Error loading lane detector {name}: {e}")

def get_lane_detector(name: str):
    """
    Function to get a lane detector.
    """
    return LaneDetectorFactory.create_detector(name)

