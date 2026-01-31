# -*- coding: utf-8 -*-
"""
Unit tests for the detectors.
"""

import pytest
import cv2
import numpy as np
from pathlib import Path
from Vision.src.lanes.lane_factory import get_lane_detector
from Vision.src.detectors.detector_factory import get_object_detector
from config.utils.path_manager import PathManager

@pytest.fixture
def sample_image():
    """
    Fixture to load a sample image.
    """
    img_path = PathManager.get("bdd_images_val") / "b1c9c847-3bda4659.jpg"
    if not img_path.exists():
        pytest.skip("Sample image not found, skipping tests that require it.")
    return cv2.imread(str(img_path))

def test_yolop_detector_loading():
    """
    Test if the YOLOP detector can be loaded.
    """
    try:
        detector = get_lane_detector("YOLOP (Panoptic)")
        assert detector is not None
    except Exception as e:
        pytest.fail(f"YOLOP detector loading failed: {e}")

def test_yolop_detector_detection(sample_image):
    """
    Test if the YOLOP detector can process an image.
    """
    detector = get_lane_detector("YOLOP (Panoptic)")
    processed_img, latency = detector.detect(sample_image)
    assert isinstance(processed_img, np.ndarray)
    assert isinstance(latency, float)
    assert processed_img.shape == sample_image.shape

def test_ufld_detector_loading():
    """
    Test if the UFLD detector can be loaded.
    """
    try:
        detector = get_lane_detector("UFLD")
        assert detector is not None
    except Exception as e:
        pytest.fail(f"UFLD detector loading failed: {e}")

def test_ufld_detector_detection(sample_image):
    """
    Test if the UFLD detector can process an image.
    """
    detector = get_lane_detector("UFLD")
    processed_img, latency = detector.detect(sample_image)
    assert isinstance(processed_img, np.ndarray)
    assert isinstance(latency, float)
    assert processed_img.shape == sample_image.shape

def test_polylanenet_detector_loading():
    """
    Test if the PolyLaneNet detector can be loaded.
    """
    try:
        detector = get_lane_detector("PolyLaneNet")
        assert detector is not None
    except Exception as e:
        pytest.fail(f"PolyLaneNet detector loading failed: {e}")

def test_polylanenet_detector_detection(sample_image):
    """
    Test if the PolyLaneNet detector can process an image.
    """
    detector = get_lane_detector("PolyLaneNet")
    processed_img, latency = detector.detect(sample_image)
    assert isinstance(processed_img, np.ndarray)
    assert isinstance(latency, float)
    assert processed_img.shape == sample_image.shape

def test_yolo_object_detector_loading():
    """
    Test if the YOLO object detector can be loaded.
    """
    try:
        model_path = PathManager.get_model("yolo11l", check_exists=True)
        detector = get_object_detector("yolo", model_path=str(model_path))
        assert detector is not None
    except Exception as e:
        pytest.fail(f"YOLO object detector loading failed: {e}")

def test_yolo_object_detector_detection(sample_image):
    """
    Test if the YOLO object detector can process an image.
    """
    model_path = PathManager.get_model("yolo11l", check_exists=True)
    detector = get_object_detector("yolo", model_path=str(model_path))
    detections, _, stats = detector.detect(sample_image)
    assert isinstance(detections, list)
    assert isinstance(stats, dict)

