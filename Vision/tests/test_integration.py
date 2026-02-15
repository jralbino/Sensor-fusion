"""Integration tests for the Vision module.

Tests object detection, batch prediction, tracking, and drawing
using real BDD100K images and model weights to validate the full
pipeline and the Vision venv.
"""

import sys
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.utils.path_manager import path_manager

# ---------------------------------------------------------------------------
# Paths & skip conditions
# ---------------------------------------------------------------------------

BDD_VAL_DIR = path_manager.get("bdd_images_val")
SAMPLE_IMAGES = sorted(BDD_VAL_DIR.glob("*.jpg"))[:2] if BDD_VAL_DIR.exists() else []

skip_no_images = pytest.mark.skipif(
    len(SAMPLE_IMAGES) == 0,
    reason="BDD100K validation images not found",
)

YOLO_MODEL_PATH = Path(PROJECT_ROOT) / "Vision" / "models" / "yolo11l.pt"
RTDETR_MODEL_PATH = Path(PROJECT_ROOT) / "Vision" / "models" / "rtdetr-l.pt"

skip_no_yolo = pytest.mark.skipif(
    not YOLO_MODEL_PATH.exists(),
    reason=f"YOLO model not found at {YOLO_MODEL_PATH}",
)
skip_no_rtdetr = pytest.mark.skipif(
    not RTDETR_MODEL_PATH.exists(),
    reason=f"RT-DETR model not found at {RTDETR_MODEL_PATH}",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sample_image():
    """Load the first BDD100K val image as BGR numpy array."""
    img = cv2.imread(str(SAMPLE_IMAGES[0]))
    assert img is not None, f"Failed to read {SAMPLE_IMAGES[0]}"
    return img


@pytest.fixture(scope="module")
def two_images():
    """Return paths to two sample images."""
    return SAMPLE_IMAGES[:2]


@pytest.fixture(scope="module")
def yolo_detector():
    """Load YOLO11-L detector (cached for the module)."""
    from Vision.src.detectors.object_detector import ObjectDetector
    return ObjectDetector(model_path=str(YOLO_MODEL_PATH), conf=0.5, device="cpu")


@pytest.fixture(scope="module")
def rtdetr_detector():
    """Load RT-DETR-L detector (cached for the module)."""
    from Vision.src.detectors.object_detector import ObjectDetector
    return ObjectDetector(model_path=str(RTDETR_MODEL_PATH), conf=0.5, device="cpu")


# ---------------------------------------------------------------------------
# ObjectDetector — YOLO
# ---------------------------------------------------------------------------

@skip_no_images
@skip_no_yolo
class TestObjectDetectorYOLO:
    def test_detect_returns_detections(self, yolo_detector, sample_image):
        detections, plot_img, stats = yolo_detector.detect(sample_image)

        assert isinstance(detections, list)
        assert isinstance(plot_img, np.ndarray)
        assert "inference_time_ms" in stats
        assert stats["inference_time_ms"] > 0

    def test_detection_format(self, yolo_detector, sample_image):
        detections, _, _ = yolo_detector.detect(sample_image)

        for det in detections:
            assert "bbox" in det
            assert "class_name" in det
            assert "confidence" in det
            assert len(det["bbox"]) == 4
            assert 0 < det["confidence"] <= 1.0

    def test_confidence_threshold(self, sample_image):
        from Vision.src.detectors.object_detector import ObjectDetector

        low = ObjectDetector(model_path=str(YOLO_MODEL_PATH), conf=0.1, device="cpu")
        high = ObjectDetector(model_path=str(YOLO_MODEL_PATH), conf=0.8, device="cpu")

        dets_low, _, _ = low.detect(sample_image)
        dets_high, _, _ = high.detect(sample_image)

        assert len(dets_low) >= len(dets_high)


# ---------------------------------------------------------------------------
# ObjectDetector — RT-DETR
# ---------------------------------------------------------------------------

@skip_no_images
@skip_no_rtdetr
class TestObjectDetectorRTDETR:
    def test_detect_returns_detections(self, rtdetr_detector, sample_image):
        detections, plot_img, stats = rtdetr_detector.detect(sample_image)

        assert isinstance(detections, list)
        assert isinstance(plot_img, np.ndarray)
        assert stats["inference_time_ms"] > 0

    def test_detection_format(self, rtdetr_detector, sample_image):
        detections, _, _ = rtdetr_detector.detect(sample_image)

        for det in detections:
            assert "bbox" in det
            assert "class_name" in det
            assert "confidence" in det


# ---------------------------------------------------------------------------
# BatchPredictor
# ---------------------------------------------------------------------------

@skip_no_images
@skip_no_yolo
class TestBatchPredictor:
    def test_run_inference_limit2(self, two_images):
        from Vision.src.predictor import BatchPredictor

        with tempfile.TemporaryDirectory() as tmpdir:
            predictor = BatchPredictor(
                images_dir=str(BDD_VAL_DIR),
                output_dir=tmpdir,
            )
            json_path = predictor.run_inference(
                model_name="YOLO11-L",
                model_path=str(YOLO_MODEL_PATH),
                conf=0.5,
                limit=2,
            )

            assert json_path is not None
            assert Path(json_path).exists()

            with open(json_path) as f:
                data = json.load(f)

            assert "meta" in data
            assert "results" in data
            assert len(data["results"]) == 2

            for result in data["results"]:
                assert "image_name" in result
                assert "inference_ms" in result
                assert "detections" in result

    def test_nonexistent_model(self):
        from Vision.src.predictor import BatchPredictor

        with tempfile.TemporaryDirectory() as tmpdir:
            predictor = BatchPredictor(images_dir=str(BDD_VAL_DIR), output_dir=tmpdir)
            result = predictor.run_inference(
                model_name="fake", model_path="/nonexistent/model.pt", limit=1,
            )
            assert result is None


# ---------------------------------------------------------------------------
# draw_custom_boxes with real detections
# ---------------------------------------------------------------------------

@skip_no_images
@skip_no_yolo
class TestDrawWithRealDetections:
    def test_draw_yolo_detections(self, yolo_detector, sample_image):
        from Vision.src.app_utils import draw_custom_boxes

        detections, _, _ = yolo_detector.detect(sample_image)
        result = draw_custom_boxes(sample_image, detections)

        assert result.shape == sample_image.shape
        assert result is not sample_image
        if detections:
            assert not np.array_equal(result, sample_image)


# ---------------------------------------------------------------------------
# Tracking integration with real images
# ---------------------------------------------------------------------------

@skip_no_images
@skip_no_yolo
class TestTrackingIntegration:
    def test_track_across_two_images(self, yolo_detector, two_images):
        from tracking import ByteTracker2D
        from tracking.bytetrack import BaseTrack
        from Vision.src.app_utils import (
            detections_to_tracker_format,
            tracker_output_to_detections,
        )

        BaseTrack.reset_id_counter()
        tracker = ByteTracker2D(high_thresh=0.4, min_hits=1)

        all_tracked = []
        for img_path in two_images:
            img = cv2.imread(str(img_path))
            assert img is not None
            detections, _, _ = yolo_detector.detect(img)

            fmt = detections_to_tracker_format(detections)
            if fmt is not None:
                dets_arr, scores_arr, labels_arr, idx_to_name = fmt
                active = tracker.update(dets_arr, scores_arr, labels_arr)
                tracked = tracker_output_to_detections(active, idx_to_name)
            else:
                tracked = []

            all_tracked.append(tracked)

        # At least one frame should have tracked objects
        assert any(len(t) > 0 for t in all_tracked)

        # Tracked detections must have track_id
        for frame in all_tracked:
            for det in frame:
                assert "track_id" in det
                assert "bbox" in det
                assert "class_name" in det

    def test_draw_tracked_detections(self, yolo_detector, sample_image):
        from tracking import ByteTracker2D
        from tracking.bytetrack import BaseTrack
        from Vision.src.app_utils import (
            detections_to_tracker_format,
            tracker_output_to_detections,
            draw_custom_boxes,
        )

        BaseTrack.reset_id_counter()
        tracker = ByteTracker2D(high_thresh=0.3, min_hits=1)

        detections, _, _ = yolo_detector.detect(sample_image)
        fmt = detections_to_tracker_format(detections)

        if fmt is not None:
            dets_arr, scores_arr, labels_arr, idx_to_name = fmt
            active = tracker.update(dets_arr, scores_arr, labels_arr)
            tracked = tracker_output_to_detections(active, idx_to_name)
        else:
            tracked = []

        result = draw_custom_boxes(sample_image, tracked)
        assert result.shape == sample_image.shape


# ---------------------------------------------------------------------------
# Model loading smoke test (benchmark prerequisite)
# ---------------------------------------------------------------------------

@skip_no_yolo
class TestModelLoading:
    def test_yolo_model_loads(self):
        from ultralytics import YOLO

        model = YOLO(str(YOLO_MODEL_PATH))
        assert model is not None
        assert hasattr(model, "predict")

    @skip_no_rtdetr
    def test_rtdetr_model_loads(self):
        from ultralytics import RTDETR

        model = RTDETR(str(RTDETR_MODEL_PATH))
        assert model is not None
        assert hasattr(model, "predict")


# ---------------------------------------------------------------------------
# detector_factory
# ---------------------------------------------------------------------------

@skip_no_yolo
class TestDetectorFactory:
    def test_get_yolo_detector(self):
        from Vision.src.detectors.detector_factory import get_object_detector

        det = get_object_detector("yolo", model_path=str(YOLO_MODEL_PATH), conf=0.5)
        assert det is not None
        assert hasattr(det, "detect")

    @skip_no_rtdetr
    def test_get_rtdetr_detector(self):
        from Vision.src.detectors.detector_factory import get_object_detector

        det = get_object_detector("rtdetr", model_path=str(RTDETR_MODEL_PATH), conf=0.5)
        assert det is not None
        assert hasattr(det, "detect")
