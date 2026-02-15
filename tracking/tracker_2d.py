"""
2D ByteTrack tracker for xyxy bounding boxes.
"""

import numpy as np
from .bytetrack import BaseTrack, ByteTracker
from .kalman_2d import KalmanFilter2D


def _xyxy_to_xyah(bbox: np.ndarray) -> np.ndarray:
    """Convert [x1, y1, x2, y2] to [cx, cy, aspect_ratio, h]."""
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cx = bbox[0] + w / 2
    cy = bbox[1] + h / 2
    a = w / max(h, 1e-6)
    return np.array([cx, cy, a, h])


def _xyah_to_xyxy(xyah: np.ndarray) -> np.ndarray:
    """Convert [cx, cy, aspect_ratio, h] to [x1, y1, x2, y2]."""
    cx, cy, a, h = xyah
    w = a * h
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


class Track2D(BaseTrack):
    """A single 2D tracked object using Kalman filtering."""

    def __init__(self, detection: np.ndarray, score: float, label: int):
        super().__init__(detection, score, label)
        self.kf = KalmanFilter2D()
        self.kf.initiate(_xyxy_to_xyah(detection))

    def predict(self):
        self.kf.predict()

    def update(self, detection: np.ndarray, score: float):
        self.kf.update(_xyxy_to_xyah(detection))
        self.score = score

    def get_state(self) -> np.ndarray:
        """Return xyxy bbox."""
        return _xyah_to_xyxy(self.kf.position)


def _iou_2d_batch(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute IoU matrix between two sets of xyxy boxes.

    Args:
        boxes_a: (M, 4) xyxy
        boxes_b: (N, 4) xyxy

    Returns:
        (M, N) IoU matrix
    """
    M, N = len(boxes_a), len(boxes_b)
    if M == 0 or N == 0:
        return np.zeros((M, N))

    # Intersection
    x1 = np.maximum(boxes_a[:, 0:1], boxes_b[:, 0].reshape(1, -1))
    y1 = np.maximum(boxes_a[:, 1:2], boxes_b[:, 1].reshape(1, -1))
    x2 = np.minimum(boxes_a[:, 2:3], boxes_b[:, 2].reshape(1, -1))
    y2 = np.minimum(boxes_a[:, 3:4], boxes_b[:, 3].reshape(1, -1))

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-6)


def _center_distance_2d(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute pairwise center-distance matrix, normalized by diagonal.

    Uses the geometric mean of both box diagonals to normalize so that
    the threshold is scale-invariant (similar to GIoU/DIoU).

    Args:
        boxes_a: (M, 4) xyxy
        boxes_b: (N, 4) xyxy

    Returns:
        (M, N) normalized distance matrix (0 = same center, >1 = far apart).
    """
    cx_a = (boxes_a[:, 0] + boxes_a[:, 2]) / 2
    cy_a = (boxes_a[:, 1] + boxes_a[:, 3]) / 2
    cx_b = (boxes_b[:, 0] + boxes_b[:, 2]) / 2
    cy_b = (boxes_b[:, 1] + boxes_b[:, 3]) / 2

    dx = cx_a[:, None] - cx_b[None, :]
    dy = cy_a[:, None] - cy_b[None, :]
    dist = np.sqrt(dx ** 2 + dy ** 2)

    # Normalize by geometric mean of diagonals
    diag_a = np.sqrt(
        (boxes_a[:, 2] - boxes_a[:, 0]) ** 2 + (boxes_a[:, 3] - boxes_a[:, 1]) ** 2
    )
    diag_b = np.sqrt(
        (boxes_b[:, 2] - boxes_b[:, 0]) ** 2 + (boxes_b[:, 3] - boxes_b[:, 1]) ** 2
    )
    norm = np.sqrt(diag_a[:, None] * np.maximum(diag_b[None, :], 1e-6))
    return dist / np.maximum(norm, 1e-6)


class ByteTracker2D(ByteTracker):
    """ByteTrack for 2D xyxy detections."""

    def _create_track(self, detection, score, label):
        return Track2D(detection, score, label)

    def _compute_iou_matrix(self, tracks, detections):
        return _iou_2d_batch(tracks, detections)

    def _compute_distance_matrix(self, tracks, detections):
        return _center_distance_2d(tracks, detections)

    def _det_dim(self):
        return 4
