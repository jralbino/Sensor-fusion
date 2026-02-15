"""
3D ByteTrack tracker for 7-parameter bounding boxes [x, y, z, l, w, h, yaw].
Uses axis-aligned BEV IoU for association (fast, no Shapely dependency).
"""

import numpy as np
from .bytetrack import BaseTrack, ByteTracker
from .kalman_3d import KalmanFilter3D


class Track3D(BaseTrack):
    """A single 3D tracked object using Kalman filtering."""

    def __init__(self, detection: np.ndarray, score: float, label: int):
        super().__init__(detection, score, label)
        self.kf = KalmanFilter3D()
        self.kf.initiate(detection[:7])

    def predict(self):
        self.kf.predict()

    def update(self, detection: np.ndarray, score: float):
        self.kf.update(detection[:7])
        self.score = score

    def get_state(self) -> np.ndarray:
        """Return [x, y, z, l, w, h, yaw]."""
        return self.kf.position


def _rotated_to_aabb(boxes: np.ndarray) -> np.ndarray:
    """Convert rotated boxes (N,7) to axis-aligned BEV boxes (N,4): [x1,y1,x2,y2]."""
    x, y = boxes[:, 0], boxes[:, 1]
    l, w, yaw = boxes[:, 3], boxes[:, 4], boxes[:, 6]
    cos, sin = np.abs(np.cos(yaw)), np.abs(np.sin(yaw))
    hx = (l * cos + w * sin) / 2
    hy = (l * sin + w * cos) / 2
    return np.stack([x - hx, y - hy, x + hx, y + hy], axis=1)


def _iou_bev_batch(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute axis-aligned BEV IoU matrix between two sets of 7-param boxes.

    Args:
        boxes_a: (M, 7) [x, y, z, l, w, h, yaw]
        boxes_b: (N, 7) [x, y, z, l, w, h, yaw]

    Returns:
        (M, N) IoU matrix
    """
    M, N = len(boxes_a), len(boxes_b)
    if M == 0 or N == 0:
        return np.zeros((M, N))

    aabb_a = _rotated_to_aabb(boxes_a)  # (M, 4)
    aabb_b = _rotated_to_aabb(boxes_b)  # (N, 4)

    x1 = np.maximum(aabb_a[:, 0:1], aabb_b[:, 0].reshape(1, -1))
    y1 = np.maximum(aabb_a[:, 1:2], aabb_b[:, 1].reshape(1, -1))
    x2 = np.minimum(aabb_a[:, 2:3], aabb_b[:, 2].reshape(1, -1))
    y2 = np.minimum(aabb_a[:, 3:4], aabb_b[:, 3].reshape(1, -1))

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    area_a = (aabb_a[:, 2] - aabb_a[:, 0]) * (aabb_a[:, 3] - aabb_a[:, 1])
    area_b = (aabb_b[:, 2] - aabb_b[:, 0]) * (aabb_b[:, 3] - aabb_b[:, 1])

    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-6)


def _center_distance_bev(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute pairwise BEV center-distance normalized by mean box diagonal.

    Args:
        boxes_a: (M, 7) [x, y, z, l, w, h, yaw]
        boxes_b: (N, 7) [x, y, z, l, w, h, yaw]

    Returns:
        (M, N) normalized distance matrix.
    """
    dx = boxes_a[:, 0:1] - boxes_b[:, 0].reshape(1, -1)
    dy = boxes_a[:, 1:2] - boxes_b[:, 1].reshape(1, -1)
    dist = np.sqrt(dx ** 2 + dy ** 2)

    diag_a = np.sqrt(boxes_a[:, 3] ** 2 + boxes_a[:, 4] ** 2)
    diag_b = np.sqrt(boxes_b[:, 3] ** 2 + boxes_b[:, 4] ** 2)
    norm = np.sqrt(diag_a[:, None] * np.maximum(diag_b[None, :], 1e-6))
    return dist / np.maximum(norm, 1e-6)


class ByteTracker3D(ByteTracker):
    """ByteTrack for 3D 7-parameter detections [x,y,z,l,w,h,yaw]."""

    def _create_track(self, detection, score, label):
        return Track3D(detection, score, label)

    def _compute_iou_matrix(self, tracks, detections):
        return _iou_bev_batch(tracks, detections)

    def _compute_distance_matrix(self, tracks, detections):
        return _center_distance_bev(tracks, detections)

    def _det_dim(self):
        return 7
