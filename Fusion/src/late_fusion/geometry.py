"""Geometry helpers for late fusion: 3D box corners, LiDAR→image projection,
2D IoU, and BEV distance. Pure NumPy, no torch/data dependencies.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def boxes_to_corners_3d(box: np.ndarray) -> np.ndarray:
    """Convert a single ``[x, y, z, l, w, h, yaw]`` box to (8, 3) corners.

    ``z`` is treated as the box centre; corners span ``z ± h/2``. ``l`` runs along
    the heading, ``w`` across it. Rotation is yaw about the z axis.
    """
    x, y, z, l, w, h, yaw = [float(v) for v in box]
    xs = (l / 2.0) * np.array([1, 1, 1, 1, -1, -1, -1, -1], dtype=np.float64)
    ys = (w / 2.0) * np.array([1, 1, -1, -1, 1, 1, -1, -1], dtype=np.float64)
    zs = (h / 2.0) * np.array([1, -1, 1, -1, 1, -1, 1, -1], dtype=np.float64)
    corners = np.stack([xs, ys, zs], axis=1)  # (8, 3)

    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    corners = corners @ rot.T
    corners += np.array([x, y, z])
    return corners.astype(np.float32)


def project_box_to_image(
    corners_3d: np.ndarray,
    lidar_to_cam: np.ndarray,
    intrinsic: np.ndarray,
    img_w: int,
    img_h: int,
    min_depth: float = 0.1,
) -> Optional[np.ndarray]:
    """Project (8, 3) LiDAR-frame corners to an axis-aligned image bbox.

    Returns ``[x1, y1, x2, y2]`` clipped to the image, or ``None`` if the box is
    entirely behind the camera or projects outside the frame.
    """
    pts = np.concatenate([corners_3d, np.ones((len(corners_3d), 1))], axis=1)  # (8,4)
    cam = (lidar_to_cam @ pts.T).T[:, :3]                                       # (8,3)

    in_front = cam[:, 2] > min_depth
    if not np.any(in_front):
        return None

    cam = cam[in_front]
    uv = (intrinsic @ cam.T).T
    uv = uv[:, :2] / uv[:, 2:3]

    x1, y1 = uv.min(axis=0)
    x2, y2 = uv.max(axis=0)
    x1, y1 = max(0.0, float(x1)), max(0.0, float(y1))
    x2, y2 = min(float(img_w), float(x2)), min(float(img_h), float(y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def transform_box(box: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a 4×4 rigid transform to a ``[x,y,z,l,w,h,yaw]`` box (e.g. LiDAR↔global)."""
    box = np.asarray(box, dtype=np.float64).copy()
    box[:3] = transform[:3, :3] @ box[:3] + transform[:3, 3]
    box[6] = box[6] + float(np.arctan2(transform[1, 0], transform[0, 0]))
    return box.astype(np.float32)


def project_corners_to_image(
    corners_3d: np.ndarray,
    lidar_to_cam: np.ndarray,
    intrinsic: np.ndarray,
    min_depth: float = 0.1,
):
    """Project (8,3) LiDAR-frame corners to image points for wireframe drawing.

    Returns ``(uv, in_front)`` where ``uv`` is (8,2) pixel coords and ``in_front``
    is an (8,) bool mask of corners ahead of the camera. Use both with the 12 box
    edges to draw a 3D box on the image.
    """
    pts = np.concatenate([corners_3d, np.ones((len(corners_3d), 1))], axis=1)
    cam = (lidar_to_cam @ pts.T).T[:, :3]
    in_front = cam[:, 2] > min_depth
    depth = np.where(in_front, cam[:, 2], 1.0)
    uv = (intrinsic @ cam.T).T
    uv = uv[:, :2] / depth[:, None]
    return uv.astype(np.float32), in_front


# 12 edges of a box as index pairs into the 8 corners (see boxes_to_corners_3d).
BOX_EDGES = [
    (0, 1), (1, 3), (3, 2), (2, 0),   # one face (constant x sign via z/y pattern)
    (4, 5), (5, 7), (7, 6), (6, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def project_points_to_image(
    points_xyz: np.ndarray,
    lidar_to_cam: np.ndarray,
    intrinsic: np.ndarray,
    img_w: int,
    img_h: int,
    min_depth: float = 0.5,
):
    """Project LiDAR points to a camera image. Returns ``(uv, depth)`` for points
    in front of and inside the image — a direct check that the calibration is
    correct (projected points should land on the scene)."""
    pts = np.concatenate([points_xyz[:, :3], np.ones((len(points_xyz), 1))], axis=1)
    cam = (lidar_to_cam @ pts.T).T[:, :3]
    depths = cam[:, 2]
    front = depths > min_depth
    cam, depths = cam[front], depths[front]
    if len(cam) == 0:
        return np.zeros((0, 2), np.float32), np.zeros(0, np.float32)
    uv = (intrinsic @ cam.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    inimg = (uv[:, 0] >= 0) & (uv[:, 0] < img_w) & (uv[:, 1] >= 0) & (uv[:, 1] < img_h)
    return uv[inimg].astype(np.float32), depths[inimg].astype(np.float32)


def iou_2d(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Axis-aligned 2D IoU of ``[x1, y1, x2, y2]`` boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _bev_aabb(box: np.ndarray):
    """Axis-aligned BEV bounding rect of a (possibly rotated) box footprint."""
    c = boxes_to_corners_3d(box)[:, :2]
    return [c[:, 0].min(), c[:, 1].min(), c[:, 0].max(), c[:, 1].max()]


def bev_iou_aabb(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Approximate BEV IoU using the boxes' axis-aligned footprint rects."""
    return iou_2d(_bev_aabb(box_a), _bev_aabb(box_b))


def nms_bev(dets, iou_thresh: float = 0.4):
    """Class-agnostic greedy BEV NMS over detections with ``.box`` and ``.score``.

    Removes overlapping duplicates (e.g. one vehicle detected as two classes), which
    per-class NMS inside a detector leaves behind. Keeps the highest-scoring box.
    """
    if not dets:
        return list(dets)
    order = sorted(range(len(dets)), key=lambda i: -float(dets[i].score))
    boxes = [d.box for d in dets]
    keep, removed = [], set()
    for i in order:
        if i in removed:
            continue
        keep.append(dets[i])
        for j in order:
            if j != i and j not in removed and bev_iou_aabb(boxes[i], boxes[j]) > iou_thresh:
                removed.add(j)
    return keep


def bev_center_distance(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Euclidean distance between two boxes' BEV (x, y) centres."""
    return float(np.linalg.norm(np.asarray(box_a)[:2] - np.asarray(box_b)[:2]))
