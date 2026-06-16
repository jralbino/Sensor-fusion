"""Cross-sensor association for late fusion.

- LiDAR ↔ camera: project each 3D box into every camera and match to 2D
  detections by image-space IoU (Hungarian, per camera).
- LiDAR ↔ radar: match by BEV centre distance (Hungarian).
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from .geometry import (
    bev_center_distance,
    boxes_to_corners_3d,
    iou_2d,
    project_box_to_image,
)
from .types import Detection2D, Detection3D


def match_by_affinity(affinity: np.ndarray, min_affinity: float) -> List[tuple]:
    """Optimal one-to-one assignment maximising affinity, above a threshold.

    Returns a list of ``(row, col)`` index pairs whose affinity ≥ ``min_affinity``.
    """
    if affinity.size == 0:
        return []
    from scipy.optimize import linear_sum_assignment

    rows, cols = linear_sum_assignment(-affinity)
    return [
        (int(r), int(c))
        for r, c in zip(rows, cols)
        if affinity[r, c] >= min_affinity
    ]


def associate_lidar_camera(
    lidar_dets: Sequence[Detection3D],
    cam_dets: Sequence[Detection2D],
    cameras: Dict[str, dict],
    iou_thresh: float = 0.1,
) -> Dict[int, Detection2D]:
    """Match LiDAR detections to camera 2D detections via projection.

    Args:
        cameras: the ``cameras`` dict from ``load_sample_data`` — per camera a dict
            with ``intrinsic`` (3,3), ``lidar_to_cam`` (4,4), ``img_w``, ``img_h``.

    Returns:
        ``{lidar_idx: Detection2D}`` keeping, per LiDAR box, its best-IoU camera
        match across all cameras.
    """
    by_cam: Dict[str, List[Detection2D]] = {}
    for d in cam_dets:
        by_cam.setdefault(d.camera, []).append(d)

    best_match: Dict[int, Detection2D] = {}
    best_iou: Dict[int, float] = {}

    for cam_name, cam in cameras.items():
        cdets = by_cam.get(cam_name, [])
        if not cdets:
            continue

        proj_boxes, proj_idx = [], []
        for li, ld in enumerate(lidar_dets):
            corners = boxes_to_corners_3d(ld.box)
            bbox = project_box_to_image(
                corners, np.asarray(cam["lidar_to_cam"]), np.asarray(cam["intrinsic"]),
                int(cam["img_w"]), int(cam["img_h"]),
            )
            if bbox is not None:
                proj_boxes.append(bbox)
                proj_idx.append(li)
        if not proj_boxes:
            continue

        affinity = np.zeros((len(proj_boxes), len(cdets)), dtype=np.float64)
        for i, b in enumerate(proj_boxes):
            for j, cd in enumerate(cdets):
                affinity[i, j] = iou_2d(b, cd.bbox)

        for i, j in match_by_affinity(affinity, iou_thresh):
            li = proj_idx[i]
            if affinity[i, j] > best_iou.get(li, 0.0):
                best_iou[li] = affinity[i, j]
                best_match[li] = cdets[j]

    return best_match


def associate_lidar_radar(
    lidar_dets: Sequence[Detection3D],
    radar_dets: Sequence[Detection3D],
    dist_thresh: float = 3.0,
) -> Dict[int, int]:
    """Match LiDAR detections to radar detections by BEV centre distance.

    Returns ``{lidar_idx: radar_idx}`` for pairs within ``dist_thresh`` metres.
    """
    if not lidar_dets or not radar_dets:
        return {}

    affinity = np.full((len(lidar_dets), len(radar_dets)), -1.0, dtype=np.float64)
    for i, ld in enumerate(lidar_dets):
        for j, rd in enumerate(radar_dets):
            d = bev_center_distance(ld.box, rd.box)
            if d <= dist_thresh:
                affinity[i, j] = 1.0 / (1.0 + d)

    min_aff = 1.0 / (1.0 + dist_thresh)
    return {i: j for i, j in match_by_affinity(affinity, min_aff)}
