"""Adapters: convert each modality's raw detector output into the common
late-fusion types (Detection3D / Detection2D), including the radar→LiDAR frame
transform. Pure NumPy — no detector imports here, so this stays import-safe in
any module context.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

from .types import Detection2D, Detection3D, NUSCENES_CLASSES, coco_to_nuscenes


def lidar_results_to_dets(results: dict) -> List[Detection3D]:
    """LiDAR detector output ``{'boxes','scores','labels'}`` → Detection3D list."""
    boxes = np.asarray(results.get("boxes", []), dtype=np.float32).reshape(-1, 7)
    scores = np.asarray(results.get("scores", []), dtype=np.float32).reshape(-1)
    labels = np.asarray(results.get("labels", []), dtype=np.int64).reshape(-1)
    out = []
    for box, score, label in zip(boxes, scores, labels):
        name = NUSCENES_CLASSES[label] if 0 <= label < len(NUSCENES_CLASSES) else "car"
        out.append(Detection3D(box=box, score=float(score), label=name, source="lidar"))
    return out


def yolo_dets_to_dets(parsed: Sequence[dict], camera: str) -> List[Detection2D]:
    """YOLO/RT-DETR parsed detections → Detection2D list (COCO→NuScenes mapped).

    ``parsed`` items follow the repo's detector output:
    ``{'class_name','confidence','bbox':[x1,y1,x2,y2]}``. Unmapped classes are
    dropped (only fusion-relevant traffic participants are kept).
    """
    out = []
    for d in parsed:
        ns = coco_to_nuscenes(d["class_name"])
        if ns is None:
            continue
        out.append(Detection2D(bbox=d["bbox"], score=float(d["confidence"]),
                               label=ns, camera=camera, raw_label=d["class_name"]))
    return out


def _transform_box(box: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a 4×4 rigid transform to a ``[x,y,z,l,w,h,yaw]`` box."""
    box = np.asarray(box, dtype=np.float64).copy()
    xyz = transform[:3, :3] @ box[:3] + transform[:3, 3]
    yaw = box[6] + float(np.arctan2(transform[1, 0], transform[0, 0]))
    box[:3] = xyz
    box[6] = yaw
    return box.astype(np.float32)


def radar_dets_to_common(
    radar_dets: Sequence,
    radar_to_lidar: np.ndarray,
) -> List[Detection3D]:
    """Radar ``Detection3D`` (RADAR_FRONT frame) → common Detection3D (LiDAR frame).

    Transforms each box and rotates its velocity vector into the LiDAR frame.
    Accepts the Radar module's own Detection3D objects (duck-typed: ``.box``,
    ``.score``, ``.label_name``/``.label``, ``.velocity``).
    """
    rot2d = np.asarray(radar_to_lidar)[:2, :2]
    out = []
    for rd in radar_dets:
        box = _transform_box(rd.box, np.asarray(radar_to_lidar))
        vel = None
        if getattr(rd, "velocity", None) is not None:
            vel = rot2d @ np.asarray(rd.velocity, dtype=np.float64)[:2]
        name = getattr(rd, "label_name", "") or "car"
        out.append(Detection3D(box=box, score=float(rd.score), label=name,
                               source="radar", velocity=vel))
    return out
