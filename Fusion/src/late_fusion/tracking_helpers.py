"""Per-sensor tracking helpers shared by the inspection videos and the multimodal
fusion pipelines. Gap-tolerant ByteTrack (3D in the global frame for LiDAR/radar so
ego motion is compensated; 2D per camera), plus the per-sensor TRACK_CONFIG.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from .geometry import transform_box
from .types import Detection2D, Detection3D, NUSCENES_CLASSES
from tracking import ByteTracker2D, ByteTracker3D

CAMERA_LIST = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
               "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]

# Per-sensor tracking config (NuScenes keyframes are 2 Hz → each frame = 0.5 s).
#   max_age:        frames a track coasts (Kalman) through MISSING detections.
#   distance_thresh: BEV/center fallback (normalized by box diagonal) for when IoU
#                    drops to 0 between low-framerate frames.
#   confirm:        post-processing — keep only tracks DETECTED in >= this many frames.
TRACK_CONFIG = {
    "lidar":  dict(high_thresh=0.20, low_thresh=0.08, match_thresh=0.10,
                   max_age=4, min_hits=1, distance_thresh=3.0, confirm=4),
    "camera": dict(high_thresh=0.30, low_thresh=0.15, match_thresh=0.30,
                   max_age=5, min_hits=2, distance_thresh=3.0, confirm=3),
    "radar":  dict(high_thresh=0.25, low_thresh=0.10, match_thresh=0.10,
                   max_age=3, min_hits=2, distance_thresh=2.5, confirm=5),
    "fused":  dict(high_thresh=0.30, low_thresh=0.10, match_thresh=0.10,
                   max_age=5, min_hits=1, distance_thresh=3.0, confirm=4),
}


def name_to_idx(name: str) -> int:
    return NUSCENES_CLASSES.index(name) if name in NUSCENES_CLASSES else 0


def idx_to_name(idx: int) -> str:
    return NUSCENES_CLASSES[idx] if 0 <= idx < len(NUSCENES_CLASSES) else "obj"


def make_tracker(sensor: str):
    c = TRACK_CONFIG[sensor]
    cls = ByteTracker2D if sensor == "camera" else ByteTracker3D
    return cls(high_thresh=c["high_thresh"], low_thresh=c["low_thresh"],
               match_thresh=c["match_thresh"], max_age=c["max_age"],
               min_hits=c["min_hits"], distance_thresh=c["distance_thresh"])


def make_camera_trackers() -> Dict[str, object]:
    return {cn: make_tracker("camera") for cn in CAMERA_LIST}


def track_3d(tracker, dets: Sequence[Detection3D], lidar_to_global: np.ndarray,
             source: str) -> List[Detection3D]:
    """Track LiDAR-frame Detection3D in the global frame (ego-compensated);
    returns tracked Detection3D back in the LiDAR frame with ``track_id`` set."""
    g2l = np.linalg.inv(lidar_to_global)
    if dets:
        boxes = np.stack([transform_box(d.box, lidar_to_global) for d in dets])
        scores = np.array([d.score for d in dets], float)
        labels = np.array([name_to_idx(d.label) for d in dets], int)
    else:
        boxes, scores, labels = np.zeros((0, 7)), np.zeros(0), np.zeros(0, int)
    out = []
    for t in tracker.update(boxes, scores, labels):
        out.append(Detection3D(box=transform_box(t.get_state(), g2l), score=float(t.score),
                               label=idx_to_name(int(t.label)), source=source,
                               track_id=int(t.track_id)))
    return out


def track_2d(trackers: Dict[str, object], cam_dets: Sequence[Detection2D]) -> List[Detection2D]:
    """Track 2D detections independently per camera (all cameras considered);
    returns tracked Detection2D with ``track_id`` set."""
    by_cam: Dict[str, List[Detection2D]] = {}
    for d in cam_dets:
        by_cam.setdefault(d.camera, []).append(d)
    out = []
    for cam_name, trk in trackers.items():
        cds = by_cam.get(cam_name, [])
        if cds:
            boxes = np.stack([d.bbox for d in cds])
            scores = np.array([d.score for d in cds], float)
            labels = np.array([name_to_idx(d.label) for d in cds], int)
        else:
            boxes, scores, labels = np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)
        for t in trk.update(boxes, scores, labels):
            out.append(Detection2D(bbox=t.get_state(), score=float(t.score),
                                   label=idx_to_name(int(t.label)), camera=cam_name,
                                   track_id=int(t.track_id)))
    return out


def attach_velocity(tracked: Sequence[Detection3D], raw: Sequence[Detection3D],
                    max_dist: float = 2.5) -> None:
    """Copy velocity from the nearest raw radar det to each tracked box (BEV)."""
    if not raw:
        return
    raw_xy = np.stack([r.box[:2] for r in raw])
    for t in tracked:
        d = np.linalg.norm(raw_xy - t.box[:2], axis=1)
        j = int(np.argmin(d))
        if d[j] < max_dist:
            t.velocity = raw[j].velocity
