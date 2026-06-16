"""Decision-level (late) fusion of LiDAR / camera / radar detections.

Strategy (LiDAR-anchored):
  - LiDAR detections provide the 3D boxes.
  - A matched camera detection confirms the object and refines its class/score
    (camera is the stronger classifier). Agreement boosts confidence (noisy-OR).
  - A matched radar detection supplies velocity.
  - Optionally, confident radar detections with no LiDAR match are emitted as
    radar-only objects (radar can catch moving objects LiDAR missed).
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from .association import associate_lidar_camera, associate_lidar_radar
from .types import Detection2D, Detection3D, FusedObject


def noisy_or(a: float, b: float) -> float:
    """Combine two independent confidences: 1 - (1-a)(1-b)."""
    return 1.0 - (1.0 - a) * (1.0 - b)


def fuse(
    lidar_dets: Sequence[Detection3D],
    cam_dets: Sequence[Detection2D],
    radar_dets: Sequence[Detection3D],
    cameras: Dict[str, dict],
    cam_iou_thresh: float = 0.1,
    radar_dist_thresh: float = 3.0,
    include_radar_only: bool = True,
    radar_only_min_score: float = 0.3,
    radar_only_min_speed: float = 1.0,
) -> List[FusedObject]:
    """Fuse per-modality detections into a list of :class:`FusedObject`.

    Args:
        lidar_dets: 3D detections (the 3D anchors).
        cam_dets: 2D detections across all cameras.
        radar_dets: 3D detections from radar (carry velocity).
        cameras: ``cameras`` calib dict from ``load_sample_data`` (for projection).
        radar_only_min_speed: a radar-only object must be moving at least this fast
            (m/s) to be emitted — radar's unique value is moving objects; this
            rejects the static radar clutter that otherwise floods the output.
    """
    cam_match = associate_lidar_camera(lidar_dets, cam_dets, cameras, cam_iou_thresh)
    radar_match = associate_lidar_radar(lidar_dets, radar_dets, radar_dist_thresh)

    fused: List[FusedObject] = []
    matched_radar = set()

    for li, ld in enumerate(lidar_dets):
        obj = FusedObject(box=ld.box.copy(), score=float(ld.score), label=ld.label,
                          sources={"lidar"}, track_id=getattr(ld, "track_id", None))
        obj.per_source_score["lidar"] = float(ld.score)

        cd = cam_match.get(li)
        if cd is not None:
            obj.sources.add("camera")
            obj.camera_confirmed = True
            obj.per_source_score["camera"] = float(cd.score)
            if cd.label == ld.label:
                obj.score = noisy_or(float(ld.score), float(cd.score))
            else:
                # Disagreement: trust the camera's class, don't boost confidence.
                obj.label = cd.label

        rj = radar_match.get(li)
        if rj is not None:
            rd = radar_dets[rj]
            matched_radar.add(rj)
            obj.sources.add("radar")
            obj.per_source_score["radar"] = float(rd.score)
            obj.velocity = rd.velocity

        fused.append(obj)

    if include_radar_only:
        for rj, rd in enumerate(radar_dets):
            if rj in matched_radar or float(rd.score) < radar_only_min_score:
                continue
            speed = float(np.linalg.norm(rd.velocity)) if rd.velocity is not None else 0.0
            if speed < radar_only_min_speed:
                continue
            obj = FusedObject(box=rd.box.copy(), score=float(rd.score), label=rd.label,
                              sources={"radar"}, velocity=rd.velocity,
                              track_id=getattr(rd, "track_id", None))
            obj.per_source_score["radar"] = float(rd.score)
            fused.append(obj)

    return fused
