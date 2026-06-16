"""Common data types for decision-level (late) sensor fusion.

Every modality (LiDAR 3D, camera 2D, radar) is adapted into one of the detection
types below so the association and fusion logic stays modality-agnostic.

Frames & conventions
--------------------
- All 3D quantities live in the **LiDAR frame** of the keyframe (x forward, y left,
  z up), matching ``Lidar/visualize_3d.load_sample_data``.
- A 3D box is ``[x, y, z, l, w, h, yaw]`` (repo convention: length along the box's
  heading, width across it). Same layout the LiDAR detectors emit.
- A 2D box is ``[x1, y1, x2, y2]`` in pixels.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# The 10 NuScenes detection classes used across the LiDAR module.
NUSCENES_CLASSES = [
    "car", "truck", "construction_vehicle", "bus", "trailer",
    "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone",
]

# Map COCO class names (what YOLO/RT-DETR emit) onto NuScenes categories so a
# camera detection can confirm/refine a LiDAR detection's class.
COCO_TO_NUSCENES = {
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "motorcycle": "motorcycle",
    "bicycle": "bicycle",
    "person": "pedestrian",
    "traffic light": "traffic_cone",   # coarse: both are small static markers
    "stop sign": "barrier",
}


def coco_to_nuscenes(name: str) -> Optional[str]:
    """Return the NuScenes category for a COCO class name, or None if unmapped."""
    return COCO_TO_NUSCENES.get(name.lower())


@dataclass
class Detection3D:
    """A 3D detection in the LiDAR frame from any 3D-capable sensor."""

    box: np.ndarray                      # (7,) [x, y, z, l, w, h, yaw]
    score: float
    label: str                           # NuScenes category name
    source: str                          # "lidar" | "radar"
    velocity: Optional[np.ndarray] = None  # (2,) [vx, vy] in m/s, if known
    track_id: Optional[int] = None         # set by a tracker, if tracked

    def __post_init__(self) -> None:
        self.box = np.asarray(self.box, dtype=np.float32).reshape(7)
        if self.velocity is not None:
            self.velocity = np.asarray(self.velocity, dtype=np.float32).reshape(2)

    @property
    def center_bev(self) -> np.ndarray:
        """(2,) [x, y] BEV center."""
        return self.box[:2]


@dataclass
class Detection2D:
    """A 2D detection in a named camera image."""

    bbox: np.ndarray                     # (4,) [x1, y1, x2, y2]
    score: float
    label: str                           # NuScenes category name (mapped from COCO)
    camera: str                          # e.g. "CAM_FRONT"
    raw_label: str = ""                  # original COCO name, for debugging
    track_id: Optional[int] = None       # set by a tracker, if tracked

    def __post_init__(self) -> None:
        self.bbox = np.asarray(self.bbox, dtype=np.float32).reshape(4)


@dataclass
class FusedObject:
    """An object after fusing evidence from one or more sensors.

    The 3D box comes from the anchoring 3D detection (LiDAR, or radar if LiDAR
    missed it). Camera evidence refines the class and confidence; radar evidence
    supplies velocity.
    """

    box: np.ndarray                      # (7,) anchored 3D box
    score: float                         # fused confidence
    label: str                           # fused class
    sources: set = field(default_factory=set)   # {"lidar","camera","radar"}
    velocity: Optional[np.ndarray] = None        # (2,) from radar, if any
    camera_confirmed: bool = False
    per_source_score: dict = field(default_factory=dict)
    track_id: Optional[int] = None               # set by a tracker, if tracked

    def __post_init__(self) -> None:
        self.box = np.asarray(self.box, dtype=np.float32).reshape(7)
        if self.velocity is not None:
            self.velocity = np.asarray(self.velocity, dtype=np.float32).reshape(2)

    @property
    def num_sensors(self) -> int:
        return len(self.sources)
