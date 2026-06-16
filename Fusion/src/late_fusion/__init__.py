"""Decision-level (late) sensor fusion for LiDAR + camera + radar.

Public API:
    Detection3D, Detection2D, FusedObject   — common types (types.py)
    fuse(...)                                — LiDAR-anchored fusion (fusion.py)
    associate_lidar_camera, associate_lidar_radar  — association (association.py)

The core (types/geometry/association/fusion) is pure NumPy and unit-tested with
no GPU/data dependencies. The pipeline/adapters wire real detectors on top.
"""
from .association import associate_lidar_camera, associate_lidar_radar, match_by_affinity
from .fusion import fuse, noisy_or
from .geometry import (
    bev_center_distance,
    boxes_to_corners_3d,
    iou_2d,
    project_box_to_image,
)
from .types import (
    COCO_TO_NUSCENES,
    NUSCENES_CLASSES,
    Detection2D,
    Detection3D,
    FusedObject,
    coco_to_nuscenes,
)

__all__ = [
    "Detection3D", "Detection2D", "FusedObject",
    "NUSCENES_CLASSES", "COCO_TO_NUSCENES", "coco_to_nuscenes",
    "fuse", "noisy_or",
    "associate_lidar_camera", "associate_lidar_radar", "match_by_affinity",
    "boxes_to_corners_3d", "project_box_to_image", "iou_2d", "bev_center_distance",
]
