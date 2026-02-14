"""
3D Geometry Utilities

Functions for 3D bounding box operations, coordinate transformations,
IoU computation, and NMS.
"""

import numpy as np
from typing import Tuple, List
import torch


def rotate_points_along_z(points: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate points around Z-axis.
    
    Args:
        points: [N, 3] points
        angle: Rotation angle in radians
    
    Returns:
        Rotated points [N, 3]
    """
    cosa = np.cos(angle)
    sina = np.sin(angle)
    
    rotation_matrix = np.array([
        [cosa, -sina, 0],
        [sina, cosa, 0],
        [0, 0, 1]
    ])
    
    return points @ rotation_matrix.T


def boxes_to_corners_3d(boxes: np.ndarray) -> np.ndarray:
    """
    Convert boxes to 8 corner representation.
    
    Args:
        boxes: [N, 7] boxes (x, y, z, l, w, h, yaw)
    
    Returns:
        corners: [N, 8, 3] corner coordinates
    """
    N = boxes.shape[0]
    x, y, z = boxes[:, 0], boxes[:, 1], boxes[:, 2]
    l, w, h = boxes[:, 3], boxes[:, 4], boxes[:, 5]
    yaw = boxes[:, 6]
    
    # Template corners (centered at origin)
    template = np.array([
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1]
    ]) / 2.0  # [8, 3]
    
    # Scale by dimensions
    corners = template[np.newaxis, :, :] * np.stack([l, w, h], axis=1)[:, np.newaxis, :]  # [N, 8, 3]
    
    # Rotate around Z-axis
    for i in range(N):
        corners[i] = rotate_points_along_z(corners[i], yaw[i])
    
    # Translate to center
    corners += np.stack([x, y, z], axis=1)[:, np.newaxis, :]
    
    return corners


def compute_iou_bev(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute Bird's Eye View (BEV) IoU between two boxes.
    
    Args:
        box1, box2: [7] boxes (x, y, z, l, w, h, yaw)
    
    Returns:
        IoU value [0, 1]
    """
    from shapely.geometry import Polygon
    
    def get_corners_2d(box):
        """Get 4 corners in BEV."""
        x, y, l, w, yaw = box[0], box[1], box[3], box[4], box[6]
        
        # Corners before rotation
        corners = np.array([
            [l/2, w/2], [-l/2, w/2],
            [-l/2, -w/2], [l/2, -w/2]
        ])
        
        # Rotation matrix
        R = np.array([
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw), np.cos(yaw)]
        ])
        
        # Rotate and translate
        corners = corners @ R.T + np.array([x, y])
        return corners
    
    # Get 2D corners
    corners1 = get_corners_2d(box1)
    corners2 = get_corners_2d(box2)
    
    # Create polygons
    poly1 = Polygon(corners1)
    poly2 = Polygon(corners2)
    
    # Compute intersection and union
    if not poly1.is_valid or not poly2.is_valid:
        return 0.0
    
    intersection = poly1.intersection(poly2).area
    union = poly1.area + poly2.area - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union


def compute_iou_3d(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute 3D IoU between two boxes.
    
    Args:
        box1, box2: [7] boxes
    
    Returns:
        3D IoU value
    """
    # BEV IoU
    iou_bev = compute_iou_bev(box1, box2)
    
    # Height overlap
    z1_min, z1_max = box1[2] - box1[5]/2, box1[2] + box1[5]/2
    z2_min, z2_max = box2[2] - box2[5]/2, box2[2] + box2[5]/2
    
    z_overlap = max(0, min(z1_max, z2_max) - max(z1_min, z2_min))
    z_union = max(z1_max, z2_max) - min(z1_min, z2_min)
    
    if z_union == 0:
        return 0.0
    
    # Combine BEV and height
    iou_3d = iou_bev * (z_overlap / z_union)
    
    return iou_3d


def nms_3d(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.0
) -> np.ndarray:
    """
    3D Non-Maximum Suppression.
    
    Args:
        boxes: [N, 7] boxes
        scores: [N] confidence scores
        iou_threshold: IoU threshold for suppression
        score_threshold: Minimum score to keep
    
    Returns:
        keep_indices: Indices of boxes to keep
    """
    # Filter by score threshold
    score_mask = scores > score_threshold
    boxes = boxes[score_mask]
    scores = scores[score_mask]
    
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)
    
    # Sort by score descending
    order = scores.argsort()[::-1]
    
    keep = []
    while len(order) > 0:
        # Keep highest scoring box
        i = order[0]
        keep.append(i)
        
        if len(order) == 1:
            break
        
        # Compute IoU with remaining boxes
        ious = np.array([compute_iou_3d(boxes[i], boxes[j]) for j in order[1:]])
        
        # Keep boxes with IoU below threshold
        inds = np.where(ious <= iou_threshold)[0]
        order = order[inds + 1]
    
    return np.array(keep, dtype=np.int64)


def _rotated_to_aabb(boxes: np.ndarray) -> np.ndarray:
    """Convert rotated boxes (N,7) to axis-aligned BEV bounding boxes (N,4): [x1,y1,x2,y2]."""
    x, y = boxes[:, 0], boxes[:, 1]
    l, w, yaw = boxes[:, 3], boxes[:, 4], boxes[:, 6]
    cos, sin = np.abs(np.cos(yaw)), np.abs(np.sin(yaw))
    # half-extents of the axis-aligned envelope
    hx = (l * cos + w * sin) / 2
    hy = (l * sin + w * cos) / 2
    return np.stack([x - hx, y - hy, x + hx, y + hy], axis=1)


def nms_bev_fast(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.0,
    max_candidates: int = 500,
) -> np.ndarray:
    """Fast BEV NMS using axis-aligned bounding boxes (vectorized numpy).

    Much faster than nms_3d which uses Shapely polygons in a Python loop.

    Args:
        boxes: (N, 7) boxes
        scores: (N,) confidence scores
        iou_threshold: IoU threshold for suppression
        score_threshold: minimum score to keep
        max_candidates: keep only top-K candidates before NMS

    Returns:
        keep_indices: indices into the original (pre-filter) array
    """
    mask = scores > score_threshold
    if not mask.any():
        return np.array([], dtype=np.int64)

    orig_idx = np.where(mask)[0]
    boxes_f = boxes[mask]
    scores_f = scores[mask]

    # Top-K cap
    if len(scores_f) > max_candidates:
        topk = np.argpartition(scores_f, -max_candidates)[-max_candidates:]
        orig_idx = orig_idx[topk]
        boxes_f = boxes_f[topk]
        scores_f = scores_f[topk]

    # Sort by score descending
    order = scores_f.argsort()[::-1]
    orig_idx = orig_idx[order]
    boxes_f = boxes_f[order]

    # Axis-aligned BEV boxes
    aabb = _rotated_to_aabb(boxes_f)  # (K, 4)
    areas = (aabb[:, 2] - aabb[:, 0]) * (aabb[:, 3] - aabb[:, 1])

    keep = []
    suppressed = np.zeros(len(boxes_f), dtype=bool)

    for i in range(len(boxes_f)):
        if suppressed[i]:
            continue
        keep.append(i)

        # Vectorized IoU with all remaining boxes
        xx1 = np.maximum(aabb[i, 0], aabb[i + 1:, 0])
        yy1 = np.maximum(aabb[i, 1], aabb[i + 1:, 1])
        xx2 = np.minimum(aabb[i, 2], aabb[i + 1:, 2])
        yy2 = np.minimum(aabb[i, 3], aabb[i + 1:, 3])

        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        union = areas[i] + areas[i + 1:] - inter
        iou = inter / np.maximum(union, 1e-6)

        suppress_mask = iou > iou_threshold
        suppressed[i + 1:][suppress_mask] = True

    return orig_idx[keep]


def limit_period(val: np.ndarray, offset: float = 0.5, period: float = np.pi) -> np.ndarray:
    """
    Limit rotation angle to [-pi, pi].
    
    Args:
        val: Angle values
        offset: Offset for period adjustment
        period: Period (default: pi for rotation)
    
    Returns:
        Limited angles
    """
    return val - np.floor(val / period + offset) * period


def rotation_3d_in_axis(points: torch.Tensor, angles: torch.Tensor, axis: int = 0) -> torch.Tensor:
    """
    Rotate points along an axis (PyTorch version).
    
    Args:
        points: [B, N, 3] points
        angles: [B] rotation angles
        axis: Rotation axis (0=x, 1=y, 2=z)
    
    Returns:
        Rotated points [B, N, 3]
    """
    rot_sin = torch.sin(angles)
    rot_cos = torch.cos(angles)
    ones = torch.ones_like(rot_cos)
    zeros = torch.zeros_like(rot_cos)
    
    if axis == 1:  # Rotation around Y-axis
        rot_mat_T = torch.stack([
            torch.stack([rot_cos, zeros, -rot_sin]),
            torch.stack([zeros, ones, zeros]),
            torch.stack([rot_sin, zeros, rot_cos])
        ])  # [3, 3, B]
    elif axis == 2 or axis == -1:  # Rotation around Z-axis
        rot_mat_T = torch.stack([
            torch.stack([rot_cos, -rot_sin, zeros]),
            torch.stack([rot_sin, rot_cos, zeros]),
            torch.stack([zeros, zeros, ones])
        ])
    elif axis == 0:  # Rotation around X-axis
        rot_mat_T = torch.stack([
            torch.stack([ones, zeros, zeros]),
            torch.stack([zeros, rot_cos, -rot_sin]),
            torch.stack([zeros, rot_sin, rot_cos])
        ])
    else:
        raise ValueError(f"Invalid axis: {axis}")
    
    # [3, 3, B] -> [B, 3, 3]
    rot_mat_T = rot_mat_T.permute(2, 0, 1)
    
    # Rotate points
    points_rot = torch.matmul(points, rot_mat_T)
    
    return points_rot


def points_in_boxes_cpu(points: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """
    Check which points are inside which boxes (CPU version).
    
    Args:
        points: [N, 3] points (x, y, z)
        boxes: [M, 7] boxes (x, y, z, l, w, h, yaw)
    
    Returns:
        point_indices: [K, 2] array where each row is [point_idx, box_idx]
    """
    N, M = len(points), len(boxes)
    point_indices = []
    
    for box_idx in range(M):
        box = boxes[box_idx]
        
        # Transform points to box frame
        local_points = points.copy()
        local_points -= box[:3]  # Translate
        local_points = rotate_points_along_z(local_points, -box[6])  # Rotate
        
        # Check if inside box
        l, w, h = box[3], box[4], box[5]
        mask = (
            (np.abs(local_points[:, 0]) <= l/2) &
            (np.abs(local_points[:, 1]) <= w/2) &
            (np.abs(local_points[:, 2]) <= h/2)
        )
        
        point_idx = np.where(mask)[0]
        for p_idx in point_idx:
            point_indices.append([p_idx, box_idx])
    
    return np.array(point_indices, dtype=np.int64) if point_indices else np.zeros((0, 2), dtype=np.int64)


def enlarge_box3d(boxes: np.ndarray, extra_width: Tuple[float, float, float]) -> np.ndarray:
    """
    Enlarge boxes by adding extra width.
    
    Args:
        boxes: [N, 7] boxes
        extra_width: (extra_l, extra_w, extra_h) to add
    
    Returns:
        Enlarged boxes [N, 7]
    """
    enlarged = boxes.copy()
    enlarged[:, 3:6] += np.array(extra_width)
    return enlarged
