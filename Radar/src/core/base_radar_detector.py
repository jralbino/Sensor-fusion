# -*- coding: utf-8 -*-
"""
Base Radar Detector — abstract interface for all radar-based 3D detectors.

Mirrors Lidar's BaseDetector but with radar-specific defaults:
- Larger voxels (0.5m) and wider range (200m) for radar's extended range
- Radar-specific feature handling (18 NuScenes features)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import logging
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _nms_bev(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> np.ndarray:
    """Fast axis-aligned BEV NMS.

    Args:
        boxes: (N, 7) — [x, y, z, l, w, h, yaw].
        scores: (N,) confidence scores.
        iou_thresh: IoU threshold.

    Returns:
        Array of kept indices.
    """
    x, y, l, w = boxes[:, 0], boxes[:, 1], boxes[:, 3], boxes[:, 4]
    half_l, half_w = l / 2, w / 2
    x1, y1, x2, y2 = x - half_l, y - half_w, x + half_l, y + half_w
    areas = l * w

    order = scores.argsort()[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / np.maximum(areas[i] + areas[rest] - inter, 1e-6)
        order = rest[iou <= iou_thresh]

    return np.array(keep, dtype=int)


@dataclass
class Detection3D:
    """Standard 3D detection output format."""

    box: np.ndarray         # [7] — x, y, z, l, w, h, yaw
    score: float
    label: int
    label_name: str = ""
    velocity: Optional[np.ndarray] = None  # [2] — vx, vy (radar can estimate)
    metadata: Optional[Dict] = field(default_factory=dict)

    def __post_init__(self):
        self.box = np.asarray(self.box, dtype=np.float32)
        if self.box.shape != (7,):
            raise ValueError(f"Box must be [7], got {self.box.shape}")
        self.score = float(np.clip(self.score, 0, 1))
        self.label = int(self.label)
        if self.velocity is not None:
            self.velocity = np.asarray(self.velocity, dtype=np.float32)
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict:
        d = {
            'box': self.box.tolist(),
            'score': self.score,
            'label': self.label,
            'label_name': self.label_name,
            'metadata': self.metadata,
        }
        if self.velocity is not None:
            d['velocity'] = self.velocity.tolist()
        return d


class BaseRadarDetector(ABC, nn.Module):
    """Abstract base class for radar-based 3D object detectors.

    Subclasses must implement ``forward()`` and ``get_loss()``.
    Provides shared ``detect()``, ``voxelize()``, and ``postprocess()`` logic.
    """

    CLASS_NAMES = [
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
        'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone',
    ]

    def __init__(
        self,
        num_classes: int = 10,
        voxel_size: Tuple[float, float, float] = (0.5, 0.5, 8.0),
        point_cloud_range: Tuple[float, ...] = (-100.0, -100.0, -5.0, 100.0, 100.0, 3.0),
        max_points_per_voxel: int = 20,
        max_voxels: int = 30000,
        in_channels: int = 6,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.voxel_size = np.array(voxel_size, dtype=np.float32)
        self.point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
        self.max_points_per_voxel = max_points_per_voxel
        self.max_voxels = max_voxels
        self.in_channels = in_channels
        self.device = torch.device('cpu')

        # Grid size derived from range and voxel size
        grid_size = (
            (self.point_cloud_range[3:] - self.point_cloud_range[:3]) / self.voxel_size
        ).astype(int)
        self.grid_size = grid_size  # [nx, ny, nz]

        # Lazy-init voxel generator (avoid import at module level)
        self._voxel_gen = None

    @property
    def voxel_generator(self):
        if self._voxel_gen is None:
            from Radar.src.utils.voxel_generator import VoxelGenerator
            self._voxel_gen = VoxelGenerator(
                voxel_size=self.voxel_size.tolist(),
                point_cloud_range=self.point_cloud_range.tolist(),
                max_num_points=self.max_points_per_voxel,
                max_voxels=self.max_voxels,
            )
        return self._voxel_gen

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def forward(self, batch_dict: Dict) -> Dict:
        """Forward pass. Must populate pred_boxes / pred_scores / pred_labels."""

    @abstractmethod
    def get_loss(self, batch_dict: Dict) -> Tuple[torch.Tensor, Dict]:
        """Compute training loss. Returns (total_loss, loss_dict)."""

    # ------------------------------------------------------------------
    # Shared logic
    # ------------------------------------------------------------------

    def voxelize(self, batch_dict: Dict) -> Dict:
        """Convert point cloud to voxel representation."""
        points = batch_dict['points']
        if isinstance(points, torch.Tensor):
            points = points.cpu().numpy()
        if points.ndim == 3:
            points = points.squeeze(0)

        # Range filter
        pc = self.point_cloud_range
        mask = (
            (points[:, 0] >= pc[0]) & (points[:, 0] <= pc[3])
            & (points[:, 1] >= pc[1]) & (points[:, 1] <= pc[4])
            & (points[:, 2] >= pc[2]) & (points[:, 2] <= pc[5])
        )
        points = points[mask]

        voxels, coords, num_points = self.voxel_generator.generate(points)
        device = next(self.parameters()).device
        batch_dict['voxels'] = torch.from_numpy(voxels).float().to(device)
        batch_dict['voxel_coords'] = torch.from_numpy(coords).int().to(device)
        batch_dict['voxel_num_points'] = torch.from_numpy(num_points).int().to(device)
        return batch_dict

    def detect(self, points: np.ndarray, conf_threshold: float = 0.3) -> List[Detection3D]:
        """Run inference on raw radar points.

        Args:
            points: (N, C) numpy array with at least ``in_channels`` columns.
            conf_threshold: Minimum confidence score.

        Returns:
            List of Detection3D objects.
        """
        self.eval()
        if isinstance(points, torch.Tensor):
            points = points.cpu().numpy()

        batch_dict = {'points': points, 'batch_size': 1}
        batch_dict = self.voxelize(batch_dict)

        # Early return if no valid voxels after range filtering
        if batch_dict['voxels'].shape[0] == 0:
            return []

        with torch.no_grad():
            pred_dict = self.forward(batch_dict)

        return self.postprocess(pred_dict, conf_threshold)

    def postprocess(
        self, pred_dict: Dict, conf_threshold: float = 0.3, nms_iou: float = 0.5,
    ) -> List[Detection3D]:
        """Convert network predictions to Detection3D objects with NMS."""
        boxes = pred_dict['pred_boxes'][0].cpu().numpy()
        scores = pred_dict['pred_scores'][0].cpu().numpy()
        labels = pred_dict['pred_labels'][0].cpu().numpy()
        velocities = pred_dict.get('pred_velocity')
        if velocities is not None:
            velocities = velocities[0].cpu().numpy()

        # Confidence filter
        mask = scores >= conf_threshold
        boxes, scores, labels = boxes[mask], scores[mask], labels[mask]
        if velocities is not None:
            velocities = velocities[mask]

        # BEV NMS
        if len(boxes) > 0:
            keep = _nms_bev(boxes, scores, nms_iou)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            if velocities is not None:
                velocities = velocities[keep]

        detections = []
        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            vel = velocities[i] if velocities is not None else None
            name = self.CLASS_NAMES[int(label)] if int(label) < len(self.CLASS_NAMES) else 'unknown'
            detections.append(Detection3D(box=box, score=float(score), label=int(label),
                                          label_name=name, velocity=vel))
        return detections

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def load_checkpoint(self, path: Union[str, Path]):
        ckpt = torch.load(str(path), map_location='cpu')
        state = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
        state = {k.replace('module.', ''): v for k, v in state.items()}
        self.load_state_dict(state, strict=False)
        logger.info("Loaded checkpoint from %s", path)

    def save_checkpoint(self, path: Union[str, Path], epoch: int = 0, **extra):
        data = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'config': {
                'num_classes': self.num_classes,
                'voxel_size': self.voxel_size.tolist(),
                'point_cloud_range': self.point_cloud_range.tolist(),
            },
        }
        data.update(extra)
        torch.save(data, str(path))
        logger.info("Saved checkpoint to %s", path)

    def to(self, device):
        self.device = torch.device(device)
        return super().to(device)

    def get_model_info(self) -> Dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'model_class': self.__class__.__name__,
            'num_classes': self.num_classes,
            'total_parameters': total,
            'trainable_parameters': trainable,
            'grid_size': self.grid_size.tolist(),
            'voxel_size': self.voxel_size.tolist(),
            'point_cloud_range': self.point_cloud_range.tolist(),
        }
