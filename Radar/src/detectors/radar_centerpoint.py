# -*- coding: utf-8 -*-
"""
RadarCenterPoint — Anchor-free heatmap-based detector for radar.

Same VFE + scatter + backbone as RadarPillars, but with a CenterPoint-style
detection head that predicts heatmaps + regression per class.

Extra branch: velocity prediction (radar provides Doppler velocity).
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from Radar.src.core.base_radar_detector import BaseRadarDetector, Detection3D
from Radar.src.detectors.radar_pillars import (
    RadarPillarFeatureNet,
    PillarScatter,
    RadarBackbone,
)


class CenterPointHead(nn.Module):
    """Anchor-free heatmap detection head with velocity branch.

    Outputs per-pixel predictions:
        - heatmap:  (B, num_classes, H, W)
        - center:   (B, 2, H, W)  — sub-pixel offset
        - dim:      (B, 3, H, W)  — l, w, h (log-space)
        - height:   (B, 1, H, W)  — z center
        - rot:      (B, 2, H, W)  — sin(yaw), cos(yaw)
        - vel:      (B, 2, H, W)  — vx, vy
    """

    def __init__(self, in_channels: int = 192, num_classes: int = 10):
        super().__init__()
        mid = 64
        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )
        self.heatmap = nn.Conv2d(mid, num_classes, 1)
        self.center = nn.Conv2d(mid, 2, 1)
        self.dim = nn.Conv2d(mid, 3, 1)
        self.height = nn.Conv2d(mid, 1, 1)
        self.rot = nn.Conv2d(mid, 2, 1)
        self.vel = nn.Conv2d(mid, 2, 1)

        # Initialize heatmap bias to -2.19 (focal loss init)
        nn.init.constant_(self.heatmap.bias, -2.19)

    def forward(self, x: torch.Tensor) -> Dict:
        feat = self.shared(x)
        return {
            'heatmap': self.heatmap(feat),
            'center': self.center(feat),
            'dim': self.dim(feat),
            'height': self.height(feat),
            'rot': self.rot(feat),
            'vel': self.vel(feat),
        }


class RadarCenterPoint(BaseRadarDetector):
    """CenterPoint-style anchor-free detector for radar.

    Args:
        num_classes: Number of detection classes.
        in_channels: Number of input point features (default 6).
        voxel_size: Pillar size in metres.
        point_cloud_range: Detection range.
        top_k: Maximum detections per frame.
        score_threshold: Minimum heatmap score.
    """

    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 6,
        voxel_size: Tuple[float, float, float] = (0.5, 0.5, 8.0),
        point_cloud_range: Tuple[float, ...] = (-100.0, -100.0, -5.0, 100.0, 100.0, 3.0),
        top_k: int = 100,
        score_threshold: float = 0.1,
        **kwargs,
    ):
        super().__init__(
            num_classes=num_classes,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            in_channels=in_channels,
            **kwargs,
        )
        self.top_k = top_k
        self.score_threshold = score_threshold

        nx, ny, nz = self.grid_size

        self.vfe = RadarPillarFeatureNet(in_channels=in_channels, out_channels=64)
        self.scatter = PillarScatter(in_channels=64, nx=nx, ny=ny)
        self.backbone = RadarBackbone(in_channels=64)
        self.head = CenterPointHead(
            in_channels=self.backbone.out_channels,
            num_classes=num_classes,
        )
        self.output_stride = int(np.prod(self.backbone.strides))

    def forward(self, batch_dict: Dict) -> Dict:
        voxels = batch_dict['voxels']
        coords = batch_dict['voxel_coords']
        num_points = batch_dict['voxel_num_points']

        vs = torch.tensor(self.voxel_size, device=voxels.device, dtype=torch.float32)
        pcr = torch.tensor(self.point_cloud_range, device=voxels.device, dtype=torch.float32)

        pillar_features = self.vfe(voxels, num_points, coords, vs, pcr)
        bev = self.scatter(pillar_features, coords)
        features = self.backbone(bev)
        preds = self.head(features)

        batch_dict.update(preds)

        if not self.training:
            batch_dict.update(self._decode_predictions(preds))

        return batch_dict

    def get_loss(self, batch_dict: Dict) -> Tuple[torch.Tensor, Dict]:
        # Placeholder — full implementation uses heatmap focal + regression L1
        loss = torch.tensor(0.0, requires_grad=True, device=self.device)
        return loss, {'total_loss': 0.0}

    def _decode_predictions(self, preds: Dict) -> Dict:
        """Decode heatmap peaks into 3D bounding boxes."""
        heatmap = preds['heatmap'].sigmoid()  # (B, C, H, W)
        center = preds['center']               # (B, 2, H, W)
        dim = preds['dim']                     # (B, 3, H, W)
        height = preds['height']               # (B, 1, H, W)
        rot = preds['rot']                     # (B, 2, H, W)
        vel = preds['vel']                     # (B, 2, H, W)

        B, C, H, W = heatmap.shape

        # Max over classes → (B, H, W)
        scores, labels = heatmap.max(dim=1)

        # Flatten spatial dims
        scores_flat = scores.view(B, -1)
        labels_flat = labels.view(B, -1)

        K = min(self.top_k, scores_flat.shape[1])
        topk_scores, topk_idx = scores_flat.topk(K, dim=1)

        # Gather spatial indices
        topk_y = topk_idx // W
        topk_x = topk_idx % W

        # Gather regression values
        def _gather(tensor):
            # tensor: (B, channels, H, W)
            ch = tensor.shape[1]
            flat = tensor.view(B, ch, -1)  # (B, ch, H*W)
            idx = topk_idx.unsqueeze(1).expand(-1, ch, -1)  # (B, ch, K)
            return flat.gather(2, idx).permute(0, 2, 1)  # (B, K, ch)

        center_vals = _gather(center)  # (B, K, 2)
        dim_vals = _gather(dim)        # (B, K, 3)
        height_vals = _gather(height)  # (B, K, 1)
        rot_vals = _gather(rot)        # (B, K, 2)
        vel_vals = _gather(vel)        # (B, K, 2)

        # Convert grid coords to metric using derived output stride
        stride = self.output_stride
        x_metric = (topk_x.float() + center_vals[:, :, 0]) * stride * self.voxel_size[0] + self.point_cloud_range[0]
        y_metric = (topk_y.float() + center_vals[:, :, 1]) * stride * self.voxel_size[1] + self.point_cloud_range[1]
        z_metric = height_vals[:, :, 0]

        # Dimensions: exp to ensure positive
        l = dim_vals[:, :, 0].exp()
        w = dim_vals[:, :, 1].exp()
        h = dim_vals[:, :, 2].exp()

        # Yaw from sin/cos
        yaw = torch.atan2(rot_vals[:, :, 0], rot_vals[:, :, 1])

        pred_boxes = torch.stack([x_metric, y_metric, z_metric, l, w, h, yaw], dim=-1)
        pred_labels = labels_flat.gather(1, topk_idx)

        return {
            'pred_boxes': pred_boxes,
            'pred_scores': topk_scores,
            'pred_labels': pred_labels,
            'pred_velocity': vel_vals,
        }
