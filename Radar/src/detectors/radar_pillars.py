# -*- coding: utf-8 -*-
"""
RadarPillars — PointPillars adapted for sparse radar point clouds.

Key adaptations vs LiDAR PointPillars:
- Input features: x, y, z, rcs, vx_comp, vy_comp (6 channels)
- Augmented VFE with cluster/pillar offsets → 12 input channels
- Uniform channel scaling (64→64→64) instead of doubling — better for sparse data
- Larger voxels (0.5m) and wider range (200m) for radar's extended range
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from Radar.src.core.base_radar_detector import BaseRadarDetector


class RadarPillarFeatureNet(nn.Module):
    """Voxel Feature Encoder for radar pillars.

    Takes raw radar features + augmented features (cluster offsets,
    pillar center offsets) and encodes each pillar to a fixed-size vector.
    """

    def __init__(self, in_channels: int = 6, out_channels: int = 64):
        super().__init__()
        # Augmented features: +3 (cluster center offset) +3 (pillar center offset) = +6
        augmented_channels = in_channels + 6
        self.linear = nn.Linear(augmented_channels, out_channels, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(
        self,
        voxels: torch.Tensor,       # (M, max_pts, C)
        num_points: torch.Tensor,    # (M,)
        coords: torch.Tensor,        # (M, 3)
        voxel_size: torch.Tensor,    # (3,)
        pc_range: torch.Tensor,      # (6,)
    ) -> torch.Tensor:
        """Encode pillars → (M, out_channels)."""
        if voxels.shape[0] == 0:
            return torch.zeros((0, self.linear.out_features), device=voxels.device)
        M, P, C = voxels.shape

        # Cluster center offset: point - mean_of_points_in_pillar
        mask = torch.arange(P, device=voxels.device).unsqueeze(0) < num_points.unsqueeze(1)
        mask_f = mask.unsqueeze(-1).float()  # (M, P, 1)

        points_sum = (voxels[:, :, :3] * mask_f).sum(dim=1, keepdim=True)
        counts = num_points.clamp(min=1).float().view(M, 1, 1)
        cluster_center = points_sum / counts  # (M, 1, 3)
        cluster_offset = voxels[:, :, :3] - cluster_center  # (M, P, 3)

        # Pillar center offset: point - geometric_center_of_pillar
        pillar_x = coords[:, 2].float() * voxel_size[0] + pc_range[0] + voxel_size[0] / 2
        pillar_y = coords[:, 1].float() * voxel_size[1] + pc_range[1] + voxel_size[1] / 2
        pillar_z = coords[:, 0].float() * voxel_size[2] + pc_range[2] + voxel_size[2] / 2
        pillar_center = torch.stack([pillar_x, pillar_y, pillar_z], dim=1)  # (M, 3)
        pillar_offset = voxels[:, :, :3] - pillar_center.unsqueeze(1)  # (M, P, 3)

        # Concatenate: [original_features, cluster_offset, pillar_offset]
        augmented = torch.cat([voxels, cluster_offset, pillar_offset], dim=-1)  # (M, P, C+6)

        # Apply padding mask
        augmented = augmented * mask_f

        # Linear → BN → ReLU → max-pool
        x = self.linear(augmented.view(-1, augmented.shape[-1]))  # (M*P, 64)
        x = self.bn(x)
        x = self.relu(x)
        x = x.view(M, P, -1)  # (M, P, 64)

        # Max-pool over points
        x = x * mask_f  # zero out padded points
        x = x.max(dim=1)[0]  # (M, 64)

        return x


class PillarScatter(nn.Module):
    """Scatter pillar features to BEV pseudo-image."""

    def __init__(self, in_channels: int = 64, nx: int = 400, ny: int = 400):
        super().__init__()
        self.in_channels = in_channels
        self.nx = nx
        self.ny = ny

    def forward(
        self,
        pillar_features: torch.Tensor,  # (M, C)
        coords: torch.Tensor,            # (M, 3) [z, y, x]
    ) -> torch.Tensor:
        """Returns (1, C, ny, nx) BEV pseudo-image."""
        canvas = torch.zeros(
            1, self.in_channels, self.ny, self.nx,
            dtype=pillar_features.dtype, device=pillar_features.device,
        )
        x_idx = coords[:, 2].long().clamp(0, self.nx - 1)
        y_idx = coords[:, 1].long().clamp(0, self.ny - 1)
        canvas[0, :, y_idx, x_idx] = pillar_features.T
        return canvas


class RadarBackbone(nn.Module):
    """2D CNN backbone with uniform channel scaling for sparse radar.

    Uses 64 channels in all blocks (instead of doubling like LiDAR).
    """

    def __init__(self, in_channels: int = 64, layer_nums: Tuple[int, ...] = (3, 5, 5)):
        super().__init__()
        C = in_channels
        self.blocks = nn.ModuleList()
        self.strides = [1, 2, 2]

        for i, (n_layers, stride) in enumerate(zip(layer_nums, self.strides)):
            layers = [
                nn.Conv2d(C, C, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(C),
                nn.ReLU(inplace=True),
            ]
            for _ in range(n_layers - 1):
                layers.extend([
                    nn.Conv2d(C, C, 3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(C),
                    nn.ReLU(inplace=True),
                ])
            self.blocks.append(nn.Sequential(*layers))

        # Deconv upsampling for FPN
        self.deblocks = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(C, C, s, stride=s, bias=False),
                nn.BatchNorm2d(C),
                nn.ReLU(inplace=True),
            )
            for s in [1, 2, 4]
        ])

        self.out_channels = C * 3  # concat of 3 scales

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input: (B, C, H, W) → Output: (B, C*3, H, W)."""
        ups = []
        for block, deblock in zip(self.blocks, self.deblocks):
            x = block(x)
            ups.append(deblock(x))
        return torch.cat(ups, dim=1)


class RadarDetectionHead(nn.Module):
    """Anchor-based detection head for radar."""

    def __init__(
        self,
        in_channels: int = 192,
        num_classes: int = 10,
        num_anchors_per_loc: int = 18,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors_per_loc

        self.cls_head = nn.Conv2d(in_channels, num_anchors_per_loc * num_classes, 1)
        self.box_head = nn.Conv2d(in_channels, num_anchors_per_loc * 7, 1)
        self.dir_head = nn.Conv2d(in_channels, num_anchors_per_loc * 2, 1)
        self.vel_head = nn.Conv2d(in_channels, num_anchors_per_loc * 2, 1)

    def forward(self, x: torch.Tensor) -> Dict:
        B, _, H, W = x.shape
        R = self.num_anchors

        cls = self.cls_head(x).view(B, H, W, R, self.num_classes)
        box = self.box_head(x).view(B, H, W, R, 7)
        dir_ = self.dir_head(x).view(B, H, W, R, 2)
        vel = self.vel_head(x).view(B, H, W, R, 2)

        return {
            'cls_preds': cls,
            'box_preds': box,
            'dir_preds': dir_,
            'vel_preds': vel,
        }


class RadarPillars(BaseRadarDetector):
    """RadarPillars: PointPillars adapted for radar point clouds.

    Args:
        num_classes: Number of detection classes.
        in_channels: Number of input point features (default 6: x,y,z,rcs,vx,vy).
        voxel_size: Pillar size [x, y, z] in metres.
        point_cloud_range: Detection range.
    """

    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 6,
        voxel_size: Tuple[float, float, float] = (0.5, 0.5, 8.0),
        point_cloud_range: Tuple[float, ...] = (-100.0, -100.0, -5.0, 100.0, 100.0, 3.0),
        **kwargs,
    ):
        super().__init__(
            num_classes=num_classes,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            in_channels=in_channels,
            **kwargs,
        )

        nx, ny, nz = self.grid_size

        self.vfe = RadarPillarFeatureNet(in_channels=in_channels, out_channels=64)
        self.scatter = PillarScatter(in_channels=64, nx=nx, ny=ny)
        self.backbone = RadarBackbone(in_channels=64)
        self.head = RadarDetectionHead(
            in_channels=self.backbone.out_channels,
            num_classes=num_classes,
        )

    def forward(self, batch_dict: Dict) -> Dict:
        voxels = batch_dict['voxels']
        coords = batch_dict['voxel_coords']
        num_points = batch_dict['voxel_num_points']

        voxel_size_t = torch.tensor(self.voxel_size, device=voxels.device, dtype=torch.float32)
        pc_range_t = torch.tensor(self.point_cloud_range, device=voxels.device, dtype=torch.float32)

        # VFE → scatter → backbone → head
        pillar_features = self.vfe(voxels, num_points, coords, voxel_size_t, pc_range_t)
        bev = self.scatter(pillar_features, coords)
        features = self.backbone(bev)
        preds = self.head(features)

        batch_dict.update(preds)

        # Generate simple predictions for inference
        if not self.training:
            batch_dict.update(self._generate_predictions(preds))

        return batch_dict

    def get_loss(self, batch_dict: Dict) -> Tuple[torch.Tensor, Dict]:
        # Placeholder — delegates to training/losses.py
        # Returns zero loss for now; implement with focal + smooth_l1 + dir_bce
        loss = torch.tensor(0.0, requires_grad=True, device=self.device)
        return loss, {'total_loss': 0.0}

    def _generate_predictions(self, preds: Dict) -> Dict:
        """Simple argmax-based prediction extraction."""
        cls = preds['cls_preds']      # (B, H, W, R, C)
        box = preds['box_preds']      # (B, H, W, R, 7)
        vel = preds['vel_preds']      # (B, H, W, R, 2)

        B = cls.shape[0]
        cls_flat = cls.reshape(B, -1, self.num_classes)
        box_flat = box.reshape(B, -1, 7)
        vel_flat = vel.reshape(B, -1, 2)

        scores, labels = cls_flat.sigmoid().max(dim=-1)

        # Top-K
        K = min(100, scores.shape[1])
        topk_scores, topk_idx = scores.topk(K, dim=1)

        pred_boxes = torch.gather(box_flat, 1, topk_idx.unsqueeze(-1).expand(-1, -1, 7))
        pred_labels = torch.gather(labels, 1, topk_idx)
        pred_vel = torch.gather(vel_flat, 1, topk_idx.unsqueeze(-1).expand(-1, -1, 2))

        return {
            'pred_boxes': pred_boxes,
            'pred_scores': topk_scores,
            'pred_labels': pred_labels,
            'pred_velocity': pred_vel,
        }
