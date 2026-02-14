"""
CenterPoint: Center-based 3D Object Detection

Anchor-free detector that predicts object centers via heatmaps
and regresses box attributes (offset, dimensions, height, rotation)
at each center location.

Uses the same pillar-based voxelization and 2D CNN backbone as PointPillars,
but replaces the anchor-based head with a center-based heatmap head.

Reference: Yin et al., "Center-based 3D Object Detection and Tracking", CVPR 2021
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from src.detectors.pointpillars import (
    PillarFeatureNet,
    PointPillarScatter,
    PointPillarsBackbone,
)
from src.core.geometry import nms_bev_fast


# ============================================================================
# CENTER-BASED DETECTION HEAD
# ============================================================================

class CenterPointHead(nn.Module):
    """Center-based detection head with heatmap + regression branches.

    Architecture:
        shared_conv (384 → 64) → 5 parallel 1×1 branches:
            - heatmap:  (B, num_classes, H, W)
            - center:   (B, 2, H, W)  — sub-pixel x,y offset
            - dim:      (B, 3, H, W)  — l, w, h (log-space)
            - height:   (B, 1, H, W)  — z center
            - rot:      (B, 2, H, W)  — sin(yaw), cos(yaw)
    """

    def __init__(self, num_input_features=384, num_classes=10, hidden_channels=64):
        super().__init__()
        self.num_classes = num_classes

        # Shared feature extraction
        self.shared_conv = nn.Sequential(
            nn.Conv2d(num_input_features, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )

        # Heatmap branch (class centers)
        self.heatmap = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, num_classes, 1),
        )

        # Center offset branch (sub-pixel x, y)
        self.center = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, 2, 1),
        )

        # Dimension branch (l, w, h)
        self.dim = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, 3, 1),
        )

        # Height branch (z center)
        self.height = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, 1, 1),
        )

        # Rotation branch (sin, cos of yaw)
        self.rot = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, 2, 1),
        )

        # Initialize heatmap bias to -2.19 (log(0.1/(1-0.1)))
        # so initial predictions are ~0.1 probability
        self.heatmap[-1].bias.data.fill_(-2.19)

    def forward(self, spatial_features):
        """
        Args:
            spatial_features: (B, C, H, W) from backbone

        Returns:
            dict with heatmap, center, dim, height, rot predictions
        """
        x = self.shared_conv(spatial_features)

        return {
            'heatmap': self.heatmap(x),    # (B, num_classes, H, W)
            'center': self.center(x),      # (B, 2, H, W)
            'dim': self.dim(x),            # (B, 3, H, W)
            'height': self.height(x),      # (B, 1, H, W)
            'rot': self.rot(x),            # (B, 2, H, W)
        }


# ============================================================================
# CENTERPOINT DETECTOR
# ============================================================================

class CenterPoint(nn.Module):
    """
    CenterPoint detector with center-based heatmap head.

    Uses the same pillar-based voxelization and 2D CNN backbone as PointPillars,
    but replaces the anchor-based detection head with a center-based heatmap head.
    """

    def __init__(
        self,
        num_classes=10,
        in_channels=4,
        voxel_size=(0.16, 0.16, 4.0),
        point_cloud_range=(0, -39.68, -3, 69.12, 39.68, 1),
        max_num_points=32,
        max_voxels=(16000, 40000),
        **kwargs
    ):
        super().__init__()

        self.num_classes = num_classes
        self.voxel_size = np.array(voxel_size)
        self.point_cloud_range = np.array(point_cloud_range)
        self.max_num_points = max_num_points
        self.max_voxels = max_voxels if isinstance(max_voxels, tuple) else (max_voxels, max_voxels)

        # Grid size
        grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / self.voxel_size
        self.grid_size = np.round(grid_size).astype(np.int64)

        print(f"CenterPoint initialized:")
        print(f"  Grid size: {self.grid_size}")
        print(f"  Voxel size: {self.voxel_size}")
        print(f"  Point cloud range: {self.point_cloud_range}")

        nx, ny = int(self.grid_size[0]), int(self.grid_size[1])

        # 1. Pillar Feature Network (shared with PointPillars)
        self.vfe = PillarFeatureNet(
            num_input_features=in_channels,
            num_filters=(64,),
            with_distance=False,
        )

        # 2. Scatter to BEV pseudo-image
        self.scatter = PointPillarScatter(
            num_input_features=64,
            nx=nx, ny=ny,
        )

        # 3. 2D CNN Backbone (shared with PointPillars)
        self.backbone = PointPillarsBackbone(num_input_features=64)

        # 4. CenterPoint Head (replaces anchor-based head)
        self.head = CenterPointHead(
            num_input_features=self.backbone.num_output_features,
            num_classes=num_classes,
        )

        # Stride factor (backbone downsamples by 2)
        self.stride = 2
        # Feature map resolution for decoding
        self.fm_h = ny // self.stride  # 248
        self.fm_w = nx // self.stride  # 216

        self.device = torch.device('cpu')

    def forward(self, batch_dict: Dict) -> Dict:
        """Forward pass."""
        # 1. Pillar Feature Encoding
        pillar_features = self.vfe(
            batch_dict['voxels'],
            batch_dict['voxel_num_points'],
            batch_dict.get('voxel_coords'),
        )

        # 2. Scatter to BEV pseudo-image
        pseudo_image = self.scatter(pillar_features, batch_dict['voxel_coords'],
                                     batch_dict['batch_size'])

        # 3. 2D CNN Backbone
        spatial_features = self.backbone(pseudo_image)

        # 4. CenterPoint Head
        predictions = self.head(spatial_features)
        batch_dict.update(predictions)
        batch_dict['batch_size_val'] = batch_dict['batch_size']

        # 5. Decode predictions for inference
        self._decode_predictions(batch_dict)

        return batch_dict

    def _decode_predictions(self, batch_dict: Dict):
        """Decode heatmap predictions into boxes, scores, labels."""
        heatmap = torch.sigmoid(batch_dict['heatmap'])  # (B, C, H, W)
        center = batch_dict['center']                     # (B, 2, H, W)
        dim = batch_dict['dim']                           # (B, 3, H, W)
        height = batch_dict['height']                     # (B, 1, H, W)
        rot = batch_dict['rot']                           # (B, 2, H, W)

        B, C, H, W = heatmap.shape

        # Grid coordinates
        device = heatmap.device
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij',
        )  # (H, W)

        # Voxel-to-world conversion
        x_stride = self.voxel_size[0] * self.stride
        y_stride = self.voxel_size[1] * self.stride
        x_min = self.point_cloud_range[0]
        y_min = self.point_cloud_range[1]

        all_boxes = []
        all_scores = []
        all_labels = []

        for b in range(B):
            # Flatten heatmap: (C, H, W) → (C*H*W,)
            hm_b = heatmap[b]  # (C, H, W)
            scores_flat = hm_b.reshape(-1)  # (C*H*W,)

            # Top-K globally
            K = min(500, scores_flat.shape[0])
            topk_scores, topk_inds = scores_flat.topk(K)

            # Decode indices
            topk_classes = topk_inds // (H * W)
            topk_spatial = topk_inds % (H * W)
            topk_ys = topk_spatial // W
            topk_xs = topk_spatial % W

            # Gather regression values at peak locations
            center_b = center[b]  # (2, H, W)
            dim_b = dim[b]        # (3, H, W)
            height_b = height[b]  # (1, H, W)
            rot_b = rot[b]        # (2, H, W)

            cx = center_b[0][topk_ys, topk_xs]  # (K,) sub-pixel offset x
            cy = center_b[1][topk_ys, topk_xs]  # (K,) sub-pixel offset y

            # World coordinates
            x_world = (topk_xs.float() + cx) * x_stride + x_min + x_stride / 2
            y_world = (topk_ys.float() + cy) * y_stride + y_min + y_stride / 2
            z_world = height_b[0][topk_ys, topk_xs]

            # Dimensions (exp to ensure positive)
            l = dim_b[0][topk_ys, topk_xs].exp()
            w = dim_b[1][topk_ys, topk_xs].exp()
            h = dim_b[2][topk_ys, topk_xs].exp()

            # Rotation
            sin_yaw = rot_b[0][topk_ys, topk_xs]
            cos_yaw = rot_b[1][topk_ys, topk_xs]
            yaw = torch.atan2(sin_yaw, cos_yaw)

            boxes = torch.stack([x_world, y_world, z_world, l, w, h, yaw], dim=-1)  # (K, 7)

            all_boxes.append(boxes)
            all_scores.append(topk_scores)
            all_labels.append(topk_classes)

        batch_dict['pred_boxes'] = torch.stack(all_boxes, dim=0)    # (B, K, 7)
        batch_dict['pred_scores'] = torch.stack(all_scores, dim=0)  # (B, K)
        batch_dict['pred_labels'] = torch.stack(all_labels, dim=0)  # (B, K)

    @torch.no_grad()
    def postprocess(self, batch_dict, score_thresh=0.1, nms_iou_thresh=0.3,
                    max_detections=500) -> List[Dict]:
        """Decode predictions and apply NMS."""
        pred_boxes = batch_dict['pred_boxes']    # (B, K, 7)
        pred_scores = batch_dict['pred_scores']  # (B, K)
        pred_labels = batch_dict['pred_labels']  # (B, K)
        B = batch_dict['batch_size']

        results = []
        for b in range(B):
            boxes_b = pred_boxes[b].cpu().numpy()
            scores_b = pred_scores[b].cpu().numpy()
            labels_b = pred_labels[b].cpu().numpy()

            all_boxes, all_scores, all_labels = [], [], []

            for cls_idx in range(self.num_classes):
                cls_mask = labels_b == cls_idx
                cls_scores = scores_b[cls_mask]
                cls_boxes = boxes_b[cls_mask]

                if len(cls_scores) == 0:
                    continue

                # Score filter
                score_mask = cls_scores > score_thresh
                if not score_mask.any():
                    continue

                cls_boxes = cls_boxes[score_mask]
                cls_scores = cls_scores[score_mask]

                # NMS
                keep = nms_bev_fast(cls_boxes, cls_scores,
                                    iou_threshold=nms_iou_thresh,
                                    score_threshold=0.0)

                if len(keep) == 0:
                    continue

                all_boxes.append(cls_boxes[keep])
                all_scores.append(cls_scores[keep])
                all_labels.append(np.full(len(keep), cls_idx, dtype=np.int64))

            if len(all_boxes) > 0:
                all_boxes = np.concatenate(all_boxes)
                all_scores = np.concatenate(all_scores)
                all_labels = np.concatenate(all_labels)

                if len(all_scores) > max_detections:
                    topk = np.argsort(all_scores)[::-1][:max_detections]
                    all_boxes = all_boxes[topk]
                    all_scores = all_scores[topk]
                    all_labels = all_labels[topk]
            else:
                all_boxes = np.zeros((0, 7), dtype=np.float32)
                all_scores = np.zeros(0, dtype=np.float32)
                all_labels = np.zeros(0, dtype=np.int64)

            results.append({
                'boxes': all_boxes,
                'scores': all_scores,
                'labels': all_labels,
            })

        return results

    def to(self, device):
        self.device = device
        return super().to(device)

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v

        self.load_state_dict(new_state_dict, strict=False)
        print(f"Loaded checkpoint from {checkpoint_path}")

    def save_checkpoint(self, save_path, epoch=0, **kwargs):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'model_type': 'centerpoint',
            'config': {
                'num_classes': self.num_classes,
                'voxel_size': self.voxel_size.tolist(),
                'point_cloud_range': self.point_cloud_range.tolist(),
                'grid_size': self.grid_size.tolist(),
            }
        }
        checkpoint.update(kwargs)
        torch.save(checkpoint, save_path)
        print(f"Saved checkpoint to {save_path}")
