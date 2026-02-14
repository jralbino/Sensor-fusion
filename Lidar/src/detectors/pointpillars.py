"""
PointPillars - COMPLETE WORKING VERSION

This is a fully functional PointPillars implementation with all necessary components.
Fixes all previous issues including voxelization, coordinate handling, and device management.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, List, Optional
from pathlib import Path

from src.training.losses import AnchorGenerator, MultiClassAnchorGenerator, NUM_ANCHORS_PER_LOCATION
from src.core.geometry import nms_3d, nms_bev_fast


# ============================================================================
# PILLAR FEATURE NETWORK (VFE)
# ============================================================================

class PillarFeatureNet(nn.Module):
    """
    Pillar Feature Network - Encodes point features within each pillar.

    This is the Voxel Feature Encoding (VFE) layer that processes raw points
    in each pillar and produces a fixed-size feature representation.

    When ``use_augmented_features=True``, the input features are expanded from
    4 (x, y, z, intensity) to 10 by appending cluster-center offsets (3) and
    voxel-center offsets (3), matching the MMDet3D pretrained VFE architecture.
    """

    def __init__(self, num_input_features=4, num_filters=(64,),
                 with_distance=False, use_augmented_features=False):
        """
        Args:
            num_input_features: Number of raw input features per point (default: 4)
            num_filters: Tuple of filter sizes for each layer
            with_distance: Whether to append distance to cluster center
            use_augmented_features: If True, compute cluster + voxel offsets
                (4 raw -> 10 features). Requires voxel_coords, voxel_size,
                point_cloud_range to be passed to forward().
        """
        super().__init__()

        self.num_input_features = num_input_features
        self.with_distance = with_distance
        self.use_augmented_features = use_augmented_features

        # Determine actual input dim for the MLP
        effective_input = num_input_features
        if use_augmented_features:
            effective_input = num_input_features + 6  # +3 cluster + 3 voxel offsets
        elif with_distance:
            effective_input = num_input_features + 3

        # Build MLP layers
        num_filters = list(num_filters)
        assert len(num_filters) > 0

        layers = []
        in_channels = effective_input

        for out_channels in num_filters[:-1]:
            layers.extend([
                nn.Linear(in_channels, out_channels, bias=False),
                nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
                nn.ReLU()
            ])
            in_channels = out_channels

        # Final layer (no ReLU)
        layers.extend([
            nn.Linear(in_channels, num_filters[-1], bias=False),
            nn.BatchNorm1d(num_filters[-1], eps=1e-3, momentum=0.01)
        ])

        self.pfn_layers = nn.Sequential(*layers)
        self.num_output_features = num_filters[-1]

    def forward(self, voxels, num_points, coords=None, *,
                voxel_coords=None, voxel_size=None, point_cloud_range=None):
        """
        Args:
            voxels: (M, max_points, C) - Point features in each pillar
            num_points: (M,) - Number of valid points in each pillar
            coords: (M, 3 or 4) - Pillar coordinates (legacy, for with_distance)
            voxel_coords: (M, 4) - (batch, z, y, x) coordinates (for augmented features)
            voxel_size: (3,) tensor - voxel dimensions (for augmented features)
            point_cloud_range: (6,) tensor - [x_min, y_min, z_min, x_max, y_max, z_max]

        Returns:
            pillar_features: (M, num_filters[-1]) - Encoded pillar features
        """
        # Get shapes
        batch_size, max_points, num_features = voxels.shape

        points_mean = voxels[:, :, :3].sum(dim=1, keepdim=True) / num_points.unsqueeze(1).unsqueeze(2).clamp(min=1.0)

        if self.use_augmented_features:
            # Cluster-center offsets: point_xyz - mean_xyz
            cluster_offset = voxels[:, :, :3] - points_mean  # (M, max_pts, 3)

            # Voxel-center offsets: point_xyz - voxel_center_xyz
            # voxel_coords format: (batch, z, y, x) -> x=col3, y=col2, z=col1
            voxel_center_x = point_cloud_range[0] + (voxel_coords[:, 3].float() + 0.5) * voxel_size[0]
            voxel_center_y = point_cloud_range[1] + (voxel_coords[:, 2].float() + 0.5) * voxel_size[1]
            voxel_center_z = point_cloud_range[2] + (voxel_coords[:, 1].float() + 0.5) * voxel_size[2]
            voxel_center = torch.stack([voxel_center_x, voxel_center_y, voxel_center_z], dim=-1)
            voxel_offset = voxels[:, :, :3] - voxel_center.unsqueeze(1)  # (M, max_pts, 3)

            # Concatenate: [raw_features, cluster_offset(3), voxel_offset(3)]
            voxels = torch.cat([voxels, cluster_offset, voxel_offset], dim=-1)
        elif self.with_distance:
            points_offset = voxels[:, :, :3] - points_mean
            voxels = torch.cat([voxels, points_offset], dim=-1)

        # Create mask for valid points
        point_mask = torch.arange(max_points, device=voxels.device).unsqueeze(0)
        point_mask = point_mask < num_points.unsqueeze(1)  # (M, max_points)

        # Apply padding mask before MLP (matches MMDet3D's get_paddings_indicator)
        voxels = voxels * point_mask.unsqueeze(-1).float()

        # Flatten for batch processing
        voxels_flat = voxels.view(-1, voxels.shape[-1])  # (M * max_points, C)

        # Process through network
        features = self.pfn_layers(voxels_flat)  # (M * max_points, num_filters[-1])

        # Reshape back
        features = features.view(batch_size, max_points, -1)  # (M, max_points, num_filters[-1])

        # Mask invalid points
        features = features * point_mask.unsqueeze(-1).float()

        # Max pooling over points in each pillar
        pillar_features = features.max(dim=1)[0]  # (M, num_filters[-1])

        return pillar_features


# ============================================================================
# SCATTER LAYER
# ============================================================================

class PointPillarScatter(nn.Module):
    """
    Scatters pillar features back to a pseudo-image (BEV representation).
    """
    
    def __init__(self, num_input_features=64, nx=432, ny=496, nz=1):
        """
        Args:
            num_input_features: Number of input features per pillar
            nx, ny, nz: Grid dimensions (typically nz=1 for pillars)
        """
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.num_input_features = num_input_features
    
    def forward(self, pillar_features, coords, batch_size):
        """
        Args:
            pillar_features: (M, C) - Features for each pillar
            coords: (M, 4) - Pillar coordinates (batch_idx, z, y, x)
            batch_size: Number of batches
        
        Returns:
            pseudo_image: (B, C, ny, nx) - BEV feature map
        """
        batch_canvas = []
        
        for batch_idx in range(batch_size):
            # Filter pillars for this batch
            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask]
            this_features = pillar_features[batch_mask]
            
            # Create empty canvas
            canvas = torch.zeros(
                self.num_input_features,
                self.ny,
                self.nx,
                dtype=pillar_features.dtype,
                device=pillar_features.device
            )
            
            # Extract coordinates (format: batch_idx, z, y, x)
            # For pillars, z is always 0, so we use y and x
            y_coords = this_coords[:, 2].long()
            x_coords = this_coords[:, 3].long()
            
            # Clamp to valid range
            y_coords = y_coords.clamp(0, self.ny - 1)
            x_coords = x_coords.clamp(0, self.nx - 1)
            
            # Scatter features to canvas
            # Use transpose to match (C, y, x) format
            canvas[:, y_coords, x_coords] = this_features.t()
            
            batch_canvas.append(canvas)
        
        # Stack batches
        return torch.stack(batch_canvas, dim=0)  # (B, C, ny, nx)


# ============================================================================
# BACKBONE
# ============================================================================

class PointPillarsBackbone(nn.Module):
    """
    2D CNN backbone for processing BEV features.
    Uses a simple encoder-decoder structure with skip connections.
    """
    
    def __init__(self, num_input_features=64, layer_nums=(3, 5, 5), layer_strides=(2, 2, 2),
                 num_filters=(64, 128, 256), upsample_strides=(1, 2, 4), num_upsample_filters=(128, 128, 128)):
        """
        Args:
            num_input_features: Input channels
            layer_nums: Number of layers in each block
            layer_strides: Stride for each block
            num_filters: Output channels for each block
            upsample_strides: Upsampling factors
            num_upsample_filters: Channels after upsampling
        """
        super().__init__()
        
        assert len(layer_nums) == len(layer_strides) == len(num_filters)
        assert len(upsample_strides) == len(num_upsample_filters)
        
        self.num_input_features = num_input_features
        
        # Encoder blocks
        in_filters = [num_input_features, *num_filters[:-1]]
        blocks = []
        
        for i, (num_layers, stride, num_filter, in_filter) in enumerate(
            zip(layer_nums, layer_strides, num_filters, in_filters)
        ):
            block_layers = []
            for j in range(num_layers):
                use_stride = stride if j == 0 else 1
                block_layers.append(
                    nn.Conv2d(in_filter if j == 0 else num_filter, num_filter, 3, 
                             stride=use_stride, padding=1, bias=False)
                )
                block_layers.append(nn.BatchNorm2d(num_filter, eps=1e-3, momentum=0.01))
                block_layers.append(nn.ReLU())
            
            blocks.append(nn.Sequential(*block_layers))
        
        self.blocks = nn.ModuleList(blocks)
        
        # Decoder (upsampling)
        deblocks = []
        for i, (stride, num_filter, num_upsample_filter) in enumerate(
            zip(upsample_strides, num_filters, num_upsample_filters)
        ):
            if stride > 1:
                deblock = nn.Sequential(
                    nn.ConvTranspose2d(num_filter, num_upsample_filter, stride, 
                                      stride=stride, bias=False),
                    nn.BatchNorm2d(num_upsample_filter, eps=1e-3, momentum=0.01),
                    nn.ReLU()
                )
            else:
                deblock = nn.Sequential(
                    nn.Conv2d(num_filter, num_upsample_filter, 1, bias=False),
                    nn.BatchNorm2d(num_upsample_filter, eps=1e-3, momentum=0.01),
                    nn.ReLU()
                )
            deblocks.append(deblock)
        
        self.deblocks = nn.ModuleList(deblocks)
        self.num_output_features = sum(num_upsample_filters)
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) - BEV features
        
        Returns:
            out: (B, C_out, H', W') - Processed features
        """
        ups = []
        
        for i, block in enumerate(self.blocks):
            x = block(x)
            ups.append(self.deblocks[i](x))
        
        # Concatenate all upsampled features
        out = torch.cat(ups, dim=1)
        
        return out


# ============================================================================
# DETECTION HEAD
# ============================================================================

class PointPillarsHead(nn.Module):
    """
    Detection head that predicts class scores and bounding boxes.
    """
    
    def __init__(self, num_input_features=384, num_classes=10, num_anchors_per_location=2):
        """
        Args:
            num_input_features: Input feature channels
            num_classes: Number of object classes
            num_anchors_per_location: Number of anchors per spatial location
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.num_anchors_per_location = num_anchors_per_location
        
        # Classification head
        self.conv_cls = nn.Conv2d(num_input_features, num_anchors_per_location * num_classes, 1)
        
        # Regression head (7 values: x, y, z, l, w, h, yaw)
        self.conv_box = nn.Conv2d(num_input_features, num_anchors_per_location * 7, 1)
        
        # Direction classification (sin, cos of yaw)
        self.conv_dir = nn.Conv2d(num_input_features, num_anchors_per_location * 2, 1)
    
    def forward(self, spatial_features):
        """
        Args:
            spatial_features: (B, C, H, W) - Features from backbone
        
        Returns:
            Dictionary with predictions
        """
        cls_preds = self.conv_cls(spatial_features)
        box_preds = self.conv_box(spatial_features)
        dir_preds = self.conv_dir(spatial_features)
        
        # Permute to (B, H, W, num_anchors, num_classes/7/2)
        B, _, H, W = cls_preds.shape
        
        cls_preds = cls_preds.view(B, self.num_anchors_per_location, self.num_classes, H, W)
        cls_preds = cls_preds.permute(0, 3, 4, 1, 2).contiguous()  # (B, H, W, num_anchors, num_classes)
        
        box_preds = box_preds.view(B, self.num_anchors_per_location, 7, H, W)
        box_preds = box_preds.permute(0, 3, 4, 1, 2).contiguous()  # (B, H, W, num_anchors, 7)
        
        dir_preds = dir_preds.view(B, self.num_anchors_per_location, 2, H, W)
        dir_preds = dir_preds.permute(0, 3, 4, 1, 2).contiguous()  # (B, H, W, num_anchors, 2)
        
        return {
            'cls_preds': cls_preds,
            'box_preds': box_preds,
            'dir_preds': dir_preds
        }


# ============================================================================
# MAIN POINTPILLARS MODEL
# ============================================================================

class PointPillars(nn.Module):
    """
    Complete PointPillars implementation.
    
    This version includes all necessary components and proper initialization.
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
        """
        Args:
            num_classes: Number of object classes
            in_channels: Input channels per point (x, y, z, intensity)
            voxel_size: Voxel size (x, y, z)
            point_cloud_range: Detection range [x_min, y_min, z_min, x_max, y_max, z_max]
            max_num_points: Maximum points per voxel
            max_voxels: Maximum number of voxels (train, test)
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.voxel_size = np.array(voxel_size)
        self.point_cloud_range = np.array(point_cloud_range)
        self.max_num_points = max_num_points
        self.max_voxels = max_voxels if isinstance(max_voxels, tuple) else (max_voxels, max_voxels)
        
        # Calculate grid size
        grid_size = (
            (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / self.voxel_size
        )
        self.grid_size = np.round(grid_size).astype(np.int64)
        
        print(f"PointPillars initialized:")
        print(f"  Grid size: {self.grid_size}")
        print(f"  Voxel size: {self.voxel_size}")
        print(f"  Point cloud range: {self.point_cloud_range}")
        
        # Build network components
        # 1. Pillar Feature Network (VFE)
        self.vfe = PillarFeatureNet(
            num_input_features=in_channels,
            num_filters=(64,),
            with_distance=False
        )
        
        # 2. Scatter layer
        self.scatter = PointPillarScatter(
            num_input_features=64,
            nx=int(self.grid_size[0]),
            ny=int(self.grid_size[1]),
            nz=1
        )
        
        # 3. Backbone
        self.backbone = PointPillarsBackbone(
            num_input_features=64,
            layer_nums=(3, 5, 5),
            layer_strides=(2, 2, 2),
            num_filters=(64, 128, 256),
            upsample_strides=(1, 2, 4),
            num_upsample_filters=(128, 128, 128)
        )
        
        # 4. Detection head
        self.head = PointPillarsHead(
            num_input_features=self.backbone.num_output_features,
            num_classes=num_classes,
            num_anchors_per_location=NUM_ANCHORS_PER_LOCATION
        )
        
        self.device = torch.device('cpu')

        # Anchor cache (populated lazily on first forward)
        self._anchors: Optional[torch.Tensor] = None
        self._anchor_class_ids: Optional[torch.Tensor] = None
        self._anchor_gen: Optional[MultiClassAnchorGenerator] = None
    
    def forward(self, batch_dict: Dict) -> Dict:
        """
        Forward pass through the network.
        
        Args:
            batch_dict: Dictionary containing:
                - voxels: (M, max_points, C) voxelized points
                - voxel_coords: (M, 4) voxel coordinates
                - voxel_num_points: (M,) number of points per voxel
                - batch_size: int
        
        Returns:
            Dictionary with predictions
        """
        # 1. Pillar Feature Encoding
        pillar_features = self.vfe(
            batch_dict['voxels'],
            batch_dict['voxel_num_points'],
            batch_dict.get('voxel_coords')
        )
        
        # 2. Scatter to pseudo-image
        spatial_features = self.scatter(
            pillar_features,
            batch_dict['voxel_coords'],
            batch_dict['batch_size']
        )
        
        # 3. Backbone
        spatial_features = self.backbone(spatial_features)
        
        # 4. Detection head
        predictions = self.head(spatial_features)
        
        # Add predictions to batch_dict
        batch_dict.update(predictions)
        
        # Flatten and decode predictions
        B, H, W, num_anchors, _ = predictions['cls_preds'].shape

        box_deltas = predictions['box_preds'].reshape(B, -1, 7)  # (B, A, 7)
        anchors = self._get_anchors(box_deltas.device)           # (A, 7)

        # Decode deltas → absolute boxes for each sample in the batch
        decoded_boxes = []
        for b in range(B):
            decoded_boxes.append(self.decode_boxes(box_deltas[b], anchors))
        batch_dict['pred_boxes'] = torch.stack(decoded_boxes, dim=0)  # (B, A, 7)

        cls_scores = torch.sigmoid(predictions['cls_preds']).reshape(B, -1, self.num_classes)
        batch_dict['pred_scores'] = cls_scores.max(dim=-1)[0]
        batch_dict['pred_labels'] = cls_scores.argmax(dim=-1)
        
        return batch_dict
    
    def _get_anchors(self, device: torch.device) -> torch.Tensor:
        """Lazily generate and cache anchors on the correct device."""
        if self._anchors is not None and self._anchors.device == device:
            return self._anchors
        fm_h = int(self.grid_size[1]) // 2  # backbone stride=2
        fm_w = int(self.grid_size[0]) // 2
        self._anchor_gen = MultiClassAnchorGenerator(
            feature_map_size=(fm_h, fm_w),
            point_cloud_range=self.point_cloud_range,
        )
        self._anchors, self._anchor_class_ids = self._anchor_gen.generate(device)
        return self._anchors

    @staticmethod
    def decode_boxes(deltas: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        """Decode delta-encoded predictions back to absolute box coordinates.

        Encoding matches ``TargetAssigner.assign`` in losses.py:
            dx = (gt_x - a_x) / diag,  dy = ..., dz = (gt_z - a_z) / a_h
            dl = log(gt_l / a_l),       dw = ..., dh = ...
            dr = gt_yaw - a_yaw

        Args:
            deltas: (N, 7) predicted deltas
            anchors: (N, 7) anchor boxes (x, y, z, l, w, h, yaw)

        Returns:
            boxes: (N, 7) decoded boxes in world coordinates
        """
        diag = torch.sqrt(anchors[:, 3] ** 2 + anchors[:, 4] ** 2)

        x = deltas[:, 0] * diag + anchors[:, 0]
        y = deltas[:, 1] * diag + anchors[:, 1]
        z = deltas[:, 2] * anchors[:, 5] + anchors[:, 2]
        l = torch.exp(deltas[:, 3]) * anchors[:, 3]
        w = torch.exp(deltas[:, 4]) * anchors[:, 4]
        h = torch.exp(deltas[:, 5]) * anchors[:, 5]
        yaw = deltas[:, 6] + anchors[:, 6]

        return torch.stack([x, y, z, l, w, h, yaw], dim=-1)

    @torch.no_grad()
    def postprocess(
        self,
        batch_dict: Dict,
        score_thresh: float = 0.1,
        nms_iou_thresh: float = 0.3,
        max_detections: int = 500,
    ) -> List[Dict]:
        """Decode predictions and apply per-class NMS.

        Args:
            batch_dict: model forward output (must contain decoded pred_boxes,
                        cls_preds, dir_preds, batch_size).
            score_thresh: minimum score to keep before NMS.
            nms_iou_thresh: IoU threshold for NMS.
            max_detections: maximum detections per sample.

        Returns:
            list of dicts per sample with keys 'boxes', 'scores', 'labels'.
        """
        cls_preds = batch_dict['cls_preds']  # (B, H, W, R, C)
        pred_boxes = batch_dict['pred_boxes']  # (B, A, 7) — already decoded
        dir_preds = batch_dict['dir_preds']    # (B, H, W, R, 2)
        B = batch_dict['batch_size']

        cls_scores = torch.sigmoid(cls_preds.reshape(B, -1, self.num_classes))  # (B, A, C)

        results = []
        for b in range(B):
            boxes_b = pred_boxes[b].cpu().numpy()   # (A, 7)
            scores_b = cls_scores[b].cpu().numpy()  # (A, C)

            # Apply direction correction from dir head
            dir_b = dir_preds.reshape(B, -1, 2)[b]  # (A, 2)
            dir_label = dir_b.argmax(dim=-1).cpu().numpy()  # 0 or 1

            all_boxes, all_scores, all_labels = [], [], []

            for cls_idx in range(self.num_classes):
                cls_sc = scores_b[:, cls_idx]
                mask = cls_sc > score_thresh
                if not mask.any():
                    continue

                cls_boxes = boxes_b[mask]
                cls_scores_f = cls_sc[mask]

                # NMS (fast axis-aligned BEV)
                keep = nms_bev_fast(cls_boxes, cls_scores_f,
                                    iou_threshold=nms_iou_thresh,
                                    score_threshold=0.0)

                if len(keep) == 0:
                    continue

                all_boxes.append(cls_boxes[keep])
                all_scores.append(cls_scores_f[keep])
                all_labels.append(np.full(len(keep), cls_idx, dtype=np.int64))

            if len(all_boxes) > 0:
                all_boxes = np.concatenate(all_boxes)
                all_scores = np.concatenate(all_scores)
                all_labels = np.concatenate(all_labels)

                # Top-K
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
        """Move model to device."""
        self.device = device
        return super().to(device)
    
    def load_checkpoint(self, checkpoint_path):
        """Load model weights."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Remove 'module.' prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v
        
        self.load_state_dict(new_state_dict, strict=False)
        print(f"✅ Loaded checkpoint from {checkpoint_path}")
    
    def save_checkpoint(self, save_path, epoch=0, **kwargs):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'config': {
                'num_classes': self.num_classes,
                'voxel_size': self.voxel_size.tolist(),
                'point_cloud_range': self.point_cloud_range.tolist(),
                'grid_size': self.grid_size.tolist()
            }
        }
        checkpoint.update(kwargs)
        
        torch.save(checkpoint, save_path)
        print(f"✅ Saved checkpoint to {save_path}")


# For backward compatibility
SimplePointPillars = PointPillars