"""
SECOND: Sparsely Embedded Convolutional Detection

Uses sparse 2D convolutions (spconv) on the BEV pseudo-image
for more efficient feature extraction compared to dense 2D CNNs.
Shares the same pillar-based voxelization and detection head as PointPillars.

Reference: Yan et al., "SECOND: Sparsely Embedded Convolutional Detection", Sensors 2018
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import spconv.pytorch as spconv

from src.detectors.pointpillars import (
    PillarFeatureNet,
    PointPillarsHead,
    PointPillars,
)
from src.training.losses import AnchorGenerator, MultiClassAnchorGenerator, NUM_ANCHORS_PER_LOCATION
from src.core.geometry import nms_3d, nms_bev_fast


# ============================================================================
# SPARSE BEV BACKBONE
# ============================================================================

def _sparse_block(in_c, out_c, n_layers, stride=1):
    """Build a sparse conv block: one strided conv + N-1 submanifold convs."""
    layers = []
    # First layer: optionally strided
    if stride > 1:
        layers.append(spconv.SparseConv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False))
    else:
        layers.append(spconv.SubMConv2d(in_c, out_c, 3, padding=1, bias=False))
    layers.append(nn.BatchNorm1d(out_c, eps=1e-3, momentum=0.01))
    layers.append(nn.ReLU())

    # Remaining submanifold layers
    for _ in range(n_layers - 1):
        layers.append(spconv.SubMConv2d(out_c, out_c, 3, padding=1, bias=False))
        layers.append(nn.BatchNorm1d(out_c, eps=1e-3, momentum=0.01))
        layers.append(nn.ReLU())

    return spconv.SparseSequential(*layers)


class SparseBEVBackbone(nn.Module):
    """Sparse 2D backbone operating on the BEV pseudo-image.

    Architecture:
        Block 1: SubM(64→64) × 3, stride=1     → (ny, nx)
        Block 2: Conv(64→128, s=2) + SubM × 2   → (ny/2, nx/2)
        Block 3: Conv(128→256, s=2) + SubM × 2  → (ny/4, nx/4)

        Then convert to dense and apply 2D transposed-conv decoder
        to produce multi-scale BEV features.
    """

    def __init__(self, in_channels=64, ny=496, nx=432):
        super().__init__()
        self.ny = ny
        self.nx = nx

        # Sparse encoder blocks
        self.block1 = _sparse_block(in_channels, 64, n_layers=3, stride=1)
        self.block2 = _sparse_block(64, 128, n_layers=3, stride=2)
        self.block3 = _sparse_block(128, 256, n_layers=3, stride=2)

        # Dense decoder (transposed convolutions to upsample)
        self.deblock1 = nn.Sequential(
            nn.Conv2d(64, 128, 1, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )
        self.deblock2 = nn.Sequential(
            nn.Conv2d(128, 128, 1, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )
        self.deblock3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )

        self.num_output_features = 128 * 3  # 384

    def forward(self, pillar_features, coords, batch_size):
        """
        Args:
            pillar_features: (M, C) sparse pillar features
            coords: (M, 4) — (batch_idx, z, y, x) voxel coordinates
            batch_size: int

        Returns:
            dense_features: (B, 384, ny/2, nx/2)
        """
        device = pillar_features.device

        # Build sparse tensor for 2D grid — coords are (batch, z, y, x)
        # For 2D spconv we need indices as (batch, y, x)
        sparse_indices = coords[:, [0, 2, 3]].int()  # (M, 3): batch, y, x

        sp_tensor = spconv.SparseConvTensor(
            features=pillar_features,
            indices=sparse_indices,
            spatial_shape=[self.ny, self.nx],
            batch_size=batch_size,
        )

        # Sparse encoder
        x1 = self.block1(sp_tensor)   # (ny, nx, 64)
        x2 = self.block2(x1)          # (ny/2, nx/2, 128)
        x3 = self.block3(x2)          # (ny/4, nx/4, 256)

        # Convert to dense for decoder
        d1 = x1.dense()  # (B, 64, ny, nx)
        d2 = x2.dense()  # (B, 128, ny/2, nx/2)
        d3 = x3.dense()  # (B, 256, ny/4, nx/4)

        # Decoder: all outputs at (ny/2, nx/2) resolution
        # d1 at full res → stride-2 downsample via the deblock
        # We use avg_pool2d to match the target spatial size
        up1 = self.deblock1(d1)
        up1 = nn.functional.avg_pool2d(up1, 2)  # (B, 128, ny/2, nx/2)

        up2 = self.deblock2(d2)                   # (B, 128, ny/2, nx/2)
        up3 = self.deblock3(d3)                   # (B, 128, ny/2, nx/2)

        # Concatenate multi-scale
        out = torch.cat([up1, up2, up3], dim=1)  # (B, 384, ny/2, nx/2)
        return out


# ============================================================================
# SECOND DETECTOR
# ============================================================================

class SECOND(nn.Module):
    """
    SECOND detector with sparse BEV backbone.

    Uses the same pillar-based voxelization and detection head as PointPillars,
    but replaces the dense 2D CNN backbone with sparse 2D convolutions.

    Default configuration uses 360-degree range and augmented VFE features
    (10 features: x,y,z,intensity + cluster offsets + voxel offsets) to match
    the architecture used by MMDet3D pretrained models.
    """

    def __init__(
        self,
        num_classes=10,
        in_channels=4,
        voxel_size=(0.2, 0.2, 8.0),
        point_cloud_range=(-50, -50, -5, 50, 50, 3),
        max_num_points=64,
        max_voxels=(40000, 40000),
        use_augmented_features=True,
        **kwargs
    ):
        super().__init__()

        self.num_classes = num_classes
        self.voxel_size = np.array(voxel_size)
        self.point_cloud_range = np.array(point_cloud_range)
        self.max_num_points = max_num_points
        self.max_voxels = max_voxels if isinstance(max_voxels, tuple) else (max_voxels, max_voxels)
        self.use_augmented_features = use_augmented_features

        # Grid size
        grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / self.voxel_size
        self.grid_size = np.round(grid_size).astype(np.int64)

        print(f"SECOND initialized:")
        print(f"  Grid size: {self.grid_size}")
        print(f"  Voxel size: {self.voxel_size}")
        print(f"  Point cloud range: {self.point_cloud_range}")
        print(f"  Augmented VFE: {use_augmented_features}")

        nx, ny = int(self.grid_size[0]), int(self.grid_size[1])

        # 1. Pillar Feature Network
        self.vfe = PillarFeatureNet(
            num_input_features=in_channels,
            num_filters=(64,),
            with_distance=False,
            use_augmented_features=use_augmented_features,
        )

        # 2. Sparse BEV Backbone
        self.backbone = SparseBEVBackbone(
            in_channels=64,
            ny=ny,
            nx=nx,
        )

        # 3. Detection Head (shared with PointPillars)
        self.head = PointPillarsHead(
            num_input_features=self.backbone.num_output_features,
            num_classes=num_classes,
            num_anchors_per_location=NUM_ANCHORS_PER_LOCATION,
        )

        self.device = torch.device('cpu')
        self._anchors: Optional[torch.Tensor] = None
        self._anchor_class_ids: Optional[torch.Tensor] = None
        self._anchor_gen: Optional[MultiClassAnchorGenerator] = None

    def forward(self, batch_dict: Dict) -> Dict:
        """Forward pass — same interface as PointPillars."""
        device = batch_dict['voxels'].device

        # 1. Pillar Feature Encoding
        vfe_kwargs = {}
        if self.use_augmented_features:
            vfe_kwargs = dict(
                voxel_coords=batch_dict['voxel_coords'],
                voxel_size=torch.tensor(self.voxel_size, dtype=torch.float32, device=device),
                point_cloud_range=torch.tensor(self.point_cloud_range, dtype=torch.float32, device=device),
            )

        pillar_features = self.vfe(
            batch_dict['voxels'],
            batch_dict['voxel_num_points'],
            batch_dict.get('voxel_coords'),
            **vfe_kwargs,
        )

        # 2. Sparse BEV Backbone (replaces scatter + dense backbone)
        spatial_features = self.backbone(
            pillar_features,
            batch_dict['voxel_coords'],
            batch_dict['batch_size'],
        )

        # 3. Detection Head
        predictions = self.head(spatial_features)
        batch_dict.update(predictions)

        # 4. Decode predictions (same as PointPillars)
        B, H, W, num_anchors, _ = predictions['cls_preds'].shape
        box_deltas = predictions['box_preds'].reshape(B, -1, 7)
        anchors = self._get_anchors(box_deltas.device)

        decoded_boxes = []
        for b in range(B):
            decoded_boxes.append(PointPillars.decode_boxes(box_deltas[b], anchors))
        batch_dict['pred_boxes'] = torch.stack(decoded_boxes, dim=0)

        cls_scores = torch.sigmoid(predictions['cls_preds']).reshape(B, -1, self.num_classes)
        batch_dict['pred_scores'] = cls_scores.max(dim=-1)[0]
        batch_dict['pred_labels'] = cls_scores.argmax(dim=-1)

        return batch_dict

    def _get_anchors(self, device: torch.device) -> torch.Tensor:
        """Lazily generate and cache anchors."""
        if self._anchors is not None and self._anchors.device == device:
            return self._anchors
        fm_h = int(self.grid_size[1]) // 2
        fm_w = int(self.grid_size[0]) // 2
        self._anchor_gen = MultiClassAnchorGenerator(
            feature_map_size=(fm_h, fm_w),
            point_cloud_range=self.point_cloud_range,
        )
        self._anchors, self._anchor_class_ids = self._anchor_gen.generate(device)
        return self._anchors

    @torch.no_grad()
    def postprocess(self, batch_dict, score_thresh=0.1, nms_iou_thresh=0.3,
                    max_detections=500) -> List[Dict]:
        """Decode predictions and apply NMS — same as PointPillars."""
        cls_preds = batch_dict['cls_preds']
        pred_boxes = batch_dict['pred_boxes']
        dir_preds = batch_dict['dir_preds']
        B = batch_dict['batch_size']

        cls_scores = torch.sigmoid(cls_preds.reshape(B, -1, self.num_classes))

        results = []
        for b in range(B):
            boxes_b = pred_boxes[b].cpu().numpy()
            scores_b = cls_scores[b].cpu().numpy()

            all_boxes, all_scores, all_labels = [], [], []

            for cls_idx in range(self.num_classes):
                cls_sc = scores_b[:, cls_idx]
                mask = cls_sc > score_thresh
                if not mask.any():
                    continue

                cls_boxes = boxes_b[mask]
                cls_scores_f = cls_sc[mask]

                keep = nms_bev_fast(cls_boxes, cls_scores_f,
                                    iou_threshold=nms_iou_thresh, score_threshold=0.0)
                if len(keep) == 0:
                    continue

                all_boxes.append(cls_boxes[keep])
                all_scores.append(cls_scores_f[keep])
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

    def init_vfe_from_pretrained(self, ckpt_path: str):
        """Initialize VFE weights from an MMDet3D PointPillars checkpoint.

        Maps ``pts_voxel_encoder.vfe_layers.{i}.{linear,norm}`` from the
        MMDet3D checkpoint to ``vfe.pfn_layers.{linear,bn}`` in our model.
        Only transfers layers with compatible dimensions.

        Args:
            ckpt_path: Path to MMDet3D .pth checkpoint file.
        """
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        sd = ckpt.get('state_dict', ckpt)

        # Our PillarFeatureNet with augmented features uses a flat Sequential:
        #   pfn_layers.0 = Linear(10, 64)
        #   pfn_layers.1 = BatchNorm1d(64)
        # MMDet3D has 2 PFNLayers:
        #   vfe_layers.0.linear (10, 64), vfe_layers.0.norm (64)
        #   vfe_layers.1.linear (128, 64), vfe_layers.1.norm (64)
        # We can only map layer 0 since our architecture is a single-layer VFE.

        loaded = 0
        # Map MMDet3D vfe_layers.0.linear.weight -> vfe.pfn_layers.0.weight
        mm_key = 'pts_voxel_encoder.vfe_layers.0.linear.weight'
        our_key = 'vfe.pfn_layers.0.weight'
        if mm_key in sd:
            mm_w = sd[mm_key]
            our_w = self.state_dict()[our_key]
            if mm_w.shape == our_w.shape:
                self.state_dict()[our_key].copy_(mm_w)
                loaded += 1
                print(f"[init_vfe] Loaded {mm_key} -> {our_key} {mm_w.shape}")
            else:
                print(f"[init_vfe] Shape mismatch {mm_key}: {mm_w.shape} vs {our_w.shape}")

        # Map BN params
        for suffix in ['weight', 'bias', 'running_mean', 'running_var']:
            mm_key = f'pts_voxel_encoder.vfe_layers.0.norm.{suffix}'
            our_key = f'vfe.pfn_layers.1.{suffix}'
            if mm_key in sd and our_key in self.state_dict():
                mm_w = sd[mm_key]
                our_w = self.state_dict()[our_key]
                if mm_w.shape == our_w.shape:
                    self.state_dict()[our_key].copy_(mm_w)
                    loaded += 1

        print(f"[init_vfe] Transferred {loaded} parameters from pretrained VFE")

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
            'model_type': 'second',
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
