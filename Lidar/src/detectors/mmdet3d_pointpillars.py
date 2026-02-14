"""
MMDetection3D PointPillars Wrapper

Replicates the exact architecture of MMDet3D's ``hv_pointpillars_secfpn_sbn-all``
so that its pre-trained checkpoint can be loaded without installing mmdetection3d.

Architecture verified against checkpoint key shapes:
- VFE: 2 PFNLayers with concat pattern (10→64, 128→64)
- Backbone: layer_nums=(4,6,6) — MMDet3D adds 1 strided conv before layer_nums
- Neck: ALL deblocks use ConvTranspose2d (even stride=1)
- Head: 14 anchors/loc, 9-param regression (7 box + 2 velocity)
- Grid: 400x400 (voxel_size=0.25, range=[-50..50])
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from math import pi

from src.detectors.pointpillars import PointPillarScatter
from src.core.geometry import nms_bev_fast


# ============================================================================
# MMDet3D PFN Layer (Pillar Feature Network Layer)
# ============================================================================

class PFNLayer(nn.Module):
    """Single PFN layer from MMDet3D.

    Matches MMDet3D exactly: no point masking before max-pool.
    BN applied via permute (M,P,C) → (M,C,P) to match pretrained stats.

    Non-last layers: output point features and max-pooled features concatenated.
    Last layer: output only max-pooled features.
    """

    def __init__(self, in_channels, out_channels, last_layer=False):
        super().__init__()
        self.last_layer = last_layer
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)

    def forward(self, features, num_points):
        """
        Args:
            features: (M, max_pts, C_in)
            num_points: (M,) — unused, kept for API compat
        Returns:
            If last: (M, out_channels)
            If not last: (M, max_pts, out_channels * 2)
        """
        # Linear on last dim: (M, P, C_in) → (M, P, C_out)
        x = self.linear(features.reshape(-1, features.shape[-1]))
        x = x.view(features.shape[0], features.shape[1], -1)

        # BN: (M, P, C) → (M, C, P) → BN → (M, C, P) → (M, P, C)
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = torch.relu(x)

        # Max-pool over points (NO masking — matches MMDet3D)
        x_max = x.max(dim=1, keepdim=True)[0]  # (M, 1, C)

        if self.last_layer:
            return x_max.squeeze(1)  # (M, C)
        else:
            # Repeat and concatenate with point-level features
            x_repeat = x_max.expand_as(x)  # (M, P, C)
            return torch.cat([x, x_repeat], dim=-1)  # (M, P, 2*C)


class MMDet3DPillarFeatureNet(nn.Module):
    """Replicates MMDet3D's PillarFeatureNet with 2 VFE layers.

    Input features per point (10):
        [x, y, z, intensity, x_c, y_c, z_c, x_p, y_p, z_p]
    Layer 0: Linear(10, 64) → BN → ReLU → max-pool → cat → (M, P, 128)
    Layer 1: Linear(128, 64) → BN → ReLU → max-pool → (M, 64)
    """

    def __init__(self):
        super().__init__()
        self.vfe_layers = nn.ModuleList([
            PFNLayer(10, 64, last_layer=False),
            PFNLayer(128, 64, last_layer=True),
        ])

    def forward(self, voxels, num_points, voxel_coords, voxel_size, point_cloud_range):
        """
        Args:
            voxels: (M, max_pts, 4) raw point features (x,y,z,intensity)
            num_points: (M,) valid point count per pillar
            voxel_coords: (M, 4) pillar coords (batch, z, y, x)
            voxel_size: (3,) tensor
            point_cloud_range: (6,) tensor

        Returns:
            pillar_features: (M, 64)
        """
        M, max_pts, C = voxels.shape

        # 0. Normalize intensity to [0, 1] — MMDet3D checkpoint was trained
        #    with normalized intensity (weight analysis confirms large weights
        #    on the intensity channel, expecting small input values).
        voxels = voxels.clone()
        voxels[:, :, 3] = voxels[:, :, 3] / 255.0

        # 1. Cluster-center offsets: point_xyz - mean_xyz
        points_mean = voxels[:, :, :3].sum(dim=1, keepdim=True) / \
            num_points.float().unsqueeze(1).unsqueeze(2).clamp(min=1.0)
        cluster_offset = voxels[:, :, :3] - points_mean  # (M, max_pts, 3)

        # 2. Voxel-center offsets: point_xyz - voxel_center_xyz
        # voxel_coords format: (batch, z, y, x) -> x=col3, y=col2, z=col1
        voxel_center_x = point_cloud_range[0] + (voxel_coords[:, 3].float() + 0.5) * voxel_size[0]
        voxel_center_y = point_cloud_range[1] + (voxel_coords[:, 2].float() + 0.5) * voxel_size[1]
        voxel_center_z = point_cloud_range[2] + (voxel_coords[:, 1].float() + 0.5) * voxel_size[2]

        voxel_center = torch.stack([voxel_center_x, voxel_center_y, voxel_center_z], dim=-1)
        voxel_offset = voxels[:, :, :3] - voxel_center.unsqueeze(1)

        # 3. Concatenate: [x,y,z,intensity, cluster_offset(3), voxel_offset(3)] = 10
        features = torch.cat([voxels, cluster_offset, voxel_offset], dim=-1)

        # 4. Apply padding mask — zero out features for padded point slots.
        #    This matches MMDet3D's HardVFE which applies get_paddings_indicator()
        #    before VFE layers, preventing padded points with large voxel-center
        #    offsets from dominating the features.
        padding_mask = torch.arange(
            max_pts, device=features.device
        ).unsqueeze(0) < num_points.unsqueeze(1)  # (M, max_pts)
        features = features * padding_mask.unsqueeze(-1).float()

        # 5. Pass through PFN layers
        for layer in self.vfe_layers:
            features = layer(features, num_points)

        return features  # (M, 64)


# ============================================================================
# MMDet3D SECOND Backbone (layer_nums = 4, 6, 6)
# ============================================================================

class MMDet3DBackbone(nn.Module):
    """SECOND backbone matching the MMDet3D checkpoint.

    In MMDet3D, layer_nums=[3,5,5] means 1 strided conv + layer_nums regular convs
    per block, giving actual conv counts of [4, 6, 6].
    """

    def __init__(self):
        super().__init__()
        # Encoder: actual conv counts = [4, 6, 6]
        layer_nums = (4, 6, 6)
        layer_strides = (2, 2, 2)
        num_filters = (64, 128, 256)
        in_filters = [64, 64, 128]

        blocks = []
        for i in range(3):
            block_layers = []
            for j in range(layer_nums[i]):
                in_c = in_filters[i] if j == 0 else num_filters[i]
                stride = layer_strides[i] if j == 0 else 1
                block_layers.extend([
                    nn.Conv2d(in_c, num_filters[i], 3, stride=stride, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[i], eps=1e-3, momentum=0.01),
                    nn.ReLU(),
                ])
            blocks.append(nn.Sequential(*block_layers))
        self.blocks = nn.ModuleList(blocks)

        # Decoder / Neck: ALL ConvTranspose2d (MMDet3D SECONDFPN uses deconv for all)
        self.deblocks = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(64, 128, 1, stride=1, bias=False),
                nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(128, 128, 2, stride=2, bias=False),
                nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(256, 128, 4, stride=4, bias=False),
                nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ),
        ])

    def forward(self, x):
        ups = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            ups.append(self.deblocks[i](x))
        return torch.cat(ups, dim=1)  # (B, 384, H/2, W/2)


# ============================================================================
# MMDet3D Detection Head
# ============================================================================

class MMDet3DHead(nn.Module):
    """Replicates ``Anchor3DHead`` from MMDet3D.

    Outputs (B, C, H, W) tensors for cls/reg/dir.
    """

    def __init__(self, in_channels=384, num_anchors_per_loc=14, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors_per_loc = num_anchors_per_loc
        self.conv_cls = nn.Conv2d(in_channels, num_anchors_per_loc * num_classes, 1)
        self.conv_reg = nn.Conv2d(in_channels, num_anchors_per_loc * 9, 1)
        self.conv_dir_cls = nn.Conv2d(in_channels, num_anchors_per_loc * 2, 1)

    def forward(self, x):
        return {
            'cls_preds': self.conv_cls(x),
            'box_preds': self.conv_reg(x),
            'dir_preds': self.conv_dir_cls(x),
        }


# ============================================================================
# Anchor configuration (MMDet3D NuScenes PointPillars)
# ============================================================================

# Exact anchor config from MMDet3D hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d.py
# 7 anchor types * 2 rotations = 14 anchors per location.
# Sizes are [l, w, h] in config. Z-center from ranges[:,2].
MMDET3D_ANCHOR_CONFIGS = [
    # sizes=[l, w, h] from config, z from ranges, rotations=[0, pi/2]
    {'size_lwh': [4.60718145, 1.95017717, 1.72270761], 'z': -1.80032795, 'rotations': (0, pi/2)},  # car
    {'size_lwh': [6.73778078, 2.4560939,  2.73004906], 'z': -1.74440365, 'rotations': (0, pi/2)},  # truck
    {'size_lwh': [12.01320693, 2.87427237, 3.81509561], 'z': -1.68526504, 'rotations': (0, pi/2)},  # trailer
    {'size_lwh': [1.68452161, 0.60058911, 1.27192197], 'z': -1.67339111, 'rotations': (0, pi/2)},  # bicycle
    {'size_lwh': [0.7256437,  0.66344886, 1.75748069], 'z': -1.61785072, 'rotations': (0, pi/2)},  # pedestrian
    {'size_lwh': [0.40359262, 0.39694519, 1.06232151], 'z': -1.80984986, 'rotations': (0, pi/2)},  # traffic_cone
    {'size_lwh': [0.48578221, 2.49008838, 0.98297065], 'z': -1.763965,   'rotations': (0, pi/2)},  # barrier
]

# Anchor x,y range (slightly smaller than point_cloud_range)
MMDET3D_ANCHOR_RANGE = [-49.6, -49.6, 49.6, 49.6]

MMDET3D_CLASS_NAMES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]


def generate_mmdet3d_anchors(
    feature_map_size: Tuple[int, int],
    anchor_configs: list,
    anchor_range: list,
    device: torch.device = torch.device('cpu'),
) -> torch.Tensor:
    """Generate anchors matching MMDet3D's AlignedAnchor3DRangeGenerator.

    Anchors are interleaved per spatial location to match the Conv2d output
    ordering: at each (h, w) position, all R_total anchor types are contiguous.

    Args:
        feature_map_size: (H, W) of the backbone output
        anchor_configs: list of dicts with 'size_lwh', 'z', 'rotations'
        anchor_range: [x_min, y_min, x_max, y_max] for anchor centers

    Returns:
        anchors: (H*W*R_total, 7) with (x, y, z, l, w, h, yaw)
    """
    H, W = feature_map_size
    x_min, y_min, x_max, y_max = anchor_range

    x_stride = (x_max - x_min) / W
    y_stride = (y_max - y_min) / H

    x_centres = torch.arange(W, dtype=torch.float32, device=device) * x_stride + x_min + x_stride / 2
    y_centres = torch.arange(H, dtype=torch.float32, device=device) * y_stride + y_min + y_stride / 2
    yy, xx = torch.meshgrid(y_centres, x_centres, indexing='ij')

    per_loc_anchors = []  # each (H, W, R_i, 7)

    for cfg in anchor_configs:
        R = len(cfg['rotations'])

        xx_r = xx.unsqueeze(-1).expand(-1, -1, R)
        yy_r = yy.unsqueeze(-1).expand(-1, -1, R)
        zz = torch.full_like(xx_r, cfg['z'])

        # Config sizes are [l, w, h]
        ll = torch.full_like(xx_r, cfg['size_lwh'][0])
        ww = torch.full_like(xx_r, cfg['size_lwh'][1])
        hh = torch.full_like(xx_r, cfg['size_lwh'][2])

        rots = torch.tensor(cfg['rotations'], dtype=torch.float32, device=device)
        rr = rots.view(1, 1, R).expand(H, W, R)

        anchors = torch.stack([xx_r, yy_r, zz, ll, ww, hh, rr], dim=-1)  # (H, W, R, 7)
        per_loc_anchors.append(anchors)

    # Concatenate along anchor dim so all types are interleaved per location
    all_anchors = torch.cat(per_loc_anchors, dim=2)  # (H, W, R_total, 7)
    return all_anchors.reshape(-1, 7)


# ============================================================================
# Full MMDet3D PointPillars Model
# ============================================================================

class MMDet3DPointPillars(nn.Module):
    """Complete wrapper for MMDet3D ``hv_pointpillars_secfpn_sbn-all`` checkpoint."""

    VOXEL_SIZE = np.array([0.25, 0.25, 8.0])
    POINT_CLOUD_RANGE = np.array([-50, -50, -5, 50, 50, 3])
    MAX_POINTS_PER_VOXEL = 64
    MAX_VOXELS = 40000

    def __init__(self, num_classes=10, num_anchors_per_loc=14):
        super().__init__()
        self.num_classes = num_classes

        # Grid size: (100/0.25, 100/0.25, 8/8) = (400, 400, 1)
        grid = ((self.POINT_CLOUD_RANGE[3:] - self.POINT_CLOUD_RANGE[:3]) / self.VOXEL_SIZE)
        self.grid_size = np.round(grid).astype(np.int64)

        # VFE — 2-layer PFN matching checkpoint
        self.vfe = MMDet3DPillarFeatureNet()

        # Scatter
        self.scatter = PointPillarScatter(
            num_input_features=64,
            nx=int(self.grid_size[0]),
            ny=int(self.grid_size[1]),
            nz=1,
        )

        # Backbone + Neck (matching checkpoint exactly)
        self.backbone = MMDet3DBackbone()

        # Head
        self.num_anchors_per_loc = num_anchors_per_loc
        self.head = MMDet3DHead(
            in_channels=384,
            num_anchors_per_loc=num_anchors_per_loc,
            num_classes=num_classes,
        )

        # Anchor cache
        self._anchors: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Weight loading from MMDet3D checkpoint
    # ------------------------------------------------------------------

    def load_mmdet3d_checkpoint(self, path: str):
        """Load weights from an MMDet3D checkpoint .pth file."""
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        if 'state_dict' in ckpt:
            sd = ckpt['state_dict']
        else:
            sd = ckpt

        # Inspect head shape to determine num_anchors_per_loc
        cls_w = sd.get('pts_bbox_head.conv_cls.weight')
        if cls_w is not None:
            n_anchors = cls_w.shape[0] // self.num_classes
            if n_anchors != self.num_anchors_per_loc:
                print(f"[MMDet3D] Adjusting num_anchors_per_loc: {self.num_anchors_per_loc} -> {n_anchors}")
                self.num_anchors_per_loc = n_anchors
                self.head = MMDet3DHead(384, n_anchors, self.num_classes)

        # Build key mapping: mmdet3d key -> our key
        key_map = {}

        # VFE — pts_voxel_encoder.vfe_layers.{0,1}.{linear,norm}
        for layer_idx in range(2):
            prefix_mm = f'pts_voxel_encoder.vfe_layers.{layer_idx}'
            prefix_ours = f'vfe.vfe_layers.{layer_idx}'
            key_map[f'{prefix_mm}.linear.weight'] = f'{prefix_ours}.linear.weight'
            for suffix in ['weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked']:
                key_map[f'{prefix_mm}.norm.{suffix}'] = f'{prefix_ours}.norm.{suffix}'

        # Backbone: pts_backbone.blocks.{i}.{j}.{param}
        for mm_key in list(sd.keys()):
            if mm_key.startswith('pts_backbone.'):
                our_key = mm_key.replace('pts_backbone.', 'backbone.')
                key_map[mm_key] = our_key

        # Neck: pts_neck.deblocks.{i}.{j}.{param}
        for mm_key in list(sd.keys()):
            if mm_key.startswith('pts_neck.'):
                our_key = mm_key.replace('pts_neck.', 'backbone.')
                key_map[mm_key] = our_key

        # Head: pts_bbox_head.conv_{cls,reg,dir_cls}
        key_map['pts_bbox_head.conv_cls.weight'] = 'head.conv_cls.weight'
        key_map['pts_bbox_head.conv_cls.bias'] = 'head.conv_cls.bias'
        key_map['pts_bbox_head.conv_reg.weight'] = 'head.conv_reg.weight'
        key_map['pts_bbox_head.conv_reg.bias'] = 'head.conv_reg.bias'
        key_map['pts_bbox_head.conv_dir_cls.weight'] = 'head.conv_dir_cls.weight'
        key_map['pts_bbox_head.conv_dir_cls.bias'] = 'head.conv_dir_cls.bias'

        # Apply mapping
        new_sd = {}
        loaded, skipped = 0, 0
        for mm_key, param in sd.items():
            our_key = key_map.get(mm_key)
            if our_key is not None:
                new_sd[our_key] = param
                loaded += 1
            else:
                skipped += 1

        missing, unexpected = self.load_state_dict(new_sd, strict=False)
        print(f"[MMDet3D] Loaded {loaded} params, skipped {skipped} mmdet3d keys")
        if missing:
            print(f"[MMDet3D] Missing keys ({len(missing)}): {missing}")
        if unexpected:
            print(f"[MMDet3D] Unexpected keys ({len(unexpected)}): {unexpected}")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, batch_dict: Dict) -> Dict:
        """Forward pass with MMDet3D-style VFE."""
        device = batch_dict['voxels'].device
        voxel_size = torch.tensor(self.VOXEL_SIZE, dtype=torch.float32, device=device)
        pc_range = torch.tensor(self.POINT_CLOUD_RANGE, dtype=torch.float32, device=device)

        # VFE
        pillar_features = self.vfe(
            batch_dict['voxels'],
            batch_dict['voxel_num_points'],
            batch_dict['voxel_coords'],
            voxel_size,
            pc_range,
        )

        # Scatter
        spatial_features = self.scatter(
            pillar_features,
            batch_dict['voxel_coords'],
            batch_dict['batch_size'],
        )

        # Backbone + Neck
        spatial_features = self.backbone(spatial_features)

        # Head
        preds = self.head(spatial_features)

        batch_dict.update(preds)
        return batch_dict

    # ------------------------------------------------------------------
    # Anchor helpers
    # ------------------------------------------------------------------

    def _get_anchors(self, device: torch.device) -> torch.Tensor:
        if self._anchors is not None and self._anchors.device == device:
            return self._anchors
        fm_h = int(self.grid_size[1]) // 2
        fm_w = int(self.grid_size[0]) // 2
        self._anchors = generate_mmdet3d_anchors(
            feature_map_size=(fm_h, fm_w),
            anchor_configs=MMDET3D_ANCHOR_CONFIGS,
            anchor_range=MMDET3D_ANCHOR_RANGE,
            device=device,
        )
        return self._anchors

    # ------------------------------------------------------------------
    # Decode + postprocess
    # ------------------------------------------------------------------

    @staticmethod
    def decode_boxes(deltas: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        """Decode 9-param deltas (7 box + 2 velocity) to 7-param boxes.

        MMDet3D NuScenes convention: anchor z is the bottom-center of the box.
        We convert to geometric center (z + h/2) for compatibility with our
        visualization and evaluation code.
        """
        diag = torch.sqrt(anchors[:, 3] ** 2 + anchors[:, 4] ** 2)

        x = deltas[:, 0] * diag + anchors[:, 0]
        y = deltas[:, 1] * diag + anchors[:, 1]
        z_bottom = deltas[:, 2] * anchors[:, 5] + anchors[:, 2]
        l = torch.exp(deltas[:, 3]) * anchors[:, 3]
        w = torch.exp(deltas[:, 4]) * anchors[:, 4]
        h = torch.exp(deltas[:, 5]) * anchors[:, 5]
        yaw = deltas[:, 6] + anchors[:, 6]
        # dims 7,8 are velocity — ignored

        # Convert from bottom-center to geometric center
        z = z_bottom + h / 2

        return torch.stack([x, y, z, l, w, h, yaw], dim=-1)

    @torch.no_grad()
    def postprocess(
        self,
        batch_dict: Dict,
        score_thresh: float = 0.1,
        nms_iou_thresh: float = 0.3,
        max_detections: int = 500,
    ) -> List[Dict]:
        """Decode predictions and apply per-class NMS."""
        cls_raw = batch_dict['cls_preds']   # (B, R*C, H, W)
        box_raw = batch_dict['box_preds']   # (B, R*9, H, W)
        dir_raw = batch_dict['dir_preds']   # (B, R*2, H, W)

        B = batch_dict['batch_size']
        C = self.num_classes
        R = self.num_anchors_per_loc

        _, _, fH, fW = cls_raw.shape
        device = cls_raw.device

        anchors = self._get_anchors(device)

        results = []
        for b in range(B):
            # Reshape: (R*X, H, W) -> (H, W, R, X) -> (H*W*R, X)
            cls_b = cls_raw[b].view(R, C, fH, fW).permute(2, 3, 0, 1).reshape(-1, C)
            scores_b = torch.sigmoid(cls_b)

            box_b = box_raw[b].view(R, 9, fH, fW).permute(2, 3, 0, 1).reshape(-1, 9)
            dir_b = dir_raw[b].view(R, 2, fH, fW).permute(2, 3, 0, 1).reshape(-1, 2)

            # Decode boxes
            decoded = self.decode_boxes(box_b, anchors)

            # Direction correction
            dir_label = dir_b.argmax(dim=-1)
            decoded[:, 6] = decoded[:, 6] + dir_label.float() * pi

            decoded_np = decoded.cpu().numpy()
            scores_np = scores_b.cpu().numpy()

            all_boxes, all_scores, all_labels = [], [], []

            for cls_idx in range(C):
                cls_sc = scores_np[:, cls_idx]
                mask = cls_sc > score_thresh
                if not mask.any():
                    continue

                cls_boxes = decoded_np[mask]
                cls_scores_f = cls_sc[mask]

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
        self._anchors = None
        return super().to(device)
