"""
MMDetection3D CenterPoint Pillar Wrapper

Replicates the exact architecture of MMDet3D's
``centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus``
so that its pre-trained checkpoint can be loaded without installing mmdetection3d.

Architecture verified against checkpoint key shapes:
- VFE: PillarFeatureNet (NOT HardVFE), in_channels=5, with_distance=False
       11 input features: [x,y,z,intensity,ring_index, cluster(3), voxel(3)]
       pfn_layers.0: Linear(11,64) → BN → ReLU → max-pool → (M,64)
- Backbone: layer_nums=(3,5,5) + 1 strided conv each = (4,6,6) convs
- Neck: upsample_strides=[0.5, 1, 2]:
       Block 0: Conv2d(64, 128, 2, stride=2) — downsample
       Block 1: ConvTranspose2d(128, 128, 1, stride=1) — identity
       Block 2: ConvTranspose2d(256, 128, 2, stride=2) — upsample
- Head: shared_conv(384→64) + 6 task heads, each with branch convs from 64-ch
- Grid: 512x512 (voxel_size=0.2, range=[-51.2..51.2])
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List
from math import pi

from src.detectors.pointpillars import PointPillarScatter
from src.core.geometry import nms_bev_fast


def circle_nms(boxes, scores, min_radius, post_max_size=83):
    """Distance-based NMS matching MMDet3D's circle_nms.

    Instead of IoU overlap, suppresses detections whose centers are within
    `min_radius` of a higher-scoring detection.

    Args:
        boxes: (N, 7+) array, columns 0,1 = x,y center
        scores: (N,) array
        min_radius: suppression radius in meters
        post_max_size: max detections to keep

    Returns:
        keep: indices of kept detections
    """
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)

    order = np.argsort(-scores)
    keep = []
    suppressed = np.zeros(len(boxes), dtype=bool)

    for i in order:
        if suppressed[i]:
            continue
        keep.append(i)
        if len(keep) >= post_max_size:
            break
        # Suppress all lower-scoring detections within radius
        dx = boxes[order, 0] - boxes[i, 0]
        dy = boxes[order, 1] - boxes[i, 1]
        dist = np.sqrt(dx ** 2 + dy ** 2)
        for j_idx in range(len(order)):
            j = order[j_idx]
            if not suppressed[j] and j != i:
                d = np.sqrt((boxes[j, 0] - boxes[i, 0]) ** 2 +
                            (boxes[j, 1] - boxes[i, 1]) ** 2)
                if d < min_radius:
                    suppressed[j] = True

    return np.array(keep, dtype=np.int64)


# ============================================================================
# CenterPoint VFE — single-layer PFN with 11 input features
# ============================================================================

class CenterPointPFNLayer(nn.Module):
    """Single PFN layer — last layer only (max-pool output)."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)

    def forward(self, features, num_points):
        # (M, P, C_in) → Linear → BN → ReLU → max-pool → (M, C_out)
        x = self.linear(features.reshape(-1, features.shape[-1]))
        x = x.view(features.shape[0], features.shape[1], -1)
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = torch.relu(x)
        return x.max(dim=1)[0]  # (M, C_out)


class CenterPointVFE(nn.Module):
    """VFE matching CenterPoint checkpoint (PillarFeatureNet, NOT HardVFE).

    Config: in_channels=5, feat_channels=[64], with_distance=False
    Input: 5 raw features [x, y, z, intensity, ring_index]
    Augmented: 5 raw + 3 cluster + 3 voxel_center = 11 features
    Single PFN layer: Linear(11, 64) → BN → ReLU → max-pool
    """

    def __init__(self):
        super().__init__()
        self.pfn_layers = nn.ModuleList([
            CenterPointPFNLayer(11, 64),
        ])

    def forward(self, voxels, num_points, voxel_coords, voxel_size, point_cloud_range):
        M, max_pts, C = voxels.shape
        # voxels: (M, max_pts, 5) — [x, y, z, intensity, time_lag]

        voxels = voxels.clone()
        # Note: NO intensity normalization for CenterPoint — its PillarFeatureNet
        # checkpoint was trained with raw intensity values (unlike PointPillars HardVFE)

        # Cluster-center offsets: sum / actual_num_points (NOT .mean() which divides by max_pts)
        points_mean = voxels[:, :, :3].sum(dim=1, keepdim=True) / \
            num_points.float().unsqueeze(1).unsqueeze(2).clamp(min=1.0)
        cluster_offset = voxels[:, :, :3] - points_mean

        # Voxel-center offsets
        voxel_center_x = point_cloud_range[0] + (voxel_coords[:, 3].float() + 0.5) * voxel_size[0]
        voxel_center_y = point_cloud_range[1] + (voxel_coords[:, 2].float() + 0.5) * voxel_size[1]
        voxel_center_z = point_cloud_range[2] + (voxel_coords[:, 1].float() + 0.5) * voxel_size[2]
        voxel_center = torch.stack([voxel_center_x, voxel_center_y, voxel_center_z], dim=-1)
        voxel_offset = voxels[:, :, :3] - voxel_center.unsqueeze(1)

        # 11 features: [x,y,z,intensity,ring, cluster(3), voxel(3)]
        # NO distance feature (with_distance=False in config)
        features = torch.cat([voxels, cluster_offset, voxel_offset], dim=-1)

        # Padding mask (applied before PFN layers, same as MMDet3D PillarFeatureNet)
        padding_mask = torch.arange(
            max_pts, device=features.device
        ).unsqueeze(0) < num_points.unsqueeze(1)
        features = features * padding_mask.unsqueeze(-1).float()

        # Single PFN layer
        return self.pfn_layers[0](features, num_points)


# ============================================================================
# CenterPoint Backbone — returns multi-scale features
# ============================================================================

class CenterPointBackbone(nn.Module):
    """SECOND backbone returning intermediate features for the neck."""

    def __init__(self):
        super().__init__()
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

    def forward(self, x):
        outs = []
        for block in self.blocks:
            x = block(x)
            outs.append(x)
        return outs


# ============================================================================
# CenterPoint Neck
# ============================================================================

class CenterPointNeck(nn.Module):
    """SECONDFPN with upsample_strides=[0.5, 1, 2].

    Verified against checkpoint:
    Block 0: Conv2d(64, 128, 2, stride=2)      — [128, 64, 2, 2]
    Block 1: ConvTranspose2d(128, 128, 1, s=1)  — [128, 128, 1, 1]
    Block 2: ConvTranspose2d(256, 128, 2, s=2)  — [256, 128, 2, 2]
    Output: concat → 384 channels
    """

    def __init__(self):
        super().__init__()
        self.deblocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(64, 128, 2, stride=2, bias=False),
                nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(128, 128, 1, stride=1, bias=False),
                nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(256, 128, 2, stride=2, bias=False),
                nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ),
        ])

    def forward(self, outs):
        ups = []
        for feat, deblock in zip(outs, self.deblocks):
            ups.append(deblock(feat))
        return torch.cat(ups, dim=1)


# ============================================================================
# CenterPoint Multi-task Head
# ============================================================================

TASK_CLASSES = [
    ['car'],
    ['truck', 'construction_vehicle'],
    ['bus', 'trailer'],
    ['barrier'],
    ['motorcycle', 'bicycle'],
    ['pedestrian', 'traffic_cone'],
]

CLASS_NAMES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]


class ConvBNReLU(nn.Module):
    """Conv2d + BN + ReLU block matching MMDet3D's ConvModule (bias=False with BN)."""
    def __init__(self, in_c, out_c, kernel=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c, eps=1e-3, momentum=0.01)

    def forward(self, x):
        return torch.relu(self.bn(self.conv(x)))


class SeparateHead(nn.Module):
    """Per-task head: each branch is ConvBNReLU(64,64) + Conv2d(64,out).

    Matches checkpoint structure:
        task_heads.{i}.{branch}.0.conv/bn  — intermediate conv
        task_heads.{i}.{branch}.1          — output conv
    """

    def __init__(self, in_channels, heads, head_conv=64):
        super().__init__()
        self.heads = nn.ModuleDict()
        for name, out_c in heads.items():
            self.heads[name] = nn.Sequential(
                ConvBNReLU(in_channels, head_conv, 3, 1),
                nn.Conv2d(head_conv, out_c, 3, padding=1, bias=True),
            )

    def forward(self, x):
        return {name: head(x) for name, head in self.heads.items()}


class CenterPointHead(nn.Module):
    """Multi-task CenterPoint head.

    Structure from checkpoint:
    - pts_bbox_head.shared_conv.conv/bn — global shared Conv2d(384, 64, 3)
    - pts_bbox_head.task_heads.{i}.{branch} — per-task branches
    """

    def __init__(self, in_channels=384, head_conv=64):
        super().__init__()
        # Global shared conv (applied before task heads)
        self.shared_conv = ConvBNReLU(in_channels, head_conv, 3, 1)

        self.task_heads = nn.ModuleList()
        for task_cls in TASK_CLASSES:
            n_cls = len(task_cls)
            heads = {
                'heatmap': n_cls,
                'reg': 2,
                'height': 1,
                'dim': 3,
                'rot': 2,
                'vel': 2,
            }
            self.task_heads.append(SeparateHead(head_conv, heads, head_conv))

    def forward(self, x):
        x = self.shared_conv(x)
        return [head(x) for head in self.task_heads]


# ============================================================================
# Full MMDet3D CenterPoint Model
# ============================================================================

class MMDet3DCenterPoint(nn.Module):
    """Complete wrapper for MMDet3D CenterPoint (pillar-based) checkpoint."""

    VOXEL_SIZE = np.array([0.2, 0.2, 8.0])
    POINT_CLOUD_RANGE = np.array([-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])
    MAX_POINTS_PER_VOXEL = 20
    MAX_VOXELS = 40000

    def __init__(self):
        super().__init__()
        grid = ((self.POINT_CLOUD_RANGE[3:] - self.POINT_CLOUD_RANGE[:3]) / self.VOXEL_SIZE)
        self.grid_size = np.round(grid).astype(np.int64)

        self.vfe = CenterPointVFE()

        self.scatter = PointPillarScatter(
            num_input_features=64,
            nx=int(self.grid_size[0]),
            ny=int(self.grid_size[1]),
            nz=1,
        )

        self.backbone = CenterPointBackbone()
        self.neck = CenterPointNeck()
        self.head = CenterPointHead(in_channels=384, head_conv=64)

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_mmdet3d_checkpoint(self, path: str):
        """Load weights from an MMDet3D CenterPoint checkpoint."""
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        sd = ckpt.get('state_dict', ckpt)

        key_map = {}

        # VFE: pts_voxel_encoder.pfn_layers.0.{linear,norm}
        pfx_mm = 'pts_voxel_encoder.pfn_layers.0'
        pfx_ours = 'vfe.pfn_layers.0'
        key_map[f'{pfx_mm}.linear.weight'] = f'{pfx_ours}.linear.weight'
        for suf in ['weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked']:
            key_map[f'{pfx_mm}.norm.{suf}'] = f'{pfx_ours}.norm.{suf}'

        # Backbone: pts_backbone → backbone
        for mm_key in sd:
            if mm_key.startswith('pts_backbone.'):
                key_map[mm_key] = mm_key.replace('pts_backbone.', 'backbone.')

        # Neck: pts_neck → neck
        for mm_key in sd:
            if mm_key.startswith('pts_neck.'):
                key_map[mm_key] = mm_key.replace('pts_neck.', 'neck.')

        # Head shared_conv: pts_bbox_head.shared_conv → head.shared_conv
        for mm_key in sd:
            if mm_key.startswith('pts_bbox_head.shared_conv.'):
                key_map[mm_key] = mm_key.replace('pts_bbox_head.shared_conv.', 'head.shared_conv.')

        # Head task_heads: pts_bbox_head.task_heads.{i}.{branch} → head.task_heads.{i}.heads.{branch}
        for mm_key in sd:
            if mm_key.startswith('pts_bbox_head.task_heads.'):
                # e.g. pts_bbox_head.task_heads.0.reg.0.conv.weight
                # →    head.task_heads.0.heads.reg.0.conv.weight
                rest = mm_key[len('pts_bbox_head.task_heads.'):]
                parts = rest.split('.', 1)  # ['0', 'reg.0.conv.weight']
                task_idx = parts[0]
                branch_rest = parts[1]
                our_key = f'head.task_heads.{task_idx}.heads.{branch_rest}'
                key_map[mm_key] = our_key

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
        print(f"[MMDet3D-CP] Loaded {loaded} params, skipped {skipped} mmdet3d keys")
        if missing:
            print(f"[MMDet3D-CP] Missing keys ({len(missing)}): {missing[:5]}...")
        if unexpected:
            print(f"[MMDet3D-CP] Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, batch_dict: Dict) -> Dict:
        device = batch_dict['voxels'].device
        voxel_size = torch.tensor(self.VOXEL_SIZE, dtype=torch.float32, device=device)
        pc_range = torch.tensor(self.POINT_CLOUD_RANGE, dtype=torch.float32, device=device)

        pillar_features = self.vfe(
            batch_dict['voxels'],
            batch_dict['voxel_num_points'],
            batch_dict['voxel_coords'],
            voxel_size,
            pc_range,
        )

        spatial_features = self.scatter(
            pillar_features,
            batch_dict['voxel_coords'],
            batch_dict['batch_size'],
        )

        backbone_outs = self.backbone(spatial_features)
        neck_out = self.neck(backbone_outs)
        task_preds = self.head(neck_out)

        batch_dict['task_preds'] = task_preds
        return batch_dict

    # ------------------------------------------------------------------
    # Decode + postprocess
    # ------------------------------------------------------------------

    # Circle NMS radii per task head (from MMDet3D test_cfg.min_radius)
    CIRCLE_NMS_RADII = [4.0, 12.0, 10.0, 1.0, 0.85, 0.175]
    # Max detections per task head (from MMDet3D test_cfg.post_max_size)
    POST_MAX_SIZE = 83

    @torch.no_grad()
    def postprocess(
        self,
        batch_dict: Dict,
        score_thresh: float = 0.1,
        nms_iou_thresh: float = 0.2,
        max_detections: int = 500,
    ) -> List[Dict]:
        """Decode heatmap-based predictions and apply circle NMS.

        Matches MMDet3D's CenterPoint test pipeline:
        - Circle NMS (distance-based) per task head with task-specific radii
        - post_max_size=83 per task head (~500 total max)
        - Z-range filtering to remove sub-ground detections
        """
        task_preds = batch_dict['task_preds']
        B = batch_dict['batch_size']

        _, _, fH, fW = task_preds[0]['heatmap'].shape

        # Feature map stride relative to point cloud range
        stride_x = (self.POINT_CLOUD_RANGE[3] - self.POINT_CLOUD_RANGE[0]) / fW
        stride_y = (self.POINT_CLOUD_RANGE[4] - self.POINT_CLOUD_RANGE[1]) / fH

        results = []
        for b in range(B):
            all_boxes, all_scores, all_labels = [], [], []

            for task_id, preds in enumerate(task_preds):
                heatmap = torch.sigmoid(preds['heatmap'][b])  # (n_cls, H, W)
                reg = preds['reg'][b]
                height = preds['height'][b]
                dim = preds['dim'][b]
                rot = preds['rot'][b]

                n_cls = heatmap.shape[0]
                task_boxes, task_scores, task_labels = [], [], []

                for cls_local in range(n_cls):
                    cls_name = TASK_CLASSES[task_id][cls_local]
                    cls_global = CLASS_NAMES.index(cls_name)

                    scores_map = heatmap[cls_local]
                    mask = scores_map > score_thresh
                    if not mask.any():
                        continue

                    scores_flat = scores_map[mask]
                    ys, xs = torch.where(mask)

                    reg_x = reg[0][mask]
                    reg_y = reg[1][mask]
                    x = (xs.float() + reg_x) * stride_x + self.POINT_CLOUD_RANGE[0]
                    y = (ys.float() + reg_y) * stride_y + self.POINT_CLOUD_RANGE[1]
                    z = height[0][mask]

                    l = torch.exp(dim[0][mask])
                    w = torch.exp(dim[1][mask])
                    h = torch.exp(dim[2][mask])

                    sin_r = rot[0][mask]
                    cos_r = rot[1][mask]
                    yaw = torch.atan2(sin_r, cos_r)

                    boxes = torch.stack([x, y, z, l, w, h, yaw], dim=-1)
                    task_boxes.append(boxes.cpu().numpy())
                    task_scores.append(scores_flat.cpu().numpy())
                    task_labels.append(np.full(len(scores_flat), cls_global, dtype=np.int64))

                # Apply circle NMS per task head (all classes in the task together)
                if task_boxes:
                    task_boxes = np.concatenate(task_boxes)
                    task_scores = np.concatenate(task_scores)
                    task_labels = np.concatenate(task_labels)

                    nms_radius = self.CIRCLE_NMS_RADII[task_id]
                    keep = circle_nms(
                        task_boxes, task_scores,
                        min_radius=nms_radius,
                        post_max_size=self.POST_MAX_SIZE,
                    )

                    if len(keep) > 0:
                        all_boxes.append(task_boxes[keep])
                        all_scores.append(task_scores[keep])
                        all_labels.append(task_labels[keep])

            if len(all_boxes) > 0:
                all_boxes = np.concatenate(all_boxes)
                all_scores = np.concatenate(all_scores)
                all_labels = np.concatenate(all_labels)

                # Z-range filter: remove detections with unreasonable z values
                z_valid = (all_boxes[:, 2] > -4.0) & (all_boxes[:, 2] < 4.0)
                all_boxes = all_boxes[z_valid]
                all_scores = all_scores[z_valid]
                all_labels = all_labels[z_valid]

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
        return super().to(device)
