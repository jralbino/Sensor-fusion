"""
Supervised Loss for PointPillars Training

Includes anchor generation, target assignment via BEV IoU,
and a composite loss (Focal + SmoothL1 + Direction BCE).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
from math import pi


# ---------------------------------------------------------------------------
# Per-class anchor configuration (NuScenes mean sizes)
# ---------------------------------------------------------------------------

ANCHOR_CONFIGS = {
    'car':                  {'size': (3.9, 1.6, 1.56), 'z': -1.0, 'rotations': (0, pi/2)},
    'truck':                {'size': (6.93, 2.51, 2.84), 'z': -0.4, 'rotations': (0, pi/2)},
    'construction_vehicle': {'size': (6.37, 2.85, 3.19), 'z': -0.2, 'rotations': (0, pi/2)},
    'bus':                  {'size': (10.5, 2.94, 3.47), 'z': -0.1, 'rotations': (0, pi/2)},
    'trailer':              {'size': (12.29, 2.90, 3.87), 'z':  0.1, 'rotations': (0, pi/2)},
    'barrier':              {'size': (0.50, 2.53, 0.98), 'z': -1.3, 'rotations': (0, pi/2)},
    'motorcycle':           {'size': (2.11, 0.77, 1.47), 'z': -1.1, 'rotations': (0, pi/2)},
    'bicycle':              {'size': (1.70, 0.60, 1.28), 'z': -1.2, 'rotations': (0, pi/2)},
    'pedestrian':           {'size': (0.73, 0.67, 1.77), 'z': -0.9, 'rotations': (0,)},
    'traffic_cone':         {'size': (0.41, 0.41, 1.07), 'z': -1.5, 'rotations': (0,)},
}

CLASS_NAMES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

# Total anchors per location: 2+2+2+2+2+2+2+2+1+1 = 18
NUM_ANCHORS_PER_LOCATION = sum(len(ANCHOR_CONFIGS[c]['rotations']) for c in CLASS_NAMES)


# ---------------------------------------------------------------------------
# Multi-Class Anchor Generator
# ---------------------------------------------------------------------------

class MultiClassAnchorGenerator:
    """Generate per-class anchors on the BEV feature-map grid.

    Each spatial location gets anchors for all 10 classes with class-specific
    sizes, heights, and rotations. Total 18 anchors per location.
    """

    def __init__(
        self,
        feature_map_size: Tuple[int, int],          # (H, W) of the head output
        point_cloud_range: np.ndarray,               # [x_min, y_min, z_min, x_max, y_max, z_max]
    ):
        self.feature_map_size = feature_map_size
        self.point_cloud_range = point_cloud_range

    def generate(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return anchors and their class IDs.

        Returns:
            anchors: (H*W*R_total, 7) — (x, y, z, l, w, h, yaw)
            anchor_class_ids: (H*W*R_total,) — class index (0-9) for each anchor
        """
        H, W = self.feature_map_size
        pc = self.point_cloud_range

        # Centre coordinates for each cell
        x_stride = (pc[3] - pc[0]) / W
        y_stride = (pc[4] - pc[1]) / H

        x_centres = torch.arange(W, dtype=torch.float32, device=device) * x_stride + pc[0] + x_stride / 2
        y_centres = torch.arange(H, dtype=torch.float32, device=device) * y_stride + pc[1] + y_stride / 2

        # Meshgrid (H, W)
        yy, xx = torch.meshgrid(y_centres, x_centres, indexing='ij')

        all_anchors = []
        all_class_ids = []

        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            cfg = ANCHOR_CONFIGS[cls_name]
            R = len(cfg['rotations'])

            # Expand for rotations → (H, W, R)
            xx_r = xx.unsqueeze(-1).expand(-1, -1, R)
            yy_r = yy.unsqueeze(-1).expand(-1, -1, R)
            zz = torch.full_like(xx_r, cfg['z'])

            ll = torch.full_like(xx_r, cfg['size'][0])
            ww = torch.full_like(xx_r, cfg['size'][1])
            hh = torch.full_like(xx_r, cfg['size'][2])

            rots = torch.tensor(cfg['rotations'], dtype=torch.float32, device=device)
            rr = rots.view(1, 1, R).expand(H, W, R)

            anchors = torch.stack([xx_r, yy_r, zz, ll, ww, hh, rr], dim=-1)  # (H, W, R, 7)
            all_anchors.append(anchors.reshape(-1, 7))

            class_ids = torch.full((H * W * R,), cls_idx, dtype=torch.long, device=device)
            all_class_ids.append(class_ids)

        anchors = torch.cat(all_anchors, dim=0)       # (H*W*R_total, 7)  — but interleaving is wrong
        class_ids = torch.cat(all_class_ids, dim=0)

        # Reorder so anchors are grouped by location: for each (h,w), all 18 anchors together
        # Current order: all H*W*R1 for class0, then H*W*R2 for class1, ...
        # We need: for each spatial location, R1+R2+...+R10 anchors
        # Reshape each class anchors to (H, W, R_cls, 7), then interleave along R dim
        reordered_anchors = []
        reordered_ids = []
        offset = 0
        class_anchors_list = []
        class_ids_list = []
        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            R = len(ANCHOR_CONFIGS[cls_name]['rotations'])
            n = H * W * R
            cls_anc = anchors[offset:offset+n].reshape(H, W, R, 7)
            cls_id = class_ids[offset:offset+n].reshape(H, W, R)
            class_anchors_list.append(cls_anc)
            class_ids_list.append(cls_id)
            offset += n

        # Concatenate along rotation dim → (H, W, 18, 7)
        all_anc = torch.cat(class_anchors_list, dim=2)  # (H, W, 18, 7)
        all_ids = torch.cat(class_ids_list, dim=2)       # (H, W, 18)

        return all_anc.reshape(-1, 7), all_ids.reshape(-1)


# Keep old class for backward compatibility if needed
class AnchorGenerator:
    """Generate anchors on the BEV feature-map grid (single class, legacy)."""

    def __init__(
        self,
        feature_map_size: Tuple[int, int],
        point_cloud_range: np.ndarray,
        anchor_size: Tuple[float, float, float] = (4.7, 2.0, 1.7),
        anchor_z: float = -1.0,
        rotations: Tuple[float, ...] = (0.0, np.pi / 2),
    ):
        self.feature_map_size = feature_map_size
        self.point_cloud_range = point_cloud_range
        self.anchor_size = anchor_size
        self.anchor_z = anchor_z
        self.rotations = rotations

    def generate(self, device: torch.device) -> torch.Tensor:
        """Return anchors of shape ``(H*W*R, 7)`` – (x, y, z, l, w, h, yaw)."""
        H, W = self.feature_map_size
        pc = self.point_cloud_range

        x_stride = (pc[3] - pc[0]) / W
        y_stride = (pc[4] - pc[1]) / H

        x_centres = torch.arange(W, dtype=torch.float32, device=device) * x_stride + pc[0] + x_stride / 2
        y_centres = torch.arange(H, dtype=torch.float32, device=device) * y_stride + pc[1] + y_stride / 2

        yy, xx = torch.meshgrid(y_centres, x_centres, indexing='ij')

        R = len(self.rotations)
        xx = xx.unsqueeze(-1).expand(-1, -1, R)
        yy = yy.unsqueeze(-1).expand(-1, -1, R)
        zz = torch.full_like(xx, self.anchor_z)

        ll = torch.full_like(xx, self.anchor_size[0])
        ww = torch.full_like(xx, self.anchor_size[1])
        hh = torch.full_like(xx, self.anchor_size[2])

        rots = torch.tensor(self.rotations, dtype=torch.float32, device=device)
        rr = rots.view(1, 1, R).expand(H, W, R)

        anchors = torch.stack([xx, yy, zz, ll, ww, hh, rr], dim=-1)
        return anchors.reshape(-1, 7)


# ---------------------------------------------------------------------------
# BEV IoU helpers
# ---------------------------------------------------------------------------

def _corners_bev(boxes: torch.Tensor) -> torch.Tensor:
    """Convert (N, 7) boxes to BEV corners (N, 4, 2) using vectorised rotation."""
    x, y = boxes[:, 0], boxes[:, 1]
    l, w = boxes[:, 3], boxes[:, 4]
    yaw = boxes[:, 6]

    cos_y = torch.cos(yaw)
    sin_y = torch.sin(yaw)

    hl, hw = l / 2, w / 2

    dx = torch.stack([hl, hl, -hl, -hl], dim=-1)
    dy = torch.stack([hw, -hw, -hw, hw], dim=-1)

    cx = cos_y.unsqueeze(-1) * dx - sin_y.unsqueeze(-1) * dy + x.unsqueeze(-1)
    cy = sin_y.unsqueeze(-1) * dx + cos_y.unsqueeze(-1) * dy + y.unsqueeze(-1)

    return torch.stack([cx, cy], dim=-1)


def _axis_aligned_bbox(corners: torch.Tensor) -> torch.Tensor:
    """Compute axis-aligned bounding boxes from corners → (N, 4) as (x1, y1, x2, y2)."""
    mins = corners.min(dim=1)[0]
    maxs = corners.max(dim=1)[0]
    return torch.cat([mins, maxs], dim=-1)


def bev_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """Approximate BEV IoU using axis-aligned bounding boxes (fast).

    Args:
        boxes_a: (N, 7)
        boxes_b: (M, 7)

    Returns:
        iou: (N, M)
    """
    aa = _axis_aligned_bbox(_corners_bev(boxes_a))
    bb = _axis_aligned_bbox(_corners_bev(boxes_b))

    x1 = torch.max(aa[:, 0].unsqueeze(1), bb[:, 0].unsqueeze(0))
    y1 = torch.max(aa[:, 1].unsqueeze(1), bb[:, 1].unsqueeze(0))
    x2 = torch.min(aa[:, 2].unsqueeze(1), bb[:, 2].unsqueeze(0))
    y2 = torch.min(aa[:, 3].unsqueeze(1), bb[:, 3].unsqueeze(0))

    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    area_a = (aa[:, 2] - aa[:, 0]) * (aa[:, 3] - aa[:, 1])
    area_b = (bb[:, 2] - bb[:, 0]) * (bb[:, 3] - bb[:, 1])

    union = area_a.unsqueeze(1) + area_b.unsqueeze(0) - inter
    return inter / union.clamp(min=1e-6)


# ---------------------------------------------------------------------------
# Target Assigner (with per-class matching)
# ---------------------------------------------------------------------------

class TargetAssigner:
    """Assign GT boxes to anchors via BEV IoU thresholds with per-class matching."""

    def __init__(self, pos_iou: float = 0.5, neg_iou: float = 0.35):
        self.pos_iou = pos_iou
        self.neg_iou = neg_iou

    @torch.no_grad()
    def assign(
        self,
        anchors: torch.Tensor,        # (A, 7)
        gt_boxes: torch.Tensor,        # (G, 7)
        gt_labels: torch.Tensor,       # (G,)
        anchor_class_ids: torch.Tensor = None,  # (A,) — class index per anchor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return per-anchor targets with per-class matching.

        When anchor_class_ids is provided, anchors only match GT boxes of the
        same class, preventing cross-class confusion.

        Returns:
            cls_targets:  (A,) – 0 = neg, class_id (>=1) = pos, -1 = ignore
            box_targets:  (A, 7) – delta-encoded regression targets
            dir_targets:  (A,) – 0 or 1 based on GT yaw > 0
        """
        A = anchors.shape[0]
        device = anchors.device

        cls_targets = torch.zeros(A, dtype=torch.long, device=device)
        box_targets = torch.zeros(A, 7, dtype=torch.float32, device=device)
        dir_targets = torch.zeros(A, dtype=torch.long, device=device)

        if gt_boxes.shape[0] == 0:
            return cls_targets, box_targets, dir_targets

        iou = bev_iou(anchors, gt_boxes)  # (A, G)

        # Per-class matching: mask out IoU for anchors that don't match GT class
        if anchor_class_ids is not None:
            # Create mask: (A, G) where True means anchor class matches GT class
            # anchor_class_ids: (A,), gt_labels: (G,)
            class_match = anchor_class_ids.unsqueeze(1) == gt_labels.unsqueeze(0)  # (A, G)
            iou = iou * class_match.float()  # zero out cross-class IoU

        # For each anchor, best GT
        max_iou, best_gt = iou.max(dim=1)  # (A,)

        # For each GT, best anchor (ensure every GT has at least one positive)
        gt_max_iou, gt_best_anchor = iou.max(dim=0)  # (G,)

        # Negative
        cls_targets[max_iou < self.neg_iou] = 0

        # Ignore
        ignore_mask = (max_iou >= self.neg_iou) & (max_iou < self.pos_iou)
        cls_targets[ignore_mask] = -1

        # Positive
        pos_mask = max_iou >= self.pos_iou
        cls_targets[pos_mask] = gt_labels[best_gt[pos_mask]] + 1  # +1 so class 0 → label 1

        # Force-assign best anchor per GT as positive
        for g_idx in range(gt_boxes.shape[0]):
            a_idx = gt_best_anchor[g_idx]
            if iou[a_idx, g_idx] > 0:  # only if class matches (IoU not zeroed)
                cls_targets[a_idx] = gt_labels[g_idx] + 1
                best_gt[a_idx] = g_idx

        pos_mask = cls_targets > 0

        # Regression targets (delta encoding)
        if pos_mask.any():
            matched_gt = gt_boxes[best_gt[pos_mask]]
            matched_anc = anchors[pos_mask]

            diag = torch.sqrt(matched_anc[:, 3] ** 2 + matched_anc[:, 4] ** 2)

            box_targets[pos_mask, 0] = (matched_gt[:, 0] - matched_anc[:, 0]) / diag
            box_targets[pos_mask, 1] = (matched_gt[:, 1] - matched_anc[:, 1]) / diag
            box_targets[pos_mask, 2] = (matched_gt[:, 2] - matched_anc[:, 2]) / matched_anc[:, 5]
            box_targets[pos_mask, 3] = torch.log(matched_gt[:, 3] / matched_anc[:, 3].clamp(min=1e-6))
            box_targets[pos_mask, 4] = torch.log(matched_gt[:, 4] / matched_anc[:, 4].clamp(min=1e-6))
            box_targets[pos_mask, 5] = torch.log(matched_gt[:, 5] / matched_anc[:, 5].clamp(min=1e-6))
            box_targets[pos_mask, 6] = matched_gt[:, 6] - matched_anc[:, 6]

            # Direction target: 1 if GT yaw > 0
            dir_targets[pos_mask] = (matched_gt[:, 6] > 0).long()

        return cls_targets, box_targets, dir_targets


# ---------------------------------------------------------------------------
# Focal Loss helper
# ---------------------------------------------------------------------------

def focal_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Sigmoid focal loss over (A, C) predictions and (A,) integer targets.

    targets:
        0   → negative (all classes get target=0)
        -1  → ignore
        >=1 → positive (target class = value - 1)
    """
    A, C = preds.shape
    device = preds.device

    valid = targets >= 0
    preds = preds[valid]
    tgt = targets[valid]

    if preds.numel() == 0:
        return preds.sum()

    one_hot = torch.zeros(preds.shape[0], C, device=device)
    pos = tgt > 0
    if pos.any():
        one_hot[pos] = F.one_hot((tgt[pos] - 1).clamp(max=C - 1), C).float()

    p = torch.sigmoid(preds)
    ce = F.binary_cross_entropy_with_logits(preds, one_hot, reduction='none')

    p_t = p * one_hot + (1 - p) * (1 - one_hot)
    modulating = (1 - p_t) ** gamma
    alpha_t = alpha * one_hot + (1 - alpha) * (1 - one_hot)

    loss = (alpha_t * modulating * ce).sum() / pos.float().sum().clamp(min=1.0)
    return loss


# ---------------------------------------------------------------------------
# PointPillars Loss
# ---------------------------------------------------------------------------

class PointPillarsLoss(nn.Module):
    """Supervised loss for PointPillars with multi-class anchor-based target assignment."""

    def __init__(
        self,
        num_classes: int = 10,
        feature_map_size: Tuple[int, int] = (248, 216),
        point_cloud_range: np.ndarray = None,
        voxel_size: np.ndarray = None,
        cls_weight: float = 1.0,
        box_weight: float = 2.0,
        dir_weight: float = 0.2,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.cls_weight = cls_weight
        self.box_weight = box_weight
        self.dir_weight = dir_weight

        if point_cloud_range is None:
            point_cloud_range = np.array([0, -39.68, -3, 69.12, 39.68, 1])
        if voxel_size is None:
            voxel_size = np.array([0.16, 0.16, 4.0])

        self.anchor_gen = MultiClassAnchorGenerator(
            feature_map_size=feature_map_size,
            point_cloud_range=point_cloud_range,
        )
        self.target_assigner = TargetAssigner(pos_iou=0.5, neg_iou=0.35)

        # Cached anchors (lazily moved to correct device)
        self._anchors: torch.Tensor | None = None
        self._anchor_class_ids: torch.Tensor | None = None
        self._anchors_device: torch.device | None = None

    def _get_anchors(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._anchors is None or self._anchors_device != device:
            self._anchors, self._anchor_class_ids = self.anchor_gen.generate(device)
            self._anchors_device = device
        return self._anchors, self._anchor_class_ids

    def forward(
        self,
        batch_dict: Dict,
        pred_dict: Dict,
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute total loss over a batch.

        Args:
            batch_dict: must contain ``gt_boxes`` (list of (G_i, 7) tensors)
                        and ``gt_labels`` (list of (G_i,) tensors)
            pred_dict: output of the model's forward pass containing
                       ``cls_preds``  (B, H, W, R, C),
                       ``box_preds``  (B, H, W, R, 7),
                       ``dir_preds``  (B, H, W, R, 2).
        """
        cls_preds = pred_dict['cls_preds']
        box_preds = pred_dict['box_preds']
        dir_preds = pred_dict['dir_preds']

        B = cls_preds.shape[0]
        device = cls_preds.device

        anchors, anchor_class_ids = self._get_anchors(device)

        # Flatten spatial dims → (B, A, ...)
        cls_flat = cls_preds.reshape(B, -1, self.num_classes)
        box_flat = box_preds.reshape(B, -1, 7)
        dir_flat = dir_preds.reshape(B, -1, 2)

        total_cls = torch.tensor(0.0, device=device)
        total_box = torch.tensor(0.0, device=device)
        total_dir = torch.tensor(0.0, device=device)

        gt_boxes_list: List[torch.Tensor] = batch_dict['gt_boxes']
        gt_labels_list: List[torch.Tensor] = batch_dict['gt_labels']

        for b in range(B):
            gt_b = gt_boxes_list[b].to(device)
            gl_b = gt_labels_list[b].to(device)

            cls_t, box_t, dir_t = self.target_assigner.assign(
                anchors, gt_b, gl_b, anchor_class_ids
            )

            # --- Classification (Focal Loss) ---
            total_cls = total_cls + focal_loss(cls_flat[b], cls_t)

            # --- Box regression (Smooth L1, positives only) ---
            pos = cls_t > 0
            if pos.any():
                total_box = total_box + F.smooth_l1_loss(
                    box_flat[b][pos], box_t[pos], beta=1.0 / 9.0, reduction='mean'
                )

                # --- Direction (BCE, positives only) ---
                dir_one_hot = F.one_hot(dir_t[pos], 2).float()
                total_dir = total_dir + F.binary_cross_entropy_with_logits(
                    dir_flat[b][pos], dir_one_hot, reduction='mean'
                )

        # Average over batch
        total_cls = total_cls / B
        total_box = total_box / B
        total_dir = total_dir / B

        loss = self.cls_weight * total_cls + self.box_weight * total_box + self.dir_weight * total_dir

        loss_dict = {
            'cls_loss': total_cls.item(),
            'box_loss': total_box.item(),
            'dir_loss': total_dir.item(),
            'total_loss': loss.item(),
        }
        return loss, loss_dict


def get_loss_function(
    num_classes: int = 10,
    feature_map_size: Tuple[int, int] = (248, 216),
    point_cloud_range: np.ndarray = None,
    voxel_size: np.ndarray = None,
) -> nn.Module:
    """Factory function to create loss."""
    return PointPillarsLoss(
        num_classes=num_classes,
        feature_map_size=feature_map_size,
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
    )
