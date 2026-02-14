"""
CenterPoint Loss: Heatmap-based target assignment and loss computation.

Generates Gaussian heatmap targets at GT object centers and computes:
- Heatmap focal loss (penalty-reduced, CornerNet-style)
- L1 regression loss for center offset, dimensions, height, rotation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple


def gaussian_radius(height, width, min_overlap=0.5):
    """Compute Gaussian radius from box dimensions.

    Based on CornerNet: given a minimum IoU overlap, compute the radius
    of a Gaussian so that a box centered at any point within the radius
    would still have IoU >= min_overlap with the GT box.
    """
    a1 = 1
    b1 = -(height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (-b1 + sq1) / 2

    a2 = 4
    b2 = -2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (-b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = 2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (-b3 + sq3) / 2

    return min(r1, r2, r3)


def draw_gaussian(heatmap, center, radius, k=1):
    """Draw a 2D Gaussian on the heatmap at the given center.

    Args:
        heatmap: (H, W) tensor to draw on (modified in-place)
        center: (2,) integer center coordinates (x, y) on the grid
        radius: Gaussian radius (sigma = radius / 3)
        k: peak value
    """
    diameter = 2 * radius + 1
    sigma = diameter / 6.0  # 3-sigma rule

    x = torch.arange(0, diameter, dtype=torch.float32, device=heatmap.device)
    y = x.unsqueeze(1)
    x0 = y0 = diameter // 2

    gaussian = torch.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    H, W = heatmap.shape
    cx, cy = int(center[0]), int(center[1])

    left = min(cx, radius)
    right = min(W - cx - 1, radius)
    top = min(cy, radius)
    bottom = min(H - cy - 1, radius)

    if left < 0 or right < 0 or top < 0 or bottom < 0:
        return

    heatmap_region = heatmap[cy - top:cy + bottom + 1, cx - left:cx + right + 1]
    gaussian_region = gaussian[radius - top:radius + bottom + 1, radius - left:radius + right + 1]

    # Take element-wise max (don't overwrite existing peaks)
    torch.max(heatmap_region, gaussian_region * k, out=heatmap_region)


class CenterPointLoss(nn.Module):
    """Supervised loss for CenterPoint with heatmap-based target assignment.

    Same interface as PointPillarsLoss: forward(batch_dict, pred_dict) → (loss, loss_dict)
    """

    def __init__(
        self,
        num_classes: int = 10,
        feature_map_size: Tuple[int, int] = (248, 216),
        point_cloud_range: np.ndarray = None,
        voxel_size: np.ndarray = None,
        stride: int = 2,
        hm_weight: float = 1.0,
        reg_weight: float = 2.0,
        min_gaussian_overlap: float = 0.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.fm_h, self.fm_w = feature_map_size
        self.stride = stride
        self.hm_weight = hm_weight
        self.reg_weight = reg_weight
        self.min_gaussian_overlap = min_gaussian_overlap

        if point_cloud_range is None:
            point_cloud_range = np.array([0, -39.68, -3, 69.12, 39.68, 1])
        if voxel_size is None:
            voxel_size = np.array([0.16, 0.16, 4.0])

        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size

        # World-to-grid conversion factors
        self.x_stride = voxel_size[0] * stride
        self.y_stride = voxel_size[1] * stride
        self.x_min = point_cloud_range[0]
        self.y_min = point_cloud_range[1]

    def _generate_targets(self, gt_boxes, gt_labels, device):
        """Generate heatmap and regression targets for one sample.

        Args:
            gt_boxes: (G, 7) — x, y, z, l, w, h, yaw
            gt_labels: (G,) — class indices (0-based)

        Returns:
            heatmap: (C, H, W) Gaussian heatmap targets
            reg_targets: (G, 8) — center_x, center_y, l, w, h, z, sin, cos
            reg_indices: (G, 2) — (y_idx, x_idx) grid cell for each GT
            reg_mask: (G,) — bool, True if GT maps to valid grid cell
        """
        H, W = self.fm_h, self.fm_w
        C = self.num_classes

        heatmap = torch.zeros(C, H, W, dtype=torch.float32, device=device)
        num_gt = gt_boxes.shape[0]
        reg_targets = torch.zeros(num_gt, 8, dtype=torch.float32, device=device)
        reg_indices = torch.zeros(num_gt, 2, dtype=torch.long, device=device)
        reg_mask = torch.zeros(num_gt, dtype=torch.bool, device=device)

        for g in range(num_gt):
            box = gt_boxes[g]  # (7,)
            cls_idx = int(gt_labels[g])

            # Convert world center to feature map coordinates
            cx_grid = (box[0] - self.x_min - self.x_stride / 2) / self.x_stride
            cy_grid = (box[1] - self.y_min - self.y_stride / 2) / self.y_stride

            cx_int = int(cx_grid.round())
            cy_int = int(cy_grid.round())

            if cx_int < 0 or cx_int >= W or cy_int < 0 or cy_int >= H:
                continue

            # Gaussian radius based on box size in grid cells
            box_w_grid = box[3] / self.x_stride  # l in grid cells
            box_h_grid = box[4] / self.y_stride  # w in grid cells
            radius = max(0, int(gaussian_radius(box_h_grid.item(), box_w_grid.item(),
                                                self.min_gaussian_overlap)))
            radius = max(radius, 1)

            # Draw Gaussian on heatmap
            draw_gaussian(heatmap[cls_idx], center=torch.tensor([cx_int, cy_int]),
                          radius=radius)

            # Regression targets at center
            reg_targets[g, 0] = cx_grid - cx_int  # sub-pixel offset x
            reg_targets[g, 1] = cy_grid - cy_int  # sub-pixel offset y
            reg_targets[g, 2] = torch.log(box[3].clamp(min=1e-6))  # log(l)
            reg_targets[g, 3] = torch.log(box[4].clamp(min=1e-6))  # log(w)
            reg_targets[g, 4] = torch.log(box[5].clamp(min=1e-6))  # log(h)
            reg_targets[g, 5] = box[2]  # z
            reg_targets[g, 6] = torch.sin(box[6])  # sin(yaw)
            reg_targets[g, 7] = torch.cos(box[6])  # cos(yaw)

            reg_indices[g, 0] = cy_int
            reg_indices[g, 1] = cx_int
            reg_mask[g] = True

        return heatmap, reg_targets, reg_indices, reg_mask

    def forward(
        self,
        batch_dict: Dict,
        pred_dict: Dict,
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute CenterPoint loss.

        Args:
            batch_dict: contains gt_boxes (list of (G, 7)) and gt_labels (list of (G,))
            pred_dict: model output with heatmap, center, dim, height, rot
        """
        heatmap_pred = pred_dict['heatmap']  # (B, C, H, W)
        center_pred = pred_dict['center']    # (B, 2, H, W)
        dim_pred = pred_dict['dim']          # (B, 3, H, W)
        height_pred = pred_dict['height']    # (B, 1, H, W)
        rot_pred = pred_dict['rot']          # (B, 2, H, W)

        B = heatmap_pred.shape[0]
        device = heatmap_pred.device

        total_hm_loss = torch.tensor(0.0, device=device)
        total_reg_loss = torch.tensor(0.0, device=device)

        gt_boxes_list = batch_dict['gt_boxes']
        gt_labels_list = batch_dict['gt_labels']

        for b in range(B):
            gt_b = gt_boxes_list[b].to(device)
            gl_b = gt_labels_list[b].to(device)

            hm_target, reg_target, reg_idx, reg_mask = self._generate_targets(
                gt_b, gl_b, device
            )

            # --- Heatmap focal loss (penalty-reduced) ---
            hm_pred_b = torch.sigmoid(heatmap_pred[b])  # (C, H, W)
            total_hm_loss = total_hm_loss + self._focal_loss_heatmap(hm_pred_b, hm_target)

            # --- Regression L1 loss (at positive locations) ---
            if reg_mask.any():
                pos_idx = reg_idx[reg_mask]     # (P, 2)  y, x
                pos_target = reg_target[reg_mask]  # (P, 8)

                yi = pos_idx[:, 0]
                xi = pos_idx[:, 1]

                # Gather predictions at positive locations
                pred_center = center_pred[b, :, yi, xi].T   # (P, 2)
                pred_dim = dim_pred[b, :, yi, xi].T          # (P, 3)
                pred_height = height_pred[b, :, yi, xi].T    # (P, 1)
                pred_rot = rot_pred[b, :, yi, xi].T          # (P, 2)

                pred_reg = torch.cat([pred_center, pred_dim, pred_height, pred_rot], dim=-1)  # (P, 8)

                reg_loss = F.l1_loss(pred_reg, pos_target, reduction='mean')
                total_reg_loss = total_reg_loss + reg_loss

        total_hm_loss = total_hm_loss / B
        total_reg_loss = total_reg_loss / B

        loss = self.hm_weight * total_hm_loss + self.reg_weight * total_reg_loss

        loss_dict = {
            'hm_loss': total_hm_loss.item(),
            'reg_loss': total_reg_loss.item(),
            'total_loss': loss.item(),
        }
        return loss, loss_dict

    @staticmethod
    def _focal_loss_heatmap(pred, target, alpha=2.0, beta=4.0):
        """Penalty-reduced focal loss for heatmaps (CornerNet-style).

        Args:
            pred: (C, H, W) sigmoid-activated predictions
            target: (C, H, W) Gaussian heatmap targets [0, 1]
        """
        pos_mask = target.eq(1).float()
        neg_mask = target.lt(1).float()

        pred = pred.clamp(min=1e-6, max=1 - 1e-6)

        # Positive locations: standard focal loss
        pos_loss = -((1 - pred) ** alpha) * torch.log(pred) * pos_mask

        # Negative locations: penalty-reduced focal loss
        neg_loss = -((1 - target) ** beta) * (pred ** alpha) * torch.log(1 - pred) * neg_mask

        num_pos = pos_mask.sum().clamp(min=1)
        loss = (pos_loss.sum() + neg_loss.sum()) / num_pos
        return loss
