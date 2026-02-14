#!/usr/bin/env python3
"""
BEV (Bird's Eye View) Visualization for PointPillars.

Modes:
  --input <file.bin>              Inference on a single point cloud
  --dataset --data-root <path>    Iterate val set with GT comparison
"""

import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CLASS_NAMES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

# Distinct colours per class (tab10 palette)
CLASS_COLORS = plt.cm.tab10(np.linspace(0, 1, 10))[:, :3]


# -----------------------------------------------------------------------
# Drawing helpers
# -----------------------------------------------------------------------

def box_corners_bev(box):
    """Return 4 BEV corners (4, 2) for a single (7,) box [x,y,z,l,w,h,yaw]."""
    x, y, l, w, yaw = box[0], box[1], box[3], box[4], box[6]
    cos, sin = np.cos(yaw), np.sin(yaw)
    hl, hw = l / 2, w / 2
    # corners: front-left, front-right, back-right, back-left
    dx = np.array([hl, hl, -hl, -hl])
    dy = np.array([hw, -hw, -hw, hw])
    cx = cos * dx - sin * dy + x
    cy = sin * dx + cos * dy + y
    return np.stack([cx, cy], axis=-1)  # (4, 2)


def draw_boxes(ax, boxes, labels, scores=None, color_override=None,
               linestyle='-', linewidth=1.5, alpha=0.9, show_labels=True):
    """Draw rotated BEV bounding boxes on a matplotlib axis."""
    for i, box in enumerate(boxes):
        corners = box_corners_bev(box)
        cls_idx = int(labels[i]) if labels is not None else 0
        color = color_override if color_override is not None else CLASS_COLORS[cls_idx % 10]

        # Draw polygon
        polygon = plt.Polygon(corners, closed=True, fill=False,
                              edgecolor=color, linewidth=linewidth,
                              linestyle=linestyle, alpha=alpha)
        ax.add_patch(polygon)

        # Draw heading line (front edge midpoint → centre)
        front_mid = (corners[0] + corners[1]) / 2
        centre = np.array([box[0], box[1]])
        ax.plot([centre[0], front_mid[0]], [centre[1], front_mid[1]],
                color=color, linewidth=1, alpha=alpha)

        # Label
        if show_labels:
            cls_name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else f'c{cls_idx}'
            txt = cls_name
            if scores is not None:
                txt += f' {scores[i]:.2f}'
            ax.text(box[0], box[1] + box[4] / 2 + 0.5, txt,
                    fontsize=5, color=color, ha='center', va='bottom',
                    alpha=alpha, clip_on=True)


def render_bev(points, pred_boxes=None, pred_labels=None, pred_scores=None,
               gt_boxes=None, gt_labels=None, title='', pc_range=None,
               point_size=0.3, fig_size=(12, 12), dpi=150):
    """Render a BEV image and return the matplotlib figure.

    Args:
        points: (N, 4) point cloud
        pred_boxes/labels/scores: model predictions (numpy)
        gt_boxes/labels: ground truth (numpy)
        title: plot title
        pc_range: [x_min, y_min, z_min, x_max, y_max, z_max]
    """
    if pc_range is None:
        pc_range = [0, -39.68, -3, 69.12, 39.68, 1]

    fig, ax = plt.subplots(1, 1, figsize=fig_size, dpi=dpi)
    ax.set_facecolor('#1a1a1a')

    # Filter points within range
    mask = (
        (points[:, 0] >= pc_range[0]) & (points[:, 0] <= pc_range[3]) &
        (points[:, 1] >= pc_range[1]) & (points[:, 1] <= pc_range[4])
    )
    pts = points[mask]

    # Colour by height (z)
    z_vals = pts[:, 2]
    z_norm = (z_vals - pc_range[2]) / (pc_range[5] - pc_range[2])
    z_norm = np.clip(z_norm, 0, 1)

    ax.scatter(pts[:, 0], pts[:, 1], c=z_norm, cmap='viridis',
               s=point_size, alpha=0.6, edgecolors='none', rasterized=True)

    # Draw GT boxes (dashed green)
    if gt_boxes is not None and len(gt_boxes) > 0:
        draw_boxes(ax, gt_boxes, gt_labels, color_override=(0.2, 1.0, 0.2),
                   linestyle='--', linewidth=1.2, alpha=0.8, show_labels=False)

    # Draw predictions (solid, class-coloured)
    if pred_boxes is not None and len(pred_boxes) > 0:
        draw_boxes(ax, pred_boxes, pred_labels, scores=pred_scores,
                   linewidth=1.8, alpha=0.95, show_labels=True)

    # Axis setup
    ax.set_xlim(pc_range[0], pc_range[3])
    ax.set_ylim(pc_range[1], pc_range[4])
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)', color='white', fontsize=9)
    ax.set_ylabel('Y (m)', color='white', fontsize=9)
    ax.tick_params(colors='white', labelsize=7)
    for spine in ax.spines.values():
        spine.set_color('white')

    # Legend
    legend_elements = []
    if gt_boxes is not None and len(gt_boxes) > 0:
        legend_elements.append(Line2D([0], [0], color=(0.2, 1.0, 0.2),
                                      linestyle='--', label='Ground Truth'))
    if pred_boxes is not None and len(pred_boxes) > 0:
        legend_elements.append(Line2D([0], [0], color='cyan',
                                      linestyle='-', label='Predictions'))
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8,
                  facecolor='#333333', edgecolor='white', labelcolor='white')

    # Title
    n_pred = len(pred_boxes) if pred_boxes is not None else 0
    n_gt = len(gt_boxes) if gt_boxes is not None else 0
    full_title = title
    if n_pred or n_gt:
        full_title += f'  |  Pred: {n_pred}  GT: {n_gt}'
    ax.set_title(full_title, color='white', fontsize=11, pad=10)

    fig.patch.set_facecolor('#111111')
    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------
# Inference helpers (reused from infer.py)
# -----------------------------------------------------------------------

def voxelize_single(points, voxel_size, point_cloud_range, max_voxels=40000,
                     max_points_per_voxel=32):
    from src.data.datasets import voxelize_points
    voxels, coords, num_points = voxelize_points(
        points, np.array(voxel_size), np.array(point_cloud_range),
        max_points_per_voxel=max_points_per_voxel, max_voxels=max_voxels,
    )
    batch_idx = np.zeros((len(coords), 1), dtype=np.int32)
    coords = np.concatenate([batch_idx, coords], axis=1)
    return {
        'voxels': torch.from_numpy(voxels).float(),
        'voxel_coords': torch.from_numpy(coords).int(),
        'voxel_num_points': torch.from_numpy(num_points).int(),
        'batch_size': 1,
    }


def run_inference(model, points, device, score_thresh=0.15, nms_iou=0.3):
    """Run model inference on a raw point cloud, return detections dict."""
    batch_dict = voxelize_single(points, (0.16, 0.16, 4.0),
                                  (0, -39.68, -3, 69.12, 39.68, 1))
    for k in batch_dict:
        if isinstance(batch_dict[k], torch.Tensor):
            batch_dict[k] = batch_dict[k].to(device)
    with torch.no_grad():
        pred_dict = model(batch_dict)
        results = model.postprocess(pred_dict, score_thresh=score_thresh,
                                     nms_iou_thresh=nms_iou)
    return results[0]


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='BEV Visualization for LiDAR Detection')
    p.add_argument('--checkpoint', required=True, help='Model checkpoint')
    p.add_argument('--model', choices=['pointpillars', 'second', 'centerpoint'], default='pointpillars',
                   help='Model architecture (default: pointpillars)')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--score-thresh', type=float, default=0.15)
    p.add_argument('--nms-iou', type=float, default=0.3)
    p.add_argument('--output-dir', default='outputs/vis', help='Where to save images')

    # Mode A: single file
    p.add_argument('--input', default=None, help='Single .bin point cloud')

    # Mode B: dataset
    p.add_argument('--dataset', action='store_true', help='Iterate val dataset')
    p.add_argument('--data-root', default=None, help='NuScenes data root (required with --dataset)')
    p.add_argument('--max-samples', type=int, default=20, help='Max samples to visualize')

    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    logger.info(f"Loading {args.model} model...")
    if args.model == 'second':
        from src.detectors.second import SECOND
        model = SECOND(num_classes=10)
    elif args.model == 'centerpoint':
        from src.detectors.centerpoint import CenterPoint
        model = CenterPoint(num_classes=10)
    else:
        from src.detectors.pointpillars import PointPillars
        model = PointPillars(num_classes=10)
    model.load_checkpoint(args.checkpoint)
    model = model.to(device)
    model.eval()

    if args.input:
        # ---- Mode A: single file ----
        pts = np.fromfile(args.input, dtype=np.float32).reshape(-1, 5)[:, :4]
        logger.info(f"Loaded {len(pts)} points from {Path(args.input).name}")

        det = run_inference(model, pts, device, args.score_thresh, args.nms_iou)
        fig = render_bev(pts, det['boxes'], det['labels'], det['scores'],
                         title=Path(args.input).stem)
        save_path = out_dir / (Path(args.input).stem + '_bev.png')
        fig.savefig(save_path, facecolor=fig.get_facecolor())
        plt.close(fig)
        logger.info(f"Saved → {save_path}")

    elif args.dataset:
        # ---- Mode B: dataset with GT ----
        if args.data_root is None:
            logger.error("--data-root required with --dataset")
            return

        from src.data.datasets import create_dataloader
        data_root = Path(args.data_root)
        val_info = data_root / 'nuscenes_infos_val.pkl'
        if not val_info.exists():
            logger.error(f"Val info not found: {val_info}")
            return

        val_loader = create_dataloader(
            data_root=str(data_root), info_path=str(val_info),
            batch_size=1, num_workers=0, split='val',
        )

        count = 0
        for batch_dict in val_loader:
            if count >= args.max_samples:
                break

            # Move tensors to device
            for k in batch_dict:
                if isinstance(batch_dict[k], torch.Tensor):
                    batch_dict[k] = batch_dict[k].to(device)

            # Forward + postprocess
            with torch.no_grad():
                pred_dict = model(batch_dict)
                dets = model.postprocess(pred_dict, score_thresh=args.score_thresh,
                                          nms_iou_thresh=args.nms_iou)

            det = dets[0]
            gt_boxes = batch_dict['gt_boxes'][0].cpu().numpy()
            gt_labels = batch_dict['gt_labels'][0].cpu().numpy()

            # We need the raw points for visualisation — reconstruct from voxels
            # The dataset stores 'points' but collate_batch doesn't keep it,
            # so we use the voxel data to approximate
            voxels = batch_dict['voxels'].cpu().numpy()
            num_pts = batch_dict['voxel_num_points'].cpu().numpy()
            pts_list = []
            for v_idx in range(len(voxels)):
                n = num_pts[v_idx]
                pts_list.append(voxels[v_idx, :n, :])
            points = np.concatenate(pts_list, axis=0) if pts_list else np.zeros((0, 4))

            fig = render_bev(points, det['boxes'], det['labels'], det['scores'],
                             gt_boxes=gt_boxes, gt_labels=gt_labels,
                             title=f'Sample {count}')
            save_path = out_dir / f'sample_{count:04d}_bev.png'
            fig.savefig(save_path, facecolor=fig.get_facecolor())
            plt.close(fig)
            logger.info(f"[{count+1}/{args.max_samples}] Saved → {save_path}")
            count += 1

        logger.info(f"Done. {count} images saved to {out_dir}/")

    else:
        logger.error("Specify --input <file.bin> or --dataset --data-root <path>")


if __name__ == '__main__':
    main()
