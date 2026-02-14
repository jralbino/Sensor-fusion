#!/usr/bin/env python3
"""
Interactive 3D Visualization + Camera Projection for PointPillars.

Features:
  1. Interactive 3D point cloud + bounding boxes (Plotly HTML)
  2. LiDAR-to-camera projection on all 6 NuScenes cameras (matplotlib PNG)

Usage:
  # GT only (no model)
  venv/bin/python visualize_3d.py --data-root ../Fusion/data/sets/nuscenes --sample-idx 0

  # With model predictions
  venv/bin/python visualize_3d.py --data-root ../Fusion/data/sets/nuscenes \
      --checkpoint outputs/test_run/best.pth --sample-idx 0 1 2
"""

import argparse
import logging
import numpy as np
import torch
import webbrowser
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import plotly.graph_objects as go

from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CLASS_NAMES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

# NuScenes category → our class index
NUSCENES_CLS_MAP = {}
_CATEGORY_PREFIXES = {
    'vehicle.car': 0, 'vehicle.truck': 1,
    'vehicle.construction': 2, 'vehicle.bus': 3,
    'vehicle.trailer': 4, 'static_object.bollard': 5,
    'movable_object.barrier': 5, 'vehicle.motorcycle': 6,
    'vehicle.bicycle': 7, 'human.pedestrian': 8,
    'movable_object.trafficcone': 9,
}

CAMERAS = [
    'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
]

# Plotly colours per class (CSS named colours)
PLOTLY_CLASS_COLORS = [
    'dodgerblue', 'orange', 'gold', 'red', 'purple',
    'gray', 'magenta', 'cyan', 'lime', 'yellow',
]

# Matplotlib tab10
MPL_CLASS_COLORS = plt.cm.tab10(np.linspace(0, 1, 10))[:, :3]


def _category_to_label(category_name: str) -> int:
    """Map NuScenes category string to class index, or -1 if not in our set."""
    for prefix, idx in _CATEGORY_PREFIXES.items():
        if category_name.startswith(prefix):
            return idx
    return -1


# ---------------------------------------------------------------------------
# NuScenes data loading
# ---------------------------------------------------------------------------

def make_transform_matrix(translation, rotation):
    """Build a 4x4 homogeneous transform from translation + quaternion."""
    mat = np.eye(4)
    mat[:3, :3] = Quaternion(rotation).rotation_matrix
    mat[:3, 3] = translation
    return mat


def load_multi_sweep_points(nusc: NuScenes, sample_token: str, data_root: Path,
                            sweeps_num: int = 9, remove_close: float = 1.0):
    """Load keyframe + previous sweeps, transformed to keyframe LiDAR coords.

    Args:
        nusc: NuScenes instance
        sample_token: sample token
        data_root: NuScenes data root
        sweeps_num: number of previous sweeps to load (default 9 → 10 total)
        remove_close: remove points within this radius of origin (ego returns)

    Returns:
        points: (N, 5) float32 array [x, y, z, intensity, time_lag]
    """
    sample = nusc.get('sample', sample_token)
    lidar_sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])

    # Keyframe ego pose and calibration
    kf_cs = nusc.get('calibrated_sensor', lidar_sd['calibrated_sensor_token'])
    kf_ego = nusc.get('ego_pose', lidar_sd['ego_pose_token'])
    kf_lidar_to_ego = make_transform_matrix(kf_cs['translation'], kf_cs['rotation'])
    kf_ego_to_global = make_transform_matrix(kf_ego['translation'], kf_ego['rotation'])
    # Inverse transforms for global → keyframe lidar
    kf_global_to_ego = np.linalg.inv(kf_ego_to_global)
    kf_ego_to_lidar = np.linalg.inv(kf_lidar_to_ego)

    kf_ts = lidar_sd['timestamp']

    # Load keyframe points
    kf_path = data_root / lidar_sd['filename']
    kf_points = np.fromfile(str(kf_path), dtype=np.float32).reshape(-1, 5)
    # Replace ring_index (col 4) with time_lag = 0 for keyframe
    kf_points[:, 4] = 0.0

    all_points = [kf_points]

    # Walk backwards through previous sweeps
    sd = lidar_sd
    collected = 0
    while sd['prev'] and collected < sweeps_num:
        sd = nusc.get('sample_data', sd['prev'])

        # Load sweep points
        sweep_path = data_root / sd['filename']
        if not sweep_path.exists():
            break
        sweep_pts = np.fromfile(str(sweep_path), dtype=np.float32).reshape(-1, 5)

        # Transform sweep points to keyframe LiDAR frame
        # sweep_lidar → sweep_ego → global → kf_ego → kf_lidar
        sw_cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
        sw_ego = nusc.get('ego_pose', sd['ego_pose_token'])
        sw_lidar_to_ego = make_transform_matrix(sw_cs['translation'], sw_cs['rotation'])
        sw_ego_to_global = make_transform_matrix(sw_ego['translation'], sw_ego['rotation'])

        # Full chain: sweep_lidar → kf_lidar
        sweep_to_kf = kf_ego_to_lidar @ kf_global_to_ego @ sw_ego_to_global @ sw_lidar_to_ego

        # Transform xyz
        xyz_hom = np.hstack([sweep_pts[:, :3], np.ones((len(sweep_pts), 1))])
        xyz_kf = (sweep_to_kf @ xyz_hom.T).T[:, :3]

        # Time lag in seconds
        time_lag = (kf_ts - sd['timestamp']) / 1e6

        # Build output: [x, y, z, intensity, time_lag]
        out = np.zeros((len(sweep_pts), 5), dtype=np.float32)
        out[:, :3] = xyz_kf
        out[:, 3] = sweep_pts[:, 3]  # intensity
        out[:, 4] = time_lag

        all_points.append(out)
        collected += 1

    # Pad with keyframe copies if fewer sweeps than requested
    while collected < sweeps_num:
        all_points.append(kf_points.copy())
        collected += 1

    points = np.concatenate(all_points, axis=0).astype(np.float32)

    # Remove points close to origin (ego vehicle returns)
    if remove_close > 0:
        dist = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
        points = points[dist > remove_close]

    return points


def load_sample_data(nusc: NuScenes, sample_token: str, data_root: Path,
                     sweeps_num: int = 0):
    """Load points, GT boxes (in LiDAR frame), camera images, and calibrations.

    Args:
        sweeps_num: if > 0, load multi-sweep points (keyframe + sweeps_num previous).
                    The 5th feature becomes time_lag instead of ring_index.
    """
    sample = nusc.get('sample', sample_token)

    # --- LiDAR points ---
    lidar_sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    if sweeps_num > 0:
        points = load_multi_sweep_points(nusc, sample_token, data_root,
                                         sweeps_num=sweeps_num)
    else:
        pts_path = data_root / lidar_sd['filename']
        points = np.fromfile(str(pts_path), dtype=np.float32).reshape(-1, 5)

    # --- Calibration: ego and lidar sensor ---
    lidar_cs = nusc.get('calibrated_sensor', lidar_sd['calibrated_sensor_token'])
    lidar_ego = nusc.get('ego_pose', lidar_sd['ego_pose_token'])

    lidar_to_ego = make_transform_matrix(lidar_cs['translation'], lidar_cs['rotation'])
    ego_to_global = make_transform_matrix(lidar_ego['translation'], lidar_ego['rotation'])

    # --- GT boxes in LiDAR frame ---
    gt_boxes_lidar = []
    gt_labels = []
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        label = _category_to_label(ann['category_name'])
        if label < 0:
            continue

        # Annotation is in global frame → transform to lidar frame
        center_global = np.array(ann['translation'])
        rot_global = Quaternion(ann['rotation'])

        # global → ego
        ego_inv = np.linalg.inv(ego_to_global)
        center_ego = ego_inv[:3, :3] @ center_global + ego_inv[:3, 3]
        rot_ego = Quaternion(matrix=ego_inv[:3, :3]) * rot_global

        # ego → lidar
        lidar_inv = np.linalg.inv(lidar_to_ego)
        center_lidar = lidar_inv[:3, :3] @ center_ego + lidar_inv[:3, 3]
        rot_lidar = Quaternion(matrix=lidar_inv[:3, :3]) * rot_ego

        # NuScenes size is (w, l, h) — we store (l, w, h) in our convention
        w, l, h = ann['size']
        yaw = rot_lidar.yaw_pitch_roll[0]

        gt_boxes_lidar.append([center_lidar[0], center_lidar[1], center_lidar[2],
                               l, w, h, yaw])
        gt_labels.append(label)

    gt_boxes_lidar = np.array(gt_boxes_lidar, dtype=np.float32) if gt_boxes_lidar else np.zeros((0, 7), dtype=np.float32)
    gt_labels = np.array(gt_labels, dtype=np.int64) if gt_labels else np.zeros(0, dtype=np.int64)

    # --- Camera images and calibrations ---
    cameras = {}
    for cam_name in CAMERAS:
        cam_sd = nusc.get('sample_data', sample['data'][cam_name])
        cam_cs = nusc.get('calibrated_sensor', cam_sd['calibrated_sensor_token'])
        cam_ego = nusc.get('ego_pose', cam_sd['ego_pose_token'])

        # LiDAR → global → cam_ego → cam_sensor
        cam_ego_to_global = make_transform_matrix(cam_ego['translation'], cam_ego['rotation'])
        cam_sensor_to_ego = make_transform_matrix(cam_cs['translation'], cam_cs['rotation'])

        # Full transform: lidar → global → cam (combining all steps)
        global_to_cam_ego = np.linalg.inv(cam_ego_to_global)
        cam_ego_to_sensor = np.linalg.inv(cam_sensor_to_ego)
        lidar_to_cam = cam_ego_to_sensor @ global_to_cam_ego @ ego_to_global @ lidar_to_ego

        intrinsic = np.array(cam_cs['camera_intrinsic'])
        img_path = data_root / cam_sd['filename']

        cameras[cam_name] = {
            'img_path': img_path,
            'intrinsic': intrinsic,
            'lidar_to_cam': lidar_to_cam,
            'img_h': cam_sd['height'],
            'img_w': cam_sd['width'],
        }

    return {
        'points': points,
        'gt_boxes': gt_boxes_lidar,
        'gt_labels': gt_labels,
        'cameras': cameras,
        'sample_token': sample_token,
    }


# ---------------------------------------------------------------------------
# 3D box corners
# ---------------------------------------------------------------------------

def boxes_to_corners_3d(boxes: np.ndarray) -> np.ndarray:
    """Convert (N, 7) boxes → (N, 8, 3) corners. Reuses logic from src/core/geometry."""
    from src.core.geometry import boxes_to_corners_3d as _b2c
    if len(boxes) == 0:
        return np.zeros((0, 8, 3), dtype=np.float32)
    return _b2c(boxes)


# 12 edges of a box (index pairs into 8 corners)
BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
    (4, 5), (5, 6), (6, 7), (7, 4),  # top
    (0, 4), (1, 5), (2, 6), (3, 7),  # vertical
]


# ---------------------------------------------------------------------------
# Plotly interactive 3D scene
# ---------------------------------------------------------------------------

def render_3d_scene(points, gt_boxes=None, gt_labels=None,
                    pred_boxes=None, pred_labels=None, pred_scores=None,
                    max_points=30000, title='3D Scene'):
    """Create an interactive Plotly 3D scatter + wireframe boxes."""
    fig = go.Figure()

    # --- Downsample points ---
    if len(points) > max_points:
        idx = np.random.choice(len(points), max_points, replace=False)
        pts = points[idx]
    else:
        pts = points

    # Colour by height
    z_vals = pts[:, 2]
    z_min, z_max = z_vals.min(), z_vals.max()
    z_range = max(z_max - z_min, 0.1)

    fig.add_trace(go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        mode='markers',
        marker=dict(
            size=1, color=pts[:, 2], colorscale='Viridis',
            cmin=z_min, cmax=z_max, opacity=0.5,
        ),
        name='Point Cloud',
        hoverinfo='skip',
    ))

    # --- GT boxes (green dashed) ---
    if gt_boxes is not None and len(gt_boxes) > 0:
        corners = boxes_to_corners_3d(gt_boxes)
        _add_boxes_to_fig(fig, corners, gt_labels, color_override='rgba(50,255,50,0.7)',
                          dash='dash', name_prefix='GT')

    # --- Pred boxes (class-coloured solid) ---
    if pred_boxes is not None and len(pred_boxes) > 0:
        corners = boxes_to_corners_3d(pred_boxes)
        _add_boxes_to_fig(fig, corners, pred_labels, scores=pred_scores,
                          dash='solid', name_prefix='Pred')

    fig.update_layout(
        scene=dict(
            xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)',
            aspectmode='data',
            bgcolor='rgb(20,20,30)',
        ),
        title=dict(text=title, font=dict(size=16)),
        paper_bgcolor='rgb(15,15,25)',
        font=dict(color='white'),
        legend=dict(font=dict(size=10)),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    return fig


def _add_boxes_to_fig(fig, corners, labels, scores=None,
                      color_override=None, dash='solid', name_prefix='Box'):
    """Add wireframe boxes as Scatter3d lines."""
    for i in range(len(corners)):
        cls_idx = int(labels[i]) if labels is not None else 0
        color = color_override or PLOTLY_CLASS_COLORS[cls_idx % 10]
        cls_name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else f'c{cls_idx}'
        name = f'{name_prefix}: {cls_name}'
        if scores is not None:
            name += f' ({scores[i]:.2f})'

        c = corners[i]  # (8, 3)
        xs, ys, zs = [], [], []
        for e0, e1 in BOX_EDGES:
            xs.extend([c[e0, 0], c[e1, 0], None])
            ys.extend([c[e0, 1], c[e1, 1], None])
            zs.extend([c[e0, 2], c[e1, 2], None])

        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode='lines',
            line=dict(color=color, width=3, dash=dash),
            name=name,
            showlegend=(i == 0),  # only first box in group shows legend
        ))


# ---------------------------------------------------------------------------
# Camera projection
# ---------------------------------------------------------------------------

def project_points_to_camera(points_lidar, lidar_to_cam, intrinsic, img_h, img_w):
    """Project LiDAR points onto a camera image.

    Returns:
        uv: (K, 2) pixel coords of visible points
        depths: (K,) depths of visible points
        mask: (N,) bool mask of which input points are visible
    """
    # Transform to camera frame
    pts_hom = np.hstack([points_lidar[:, :3], np.ones((len(points_lidar), 1))])
    pts_cam = (lidar_to_cam @ pts_hom.T).T[:, :3]  # (N, 3)

    # Filter behind camera
    depths = pts_cam[:, 2]
    mask = depths > 0.5  # minimum depth

    pts_cam_valid = pts_cam[mask]
    if len(pts_cam_valid) == 0:
        return np.zeros((0, 2)), np.zeros(0), mask

    # Project with intrinsic
    uv_hom = (intrinsic @ pts_cam_valid.T).T  # (K, 3)
    uv = uv_hom[:, :2] / uv_hom[:, 2:3]

    # Filter outside image
    in_img = (uv[:, 0] >= 0) & (uv[:, 0] < img_w) & \
             (uv[:, 1] >= 0) & (uv[:, 1] < img_h)

    # Update mask to combine depth + image bounds
    valid_indices = np.where(mask)[0]
    full_in_img = np.zeros(len(mask), dtype=bool)
    full_in_img[valid_indices[in_img]] = True

    return uv[in_img], depths[mask][in_img], full_in_img


def project_box_corners_to_camera(corners_3d, lidar_to_cam, intrinsic):
    """Project (8, 3) box corners to 2D. Returns (8, 2) uv and visibility mask."""
    pts_hom = np.hstack([corners_3d, np.ones((8, 1))])
    pts_cam = (lidar_to_cam @ pts_hom.T).T[:, :3]

    # All corners must be in front of camera
    visible = pts_cam[:, 2] > 0.1
    if not visible.any():
        return None, None

    uv_hom = (intrinsic @ pts_cam.T).T
    uv = uv_hom[:, :2] / uv_hom[:, 2:3]
    return uv, visible


def render_camera_projection(img_path, points_lidar, lidar_to_cam, intrinsic,
                             img_h, img_w, gt_boxes=None, gt_labels=None,
                             pred_boxes=None, pred_labels=None, pred_scores=None,
                             ax=None):
    """Render a single camera view with projected points and boxes."""
    import cv2
    img = cv2.imread(str(img_path))
    if img is None:
        logger.warning(f"Could not load image: {img_path}")
        return ax
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(16, 9))

    ax.imshow(img)

    # --- Project points ---
    uv, depths, _ = project_points_to_camera(points_lidar, lidar_to_cam, intrinsic, img_h, img_w)
    if len(uv) > 0:
        # Colour by depth (jet colormap)
        depth_norm = np.clip(depths / 50.0, 0, 1)  # normalise to ~50m
        ax.scatter(uv[:, 0], uv[:, 1], c=depths, cmap='jet',
                   s=1, alpha=0.6, vmin=0, vmax=50, edgecolors='none')

    # --- Project GT boxes (green dashed) ---
    if gt_boxes is not None and len(gt_boxes) > 0:
        gt_corners = boxes_to_corners_3d(gt_boxes)
        _draw_projected_boxes(ax, gt_corners, gt_labels, lidar_to_cam, intrinsic,
                              img_h, img_w, color_override=(0.2, 1.0, 0.2),
                              linestyle='--', linewidth=1.5)

    # --- Project pred boxes (class-coloured solid) ---
    if pred_boxes is not None and len(pred_boxes) > 0:
        pred_corners = boxes_to_corners_3d(pred_boxes)
        _draw_projected_boxes(ax, pred_corners, pred_labels, lidar_to_cam, intrinsic,
                              img_h, img_w, linestyle='-', linewidth=2.0,
                              scores=pred_scores)

    ax.set_xlim(0, img_w)
    ax.set_ylim(img_h, 0)
    ax.axis('off')
    return ax


def _draw_projected_boxes(ax, all_corners, labels, lidar_to_cam, intrinsic,
                          img_h, img_w, color_override=None, linestyle='-',
                          linewidth=2.0, scores=None):
    """Draw wireframe boxes projected onto a camera image."""
    for i in range(len(all_corners)):
        uv, visible = project_box_corners_to_camera(
            all_corners[i], lidar_to_cam, intrinsic)
        if uv is None:
            continue

        # Check if at least some corners are visible and roughly in image
        center_u, center_v = uv[:, 0].mean(), uv[:, 1].mean()
        margin = 200
        if center_u < -margin or center_u > img_w + margin or \
           center_v < -margin or center_v > img_h + margin:
            continue

        cls_idx = int(labels[i]) if labels is not None else 0
        color = color_override if color_override else MPL_CLASS_COLORS[cls_idx % 10]

        # Draw 12 edges
        for e0, e1 in BOX_EDGES:
            if visible[e0] and visible[e1]:
                ax.plot([uv[e0, 0], uv[e1, 0]], [uv[e0, 1], uv[e1, 1]],
                        color=color, linewidth=linewidth, linestyle=linestyle,
                        alpha=0.85)


def render_all_cameras(sample_data, pred_boxes=None, pred_labels=None,
                       pred_scores=None, title=''):
    """Render 3x2 grid of all 6 cameras with projected points and boxes."""
    fig, axes = plt.subplots(2, 3, figsize=(30, 14), dpi=100)
    fig.patch.set_facecolor('#111111')

    cam_layout = [
        ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT'],
        ['CAM_BACK_LEFT',  'CAM_BACK',  'CAM_BACK_RIGHT'],
    ]

    for row_idx, row_cams in enumerate(cam_layout):
        for col_idx, cam_name in enumerate(row_cams):
            ax = axes[row_idx, col_idx]
            cam_info = sample_data['cameras'][cam_name]

            render_camera_projection(
                img_path=cam_info['img_path'],
                points_lidar=sample_data['points'],
                lidar_to_cam=cam_info['lidar_to_cam'],
                intrinsic=cam_info['intrinsic'],
                img_h=cam_info['img_h'],
                img_w=cam_info['img_w'],
                gt_boxes=sample_data['gt_boxes'],
                gt_labels=sample_data['gt_labels'],
                pred_boxes=pred_boxes,
                pred_labels=pred_labels,
                pred_scores=pred_scores,
                ax=ax,
            )
            ax.set_title(cam_name, color='white', fontsize=11, pad=4)

    n_gt = len(sample_data['gt_boxes'])
    n_pred = len(pred_boxes) if pred_boxes is not None else 0
    suptitle = title or 'Camera Projections'
    if n_gt or n_pred:
        suptitle += f'  |  GT: {n_gt}  Pred: {n_pred}'
    fig.suptitle(suptitle, color='white', fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------

def run_model_inference(checkpoint_path, points, device, score_thresh=0.15, nms_iou=0.3, model_type='pointpillars'):
    """Load model and run inference on a single point cloud."""
    from visualize import voxelize_single, run_inference

    if model_type == 'second':
        from src.detectors.second import SECOND
        model = SECOND(num_classes=10)
    elif model_type == 'centerpoint':
        from src.detectors.centerpoint import CenterPoint
        model = CenterPoint(num_classes=10)
    else:
        from src.detectors.pointpillars import PointPillars
        model = PointPillars(num_classes=10)
    model.load_checkpoint(checkpoint_path)
    model = model.to(device)
    model.eval()

    det = run_inference(model, points, device, score_thresh, nms_iou)
    return det, model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='3D Interactive Visualization + Camera Projection')
    p.add_argument('--data-root', required=True, help='NuScenes data root')
    p.add_argument('--version', default='v1.0-mini', help='NuScenes version')
    p.add_argument('--sample-idx', type=int, nargs='+', default=[0],
                   help='Sample indices to visualize')
    p.add_argument('--checkpoint', default=None, help='Model checkpoint (optional)')
    p.add_argument('--model', choices=['pointpillars', 'second', 'centerpoint'], default='pointpillars',
                   help='Model architecture (default: pointpillars)')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--score-thresh', type=float, default=0.15)
    p.add_argument('--nms-iou', type=float, default=0.3)
    p.add_argument('--output-dir', default='outputs/vis_3d')
    p.add_argument('--no-browser', action='store_true', help='Do not open HTML in browser')
    p.add_argument('--max-points-3d', type=int, default=30000,
                   help='Max points for Plotly 3D (downsample)')
    return p.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    logger.info(f"Loading NuScenes {args.version} from {data_root}")
    nusc = NuScenes(version=args.version, dataroot=str(data_root), verbose=False)
    logger.info(f"Loaded {len(nusc.sample)} samples")

    # Optionally load model
    model = None
    if args.checkpoint:
        from visualize import run_inference
        logger.info(f"Loading {args.model} model from {args.checkpoint}")
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

    for sample_idx in args.sample_idx:
        if sample_idx >= len(nusc.sample):
            logger.warning(f"Sample index {sample_idx} out of range (max {len(nusc.sample)-1}), skipping")
            continue

        sample_token = nusc.sample[sample_idx]['token']
        logger.info(f"Processing sample {sample_idx} (token={sample_token[:8]}...)")

        # Load data
        sample_data = load_sample_data(nusc, sample_token, data_root)

        # Run inference if model is available
        pred_boxes, pred_labels, pred_scores = None, None, None
        if model is not None:
            from visualize import run_inference
            det = run_inference(model, sample_data['points'], device,
                                args.score_thresh, args.nms_iou)
            pred_boxes = det['boxes']
            pred_labels = det['labels']
            pred_scores = det['scores']
            logger.info(f"  Model predictions: {len(pred_boxes)} detections")

        logger.info(f"  GT boxes: {len(sample_data['gt_boxes'])}")

        # --- 3D Interactive HTML ---
        fig_3d = render_3d_scene(
            sample_data['points'],
            gt_boxes=sample_data['gt_boxes'],
            gt_labels=sample_data['gt_labels'],
            pred_boxes=pred_boxes,
            pred_labels=pred_labels,
            pred_scores=pred_scores,
            max_points=args.max_points_3d,
            title=f'Sample {sample_idx} — 3D Scene',
        )
        html_path = out_dir / f'sample_{sample_idx:04d}_3d.html'
        fig_3d.write_html(str(html_path), include_plotlyjs='cdn')
        logger.info(f"  Saved 3D scene → {html_path}")

        if not args.no_browser:
            webbrowser.open(f'file://{html_path.resolve()}')

        # --- Camera projections PNG ---
        fig_cam = render_all_cameras(
            sample_data,
            pred_boxes=pred_boxes,
            pred_labels=pred_labels,
            pred_scores=pred_scores,
            title=f'Sample {sample_idx}',
        )
        cam_path = out_dir / f'sample_{sample_idx:04d}_cameras.png'
        fig_cam.savefig(str(cam_path), facecolor=fig_cam.get_facecolor(),
                        bbox_inches='tight')
        plt.close(fig_cam)
        logger.info(f"  Saved cameras → {cam_path}")

    logger.info(f"Done. Outputs in {out_dir}/")


if __name__ == '__main__':
    main()
