#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NuScenes Radar Integration Test and Visualization Script.

Tests all 3 radar detectors (CFAR+DBSCAN, RadarPillars, RadarCenterPoint) on
NuScenes mini samples and generates comprehensive visualizations:
- BEV plots with radar points, GT boxes, detections, velocity arrows
- Comparison grid (3 detectors × 5 samples)
- Front camera images with projected detections
- Summary statistics table
"""
import sys
from pathlib import Path
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.collections import LineCollection
from PIL import Image

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import project modules
from config.utils.path_manager import path_manager
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from Radar.src.data.radar_utils import (
    load_radar_points,
    filter_radar_quality,
    filter_radar_rcs,
    select_features,
    ALL_RADAR_SENSORS
)
from Radar.src.detectors.detector_factory import get_radar_detector
from Radar.src.core.base_radar_detector import Detection3D
import plotly.graph_objects as go
from Lidar.visualize_3d import make_transform_matrix, load_multi_sweep_points
from Lidar.src.core.geometry import boxes_to_corners_3d


# Class colors (RGB normalized for matplotlib)
CLASS_COLORS = {
    'car': (0, 1, 1),
    'truck': (1, 0.65, 0),
    'bus': (1, 0, 1),
    'pedestrian': (1, 0, 0),
    'motorcycle': (0, 0.65, 1),
    'bicycle': (1, 1, 0),
    'barrier': (0.5, 0.5, 0.5),
    'traffic_cone': (0, 0.5, 1),
    'trailer': (0.78, 0.78, 0),
    'construction_vehicle': (0.5, 0, 0.5),
}

# NuScenes 10 classes
NUSCENES_CLASSES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

# Detector configurations
DETECTOR_CONFIGS = {
    'CFAR+DBSCAN': {
        'detector_type': 'cfar_dbscan',
        'rcs_threshold': -10.0,
        'dbscan_eps': 5.0,
        'dbscan_min_samples': 2,
    },
    'RadarPillars': {
        'detector_type': 'radar_pillars',
    },
    'RadarCenterPoint': {
        'detector_type': 'radar_centerpoint',
    },
}

# Test sample indices
TEST_SAMPLES = [0, 50, 100, 150, 200]


def load_gt_annotations(nusc: NuScenes, sample_token: str) -> List[Detection3D]:
    """
    Load ground truth annotations from NuScenes and transform to RADAR_FRONT frame.

    Args:
        nusc: NuScenes instance
        sample_token: Sample token

    Returns:
        List of Detection3D objects with GT annotations
    """
    sample = nusc.get('sample', sample_token)

    # Get RADAR_FRONT sensor data
    radar_front_token = sample['data']['RADAR_FRONT']
    radar_data = nusc.get('sample_data', radar_front_token)

    # Get calibration and ego pose for RADAR_FRONT
    cs_record = nusc.get('calibrated_sensor', radar_data['calibrated_sensor_token'])
    ego_pose = nusc.get('ego_pose', radar_data['ego_pose_token'])

    # Transformation from global to radar frame
    # global -> ego -> radar
    radar_from_ego = Quaternion(cs_record['rotation']).inverse
    ego_from_global = Quaternion(ego_pose['rotation']).inverse

    # Category mapping using prefix matching
    _CATEGORY_MAP = {
        'vehicle.car': (0, 'car'),
        'vehicle.truck': (1, 'truck'),
        'vehicle.construction': (2, 'construction_vehicle'),
        'vehicle.bus.bendy': (3, 'bus'),
        'vehicle.bus.rigid': (3, 'bus'),
        'vehicle.trailer': (4, 'trailer'),
        'movable_object.barrier': (5, 'barrier'),
        'vehicle.motorcycle': (6, 'motorcycle'),
        'vehicle.bicycle': (7, 'bicycle'),
        'human.pedestrian': (8, 'pedestrian'),
        'movable_object.trafficcone': (9, 'traffic_cone'),
    }

    gt_detections = []

    # Get all annotations for this sample
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)

        # Filter by class using prefix matching
        category = ann['category_name']
        label, label_name = None, None
        for prefix, (lbl, lname) in _CATEGORY_MAP.items():
            if category.startswith(prefix):
                label, label_name = lbl, lname
                break

        if label is None:
            continue

        # Box in global frame: [x, y, z, w, l, h, qw, qx, qy, qz]
        box_global = np.array(ann['translation'])
        quat_global = Quaternion(ann['rotation'])
        size = ann['size']  # [w, l, h]

        # Transform to ego frame
        box_ego = ego_from_global.rotate(box_global - np.array(ego_pose['translation']))
        quat_ego = ego_from_global * quat_global

        # Transform to radar frame
        box_radar = radar_from_ego.rotate(box_ego - np.array(cs_record['translation']))
        quat_radar = radar_from_ego * quat_ego

        # Convert quaternion to yaw
        yaw = np.arctan2(
            2.0 * (quat_radar.w * quat_radar.z + quat_radar.x * quat_radar.y),
            1.0 - 2.0 * (quat_radar.y**2 + quat_radar.z**2)
        )

        # Create Detection3D: [x, y, z, l, w, h, yaw]
        # Note: NuScenes uses [w, l, h], we use [l, w, h]
        box_7dof = np.array([
            box_radar[0], box_radar[1], box_radar[2],
            size[1], size[0], size[2],  # swap w,l to l,w
            yaw
        ])

        # Velocity in global frame
        try:
            velocity_global = nusc.box_velocity(ann_token)[:2]
            if np.isnan(velocity_global).any():
                velocity_global = np.array([0.0, 0.0])
        except:
            velocity_global = np.array([0.0, 0.0])

        # Transform velocity to radar frame (rotation only)
        velocity_3d_global = np.array([velocity_global[0], velocity_global[1], 0.0])
        velocity_3d_ego = ego_from_global.rotate(velocity_3d_global)
        velocity_3d_radar = radar_from_ego.rotate(velocity_3d_ego)
        velocity_radar = velocity_3d_radar[:2]

        gt_detections.append(Detection3D(
            box=box_7dof,
            score=1.0,
            label=label,
            label_name=label_name,
            velocity=velocity_radar,
        ))

    return gt_detections


def draw_bev_box(ax, box: np.ndarray, color, linestyle='-', linewidth=2, alpha=1.0):
    """
    Draw 3D box on BEV plot.

    Args:
        ax: Matplotlib axes
        box: [x, y, z, l, w, h, yaw]
        color: Box color
        linestyle: Line style
        linewidth: Line width
        alpha: Transparency
    """
    x, y, z, l, w, h, yaw = box

    # Create rectangle corners in local frame
    corners = np.array([
        [-l/2, -w/2],
        [l/2, -w/2],
        [l/2, w/2],
        [-l/2, w/2],
        [-l/2, -w/2],  # close the loop
    ])

    # Rotate by yaw
    rot_mat = np.array([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw), np.cos(yaw)]
    ])
    corners_rot = corners @ rot_mat.T

    # Translate to box center
    corners_world = corners_rot + np.array([x, y])

    # Draw
    ax.plot(corners_world[:, 0], corners_world[:, 1],
            color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha)


def draw_velocity_arrow(ax, pos: np.ndarray, vel: np.ndarray, color, scale=2.0, alpha=0.7):
    """
    Draw velocity arrow on BEV plot.

    Args:
        ax: Matplotlib axes
        pos: Position [x, y]
        vel: Velocity [vx, vy]
        color: Arrow color
        scale: Arrow length scale
        alpha: Transparency
    """
    if vel is None or np.linalg.norm(vel) < 0.1:
        return

    ax.arrow(pos[0], pos[1], vel[0]*scale, vel[1]*scale,
             head_width=1.0, head_length=1.5, fc=color, ec=color, alpha=alpha)


def draw_bev_plot(
    radar_points: np.ndarray,
    gt_boxes: List[Detection3D],
    detections: List[Detection3D],
    title: str,
    output_path: Path,
    x_range=(0, 80),
    y_range=(-40, 40),
):
    """
    Generate BEV visualization with radar points, GT boxes, and detections.

    Args:
        radar_points: Radar point cloud (N, 6) with selected features [x, y, z, rcs, vx_comp, vy_comp]
        gt_boxes: Ground truth Detection3D list
        detections: Predicted Detection3D list
        title: Plot title
        output_path: Save path
        x_range: X-axis range (forward)
        y_range: Y-axis range (lateral)
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    # Draw grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    # Draw major grid every 20m
    ax.set_xticks(np.arange(x_range[0], x_range[1]+1, 20))
    ax.set_yticks(np.arange(y_range[0], y_range[1]+1, 20))

    # Origin cross
    ax.plot([0, 0], [y_range[0], y_range[1]], 'k--', linewidth=1, alpha=0.5)
    ax.plot([x_range[0], x_range[1]], [0, 0], 'k--', linewidth=1, alpha=0.5)

    # Draw radar points
    if len(radar_points) > 0:
        rcs_values = radar_points[:, 3]
        # Clip RCS for size scaling
        rcs_clipped = np.clip(rcs_values, -10, 40)
        sizes = (rcs_clipped + 10) / 50 * 100 + 10  # scale to 10-110

        ax.scatter(radar_points[:, 0], radar_points[:, 1],
                  c='green', s=sizes, alpha=0.6, label='Radar Points', marker='o')

        # Draw velocity arrows for radar points (subsample for clarity)
        if radar_points.shape[1] >= 6:
            step = max(1, len(radar_points) // 50)  # max 50 arrows
            for i in range(0, len(radar_points), step):
                pt = radar_points[i]
                vel = pt[4:6]
                if np.linalg.norm(vel) > 0.5:
                    draw_velocity_arrow(ax, pt[:2], vel, color='green', scale=1.5, alpha=0.4)

    # Draw GT boxes
    for gt in gt_boxes:
        color = CLASS_COLORS.get(gt.label_name, (0.7, 0.7, 0.7))
        draw_bev_box(ax, gt.box, color=color, linestyle='--', linewidth=2, alpha=0.7)

        # Add label
        x, y = gt.box[0], gt.box[1]
        ax.text(x, y + gt.box[4]/2 + 1, gt.label_name,
               fontsize=8, ha='center', va='bottom', color='white',
               bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.6))

        # Draw velocity arrow
        if gt.velocity is not None:
            draw_velocity_arrow(ax, gt.box[:2], gt.velocity, color=color, scale=2.0, alpha=0.6)

    # Draw detection boxes
    for det in detections:
        color = CLASS_COLORS.get(det.label_name, (1, 1, 1))
        draw_bev_box(ax, det.box, color=color, linestyle='-', linewidth=2.5, alpha=0.9)

        # Add label with score
        x, y = det.box[0], det.box[1]
        label_text = f"{det.label_name} {det.score:.2f}"
        ax.text(x, y - det.box[4]/2 - 1, label_text,
               fontsize=8, ha='center', va='top', color='black',
               bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.9))

        # Draw velocity arrow (red for detections)
        if det.velocity is not None:
            draw_velocity_arrow(ax, det.box[:2], det.velocity, color='red', scale=2.0, alpha=0.8)

    # Legend
    legend_handles = [
        mpatches.Patch(color='green', alpha=0.6, label='Radar Points'),
        mpatches.Patch(color='white', alpha=0.7, label='GT Boxes (dashed)'),
    ]
    for class_name, color in CLASS_COLORS.items():
        legend_handles.append(mpatches.Patch(color=color, label=class_name))

    ax.legend(handles=legend_handles, loc='upper right', fontsize=8, framealpha=0.9)

    # Labels and title
    ax.set_xlabel('X (m) - Forward', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y (m) - Lateral', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    # Set limits
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def draw_camera_with_detections(
    nusc: NuScenes,
    sample_token: str,
    detections: List[Detection3D],
    output_path: Path,
):
    """
    Draw front camera image with projected detection positions.

    Args:
        nusc: NuScenes instance
        sample_token: Sample token
        detections: Detection3D list
        output_path: Save path
    """
    sample = nusc.get('sample', sample_token)
    cam_front_token = sample['data'].get('CAM_FRONT')

    if not cam_front_token:
        print(f"Warning: No CAM_FRONT data for sample {sample_token}")
        return

    cam_data = nusc.get('sample_data', cam_front_token)
    img_path = Path(nusc.dataroot) / cam_data['filename']

    if not img_path.exists():
        print(f"Warning: Camera image not found: {img_path}")
        return

    # Load image
    img = Image.open(img_path)
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.imshow(img)

    # Get camera calibration
    cs_record = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
    cam_intrinsic = np.array(cs_record['camera_intrinsic'])

    # For simplicity, just draw circles at approximate positions
    # (Full 3D projection requires RADAR_FRONT -> ego -> CAM_FRONT transform)
    # Here we do a simplified visualization

    for det in detections:
        # Simplified: assume camera looks forward, draw based on x,y position
        # This is approximate - full projection would use proper transforms
        x, y = det.box[0], det.box[1]

        # Simple heuristic: map to image space
        if x > 0:  # in front of vehicle
            img_x = img.width / 2 + y * 10  # lateral offset
            img_y = img.height - x * 5  # distance (closer = lower in image)

            if 0 <= img_x < img.width and 0 <= img_y < img.height:
                color = CLASS_COLORS.get(det.label_name, (1, 1, 1))
                circle = plt.Circle((img_x, img_y), 15, color=color, fill=True, alpha=0.7)
                ax.add_patch(circle)

                ax.text(img_x, img_y - 20, f"{det.label_name}\n{det.score:.2f}",
                       fontsize=8, ha='center', va='bottom', color='white',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8))

    ax.axis('off')
    ax.set_title(f"Front Camera - Sample {sample['token'][:8]}", fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_comparison_grid(
    all_results: Dict[str, Dict[int, Path]],
    output_path: Path,
):
    """
    Create comparison grid showing all detectors side-by-side.

    Args:
        all_results: Dict[detector_name -> Dict[sample_idx -> bev_image_path]]
        output_path: Save path for grid
    """
    detector_names = list(all_results.keys())
    # Collect all sample indices across all detectors
    all_indices = set()
    for det_results in all_results.values():
        all_indices.update(det_results.keys())
    sample_indices = sorted(all_indices)

    if not sample_indices:
        print("Warning: No BEV images to create grid from")
        return

    n_rows = len(sample_indices)
    n_cols = len(detector_names)

    fig = plt.figure(figsize=(n_cols * 6, n_rows * 5))
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.3, wspace=0.2)

    for row_idx, sample_idx in enumerate(sample_indices):
        for col_idx, detector_name in enumerate(detector_names):
            ax = fig.add_subplot(gs[row_idx, col_idx])

            img_path = all_results[detector_name].get(sample_idx)
            if img_path and img_path.exists():
                img = Image.open(img_path)
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, 'No image', ha='center', va='center', fontsize=12)

            ax.axis('off')

            # Add title to first row
            if row_idx == 0:
                ax.set_title(detector_name, fontsize=14, fontweight='bold', pad=10)

            # Add sample index to first column
            if col_idx == 0:
                ax.text(-0.05, 0.5, f'Sample {sample_idx}',
                       transform=ax.transAxes, fontsize=12, fontweight='bold',
                       rotation=90, va='center', ha='right')

    plt.suptitle('Radar Detector Comparison on NuScenes Mini',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# Plotly colours per class (CSS named colours) — matches Lidar/visualize_3d.py
PLOTLY_CLASS_COLORS = [
    'dodgerblue', 'orange', 'gold', 'red', 'purple',
    'gray', 'magenta', 'cyan', 'lime', 'yellow',
]

# 12 edges of a 3D box (index pairs into 8 corners)
BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
    (4, 5), (5, 6), (6, 7), (7, 4),  # top
    (0, 4), (1, 5), (2, 6), (3, 7),  # vertical
]


def _compute_radar_to_lidar_transform(nusc: NuScenes, sample_token: str) -> np.ndarray:
    """Compute the 4x4 RADAR_FRONT -> LIDAR_TOP transform matrix.

    Chain: radar_sensor -> radar_ego -> global -> lidar_ego -> lidar_sensor
    """
    sample = nusc.get('sample', sample_token)

    # RADAR_FRONT calibration and ego pose
    radar_sd = nusc.get('sample_data', sample['data']['RADAR_FRONT'])
    radar_cs = nusc.get('calibrated_sensor', radar_sd['calibrated_sensor_token'])
    radar_ego = nusc.get('ego_pose', radar_sd['ego_pose_token'])

    # LIDAR_TOP calibration and ego pose
    lidar_sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    lidar_cs = nusc.get('calibrated_sensor', lidar_sd['calibrated_sensor_token'])
    lidar_ego = nusc.get('ego_pose', lidar_sd['ego_pose_token'])

    # Build individual transforms
    radar_to_ego = make_transform_matrix(radar_cs['translation'], radar_cs['rotation'])
    ego_to_global = make_transform_matrix(radar_ego['translation'], radar_ego['rotation'])
    global_to_lidar_ego = np.linalg.inv(
        make_transform_matrix(lidar_ego['translation'], lidar_ego['rotation'])
    )
    lidar_ego_to_sensor = np.linalg.inv(
        make_transform_matrix(lidar_cs['translation'], lidar_cs['rotation'])
    )

    return lidar_ego_to_sensor @ global_to_lidar_ego @ ego_to_global @ radar_to_ego


def _transform_detections_to_lidar(
    detections: List[Detection3D],
    radar_to_lidar: np.ndarray,
) -> List[Detection3D]:
    """Transform Detection3D objects from RADAR_FRONT frame to LIDAR_TOP frame."""
    rot3x3 = radar_to_lidar[:3, :3]
    # Extract yaw offset from the rotation matrix (rotation around Z)
    yaw_offset = np.arctan2(rot3x3[1, 0], rot3x3[0, 0])

    transformed = []
    for det in detections:
        # Transform center
        center_radar = np.array([det.box[0], det.box[1], det.box[2], 1.0])
        center_lidar = (radar_to_lidar @ center_radar)[:3]

        # Transform yaw
        yaw_lidar = det.box[6] + yaw_offset

        # Dimensions stay the same
        box_lidar = np.array([
            center_lidar[0], center_lidar[1], center_lidar[2],
            det.box[3], det.box[4], det.box[5], yaw_lidar,
        ])

        # Transform velocity (rotation only, 2D)
        vel_lidar = None
        if det.velocity is not None:
            vel_3d = np.array([det.velocity[0], det.velocity[1], 0.0])
            vel_3d_lidar = rot3x3 @ vel_3d
            vel_lidar = vel_3d_lidar[:2]

        transformed.append(Detection3D(
            box=box_lidar,
            score=det.score,
            label=det.label,
            label_name=det.label_name,
            velocity=vel_lidar,
        ))
    return transformed


def _add_wireframe_boxes(
    fig: go.Figure,
    boxes_7dof: np.ndarray,
    labels: np.ndarray,
    color_override: str = None,
    dash: str = 'solid',
    name_prefix: str = 'Box',
    scores: np.ndarray = None,
):
    """Add wireframe boxes to a Plotly figure (same approach as Lidar/visualize_3d.py)."""
    if len(boxes_7dof) == 0:
        return
    corners = boxes_to_corners_3d(boxes_7dof)
    for i in range(len(corners)):
        cls_idx = int(labels[i]) if labels is not None else 0
        color = color_override or PLOTLY_CLASS_COLORS[cls_idx % len(PLOTLY_CLASS_COLORS)]
        cls_name = NUSCENES_CLASSES[cls_idx] if cls_idx < len(NUSCENES_CLASSES) else f'c{cls_idx}'
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
            showlegend=(i == 0),
        ))


def generate_interactive_3d(
    nusc: NuScenes,
    sample_idx: int,
    detections_radar_frame: List[Detection3D],
    output_dir: Path,
    data_root: str,
    max_lidar_points: int = 30000,
):
    """Generate interactive 3D HTML showing radar detections on LiDAR point cloud.

    All data is transformed to the LIDAR_TOP coordinate frame.

    Args:
        nusc: NuScenes instance.
        sample_idx: Index into nusc.sample[].
        detections_radar_frame: CFAR+DBSCAN detections in RADAR_FRONT frame.
        output_dir: Directory to save HTML files.
        data_root: NuScenes data root path.
        max_lidar_points: Max LiDAR points for Plotly (downsample for performance).
    """
    sample = nusc.sample[sample_idx]
    sample_token = sample['token']

    # --- 1) Load LiDAR point cloud (already in LIDAR_TOP frame) ---
    lidar_points = load_multi_sweep_points(
        nusc, sample_token, Path(data_root), sweeps_num=0
    )  # (N, 5): x, y, z, intensity, time_lag

    # --- 2) Compute RADAR_FRONT -> LIDAR_TOP transform ---
    radar_to_lidar = _compute_radar_to_lidar_transform(nusc, sample_token)

    # --- 3) Load radar points and transform to LiDAR frame ---
    radar_raw = load_radar_points(nusc, sample, sensors=['RADAR_FRONT'])
    radar_raw = filter_radar_quality(radar_raw)
    radar_sel = select_features(radar_raw) if len(radar_raw) > 0 else np.zeros((0, 6), dtype=np.float32)
    # radar_sel: (N, 6) = [x, y, z, rcs, vx_comp, vy_comp] in RADAR_FRONT frame

    radar_lidar = np.zeros_like(radar_sel)
    if len(radar_sel) > 0:
        # Transform xyz
        xyz_hom = np.hstack([radar_sel[:, :3], np.ones((len(radar_sel), 1))])
        radar_lidar[:, :3] = (radar_to_lidar @ xyz_hom.T).T[:, :3]
        # Keep RCS
        radar_lidar[:, 3] = radar_sel[:, 3]
        # Rotate velocity vectors
        rot3x3 = radar_to_lidar[:3, :3]
        for i in range(len(radar_sel)):
            v3 = np.array([radar_sel[i, 4], radar_sel[i, 5], 0.0])
            v3_lidar = rot3x3 @ v3
            radar_lidar[i, 4:6] = v3_lidar[:2]

    # --- 4) Load GT annotations in LIDAR_TOP frame ---
    gt_boxes_lidar, gt_labels = _load_gt_in_lidar_frame(nusc, sample_token)

    # --- 5) Transform radar detections to LiDAR frame ---
    dets_lidar = _transform_detections_to_lidar(detections_radar_frame, radar_to_lidar)

    # --- 6) Build Plotly figure ---
    fig = go.Figure()

    # Trace 1: LiDAR point cloud (gray, small, semi-transparent)
    if len(lidar_points) > max_lidar_points:
        idx = np.random.choice(len(lidar_points), max_lidar_points, replace=False)
        pts = lidar_points[idx]
    else:
        pts = lidar_points

    fig.add_trace(go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        mode='markers',
        marker=dict(
            size=1, color=pts[:, 2], colorscale='Viridis',
            cmin=pts[:, 2].min(), cmax=pts[:, 2].max(), opacity=0.3,
        ),
        name=f'LiDAR ({len(pts):,} pts)',
        hoverinfo='skip',
    ))

    # Trace 2: Radar points (bright green, larger, RCS-based sizing)
    if len(radar_lidar) > 0:
        rcs = radar_lidar[:, 3]
        rcs_clipped = np.clip(rcs, -10, 40)
        sizes = (rcs_clipped + 10) / 50 * 8 + 3  # range 3-11

        fig.add_trace(go.Scatter3d(
            x=radar_lidar[:, 0], y=radar_lidar[:, 1], z=radar_lidar[:, 2],
            mode='markers',
            marker=dict(
                size=sizes, color='lime', opacity=0.8,
                line=dict(width=0.5, color='darkgreen'),
            ),
            name=f'Radar ({len(radar_lidar)} pts)',
            text=[f'RCS={r:.1f} dBsm' for r in rcs],
            hoverinfo='text',
        ))

    # Trace 3: GT boxes (green dashed wireframes)
    if len(gt_boxes_lidar) > 0:
        _add_wireframe_boxes(
            fig, gt_boxes_lidar, gt_labels,
            color_override='rgba(50,255,50,0.7)',
            dash='dash', name_prefix='GT',
        )

    # Trace 4: Radar detection boxes (class-colored solid wireframes)
    if dets_lidar:
        det_boxes = np.array([d.box for d in dets_lidar], dtype=np.float32)
        det_labels = np.array([d.label for d in dets_lidar], dtype=np.int64)
        det_scores = np.array([d.score for d in dets_lidar], dtype=np.float32)
        _add_wireframe_boxes(
            fig, det_boxes, det_labels,
            dash='solid', name_prefix='Radar Det',
            scores=det_scores,
        )

    # Trace 5: Velocity arrows for radar detections
    for det in dets_lidar:
        if det.velocity is not None and np.linalg.norm(det.velocity) > 0.5:
            cx, cy, cz = det.box[0], det.box[1], det.box[2]
            vx, vy = det.velocity[0], det.velocity[1]
            scale = 2.0
            fig.add_trace(go.Scatter3d(
                x=[cx, cx + vx * scale],
                y=[cy, cy + vy * scale],
                z=[cz, cz],
                mode='lines+markers',
                line=dict(color='red', width=4),
                marker=dict(size=[0, 3], color='red', symbol='diamond'),
                name='Velocity',
                showlegend=False,
            ))

    # Layout
    fig.update_layout(
        scene=dict(
            xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)',
            aspectmode='data',
            bgcolor='rgb(20,20,30)',
        ),
        title=dict(
            text=f'Sample {sample_idx} — Radar + LiDAR 3D '
                 f'| {len(radar_lidar)} radar pts | {len(dets_lidar)} dets '
                 f'| {len(gt_boxes_lidar)} GT',
            font=dict(size=16),
        ),
        paper_bgcolor='rgb(15,15,25)',
        font=dict(color='white'),
        legend=dict(font=dict(size=10)),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    html_path = output_dir / f'radar_lidar_3d_{sample_idx}.html'
    fig.write_html(str(html_path), include_plotlyjs='cdn')
    print(f"  Saved 3D interactive: {html_path.name}")
    return html_path


def _load_gt_in_lidar_frame(
    nusc: NuScenes, sample_token: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load GT annotations transformed to LIDAR_TOP frame.

    Returns:
        gt_boxes: (M, 7) float32 [x, y, z, l, w, h, yaw]
        gt_labels: (M,) int64
    """
    sample = nusc.get('sample', sample_token)
    lidar_sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    lidar_cs = nusc.get('calibrated_sensor', lidar_sd['calibrated_sensor_token'])
    lidar_ego = nusc.get('ego_pose', lidar_sd['ego_pose_token'])

    lidar_to_ego = make_transform_matrix(lidar_cs['translation'], lidar_cs['rotation'])
    ego_to_global = make_transform_matrix(lidar_ego['translation'], lidar_ego['rotation'])
    ego_inv = np.linalg.inv(ego_to_global)
    lidar_inv = np.linalg.inv(lidar_to_ego)

    _CATEGORY_MAP = {
        'vehicle.car': (0, 'car'),
        'vehicle.truck': (1, 'truck'),
        'vehicle.construction': (2, 'construction_vehicle'),
        'vehicle.bus.bendy': (3, 'bus'),
        'vehicle.bus.rigid': (3, 'bus'),
        'vehicle.trailer': (4, 'trailer'),
        'movable_object.barrier': (5, 'barrier'),
        'vehicle.motorcycle': (6, 'motorcycle'),
        'vehicle.bicycle': (7, 'bicycle'),
        'human.pedestrian': (8, 'pedestrian'),
        'movable_object.trafficcone': (9, 'traffic_cone'),
    }

    gt_boxes, gt_labels = [], []
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        category = ann['category_name']
        label = None
        for prefix, (lbl, _) in _CATEGORY_MAP.items():
            if category.startswith(prefix):
                label = lbl
                break
        if label is None:
            continue

        # Global -> ego -> lidar
        center_global = np.array(ann['translation'])
        rot_global = Quaternion(ann['rotation'])

        center_ego = ego_inv[:3, :3] @ center_global + ego_inv[:3, 3]
        rot_ego = Quaternion(matrix=ego_inv[:3, :3]) * rot_global

        center_lidar = lidar_inv[:3, :3] @ center_ego + lidar_inv[:3, 3]
        rot_lidar = Quaternion(matrix=lidar_inv[:3, :3]) * rot_ego

        w, l, h = ann['size']  # NuScenes: [w, l, h]
        yaw = rot_lidar.yaw_pitch_roll[0]

        gt_boxes.append([center_lidar[0], center_lidar[1], center_lidar[2],
                         l, w, h, yaw])
        gt_labels.append(label)

    if gt_boxes:
        return np.array(gt_boxes, dtype=np.float32), np.array(gt_labels, dtype=np.int64)
    return np.zeros((0, 7), dtype=np.float32), np.zeros(0, dtype=np.int64)


def main():
    """Main execution function."""
    print("=" * 80)
    print("NuScenes Radar Integration Test")
    print("=" * 80)

    # Setup paths
    output_dir = PROJECT_ROOT / 'Radar' / 'outputs' / 'nuscenes_test'
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Load NuScenes
    print("\nLoading NuScenes...")
    try:
        data_root = str(path_manager.get('nuscenes'))
        nusc = NuScenes(version='v1.0-mini', dataroot=data_root, verbose=False)
        print(f"Loaded {len(nusc.sample)} samples from {data_root}")
    except Exception as e:
        print(f"Error loading NuScenes: {e}")
        return

    # Initialize detectors
    print("\nInitializing detectors...")
    detectors = {}
    for name, config in DETECTOR_CONFIGS.items():
        try:
            # Make a copy to avoid mutating DETECTOR_CONFIGS
            config_copy = config.copy()
            dtype = config_copy.pop('detector_type')
            detector = get_radar_detector(dtype, **config_copy)
            detectors[name] = detector
            print(f"  ✓ {name}")
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            continue

    if not detectors:
        print("Error: No detectors initialized successfully")
        return

    # Test samples
    print(f"\nTesting on {len(TEST_SAMPLES)} samples: {TEST_SAMPLES}")

    # Storage for results
    all_bev_results = {name: {} for name in detectors.keys()}
    summary_data = []

    # Process each sample
    for sample_idx in TEST_SAMPLES:
        if sample_idx >= len(nusc.sample):
            print(f"Warning: Sample index {sample_idx} out of range, skipping")
            continue

        sample = nusc.sample[sample_idx]
        sample_token = sample['token']

        print(f"\n{'=' * 80}")
        print(f"Processing Sample {sample_idx} (token: {sample_token[:8]}...)")
        print(f"{'=' * 80}")

        # Load radar points
        try:
            radar_points = load_radar_points(nusc, sample)
            radar_points = filter_radar_quality(radar_points)
            n_points = len(radar_points)
            print(f"Loaded {n_points} radar points (after quality filter)")
        except Exception as e:
            print(f"Error loading radar points: {e}")
            continue

        if n_points == 0:
            print("Warning: No valid radar points, skipping")
            continue

        # Load GT annotations
        try:
            gt_boxes = load_gt_annotations(nusc, sample_token)
            print(f"Loaded {len(gt_boxes)} ground truth annotations")
        except Exception as e:
            print(f"Error loading GT: {e}")
            gt_boxes = []

        # Process with each detector
        sample_summary = {
            'sample_idx': sample_idx,
            'n_points': n_points,
            'n_gt': len(gt_boxes),
        }

        for detector_name, detector in detectors.items():
            print(f"\n  Running {detector_name}...")

            # Run detection
            try:
                start_time = time.time()

                # Feature selection: DL models need 6 selected features, CFAR+DBSCAN uses raw
                if detector_name in ('RadarPillars', 'RadarCenterPoint'):
                    det_points = select_features(radar_points) if len(radar_points) > 0 else np.zeros((0, 6), dtype=np.float32)
                else:
                    det_points = radar_points

                detections = detector.detect(det_points)
                latency_ms = (time.time() - start_time) * 1000

                print(f"    Detections: {len(detections)}, Latency: {latency_ms:.2f} ms")
                sample_summary[f'{detector_name}_dets'] = len(detections)
                sample_summary[f'{detector_name}_latency'] = latency_ms
            except Exception as e:
                print(f"    Detection error: {e}")
                detections = []
                sample_summary[f'{detector_name}_dets'] = 0
                sample_summary[f'{detector_name}_latency'] = 0.0

            # Generate BEV plot (separate try to preserve detection stats)
            try:
                viz_points = select_features(radar_points) if len(radar_points) > 0 else np.zeros((0, 6), dtype=np.float32)

                title = f"Sample {sample_idx} | {detector_name} | Points: {n_points} | Dets: {len(detections)} | {latency_ms:.2f}ms"
                bev_path = output_dir / f"bev_{detector_name.replace('+', '_')}_{sample_idx}.png"

                draw_bev_plot(
                    radar_points=viz_points,
                    gt_boxes=gt_boxes,
                    detections=detections,
                    title=title,
                    output_path=bev_path,
                )

                all_bev_results[detector_name][sample_idx] = bev_path
                print(f"    Saved BEV: {bev_path.name}")
            except Exception as e:
                print(f"    BEV plot error: {e}")

        # Generate camera image with detections (using first detector's results)
        try:
            first_detector = list(detectors.keys())[0]
            cam_path = output_dir / f"cam_front_{sample_idx}.png"

            # Get detections from first detector - apply feature selection if needed
            if first_detector in ('RadarPillars', 'RadarCenterPoint'):
                det_points = select_features(radar_points) if len(radar_points) > 0 else np.zeros((0, 6), dtype=np.float32)
            else:
                det_points = radar_points

            first_dets = detectors[first_detector].detect(det_points)
            draw_camera_with_detections(nusc, sample_token, first_dets, cam_path)
            print(f"\n  Saved camera: {cam_path.name}")
        except Exception as e:
            print(f"  Warning: Could not generate camera image: {e}")

        summary_data.append(sample_summary)

    # Create comparison grid
    print(f"\n{'=' * 80}")
    print("Creating comparison grid...")
    try:
        grid_path = output_dir / 'comparison_grid.png'
        create_comparison_grid(all_bev_results, grid_path)
        print(f"Saved: {grid_path}")
    except Exception as e:
        print(f"Error creating grid: {e}")

    # Generate interactive 3D Radar+LiDAR visualizations (CFAR+DBSCAN only)
    print(f"\n{'=' * 80}")
    print("Generating interactive 3D Radar+LiDAR visualizations...")
    if 'CFAR+DBSCAN' in detectors:
        cfar_detector = detectors['CFAR+DBSCAN']
        for sample_idx in TEST_SAMPLES:
            if sample_idx >= len(nusc.sample):
                continue
            try:
                sample = nusc.sample[sample_idx]
                # Load radar points and run CFAR+DBSCAN
                radar_pts = load_radar_points(nusc, sample)
                radar_pts = filter_radar_quality(radar_pts)
                cfar_dets = cfar_detector.detect(radar_pts)

                generate_interactive_3d(
                    nusc=nusc,
                    sample_idx=sample_idx,
                    detections_radar_frame=cfar_dets,
                    output_dir=output_dir,
                    data_root=data_root,
                )
            except Exception as e:
                print(f"  Error generating 3D for sample {sample_idx}: {e}")
                import traceback
                traceback.print_exc()
    else:
        print("  Skipped: CFAR+DBSCAN detector not available")

    # Generate summary table
    print(f"\n{'=' * 80}")
    print("SUMMARY STATISTICS")
    print(f"{'=' * 80}")

    # Print table header
    detector_names = list(detectors.keys())
    header = f"{'Sample':<8} {'Points':<8} {'GT':<5}"
    for name in detector_names:
        header += f" {name[:10]:<12} {name[:10]+'_ms':<10}"
    print(header)
    print("-" * len(header))

    # Print rows
    total_dets = {name: 0 for name in detector_names}
    total_latency = {name: 0.0 for name in detector_names}
    class_distribution = {name: {cls: 0 for cls in NUSCENES_CLASSES} for name in detector_names}

    for row in summary_data:
        line = f"{row['sample_idx']:<8} {row['n_points']:<8} {row['n_gt']:<5}"
        for name in detector_names:
            dets = row.get(f'{name}_dets', 0)
            lat = row.get(f'{name}_latency', 0.0)
            line += f" {dets:<12} {lat:<10.2f}"
            total_dets[name] += dets
            total_latency[name] += lat
        print(line)

    print("-" * len(header))

    # Print totals
    n_samples = len(summary_data)
    print(f"\nTotal Samples: {n_samples}")
    print(f"\nDetector Performance:")
    for name in detector_names:
        avg_dets = total_dets[name] / max(n_samples, 1)
        avg_lat = total_latency[name] / max(n_samples, 1)
        print(f"  {name:<20}: {total_dets[name]} total dets, {avg_dets:.1f} avg/sample, {avg_lat:.2f} ms avg latency")

    # Save summary to file
    summary_path = output_dir / 'summary.txt'
    with open(summary_path, 'w') as f:
        f.write("NuScenes Radar Integration Test Summary\n")
        f.write("=" * 80 + "\n\n")
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for row in summary_data:
            line = f"{row['sample_idx']:<8} {row['n_points']:<8} {row['n_gt']:<5}"
            for name in detector_names:
                dets = row.get(f'{name}_dets', 0)
                lat = row.get(f'{name}_latency', 0.0)
                line += f" {dets:<12} {lat:<10.2f}"
            f.write(line + "\n")
        f.write("-" * len(header) + "\n\n")
        f.write(f"Total Samples: {n_samples}\n\n")
        f.write("Detector Performance:\n")
        for name in detector_names:
            avg_dets = total_dets[name] / max(n_samples, 1)
            avg_lat = total_latency[name] / max(n_samples, 1)
            f.write(f"  {name:<20}: {total_dets[name]} total dets, {avg_dets:.1f} avg/sample, {avg_lat:.2f} ms avg latency\n")

    print(f"\nSaved summary: {summary_path}")
    print(f"\n{'=' * 80}")
    print("Test completed successfully!")
    print(f"All outputs saved to: {output_dir}")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
