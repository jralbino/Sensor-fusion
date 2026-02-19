# -*- coding: utf-8 -*-
"""
NuScenes radar dataset for training and evaluation.

Loads multi-sweep radar point clouds with ground-truth 3D annotations.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from Radar.src.data.radar_utils import (
    USEFUL_FEATURE_INDICES,
    ALL_RADAR_SENSORS,
    load_radar_points,
    filter_radar_quality,
    filter_radar_rcs,
    select_features,
    transform_velocity_augmentation,
)

logger = logging.getLogger(__name__)


class RadarNuScenesDataset(Dataset):
    """NuScenes radar dataset.

    Args:
        data_root: Path to NuScenes data root.
        version: NuScenes version (e.g., 'v1.0-mini').
        split: 'train' or 'val'.
        nsweeps: Number of radar sweeps to accumulate.
        sensors: Radar sensors to use (default: all 5).
        point_cloud_range: Detection range [x_min, y_min, z_min, x_max, y_max, z_max].
        feature_indices: Which radar features to select. Defaults to USEFUL_FEATURE_INDICES.
        min_rcs: Minimum RCS filter (dBsm). Set to None to skip.
        augment: Whether to apply data augmentation (train only).
    """

    CLASS_NAMES = [
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
        'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone',
    ]
    CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

    def __init__(
        self,
        data_root: str,
        version: str = 'v1.0-mini',
        split: str = 'train',
        nsweeps: int = 6,
        sensors: Optional[List[str]] = None,
        point_cloud_range: Tuple[float, ...] = (-100, -100, -5, 100, 100, 3),
        feature_indices: Optional[List[int]] = None,
        min_rcs: Optional[float] = -5.0,
        augment: bool = False,
    ):
        from nuscenes.nuscenes import NuScenes
        from nuscenes.utils.splits import create_splits_scenes

        self.data_root = data_root
        self.nsweeps = nsweeps
        self.sensors = sensors or ALL_RADAR_SENSORS
        self.point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
        self.feature_indices = feature_indices or USEFUL_FEATURE_INDICES
        self.min_rcs = min_rcs
        self.augment = augment

        # Compute velocity column indices in the *selected* feature array
        self._vx_col = self.feature_indices.index(8) if 8 in self.feature_indices else None
        self._vy_col = self.feature_indices.index(9) if 9 in self.feature_indices else None

        self.nusc = NuScenes(version=version, dataroot=data_root, verbose=False)

        # Build sample list from split
        splits = create_splits_scenes()
        split_key = 'mini_train' if 'mini' in version and split == 'train' else \
                     'mini_val' if 'mini' in version and split == 'val' else split
        scene_names = set(splits.get(split_key, []))

        self.samples = []
        for sample in self.nusc.sample:
            scene = self.nusc.get('scene', sample['scene_token'])
            if scene['name'] in scene_names:
                self.samples.append(sample)

        logger.info(
            "RadarNuScenesDataset: %s split, %d samples, %d sensors, %d sweeps",
            split, len(self.samples), len(self.sensors), self.nsweeps,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        # Load multi-sweep radar points (N, 19)
        raw_points = load_radar_points(
            self.nusc, sample, sensors=self.sensors, nsweeps=self.nsweeps,
        )

        # Quality + RCS filtering (on full 18-feature points)
        if len(raw_points) > 0:
            raw_points = filter_radar_quality(raw_points)
            if self.min_rcs is not None and len(raw_points) > 0:
                raw_points = filter_radar_rcs(raw_points, self.min_rcs)

        # Select useful features
        if len(raw_points) > 0:
            points = select_features(raw_points, self.feature_indices)
        else:
            points = np.zeros((0, len(self.feature_indices)), dtype=np.float32)

        # Range filter
        points = self._filter_range(points)

        # Load GT annotations
        gt_boxes, gt_labels = self._load_gt(sample)

        # Data augmentation
        if self.augment and len(points) > 0:
            points, gt_boxes = self._augment(points, gt_boxes)

        return {
            'points': points.astype(np.float32),
            'gt_boxes': gt_boxes.astype(np.float32),
            'gt_labels': gt_labels.astype(np.int64),
            'sample_token': sample['token'],
        }

    def _filter_range(self, points: np.ndarray) -> np.ndarray:
        if len(points) == 0:
            return points
        pc = self.point_cloud_range
        mask = (
            (points[:, 0] >= pc[0]) & (points[:, 0] <= pc[3])
            & (points[:, 1] >= pc[1]) & (points[:, 1] <= pc[4])
        )
        # z filter only if we have z column
        if points.shape[1] > 2:
            mask &= (points[:, 2] >= pc[2]) & (points[:, 2] <= pc[5])
        return points[mask]

    def _load_gt(self, sample: dict) -> Tuple[np.ndarray, np.ndarray]:
        """Load ground-truth 3D boxes from NuScenes annotations."""
        from nuscenes.utils.data_classes import Box
        from pyquaternion import Quaternion

        boxes_list = []
        labels_list = []

        for ann_token in sample['anns']:
            ann = self.nusc.get('sample_annotation', ann_token)
            cat = ann['category_name']

            # Map NuScenes category to our class index
            label = self._category_to_label(cat)
            if label is None:
                continue

            # Get box in global frame
            box = Box(
                ann['translation'], ann['size'], Quaternion(ann['rotation']),
            )

            # Transform to ego frame at sample timestamp
            sd = self.nusc.get('sample_data', sample['data']['RADAR_FRONT'])
            cs = self.nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
            ep = self.nusc.get('ego_pose', sd['ego_pose_token'])

            # Global → ego
            box.translate(-np.array(ep['translation']))
            box.rotate(Quaternion(ep['rotation']).inverse)

            # Ego → sensor
            box.translate(-np.array(cs['translation']))
            box.rotate(Quaternion(cs['rotation']).inverse)

            # Range filter
            if not self._box_in_range(box):
                continue

            # [x, y, z, l, w, h, yaw]
            yaw = box.orientation.yaw_pitch_roll[0]
            gt_box = np.array([
                box.center[0], box.center[1], box.center[2],
                box.wlh[1], box.wlh[0], box.wlh[2],  # NuScenes: w,l,h → l,w,h
                yaw,
            ], dtype=np.float32)

            boxes_list.append(gt_box)
            labels_list.append(label)

        if not boxes_list:
            return np.zeros((0, 7), dtype=np.float32), np.zeros((0,), dtype=np.int64)

        return np.stack(boxes_list), np.array(labels_list, dtype=np.int64)

    def _category_to_label(self, category: str) -> Optional[int]:
        """Map NuScenes category string to class index."""
        mapping = {
            'vehicle.car': 0,
            'vehicle.truck': 1,
            'vehicle.construction': 2,
            'vehicle.bus.bendy': 3,
            'vehicle.bus.rigid': 3,
            'vehicle.trailer': 4,
            'movable_object.barrier': 5,
            'vehicle.motorcycle': 6,
            'vehicle.bicycle': 7,
            'human.pedestrian': 8,
            'movable_object.trafficcone': 9,
        }
        for prefix, label in mapping.items():
            if category.startswith(prefix):
                return label
        return None

    def _box_in_range(self, box) -> bool:
        pc = self.point_cloud_range
        c = box.center
        return pc[0] <= c[0] <= pc[3] and pc[1] <= c[1] <= pc[4]

    def _augment(
        self, points: np.ndarray, gt_boxes: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply random augmentations (flip, rotation, scaling).

        Velocity columns are identified dynamically from ``self._vx_col``
        and ``self._vy_col`` so indexing is correct regardless of feature selection.
        """
        vx_i = self._vx_col
        vy_i = self._vy_col

        # Random horizontal flip
        if np.random.rand() < 0.5:
            points[:, 1] *= -1
            if len(gt_boxes) > 0:
                gt_boxes[:, 1] *= -1
                gt_boxes[:, 6] *= -1  # flip yaw
            if vy_i is not None:
                points[:, vy_i] *= -1

        # Random global rotation
        angle = np.random.uniform(-np.pi / 4, np.pi / 4)
        cos_a, sin_a = np.cos(angle), np.sin(angle)

        # Rotate positions
        xy = points[:, :2].copy()
        points[:, 0] = xy[:, 0] * cos_a - xy[:, 1] * sin_a
        points[:, 1] = xy[:, 0] * sin_a + xy[:, 1] * cos_a
        if len(gt_boxes) > 0:
            gt_xy = gt_boxes[:, :2].copy()
            gt_boxes[:, 0] = gt_xy[:, 0] * cos_a - gt_xy[:, 1] * sin_a
            gt_boxes[:, 1] = gt_xy[:, 0] * sin_a + gt_xy[:, 1] * cos_a
            gt_boxes[:, 6] += angle

        # Rotate velocities
        if vx_i is not None and vy_i is not None:
            vel = np.stack([points[:, vx_i], points[:, vy_i]], axis=1).copy()
            points[:, vx_i] = vel[:, 0] * cos_a - vel[:, 1] * sin_a
            points[:, vy_i] = vel[:, 0] * sin_a + vel[:, 1] * cos_a

        # Random scaling
        scale = np.random.uniform(0.95, 1.05)
        points[:, :3] *= scale
        if len(gt_boxes) > 0:
            gt_boxes[:, :3] *= scale
            gt_boxes[:, 3:6] *= scale

        return points, gt_boxes

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        """Custom collate for variable-size point clouds and GT boxes."""
        points_list = [b['points'] for b in batch]
        gt_boxes_list = [torch.from_numpy(b['gt_boxes']) for b in batch]
        gt_labels_list = [torch.from_numpy(b['gt_labels']) for b in batch]
        tokens = [b['sample_token'] for b in batch]

        # Pad points to same size
        max_pts = max(len(p) for p in points_list)
        C = points_list[0].shape[1] if len(points_list[0]) > 0 else 6
        padded = np.zeros((len(batch), max_pts, C), dtype=np.float32)
        for i, pts in enumerate(points_list):
            if len(pts) > 0:
                padded[i, :len(pts)] = pts

        return {
            'points': torch.from_numpy(padded),
            'gt_boxes': gt_boxes_list,
            'gt_labels': gt_labels_list,
            'batch_size': len(batch),
            'sample_tokens': tokens,
        }
