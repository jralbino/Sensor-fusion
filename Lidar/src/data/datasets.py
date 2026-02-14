"""
NuScenes Dataset with Voxelization - FIXED VERSION

Properly voxelizes point clouds for PointPillars.
"""

import numpy as np
import pickle
from pathlib import Path
from torch.utils.data import Dataset
import torch
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def voxelize_points(
    points: np.ndarray,
    voxel_size: np.ndarray,
    point_cloud_range: np.ndarray,
    max_points_per_voxel: int = 32,
    max_voxels: int = 16000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Voxelize point cloud.

    Args:
        points: [N, 4] points (x, y, z, intensity)
        voxel_size: [3] voxel size (x, y, z)
        point_cloud_range: [6] detection range
        max_points_per_voxel: Maximum points per voxel
        max_voxels: Maximum number of voxels

    Returns:
        voxels: [M, max_points_per_voxel, C] voxel features (C = points.shape[1])
        coords: [M, 3] voxel coordinates (z, y, x)
        num_points: [M] number of points per voxel
    """
    # Filter points by range
    valid_mask = (
        (points[:, 0] >= point_cloud_range[0]) & (points[:, 0] <= point_cloud_range[3]) &
        (points[:, 1] >= point_cloud_range[1]) & (points[:, 1] <= point_cloud_range[4]) &
        (points[:, 2] >= point_cloud_range[2]) & (points[:, 2] <= point_cloud_range[5])
    )
    points = points[valid_mask]

    # Compute voxel coordinates
    voxel_coords = np.floor(
        (points[:, :3] - point_cloud_range[:3]) / voxel_size
    ).astype(np.int32)

    # Create unique voxel mapping
    # Use a hash to identify unique voxels
    grid_size = np.round(
        (point_cloud_range[3:6] - point_cloud_range[0:3]) / voxel_size
    ).astype(np.int64)
    voxel_hash = (
        voxel_coords[:, 0].astype(np.int64) * (grid_size[1] * grid_size[2]) +
        voxel_coords[:, 1].astype(np.int64) * grid_size[2] +
        voxel_coords[:, 2].astype(np.int64)
    )

    unique_hashes, inverse_indices = np.unique(voxel_hash, return_inverse=True)
    num_unique_voxels = len(unique_hashes)

    # Limit number of voxels
    if num_unique_voxels > max_voxels:
        keep_indices = np.random.choice(num_unique_voxels, max_voxels, replace=False)
        keep_mask = np.isin(inverse_indices, keep_indices)
        points = points[keep_mask]
        inverse_indices = inverse_indices[keep_mask]

        # Re-map inverse indices
        old_to_new = {old: new for new, old in enumerate(keep_indices)}
        inverse_indices = np.array([old_to_new[old] for old in inverse_indices])
        num_unique_voxels = max_voxels

    # Initialize voxel arrays
    n_features = points.shape[1]
    voxels = np.zeros((num_unique_voxels, max_points_per_voxel, n_features), dtype=np.float32)
    coords = np.zeros((num_unique_voxels, 3), dtype=np.int32)
    num_points_per_voxel = np.zeros(num_unique_voxels, dtype=np.int32)

    # Fill voxels
    for i, point in enumerate(points):
        voxel_idx = inverse_indices[i]
        point_idx = num_points_per_voxel[voxel_idx]

        if point_idx < max_points_per_voxel:
            voxels[voxel_idx, point_idx] = point
            num_points_per_voxel[voxel_idx] += 1

            # Store coordinates (only once per voxel)
            if point_idx == 0:
                coords[voxel_idx] = voxel_coords[i][[2, 1, 0]]  # z, y, x order

    # Filter out empty voxels
    non_empty_mask = num_points_per_voxel > 0
    voxels = voxels[non_empty_mask]
    coords = coords[non_empty_mask]
    num_points_per_voxel = num_points_per_voxel[non_empty_mask]

    return voxels, coords, num_points_per_voxel


# ---------------------------------------------------------------------------
# GT Database Sampler
# ---------------------------------------------------------------------------

class GTDatabaseSampler:
    """Paste GT objects from a pre-built database into training scenes.

    The database is created by ``scripts/create_gt_database.py`` and consists of:
    - ``gt_database/`` directory with per-object ``.bin`` point cloud files
    - ``gt_database_info.pkl`` with metadata (class, box, num_points, path)
    """

    def __init__(
        self,
        data_root: str,
        class_names: List[str],
        min_points: int = 5,
        sample_counts: Optional[Dict[str, int]] = None,
    ):
        self.data_root = Path(data_root)
        self.class_names = class_names
        self.min_points = min_points

        # Default: sample more rare classes, fewer common ones
        if sample_counts is None:
            sample_counts = {
                'car': 3, 'truck': 5, 'construction_vehicle': 7, 'bus': 5,
                'trailer': 5, 'barrier': 5, 'motorcycle': 7, 'bicycle': 7,
                'pedestrian': 5, 'traffic_cone': 5,
            }
        self.sample_counts = sample_counts

        # Load database info
        db_info_path = self.data_root / 'gt_database_info.pkl'
        if db_info_path.exists():
            with open(db_info_path, 'rb') as f:
                self.db_infos = pickle.load(f)
            # Filter by min points
            for cls_name in list(self.db_infos.keys()):
                self.db_infos[cls_name] = [
                    info for info in self.db_infos[cls_name]
                    if info['num_points'] >= self.min_points
                ]
            total = sum(len(v) for v in self.db_infos.values())
            logger.info(f"GT database loaded: {total} objects across {len(self.db_infos)} classes")
            self.enabled = True
        else:
            logger.warning(f"GT database not found at {db_info_path}, GT-sampling disabled")
            self.db_infos = {}
            self.enabled = False

    def sample(
        self,
        points: np.ndarray,
        gt_boxes: np.ndarray,
        gt_labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Add sampled GT objects to the scene.

        Args:
            points: (N, 4) scene points
            gt_boxes: (G, 7) existing GT boxes
            gt_labels: (G,) existing GT labels

        Returns:
            augmented points, gt_boxes, gt_labels
        """
        if not self.enabled:
            return points, gt_boxes, gt_labels

        new_boxes = []
        new_labels = []
        new_points_list = []

        for cls_idx, cls_name in enumerate(self.class_names):
            if cls_name not in self.db_infos or cls_name not in self.sample_counts:
                continue

            db_list = self.db_infos[cls_name]
            if len(db_list) == 0:
                continue

            # Count existing objects of this class
            existing_count = (gt_labels == cls_idx).sum() if len(gt_labels) > 0 else 0
            n_sample = max(0, self.sample_counts[cls_name] - existing_count)
            if n_sample == 0:
                continue

            # Random sample from database
            n_sample = min(n_sample, len(db_list))
            indices = np.random.choice(len(db_list), n_sample, replace=False)

            for idx in indices:
                info = db_list[idx]
                obj_points_path = self.data_root / info['path']
                if not obj_points_path.exists():
                    continue

                obj_points = np.fromfile(str(obj_points_path), dtype=np.float32).reshape(-1, 4)
                obj_box = np.array(info['box'], dtype=np.float32)

                # Check collision with existing boxes
                if len(gt_boxes) > 0 and self._check_collision(obj_box, gt_boxes):
                    continue
                if len(new_boxes) > 0 and self._check_collision(obj_box, np.array(new_boxes)):
                    continue

                new_boxes.append(obj_box)
                new_labels.append(cls_idx)
                new_points_list.append(obj_points)

        if len(new_boxes) > 0:
            # Remove scene points inside new GT boxes
            new_boxes_arr = np.array(new_boxes, dtype=np.float32)
            points = self._remove_points_in_boxes(points, new_boxes_arr)

            # Add new points
            all_new_points = np.concatenate(new_points_list, axis=0)
            points = np.concatenate([points, all_new_points], axis=0)

            # Update GT
            gt_boxes = np.concatenate([gt_boxes, new_boxes_arr], axis=0) if len(gt_boxes) > 0 else new_boxes_arr
            new_labels_arr = np.array(new_labels, dtype=np.int64)
            gt_labels = np.concatenate([gt_labels, new_labels_arr], axis=0) if len(gt_labels) > 0 else new_labels_arr

        return points, gt_boxes, gt_labels

    @staticmethod
    def _check_collision(box: np.ndarray, existing_boxes: np.ndarray, margin: float = 0.5) -> bool:
        """Check if box collides with any existing box (axis-aligned BEV check)."""
        # Simple center-distance check
        cx, cy = box[0], box[1]
        ex, ey = existing_boxes[:, 0], existing_boxes[:, 1]
        el, ew = existing_boxes[:, 3], existing_boxes[:, 4]

        bl, bw = box[3], box[4]

        dx = np.abs(cx - ex)
        dy = np.abs(cy - ey)

        # Collision if centers are closer than half-lengths + margin
        collide = (dx < (bl / 2 + el / 2 + margin)) & (dy < (bw / 2 + ew / 2 + margin))
        return collide.any()

    @staticmethod
    def _remove_points_in_boxes(points: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Remove points that fall inside any of the given boxes (axis-aligned approx)."""
        mask = np.ones(len(points), dtype=bool)
        for box in boxes:
            cx, cy, cz, l, w, h = box[0], box[1], box[2], box[3], box[4], box[5]
            in_box = (
                (np.abs(points[:, 0] - cx) < l / 2 + 0.1) &
                (np.abs(points[:, 1] - cy) < w / 2 + 0.1) &
                (np.abs(points[:, 2] - cz) < h / 2 + 0.1)
            )
            mask &= ~in_box
        return points[mask]


class NuScenesDataset(Dataset):
    """NuScenes dataset with proper voxelization."""

    def __init__(
        self,
        data_root: str,
        info_path: str,
        split: str = 'train',
        voxel_size: tuple = (0.16, 0.16, 4.0),
        point_cloud_range: tuple = (0, -39.68, -3, 69.12, 39.68, 1),
        max_points_per_voxel: int = 32,
        max_voxels: int = 16000,
        augmentation: bool = True
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.augmentation = augmentation and (split == 'train')
        self.voxel_size = np.array(voxel_size)
        self.point_cloud_range = np.array(point_cloud_range)
        self.max_points_per_voxel = max_points_per_voxel
        self.max_voxels = max_voxels

        # Load info file
        logger.info(f"Loading {split} data from {info_path}")
        with open(info_path, 'rb') as f:
            data = pickle.load(f)

        if isinstance(data, dict):
            self.infos = data.get('data_list', data.get('infos', []))
        else:
            self.infos = data

        self.class_names = [
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]

        # GT database sampler (only for training)
        self.gt_sampler = None
        if self.augmentation:
            self.gt_sampler = GTDatabaseSampler(
                data_root=data_root,
                class_names=self.class_names,
            )

        logger.info(f"Loaded {len(self.infos)} {split} samples")

    def __len__(self):
        return len(self.infos)

    def __getitem__(self, idx: int) -> Dict:
        info = self.infos[idx]

        # Load point cloud
        points = self._load_points(info)

        # Load ground truth
        gt_boxes, gt_labels = self._load_annotations(info)

        # GT-sampling augmentation (before other augmentations)
        if self.gt_sampler is not None and self.gt_sampler.enabled:
            points, gt_boxes, gt_labels = self.gt_sampler.sample(points, gt_boxes, gt_labels)

        # Apply augmentation
        if self.augmentation:
            points, gt_boxes = self._augment(points, gt_boxes)

        # Voxelize
        voxels, coords, num_points = voxelize_points(
            points,
            self.voxel_size,
            self.point_cloud_range,
            self.max_points_per_voxel,
            self.max_voxels
        )

        return {
            'voxels': torch.from_numpy(voxels).float(),
            'voxel_coords': torch.from_numpy(coords).int(),
            'voxel_num_points': torch.from_numpy(num_points).int(),
            'gt_boxes': torch.from_numpy(gt_boxes).float(),
            'gt_labels': torch.from_numpy(gt_labels).long(),
            'points': torch.from_numpy(points).float(),  # Keep for reference
        }

    def _load_points(self, info: Dict) -> np.ndarray:
        """Load point cloud from file."""
        if 'lidar_points' in info:
            lidar_path = info['lidar_points']['lidar_path']
        elif 'lidar_path' in info:
            lidar_path = info['lidar_path']
        else:
            raise KeyError("Cannot find lidar path")

        full_path = self.data_root / lidar_path

        if not full_path.exists():
            full_path = Path(lidar_path)

        if not full_path.exists():
            raise FileNotFoundError(f"Point cloud not found: {full_path}")

        points = np.fromfile(str(full_path), dtype=np.float32)
        points = points.reshape(-1, 5)[:, :4]  # x, y, z, intensity

        return points

    def _load_annotations(self, info: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Load ground truth boxes and labels."""
        gt_boxes = []
        gt_labels = []

        if 'instances' in info:
            for instance in info['instances']:
                if not instance.get('bbox_3d_isvalid', True):
                    continue

                box = instance['bbox_3d']
                label = instance['bbox_label_3d']

                if label < len(self.class_names):
                    gt_boxes.append(box)
                    gt_labels.append(label)

        elif 'gt_boxes' in info:
            gt_boxes = info['gt_boxes']
            gt_labels = info.get('gt_labels', np.zeros(len(gt_boxes)))

        if len(gt_boxes) == 0:
            return np.zeros((0, 7), dtype=np.float32), np.zeros(0, dtype=np.int64)

        return np.array(gt_boxes, dtype=np.float32), np.array(gt_labels, dtype=np.int64)

    def _augment(self, points: np.ndarray, gt_boxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply data augmentation."""
        # Random flip along Y axis
        if np.random.rand() < 0.5:
            points[:, 1] = -points[:, 1]
            if len(gt_boxes) > 0:
                gt_boxes[:, 1] = -gt_boxes[:, 1]
                gt_boxes[:, 6] = -gt_boxes[:, 6]

        # Random flip along X axis
        if np.random.rand() < 0.5:
            points[:, 0] = -points[:, 0]
            if len(gt_boxes) > 0:
                gt_boxes[:, 0] = -gt_boxes[:, 0]
                gt_boxes[:, 6] = np.pi - gt_boxes[:, 6]

        # Random rotation
        if np.random.rand() < 0.5:
            angle = np.random.uniform(-np.pi / 4, np.pi / 4)
            rot_mat = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)]
            ])
            points[:, :2] = points[:, :2] @ rot_mat.T
            if len(gt_boxes) > 0:
                gt_boxes[:, :2] = gt_boxes[:, :2] @ rot_mat.T
                gt_boxes[:, 6] += angle

        # Random scaling (wider range)
        if np.random.rand() < 0.5:
            scale = np.random.uniform(0.90, 1.10)
            points[:, :3] *= scale
            if len(gt_boxes) > 0:
                gt_boxes[:, :6] *= scale

        # Random translation
        if np.random.rand() < 0.5:
            tx = np.random.uniform(-0.5, 0.5)
            ty = np.random.uniform(-0.5, 0.5)
            points[:, 0] += tx
            points[:, 1] += ty
            if len(gt_boxes) > 0:
                gt_boxes[:, 0] += tx
                gt_boxes[:, 1] += ty

        return points, gt_boxes

    def collate_batch(self, batch: List[Dict]) -> Dict:
        """Collate batch."""
        batch_size = len(batch)

        # Collect voxels with batch indices
        voxels_list = []
        coords_list = []
        num_points_list = []

        for i, sample in enumerate(batch):
            voxels_list.append(sample['voxels'])

            # Add batch index to coordinates
            coords = sample['voxel_coords']
            batch_idx = torch.full((len(coords), 1), i, dtype=coords.dtype)
            coords_with_batch = torch.cat([batch_idx, coords], dim=1)
            coords_list.append(coords_with_batch)

            num_points_list.append(sample['voxel_num_points'])

        # Concatenate
        voxels = torch.cat(voxels_list, dim=0)
        coords = torch.cat(coords_list, dim=0)
        num_points = torch.cat(num_points_list, dim=0)

        # Ground truth (keep as list for now)
        gt_boxes_list = [sample['gt_boxes'] for sample in batch]
        gt_labels_list = [sample['gt_labels'] for sample in batch]

        return {
            'voxels': voxels,
            'voxel_coords': coords,
            'voxel_num_points': num_points,
            'batch_size': batch_size,
            'gt_boxes': gt_boxes_list,
            'gt_labels': gt_labels_list
        }


def create_dataloader(
    data_root: str,
    info_path: str,
    batch_size: int = 4,
    num_workers: int = 4,
    split: str = 'train',
    **kwargs
):
    """Create DataLoader."""
    from torch.utils.data import DataLoader

    dataset = NuScenesDataset(
        data_root=data_root,
        info_path=info_path,
        split=split,
        **kwargs
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        collate_fn=dataset.collate_batch,
        pin_memory=True,
        drop_last=(split == 'train')
    )

    return dataloader
