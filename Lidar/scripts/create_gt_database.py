#!/usr/bin/env python3
"""
Create GT Database for GT-Sampling Augmentation

Scans the training set, crops the points inside each GT box,
and saves per-object .bin files + a gt_database_info.pkl index.

Usage:
    python scripts/create_gt_database.py \
        --data-root ../Fusion/data/sets/nuscenes
"""

import argparse
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm


CLASS_NAMES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]


def points_in_box(points: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Return mask of points inside a 3D box (x, y, z, l, w, h, yaw).

    Uses a rotation-aware check in BEV + height check.
    """
    cx, cy, cz, l, w, h, yaw = box

    # Translate to box center
    dx = points[:, 0] - cx
    dy = points[:, 1] - cy
    dz = points[:, 2] - cz

    # Rotate into box-local frame
    cos_y = np.cos(-yaw)
    sin_y = np.sin(-yaw)
    local_x = cos_y * dx - sin_y * dy
    local_y = sin_y * dx + cos_y * dy

    # Check bounds
    mask = (
        (np.abs(local_x) <= l / 2) &
        (np.abs(local_y) <= w / 2) &
        (np.abs(dz) <= h / 2)
    )
    return mask


def parse_args():
    parser = argparse.ArgumentParser(description='Create GT database for NuScenes')
    parser.add_argument('--data-root', required=True, help='NuScenes root directory')
    parser.add_argument('--info-path', default=None,
                        help='Path to train info pkl (default: <data-root>/nuscenes_infos_train.pkl)')
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root)

    info_path = Path(args.info_path) if args.info_path else data_root / 'nuscenes_infos_train.pkl'
    if not info_path.exists():
        print(f"Error: info file not found at {info_path}")
        print("Run prepare_data.py first.")
        return

    # Load training info
    with open(info_path, 'rb') as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        infos = data.get('data_list', data.get('infos', []))
    else:
        infos = data

    print(f"Processing {len(infos)} training samples...")

    # Create output directory
    db_dir = data_root / 'gt_database'
    db_dir.mkdir(parents=True, exist_ok=True)

    # Collect per-class database info
    db_infos = {name: [] for name in CLASS_NAMES}
    total_objects = 0

    for sample_idx, info in enumerate(tqdm(infos, desc="Extracting GT objects")):
        # Load point cloud
        if 'lidar_points' in info:
            lidar_path = info['lidar_points']['lidar_path']
        elif 'lidar_path' in info:
            lidar_path = info['lidar_path']
        else:
            continue

        full_path = data_root / lidar_path
        if not full_path.exists():
            full_path = Path(lidar_path)
        if not full_path.exists():
            continue

        points = np.fromfile(str(full_path), dtype=np.float32).reshape(-1, 5)[:, :4]

        # Process annotations
        if 'instances' not in info:
            continue

        for obj_idx, instance in enumerate(info['instances']):
            if not instance.get('bbox_3d_isvalid', True):
                continue

            box = np.array(instance['bbox_3d'], dtype=np.float32)
            label = instance['bbox_label_3d']

            if label < 0 or label >= len(CLASS_NAMES):
                continue

            cls_name = CLASS_NAMES[label]

            # Crop points inside this box
            mask = points_in_box(points, box)
            obj_points = points[mask]

            if len(obj_points) == 0:
                continue

            # Save object points
            filename = f'{cls_name}_{sample_idx:04d}_{obj_idx:03d}.bin'
            rel_path = f'gt_database/{filename}'
            obj_points.astype(np.float32).tofile(str(data_root / rel_path))

            # Record info
            db_infos[cls_name].append({
                'class': cls_name,
                'label': label,
                'box': box.tolist(),
                'num_points': len(obj_points),
                'path': rel_path,
            })
            total_objects += 1

    # Save database info
    info_save_path = data_root / 'gt_database_info.pkl'
    with open(info_save_path, 'wb') as f:
        pickle.dump(db_infos, f)

    # Print summary
    print(f"\nGT Database created: {total_objects} objects total")
    for cls_name in CLASS_NAMES:
        count = len(db_infos[cls_name])
        if count > 0:
            avg_pts = np.mean([info['num_points'] for info in db_infos[cls_name]])
            print(f"  {cls_name:25s}: {count:4d} objects, avg {avg_pts:.0f} points")
    print(f"\nSaved to: {db_dir}")
    print(f"Info file: {info_save_path}")


if __name__ == '__main__':
    main()
