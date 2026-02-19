# -*- coding: utf-8 -*-
"""
Radar data loading and preprocessing utilities for NuScenes.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple

# NuScenes radar feature indices (18 features per point)
RADAR_FEATURE_NAMES = [
    'x', 'y', 'z',              # 0-2: position (m)
    'dyn_prop',                   # 3: dynamic property
    'id',                         # 4: radar id
    'rcs',                        # 5: radar cross section (dBsm)
    'vx', 'vy',                   # 6-7: velocity (m/s, global frame)
    'vx_comp', 'vy_comp',         # 8-9: compensated velocity
    'is_quality_valid',           # 10: quality flag
    'ambig_state',                # 11: ambiguity state
    'x_rms', 'y_rms',            # 12-13: position uncertainty
    'invalid_state',              # 14: invalid flag
    'pdh0',                       # 15: false alarm probability
    'vx_rms', 'vy_rms',          # 16-17: velocity uncertainty
]

# Most useful feature indices for detection
USEFUL_FEATURE_INDICES = [0, 1, 2, 5, 8, 9]  # x, y, z, rcs, vx_comp, vy_comp
USEFUL_FEATURE_NAMES = ['x', 'y', 'z', 'rcs', 'vx_comp', 'vy_comp']

# All 5 NuScenes radar sensors
ALL_RADAR_SENSORS = [
    'RADAR_FRONT',
    'RADAR_FRONT_LEFT',
    'RADAR_FRONT_RIGHT',
    'RADAR_BACK_LEFT',
    'RADAR_BACK_RIGHT',
]


def load_radar_points(
    nusc,
    sample_rec: dict,
    sensors: List[str] = None,
    nsweeps: int = 6,
    min_distance: float = 1.0,
) -> np.ndarray:
    """Load and merge radar points from multiple sensors with multi-sweep.

    Args:
        nusc: NuScenes instance.
        sample_rec: Sample record dict (from nusc.sample[i]).
        sensors: List of radar sensor names. Defaults to all 5.
        nsweeps: Number of sweeps to accumulate per sensor.
        min_distance: Minimum distance filter (m).

    Returns:
        (N, 19) array — 18 radar features + time_lag column.
    """
    from nuscenes.utils.data_classes import RadarPointCloud

    if sensors is None:
        sensors = ALL_RADAR_SENSORS

    all_points = []
    for sensor in sensors:
        if sensor not in sample_rec['data']:
            continue
        pc, times = RadarPointCloud.from_file_multisweep(
            nusc,
            sample_rec,
            chan=sensor,
            ref_chan='RADAR_FRONT',
            nsweeps=nsweeps,
            min_distance=min_distance,
        )
        # pc.points shape: (18, N), times shape: (1, N)
        pts = np.vstack([pc.points, times]).T  # (N, 19)
        all_points.append(pts)

    if not all_points:
        return np.zeros((0, 19), dtype=np.float32)

    return np.concatenate(all_points, axis=0).astype(np.float32)


def filter_radar_quality(points: np.ndarray) -> np.ndarray:
    """Filter radar points by quality flags.

    Keeps points where is_quality_valid != 0 and invalid_state == 0.

    Args:
        points: (N, 18+) radar points.

    Returns:
        Filtered points array.
    """
    if len(points) == 0:
        return points
    mask = (points[:, 10] != 0) & (points[:, 14] == 0)
    return points[mask]


def filter_radar_rcs(points: np.ndarray, min_rcs: float = -5.0) -> np.ndarray:
    """Filter radar points by minimum RCS threshold.

    Args:
        points: (N, 18+) radar points.
        min_rcs: Minimum radar cross section in dBsm.

    Returns:
        Filtered points array.
    """
    if len(points) == 0:
        return points
    return points[points[:, 5] >= min_rcs]


def select_features(
    points: np.ndarray,
    feature_indices: List[int] = None,
) -> np.ndarray:
    """Select specific feature columns from radar points.

    Args:
        points: (N, 18+) full radar points.
        feature_indices: Column indices to select. Defaults to USEFUL_FEATURE_INDICES.

    Returns:
        (N, C) array with selected features.
    """
    if feature_indices is None:
        feature_indices = USEFUL_FEATURE_INDICES
    return points[:, feature_indices].copy()


def transform_velocity_augmentation(
    points: np.ndarray,
    rotation_matrix: np.ndarray,
    vx_idx: int = 8,
    vy_idx: int = 9,
) -> np.ndarray:
    """Apply rotation augmentation to velocity components.

    When rotating radar points, velocities must also be rotated.

    Args:
        points: (N, C) radar points.
        rotation_matrix: (2, 2) or (3, 3) rotation matrix.
        vx_idx: Column index of vx_comp.
        vy_idx: Column index of vy_comp.

    Returns:
        Points with rotated velocities.
    """
    points = points.copy()
    vel = np.stack([points[:, vx_idx], points[:, vy_idx]], axis=1)  # (N, 2)
    rot_2d = rotation_matrix[:2, :2] if rotation_matrix.shape[0] > 2 else rotation_matrix
    vel_rot = vel @ rot_2d.T
    points[:, vx_idx] = vel_rot[:, 0]
    points[:, vy_idx] = vel_rot[:, 1]
    return points
