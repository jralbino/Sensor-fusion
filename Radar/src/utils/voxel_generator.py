# -*- coding: utf-8 -*-
"""
Simple voxel generator for radar point clouds.

Identical logic to Lidar/src/utils/voxel_generator.py but kept local
so that Radar can be used as a standalone module.
"""
from __future__ import annotations

import numpy as np
from typing import List, Tuple


class VoxelGenerator:
    """Convert a point cloud into a voxel (pillar) representation.

    Args:
        voxel_size: [vx, vy, vz] in metres.
        point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max].
        max_num_points: Maximum points per voxel.
        max_voxels: Maximum number of voxels to keep.
    """

    def __init__(
        self,
        voxel_size: List[float],
        point_cloud_range: List[float],
        max_num_points: int = 20,
        max_voxels: int = 30000,
    ):
        self.voxel_size = np.array(voxel_size, dtype=np.float32)
        self.point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
        self.max_num_points = max_num_points
        self.max_voxels = max_voxels

        self.grid_size = np.round(
            (self.point_cloud_range[3:] - self.point_cloud_range[:3]) / self.voxel_size
        ).astype(np.int32)

    def generate(
        self, points: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Voxelize a point cloud.

        Args:
            points: (N, C) float32 array.

        Returns:
            voxels:     (M, max_num_points, C) — padded point features per voxel.
            coords:     (M, 3) — [z_idx, y_idx, x_idx] voxel grid indices.
            num_points: (M,) — actual number of points in each voxel.
        """
        if len(points) == 0:
            C = points.shape[1] if points.ndim == 2 else 1
            return (
                np.zeros((0, self.max_num_points, C), dtype=np.float32),
                np.zeros((0, 3), dtype=np.int32),
                np.zeros((0,), dtype=np.int32),
            )

        # Compute voxel indices for each point
        grid_idx = np.floor(
            (points[:, :3] - self.point_cloud_range[:3]) / self.voxel_size
        ).astype(np.int32)

        # Filter out-of-range
        valid = (
            (grid_idx[:, 0] >= 0) & (grid_idx[:, 0] < self.grid_size[0])
            & (grid_idx[:, 1] >= 0) & (grid_idx[:, 1] < self.grid_size[1])
            & (grid_idx[:, 2] >= 0) & (grid_idx[:, 2] < self.grid_size[2])
        )
        points = points[valid]
        grid_idx = grid_idx[valid]

        if len(points) == 0:
            C = points.shape[1] if points.ndim == 2 else 1
            return (
                np.zeros((0, self.max_num_points, C), dtype=np.float32),
                np.zeros((0, 3), dtype=np.int32),
                np.zeros((0,), dtype=np.int32),
            )

        # Group points by voxel using a hash
        _, unique_idx, inverse = np.unique(
            grid_idx[:, 0] * self.grid_size[1] * self.grid_size[2]
            + grid_idx[:, 1] * self.grid_size[2]
            + grid_idx[:, 2],
            return_index=True,
            return_inverse=True,
        )

        n_voxels = min(len(unique_idx), self.max_voxels)
        C = points.shape[1]
        voxels = np.zeros((n_voxels, self.max_num_points, C), dtype=np.float32)
        coords = np.zeros((n_voxels, 3), dtype=np.int32)
        num_points = np.zeros(n_voxels, dtype=np.int32)

        for vi in range(n_voxels):
            mask = inverse == vi
            pts = points[mask]
            n = min(len(pts), self.max_num_points)
            voxels[vi, :n] = pts[:n]
            num_points[vi] = n
            # Coords in (z, y, x) order for compatibility with scatter
            idx = grid_idx[unique_idx[vi]]
            coords[vi] = [idx[2], idx[1], idx[0]]

        return voxels, coords, num_points
