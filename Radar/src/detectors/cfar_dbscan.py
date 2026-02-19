# -*- coding: utf-8 -*-
"""
Classical Radar Detector: CFAR thresholding + DBSCAN clustering.

No deep learning — uses signal-processing techniques to detect objects
from sparse NuScenes radar point clouds.

Pipeline:
    1. Quality filtering (invalid_state, is_quality_valid)
    2. Adaptive RCS thresholding (simplified CFAR)
    3. DBSCAN spatial clustering on (x, y)
    4. Bounding box fitting per cluster
    5. Heuristic classification by cluster geometry / RCS
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import DBSCAN

from Radar.src.core.base_radar_detector import Detection3D

logger = logging.getLogger(__name__)


# Simple heuristic class mapping based on cluster stats
_HEURISTIC_CLASSES = {
    'large_high_rcs':  (0, 'car'),         # large cluster, high RCS
    'very_large':      (1, 'truck'),        # very large cluster
    'medium':          (0, 'car'),          # medium cluster
    'small_moving':    (6, 'motorcycle'),   # small + has velocity
    'small_static':    (5, 'barrier'),      # small + static
    'tiny_moving':     (8, 'pedestrian'),   # tiny + moving
    'tiny_static':     (9, 'traffic_cone'), # tiny + static
}


class CFARDBSCANDetector:
    """Classical radar object detector using CFAR + DBSCAN.

    This detector does NOT use neural networks.

    Args:
        rcs_threshold: Minimum RCS in dBsm. Set to None for adaptive CFAR.
        cfar_guard_cells: Number of guard cells for CFAR window.
        cfar_training_cells: Number of training cells for CFAR window.
        cfar_pfa: Probability of false alarm (lower = stricter).
        dbscan_eps: DBSCAN neighbourhood radius in metres.
        dbscan_min_samples: Minimum points to form a cluster.
        min_cluster_size: Discard clusters smaller than this.
        point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max].
    """

    CLASS_NAMES = [
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
        'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone',
    ]

    def __init__(
        self,
        rcs_threshold: Optional[float] = None,
        cfar_guard_cells: int = 2,
        cfar_training_cells: int = 8,
        cfar_pfa: float = 1e-3,
        dbscan_eps: float = 3.0,
        dbscan_min_samples: int = 2,
        min_cluster_size: int = 2,
        point_cloud_range: Tuple[float, ...] = (-100, -100, -5, 100, 100, 3),
    ):
        self.rcs_threshold = rcs_threshold
        self.cfar_guard_cells = cfar_guard_cells
        self.cfar_training_cells = cfar_training_cells
        self.cfar_pfa = cfar_pfa
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.min_cluster_size = min_cluster_size
        self.point_cloud_range = np.array(point_cloud_range, dtype=np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, points: np.ndarray, conf_threshold: float = 0.3) -> List[Detection3D]:
        """Detect objects from raw radar points.

        Args:
            points: (N, 18+) NuScenes radar points (all features).
            conf_threshold: Not used directly (kept for API compatibility).

        Returns:
            List of Detection3D.
        """
        if len(points) == 0:
            return []

        # 1. Quality filter
        pts = self._filter_quality(points)
        if len(pts) == 0:
            return []

        # 2. Range filter
        pts = self._filter_range(pts)
        if len(pts) == 0:
            return []

        # 3. RCS threshold (adaptive CFAR or fixed)
        pts = self._cfar_threshold(pts)
        if len(pts) == 0:
            return []

        # 4. DBSCAN clustering
        clusters = self._cluster(pts)

        # 5. Fit boxes + classify
        detections = []
        for cluster_pts in clusters:
            det = self._fit_box(cluster_pts)
            if det is not None:
                detections.append(det)

        return detections

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def _filter_quality(self, points: np.ndarray) -> np.ndarray:
        """Keep only quality-valid, non-invalid points."""
        mask = (points[:, 10] != 0) & (points[:, 14] == 0)
        return points[mask]

    def _filter_range(self, points: np.ndarray) -> np.ndarray:
        pc = self.point_cloud_range
        mask = (
            (points[:, 0] >= pc[0]) & (points[:, 0] <= pc[3])
            & (points[:, 1] >= pc[1]) & (points[:, 1] <= pc[4])
            & (points[:, 2] >= pc[2]) & (points[:, 2] <= pc[5])
        )
        return points[mask]

    def _cfar_threshold(self, points: np.ndarray) -> np.ndarray:
        """Simplified CA-CFAR: adaptive RCS thresholding using spatial neighbours.

        For each point, computes the mean RCS of its spatial (range-based)
        neighbours (excluding guard cells) and sets a local threshold.
        Points with RCS above the local threshold are kept.
        """
        rcs = points[:, 5]

        if self.rcs_threshold is not None:
            return points[rcs >= self.rcs_threshold]

        n = len(points)
        if n < 2 * (self.cfar_guard_cells + self.cfar_training_cells) + 1:
            return points

        # Sort by range (distance from origin in BEV) to create a 1-D signal
        ranges = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
        range_order = np.argsort(ranges)
        rcs_ordered = rcs[range_order]

        # CFAR scale factor: alpha = N * (PFA^(-1/N) - 1)
        alpha_n = self.cfar_training_cells * 2
        cfar_pfa_safe = np.clip(self.cfar_pfa, 1e-10, 1.0)
        alpha = alpha_n * (cfar_pfa_safe ** (-1.0 / alpha_n) - 1.0)

        guard = self.cfar_guard_cells
        train = self.cfar_training_cells
        keep_ordered = np.ones(n, dtype=bool)

        for i in range(n):
            # Training cells: [i-guard-train, i-guard) and (i+guard, i+guard+train]
            lo_start = max(0, i - guard - train)
            lo_end = max(0, i - guard)
            hi_start = min(n, i + guard + 1)
            hi_end = min(n, i + guard + train + 1)

            training_vals = np.concatenate([
                rcs_ordered[lo_start:lo_end],
                rcs_ordered[hi_start:hi_end],
            ])
            if len(training_vals) == 0:
                continue
            threshold = np.mean(training_vals) + alpha * np.std(training_vals)
            if rcs_ordered[i] < threshold:
                keep_ordered[i] = False

        # Map back to original order
        keep = np.zeros(n, dtype=bool)
        keep[range_order] = keep_ordered
        return points[keep]

    def _cluster(self, points: np.ndarray) -> List[np.ndarray]:
        """DBSCAN clustering on (x, y) coordinates."""
        xy = points[:, :2]
        db = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples)
        labels = db.fit_predict(xy)

        clusters = []
        for cid in set(labels):
            if cid == -1:  # noise
                continue
            cluster_pts = points[labels == cid]
            if len(cluster_pts) >= self.min_cluster_size:
                clusters.append(cluster_pts)

        return clusters

    def _fit_box(self, cluster_pts: np.ndarray) -> Optional[Detection3D]:
        """Fit an axis-aligned BEV bounding box to a cluster and classify."""
        x = cluster_pts[:, 0]
        y = cluster_pts[:, 1]
        z = cluster_pts[:, 2]

        cx = (x.min() + x.max()) / 2
        cy = (y.min() + y.max()) / 2
        cz = (z.min() + z.max()) / 2

        length = max(x.max() - x.min(), 0.5)   # min 0.5m
        width = max(y.max() - y.min(), 0.5)
        height = max(z.max() - z.min(), 0.5)

        # Heuristic classification
        label, label_name, score = self._classify_cluster(cluster_pts, length, width)

        # Velocity estimate from compensated velocities
        vx = np.median(cluster_pts[:, 8]) if cluster_pts.shape[1] > 8 else 0.0
        vy = np.median(cluster_pts[:, 9]) if cluster_pts.shape[1] > 9 else 0.0

        box = np.array([cx, cy, cz, length, width, height, 0.0], dtype=np.float32)
        return Detection3D(
            box=box,
            score=score,
            label=label,
            label_name=label_name,
            velocity=np.array([vx, vy], dtype=np.float32),
            metadata={'num_radar_points': len(cluster_pts),
                      'mean_rcs': float(cluster_pts[:, 5].mean())},
        )

    def _classify_cluster(
        self, pts: np.ndarray, length: float, width: float,
    ) -> Tuple[int, str, float]:
        """Heuristic classification based on geometry and RCS."""
        area = length * width
        mean_rcs = pts[:, 5].mean()
        n_pts = len(pts)

        # Velocity magnitude
        speed = 0.0
        if pts.shape[1] > 9:
            speed = np.sqrt(pts[:, 8] ** 2 + pts[:, 9] ** 2).mean()
        is_moving = speed > 0.5

        if area > 20.0:
            # Very large → truck / bus
            label, name = (1, 'truck') if mean_rcs > 10 else (3, 'bus')
            score = min(0.7, 0.3 + n_pts * 0.02)
        elif area > 6.0:
            label, name = 0, 'car'
            score = min(0.8, 0.4 + n_pts * 0.03)
        elif area > 2.0:
            if is_moving:
                label, name = 0, 'car'
                score = 0.5
            else:
                label, name = 5, 'barrier'
                score = 0.4
        elif area > 0.5:
            if is_moving:
                label, name = 6, 'motorcycle'
                score = 0.4
            else:
                label, name = 9, 'traffic_cone'
                score = 0.35
        else:
            if is_moving:
                label, name = 8, 'pedestrian'
                score = 0.35
            else:
                label, name = 9, 'traffic_cone'
                score = 0.3

        return label, name, score

    def get_model_info(self) -> Dict:
        return {
            'model_class': 'CFARDBSCANDetector',
            'type': 'classical (no DL)',
            'dbscan_eps': self.dbscan_eps,
            'rcs_threshold': self.rcs_threshold,
            'point_cloud_range': self.point_cloud_range.tolist(),
        }
