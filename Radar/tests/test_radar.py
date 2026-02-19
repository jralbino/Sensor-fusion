# -*- coding: utf-8 -*-
"""
Unit tests for Radar module.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ========== radar_utils tests ==========

class TestRadarUtils:
    def test_filter_quality(self):
        from Radar.src.data.radar_utils import filter_radar_quality
        pts = np.zeros((5, 18), dtype=np.float32)
        pts[:, 10] = [1, 0, 1, 1, 0]   # is_quality_valid
        pts[:, 14] = [0, 0, 0, 1, 0]   # invalid_state
        result = filter_radar_quality(pts)
        assert len(result) == 2  # only rows 0 and 2

    def test_filter_quality_empty(self):
        from Radar.src.data.radar_utils import filter_radar_quality
        pts = np.zeros((0, 18), dtype=np.float32)
        result = filter_radar_quality(pts)
        assert len(result) == 0

    def test_filter_rcs(self):
        from Radar.src.data.radar_utils import filter_radar_rcs
        pts = np.zeros((4, 18), dtype=np.float32)
        pts[:, 5] = [-10, -5, 0, 10]
        result = filter_radar_rcs(pts, min_rcs=-5.0)
        assert len(result) == 3

    def test_select_features(self):
        from Radar.src.data.radar_utils import select_features, USEFUL_FEATURE_INDICES
        pts = np.random.randn(10, 18).astype(np.float32)
        result = select_features(pts)
        assert result.shape == (10, len(USEFUL_FEATURE_INDICES))
        np.testing.assert_array_equal(result[:, 0], pts[:, 0])  # x
        np.testing.assert_array_equal(result[:, 3], pts[:, 5])  # rcs

    def test_velocity_augmentation(self):
        from Radar.src.data.radar_utils import transform_velocity_augmentation
        pts = np.zeros((3, 18), dtype=np.float32)
        pts[:, 8] = [1, 0, -1]   # vx_comp
        pts[:, 9] = [0, 1, 0]    # vy_comp
        # 90-degree rotation
        rot = np.array([[0, -1], [1, 0]], dtype=np.float32)
        result = transform_velocity_augmentation(pts, rot)
        np.testing.assert_allclose(result[0, 8], 0.0, atol=1e-6)
        np.testing.assert_allclose(result[0, 9], 1.0, atol=1e-6)


# ========== VoxelGenerator tests ==========

class TestVoxelGenerator:
    def test_basic(self):
        from Radar.src.utils.voxel_generator import VoxelGenerator
        vg = VoxelGenerator(
            voxel_size=[1.0, 1.0, 8.0],
            point_cloud_range=[-10, -10, -4, 10, 10, 4],
            max_num_points=5,
            max_voxels=100,
        )
        pts = np.array([
            [0, 0, 0, 1, 2, 3],
            [0.5, 0.5, 0, 1, 2, 3],
            [5, 5, 0, 1, 2, 3],
        ], dtype=np.float32)
        voxels, coords, num_pts = vg.generate(pts)
        assert voxels.shape[2] == 6
        assert len(num_pts) == len(coords)
        assert num_pts.sum() == 3

    def test_empty(self):
        from Radar.src.utils.voxel_generator import VoxelGenerator
        vg = VoxelGenerator([1, 1, 8], [-10, -10, -4, 10, 10, 4])
        voxels, coords, num_pts = vg.generate(np.zeros((0, 6), dtype=np.float32))
        assert len(voxels) == 0


# ========== CFAR+DBSCAN tests ==========

class TestCFARDBSCAN:
    def test_detect_empty(self):
        from Radar.src.detectors.cfar_dbscan import CFARDBSCANDetector
        det = CFARDBSCANDetector()
        result = det.detect(np.zeros((0, 18), dtype=np.float32))
        assert result == []

    def test_detect_single_cluster(self):
        from Radar.src.detectors.cfar_dbscan import CFARDBSCANDetector
        det = CFARDBSCANDetector(rcs_threshold=-20.0, dbscan_eps=5.0, min_cluster_size=2)
        pts = np.zeros((5, 18), dtype=np.float32)
        pts[:, 0] = [10, 10.5, 11, 10.2, 10.8]  # x
        pts[:, 1] = [5, 5.3, 5.1, 4.8, 5.5]      # y
        pts[:, 2] = [0, 0, 0, 0, 0]                # z
        pts[:, 5] = [10, 12, 8, 15, 11]            # rcs
        pts[:, 10] = 1                              # quality valid
        pts[:, 14] = 0                              # not invalid
        result = det.detect(pts)
        assert len(result) >= 1
        assert result[0].label_name in det.CLASS_NAMES

    def test_get_model_info(self):
        from Radar.src.detectors.cfar_dbscan import CFARDBSCANDetector
        det = CFARDBSCANDetector()
        info = det.get_model_info()
        assert info['type'] == 'classical (no DL)'


# ========== RadarPillars tests ==========

class TestRadarPillars:
    def test_forward(self):
        import torch
        from Radar.src.detectors.radar_pillars import RadarPillars

        model = RadarPillars(num_classes=10, in_channels=6,
                             voxel_size=(1.0, 1.0, 8.0),
                             point_cloud_range=(-10, -10, -4, 10, 10, 4))
        # Fake voxelized input
        M = 5
        batch = {
            'voxels': torch.randn(M, 20, 6),
            'voxel_coords': torch.randint(0, 10, (M, 3)),
            'voxel_num_points': torch.randint(1, 20, (M,)),
            'batch_size': 1,
        }
        model.eval()
        with torch.no_grad():
            out = model(batch)
        assert 'pred_boxes' in out
        assert 'pred_scores' in out

    def test_model_info(self):
        from Radar.src.detectors.radar_pillars import RadarPillars
        model = RadarPillars(num_classes=10, voxel_size=(1.0, 1.0, 8.0),
                             point_cloud_range=(-10, -10, -4, 10, 10, 4))
        info = model.get_model_info()
        assert info['model_class'] == 'RadarPillars'
        assert info['total_parameters'] > 0


# ========== RadarCenterPoint tests ==========

class TestRadarCenterPoint:
    def test_forward(self):
        import torch
        from Radar.src.detectors.radar_centerpoint import RadarCenterPoint

        model = RadarCenterPoint(num_classes=10, in_channels=6,
                                  voxel_size=(1.0, 1.0, 8.0),
                                  point_cloud_range=(-10, -10, -4, 10, 10, 4))
        M = 5
        batch = {
            'voxels': torch.randn(M, 20, 6),
            'voxel_coords': torch.randint(0, 10, (M, 3)),
            'voxel_num_points': torch.randint(1, 20, (M,)),
            'batch_size': 1,
        }
        model.eval()
        with torch.no_grad():
            out = model(batch)
        assert 'pred_boxes' in out
        assert 'pred_velocity' in out


# ========== Factory tests ==========

class TestFactory:
    def test_cfar(self):
        from Radar.src.detectors.detector_factory import get_radar_detector
        d = get_radar_detector('cfar_dbscan')
        assert hasattr(d, 'detect')

    def test_pillars(self):
        from Radar.src.detectors.detector_factory import get_radar_detector
        d = get_radar_detector('radar_pillars')
        assert hasattr(d, 'forward')

    def test_centerpoint(self):
        from Radar.src.detectors.detector_factory import get_radar_detector
        d = get_radar_detector('radar_centerpoint')
        assert hasattr(d, 'forward')

    def test_unknown(self):
        from Radar.src.detectors.detector_factory import get_radar_detector
        with pytest.raises(ValueError):
            get_radar_detector('nonexistent')


# ========== Detection3D tests ==========

class TestDetection3D:
    def test_basic(self):
        from Radar.src.core.base_radar_detector import Detection3D
        d = Detection3D(
            box=[1, 2, 3, 4, 5, 6, 0.5],
            score=0.9,
            label=0,
            label_name='car',
            velocity=np.array([1.0, 2.0]),
        )
        assert d.box.shape == (7,)
        assert d.velocity.shape == (2,)
        data = d.to_dict()
        assert 'velocity' in data

    def test_invalid_box(self):
        from Radar.src.core.base_radar_detector import Detection3D
        with pytest.raises(ValueError):
            Detection3D(box=[1, 2, 3], score=0.5, label=0)
