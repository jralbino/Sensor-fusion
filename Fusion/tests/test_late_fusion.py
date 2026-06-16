"""Unit tests for the late-fusion core. Pure CPU, no data/GPU needed.

Run inside the fusion container:
    docker compose run --rm fusion python -m pytest Fusion/tests/test_late_fusion.py -v
or on the host with any env that has numpy + scipy + pytest.
"""
import os
import sys

import numpy as np
import pytest

# Make `src.late_fusion` importable (Fusion/ on path → src = Fusion/src).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.late_fusion import (  # noqa: E402
    Detection2D,
    Detection3D,
    associate_lidar_camera,
    associate_lidar_radar,
    bev_center_distance,
    boxes_to_corners_3d,
    fuse,
    iou_2d,
    match_by_affinity,
    noisy_or,
    project_box_to_image,
)


# --- a synthetic forward-looking camera (LiDAR x-forward -> camera z-forward) ---
def _front_camera():
    R = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=np.float64)
    lidar_to_cam = np.eye(4)
    lidar_to_cam[:3, :3] = R
    intrinsic = np.array([[1000.0, 0, 800.0], [0, 1000.0, 450.0], [0, 0, 1.0]])
    return {
        "CAM_FRONT": {
            "intrinsic": intrinsic,
            "lidar_to_cam": lidar_to_cam,
            "img_w": 1600,
            "img_h": 900,
        }
    }


# ------------------------------- geometry ----------------------------------
def test_corners_shape_and_center():
    box = np.array([1, 2, 3, 4, 2, 1.5, 0.0])
    corners = boxes_to_corners_3d(box)
    assert corners.shape == (8, 3)
    np.testing.assert_allclose(corners.mean(axis=0), [1, 2, 3], atol=1e-5)


def test_corners_extent_axis_aligned():
    box = np.array([0, 0, 0, 4, 2, 1.5, 0.0])
    corners = boxes_to_corners_3d(box)
    assert pytest.approx(corners[:, 0].max() - corners[:, 0].min(), abs=1e-5) == 4.0
    assert pytest.approx(corners[:, 1].max() - corners[:, 1].min(), abs=1e-5) == 2.0
    assert pytest.approx(corners[:, 2].max() - corners[:, 2].min(), abs=1e-5) == 1.5


def test_projection_in_front_centered():
    cam = _front_camera()["CAM_FRONT"]
    corners = boxes_to_corners_3d([10, 0, 0, 4, 2, 1.5, 0.0])
    bbox = project_box_to_image(corners, cam["lidar_to_cam"], cam["intrinsic"],
                                cam["img_w"], cam["img_h"])
    assert bbox is not None
    cx = 0.5 * (bbox[0] + bbox[2])
    cy = 0.5 * (bbox[1] + bbox[3])
    assert pytest.approx(cx, abs=2.0) == 800.0
    assert pytest.approx(cy, abs=2.0) == 450.0


def test_projection_behind_returns_none():
    cam = _front_camera()["CAM_FRONT"]
    corners = boxes_to_corners_3d([-10, 0, 0, 4, 2, 1.5, 0.0])
    assert project_box_to_image(corners, cam["lidar_to_cam"], cam["intrinsic"],
                                cam["img_w"], cam["img_h"]) is None


def test_iou_2d():
    assert iou_2d([0, 0, 2, 2], [0, 0, 2, 2]) == pytest.approx(1.0)
    assert iou_2d([0, 0, 2, 2], [2, 2, 4, 4]) == 0.0
    assert iou_2d([0, 0, 2, 2], [1, 0, 3, 2]) == pytest.approx(1.0 / 3.0)


def test_bev_distance():
    assert bev_center_distance([0, 0, 5], [3, 4, 9]) == pytest.approx(5.0)


# ------------------------------ association --------------------------------
def test_match_by_affinity_optimal():
    aff = np.array([[0.9, 0.1], [0.2, 0.8]])
    assert sorted(match_by_affinity(aff, 0.5)) == [(0, 0), (1, 1)]
    assert match_by_affinity(np.zeros((0, 0)), 0.5) == []


def test_associate_lidar_camera():
    cams = _front_camera()
    lidar = [Detection3D([10, 0, 0, 4, 2, 1.5, 0], 0.6, "car", "lidar")]
    cam = cams["CAM_FRONT"]
    bbox = project_box_to_image(boxes_to_corners_3d(lidar[0].box),
                                cam["lidar_to_cam"], cam["intrinsic"],
                                cam["img_w"], cam["img_h"])
    cam_dets = [Detection2D(bbox, 0.9, "car", "CAM_FRONT")]
    match = associate_lidar_camera(lidar, cam_dets, cams, iou_thresh=0.3)
    assert match == {0: cam_dets[0]}


def test_associate_lidar_camera_no_match_when_disjoint():
    cams = _front_camera()
    lidar = [Detection3D([10, 0, 0, 4, 2, 1.5, 0], 0.6, "car", "lidar")]
    cam_dets = [Detection2D([0, 0, 5, 5], 0.9, "car", "CAM_FRONT")]
    assert associate_lidar_camera(lidar, cam_dets, cams, iou_thresh=0.3) == {}


def test_associate_lidar_radar():
    lidar = [
        Detection3D([10, 0, 0, 4, 2, 1.5, 0], 0.6, "car", "lidar"),
        Detection3D([20, 5, 0, 4, 2, 1.5, 0], 0.6, "car", "lidar"),
    ]
    radar = [
        Detection3D([20.5, 5.2, 0, 1, 1, 1, 0], 0.5, "car", "radar", [3, 0]),
        Detection3D([10.3, 0.1, 0, 1, 1, 1, 0], 0.5, "car", "radar", [5, 0]),
    ]
    assert associate_lidar_radar(lidar, radar, dist_thresh=3.0) == {0: 1, 1: 0}


def test_associate_lidar_radar_distance_gate():
    lidar = [Detection3D([10, 0, 0, 4, 2, 1.5, 0], 0.6, "car", "lidar")]
    radar = [Detection3D([30, 30, 0, 1, 1, 1, 0], 0.5, "car", "radar", [0, 0])]
    assert associate_lidar_radar(lidar, radar, dist_thresh=3.0) == {}


# -------------------------------- fusion -----------------------------------
def test_noisy_or():
    assert noisy_or(0.6, 0.9) == pytest.approx(0.96)
    assert noisy_or(0.0, 0.5) == pytest.approx(0.5)


def test_fuse_three_sensors_agree():
    cams = _front_camera()
    lidar = [Detection3D([10, 0, 0, 4, 2, 1.5, 0], 0.6, "car", "lidar")]
    cam = cams["CAM_FRONT"]
    bbox = project_box_to_image(boxes_to_corners_3d(lidar[0].box),
                                cam["lidar_to_cam"], cam["intrinsic"],
                                cam["img_w"], cam["img_h"])
    cam_dets = [Detection2D(bbox, 0.9, "car", "CAM_FRONT")]
    radar = [Detection3D([10.3, 0.1, 0, 1, 1, 1, 0], 0.5, "car", "radar", [5.0, 0.0])]

    fused = fuse(lidar, cam_dets, radar, cams, include_radar_only=False)
    assert len(fused) == 1
    obj = fused[0]
    assert obj.sources == {"lidar", "camera", "radar"}
    assert obj.camera_confirmed
    assert obj.score == pytest.approx(noisy_or(0.6, 0.9))
    assert obj.velocity is not None
    np.testing.assert_allclose(obj.velocity, [5.0, 0.0])
    assert obj.num_sensors == 3


def test_fuse_class_disagreement_prefers_camera():
    cams = _front_camera()
    lidar = [Detection3D([10, 0, 0, 4, 2, 1.5, 0], 0.6, "car", "lidar")]
    cam = cams["CAM_FRONT"]
    bbox = project_box_to_image(boxes_to_corners_3d(lidar[0].box),
                                cam["lidar_to_cam"], cam["intrinsic"],
                                cam["img_w"], cam["img_h"])
    cam_dets = [Detection2D(bbox, 0.9, "truck", "CAM_FRONT")]
    fused = fuse(lidar, cam_dets, [], cams, include_radar_only=False)
    assert fused[0].label == "truck"          # camera class wins
    assert fused[0].score == pytest.approx(0.6)  # no boost on disagreement


def test_fuse_includes_radar_only():
    cams = _front_camera()
    lidar = [Detection3D([10, 0, 0, 4, 2, 1.5, 0], 0.6, "car", "lidar")]
    radar = [Detection3D([30, 30, 0, 1, 1, 1, 0], 0.5, "car", "radar", [2.0, 0.0])]
    fused = fuse(lidar, [], radar, cams, include_radar_only=True, radar_only_min_score=0.3)
    assert len(fused) == 2
    radar_only = [o for o in fused if o.sources == {"radar"}]
    assert len(radar_only) == 1
    assert radar_only[0].velocity is not None
