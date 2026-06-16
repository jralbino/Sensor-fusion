"""Unit tests for the multi-sensor fusion architectures (`multimodal.py`).

Covers Method C (covariance-weighted central fusion, `fuse_cov` + helpers) and the
shared post-processing (`confirm_filter`, `evaluate`). Pure CPU, no data/GPU/torch
needed — only the per-frame fusion math and the scoring logic are exercised.

Run inside the fusion container:
    docker compose run --rm fusion python -m pytest Fusion/tests/test_multimodal.py -v
or on the host with any env that has numpy + scipy + pytest.
"""
import os
import sys

import numpy as np
import pytest

# Make `src.late_fusion` importable (Fusion/ on path → src = Fusion/src).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.late_fusion.geometry import boxes_to_corners_3d, project_box_to_image  # noqa: E402
from src.late_fusion.multimodal import (  # noqa: E402
    _cov_fuse_xy,
    _logodds,
    _sigmoid,
    confirm_filter,
    evaluate,
    fuse_cov,
)
from src.late_fusion.types import Detection2D, Detection3D, FusedObject  # noqa: E402


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


def _cam_box_for(lidar_det, cams, label="car", score=0.9):
    cam = cams["CAM_FRONT"]
    bbox = project_box_to_image(boxes_to_corners_3d(lidar_det.box),
                                cam["lidar_to_cam"], cam["intrinsic"],
                                cam["img_w"], cam["img_h"])
    return Detection2D(bbox, score, label, "CAM_FRONT")


# --------------------------- log-odds / sigmoid ----------------------------
def test_logodds_sigmoid_roundtrip():
    for p in (0.05, 0.5, 0.73, 0.99):
        assert _sigmoid(_logodds(p)) == pytest.approx(p, abs=1e-6)


def test_logodds_clamps_extremes():
    # Guards against +/-inf for p in {0, 1}.
    assert np.isfinite(_logodds(0.0))
    assert np.isfinite(_logodds(1.0))


# ------------------ covariance-weighted position fusion --------------------
def test_cov_fuse_lidar_dominates():
    # Radar 1 m away barely moves the accurate LiDAR estimate.
    fused = _cov_fuse_xy([10, 0], [11, 0], rng=10.0)
    np.testing.assert_allclose(fused, [10.1, 0.0], atol=1e-6)


def test_cov_fuse_radar_weaker_at_range():
    # Same 1 m radar offset, but farther away → even smaller pull toward radar.
    near = _cov_fuse_xy([10, 0], [11, 0], rng=5.0)[0]
    far = _cov_fuse_xy([100, 0], [101, 0], rng=100.0)[0]
    assert (near - 10.0) > (far - 100.0) > 0.0


# ------------------------------- fuse_cov ----------------------------------
def test_cov_three_sensors_agree():
    cams = _front_camera()
    lidar = [Detection3D([10, 0, 0, 4, 2, 1.5, 0], 0.6, "car", "lidar")]
    cam = [_cam_box_for(lidar[0], cams, "car", 0.9)]
    radar = [Detection3D([11, 0, 0, 1, 1, 1, 0], 0.5, "car", "radar", [5.0, 0.0])]

    fused = fuse_cov(lidar, cam, radar, cams, include_radar_only=False)
    assert len(fused) == 1
    o = fused[0]
    assert o.sources == {"lidar", "camera", "radar"}
    assert o.camera_confirmed
    # Bayesian existence boost: 0.6 (L) + 0.9 (C) + down-weighted 0.5 (R) → ~0.931.
    assert o.score == pytest.approx(0.931, abs=2e-3)
    assert o.score > 0.6
    # Radar nudges position only slightly; LiDAR box still dominates.
    np.testing.assert_allclose(o.box[:2], [10.1, 0.0], atol=1e-6)
    np.testing.assert_allclose(o.velocity, [5.0, 0.0])


def test_cov_class_from_camera():
    cams = _front_camera()
    lidar = [Detection3D([10, 0, 0, 4, 2, 1.5, 0], 0.6, "car", "lidar")]
    cam = [_cam_box_for(lidar[0], cams, "truck", 0.9)]   # camera says truck
    fused = fuse_cov(lidar, cam, [], cams, include_radar_only=False)
    assert fused[0].label == "truck"                     # camera refines class


def test_cov_radar_evidence_downweighted():
    # A matched radar boosts existence less than an equally-confident camera would.
    cams = _front_camera()
    lidar = [Detection3D([10, 0, 0, 4, 2, 1.5, 0], 0.6, "car", "lidar")]
    radar = [Detection3D([10, 0, 0, 1, 1, 1, 0], 0.8, "car", "radar", [1.0, 0.0])]
    cam = [_cam_box_for(lidar[0], cams, "car", 0.8)]

    with_radar = fuse_cov(lidar, [], radar, cams, include_radar_only=False)[0].score
    with_camera = fuse_cov(lidar, cam, [], cams, include_radar_only=False)[0].score
    assert with_camera > with_radar > 0.6


def test_cov_radar_only_moving_kept_static_dropped():
    cams = _front_camera()
    moving = [Detection3D([30, 30, 0, 1, 1, 1, 0], 0.5, "car", "radar", [2.0, 0.0])]
    static = [Detection3D([30, 30, 0, 1, 1, 1, 0], 0.5, "car", "radar", [0.0, 0.0])]
    assert len(fuse_cov([], [], moving, cams, radar_only_min_speed=1.0)) == 1
    assert len(fuse_cov([], [], static, cams, radar_only_min_speed=1.0)) == 0


def test_cov_empty_inputs():
    assert fuse_cov([], [], [], _front_camera()) == []


# ------------------------------ confirm_filter -----------------------------
def _obj(track_id, xy=(0, 0)):
    return FusedObject([xy[0], xy[1], 0, 1, 1, 1, 0], 0.5, "car", {"lidar"},
                       track_id=track_id)


def test_confirm_filter_drops_short_tracks():
    per_frame = [
        [_obj(1), _obj(2, (5, 5))],   # track 2 appears only here
        [_obj(1)],                    # track 1 appears in 2 frames
    ]
    out = confirm_filter(per_frame, min_frames=2)
    assert [[o.track_id for o in fr] for fr in out] == [[1], [1]]


def test_confirm_filter_keeps_all_when_threshold_one():
    per_frame = [[_obj(1)], [_obj(2)]]
    out = confirm_filter(per_frame, min_frames=1)
    assert sum(len(fr) for fr in out) == 2


# -------------------------------- evaluate ---------------------------------
def test_evaluate_perfect_match():
    per_frame = [[_obj(1)], [_obj(1), _obj(2, (5, 5))]]
    gt = [
        np.array([[0, 0, 0, 4, 2, 1.5, 0]]),
        np.array([[0, 0, 0, 4, 2, 1.5, 0], [5, 5, 0, 4, 2, 1.5, 0]]),
    ]
    m = evaluate(per_frame, gt)
    assert (m["TP"], m["FP"], m["FN"]) == (3, 0, 0)
    assert m["recall"] == pytest.approx(1.0)
    assert m["precision"] == pytest.approx(1.0)
    assert m["f1"] == pytest.approx(1.0)
    assert m["num_tracks"] == 2
    assert m["mean_track_len"] == pytest.approx(1.5)


def test_evaluate_counts_fp_and_fn():
    # One prediction far from the single GT → 1 FP and 1 FN, no TP.
    per_frame = [[_obj(1, (50, 50))]]
    gt = [np.array([[0, 0, 0, 4, 2, 1.5, 0]])]
    m = evaluate(per_frame, gt)
    assert (m["TP"], m["FP"], m["FN"]) == (0, 1, 1)
    assert m["recall"] == 0.0
    assert m["precision"] == 0.0
    assert m["f1"] == 0.0


def test_evaluate_distance_gate():
    # Prediction just outside the 2 m gate is not a TP.
    per_frame = [[_obj(1, (2.5, 0))]]
    gt = [np.array([[0, 0, 0, 4, 2, 1.5, 0]])]
    m = evaluate(per_frame, gt, dist_thresh=2.0)
    assert m["TP"] == 0 and m["FP"] == 1 and m["FN"] == 1
    # Within the gate it matches.
    m2 = evaluate(per_frame, gt, dist_thresh=3.0)
    assert m2["TP"] == 1 and m2["FP"] == 0 and m2["FN"] == 0
