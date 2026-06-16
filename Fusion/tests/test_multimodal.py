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


# --------------------- per-class / per-distance metrics --------------------
from src.late_fusion.multimodal import (  # noqa: E402
    _gt_class_name, per_class_counts, per_distance_counts,
)
from src.late_fusion.types import NUSCENES_CLASSES  # noqa: E402


def _obj_l(label, xy=(0, 0), track_id=1):
    return FusedObject([xy[0], xy[1], 0, 1, 1, 1, 0], 0.5, label, {"lidar"},
                       track_id=track_id)


def test_gt_class_name_maps_index_and_name():
    assert _gt_class_name(0, NUSCENES_CLASSES) == "car"
    assert _gt_class_name("pedestrian", NUSCENES_CLASSES) == "pedestrian"
    assert _gt_class_name(999, NUSCENES_CLASSES) is None   # out of range
    assert _gt_class_name("nope", NUSCENES_CLASSES) is None


def test_per_class_counts_class_aware():
    # car GT detected as car (TP car); pedestrian GT detected as a car at the same
    # spot (FP car + FN pedestrian, because matching is class-aware); a third car
    # GT is missed (FN car).
    car, ped = NUSCENES_CLASSES.index("car"), NUSCENES_CLASSES.index("pedestrian")
    per_frame = [[_obj_l("car", (0, 0)), _obj_l("car", (10, 0))]]
    gt = [np.array([[0, 0, 0, 4, 2, 1.5, 0],      # car  @ (0,0)   -> matched
                    [10, 0, 0, 1, 1, 1.7, 0],     # ped  @ (10,0)  -> class mismatch
                    [20, 0, 0, 4, 2, 1.5, 0]])]   # car  @ (20,0)  -> missed
    gt_labels = [np.array([car, ped, car])]
    c = per_class_counts(per_frame, gt, gt_labels)
    assert c["car"] == {"TP": 1, "FP": 1, "FN": 1}
    assert c["pedestrian"] == {"TP": 0, "FP": 0, "FN": 1}
    assert c["truck"] == {"TP": 0, "FP": 0, "FN": 0}   # untouched class


def test_per_distance_counts_bins_by_range():
    # near match (TP), far GT missed (FN), far spurious prediction (FP).
    per_frame = [[_obj_l("car", (5, 0)), _obj_l("car", (40, 0))]]
    gt = [np.array([[5, 0, 0, 4, 2, 1.5, 0],       # near  -> matched
                    [50, 0, 0, 4, 2, 1.5, 0]])]    # far   -> unmatched (>2 m away)
    c = per_distance_counts(per_frame, gt)
    assert c["0-20m"]["TP_gt"] == 1 and c["0-20m"]["TP_pred"] == 1
    assert c["0-20m"]["FN"] == 0 and c["0-20m"]["FP"] == 0
    assert c["35m+"]["FN"] == 1     # the 50 m GT is missed
    assert c["35m+"]["FP"] == 1     # the 40 m prediction is spurious


# ------------------------------- ablation ----------------------------------
from src.late_fusion.multimodal import fuse_then_track_ablation  # noqa: E402


def _scene(n_frames=5):
    """A tiny scene: one static car seen by LiDAR + camera + radar every frame."""
    cams = _front_camera()
    cam = cams["CAM_FRONT"]
    box = [10, 0, 0, 4, 2, 1.5, 0]
    bbox = project_box_to_image(boxes_to_corners_3d(box), cam["lidar_to_cam"],
                                cam["intrinsic"], cam["img_w"], cam["img_h"])
    results = []
    for i in range(n_frames):
        results.append({
            "index": i,
            "lidar_dets": [Detection3D(box, 0.6, "car", "lidar")],
            "cam_dets": [Detection2D(bbox, 0.9, "car", "CAM_FRONT")],
            "radar_dets": [Detection3D([10, 0, 0, 1, 1, 1, 0], 0.5, "car", "radar", [3.0, 0.0])],
            "sample_data": {"cameras": cams, "lidar_to_global": np.eye(4)},
        })
    return results


def _sources_union(per_frame):
    u = set()
    for objs in per_frame:
        for o in objs:
            u |= o.sources
    return u


def test_ablation_lidar_only_has_no_other_sources():
    pf = fuse_then_track_ablation(_scene(), use_camera=False, use_radar=False)
    assert any(objs for objs in pf)                 # the car is tracked
    assert _sources_union(pf) == {"lidar"}          # no camera/radar leaked in


def test_ablation_progressively_adds_modalities():
    cam = fuse_then_track_ablation(_scene(), use_camera=True, use_radar=False)
    full = fuse_then_track_ablation(_scene(), use_camera=True, use_radar=True)
    assert "camera" in _sources_union(cam) and "radar" not in _sources_union(cam)
    assert {"lidar", "camera", "radar"} <= _sources_union(full)


def test_aggregate_mean_std_and_micro():
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from fusion_evaluate import aggregate

    m = [{"f1": 0.4, "TP": 2, "FP": 1, "FN": 2},
         {"f1": 0.6, "TP": 4, "FP": 1, "FN": 0}]
    a = aggregate(m, ["f1"])
    assert a["f1"][0] == pytest.approx(0.5)         # mean
    assert a["f1"][1] == pytest.approx(0.1)         # std
    # pooled micro: TP=6, FP=2, FN=2 -> P=0.75, R=0.75, F1=0.75
    assert a["_micro"]["precision"] == pytest.approx(0.75)
    assert a["_micro"]["f1"] == pytest.approx(0.75)
