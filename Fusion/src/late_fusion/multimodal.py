"""Three multi-sensor fusion+tracking architectures, for comparison:

  A) track_then_fuse  (hierarchical/late): track each modality independently —
     camera across all 6 views, LiDAR, radar — then fuse the *tracks*.
  B) fuse_then_track  (flat/central): fuse all sensors' raw detections per frame,
     then track the fused result.
  C) cov_central      (covariance-weighted central, the researched best practice):
     like B but the per-frame fusion (``fuse_cov``) is principled — radar position
     fused inverse-variance with range-dependent noise, existence combined in
     Bayesian log-odds with radar evidence down-weighted.

All yield, per frame, a list of FusedObject with a ``track_id``. A shared
confirm-length filter (drop tracks seen in < N frames) removes false positives,
and ``evaluate`` scores recall/precision/F1 + ID stability against NuScenes GT.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from .association import associate_lidar_camera, associate_lidar_radar
from .fusion import fuse
from .geometry import transform_box
from .tracking_helpers import (
    attach_velocity, make_camera_trackers, make_tracker, name_to_idx, track_2d, track_3d,
)
from .types import NUSCENES_CLASSES, Detection2D, Detection3D, FusedObject
from tracking.bytetrack import BaseTrack


# ----- Method C: covariance-weighted central-level fusion (the researched best
# practice for 2D-camera + 3D-LiDAR + noisy radar) -------------------------------
# - LiDAR provides the 3D box (low position noise) -> the geometry anchor.
# - Radar position is fused covariance-weighted with a RANGE-DEPENDENT noise, so the
#   noisy radar barely moves the accurate LiDAR estimate (and less so far away);
#   radar supplies velocity (its strength).
# - Existence/confidence is combined in Bayesian log-odds (independent evidence),
#   with radar evidence down-weighted because it is unreliable. Camera refines class.
_SIGMA_LIDAR = 0.3                 # LiDAR BEV position std (m)
_RADAR_SIGMA0, _RADAR_SIGMA_K = 0.6, 0.03   # radar std = sigma0 + k*range (m)
_RADAR_EXIST_W = 0.5               # down-weight noisy-radar existence evidence


def _logodds(p):
    p = min(max(float(p), 1e-4), 1 - 1e-4)
    return np.log(p / (1 - p))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _cov_fuse_xy(xl, xr, rng):
    """Inverse-variance (covariance) weighted fusion of two BEV positions."""
    wl = 1.0 / _SIGMA_LIDAR ** 2
    wr = 1.0 / (_RADAR_SIGMA0 + _RADAR_SIGMA_K * rng) ** 2
    return (wl * np.asarray(xl) + wr * np.asarray(xr)) / (wl + wr)


def fuse_cov(lidar_dets, cam_dets, radar_dets, cameras,
             cam_iou_thresh=0.1, radar_dist_thresh=3.0,
             include_radar_only=True, radar_only_min_score=0.3, radar_only_min_speed=1.0):
    """Covariance-weighted, Bayesian-existence central fusion (Method C)."""
    cam_match = associate_lidar_camera(lidar_dets, cam_dets, cameras, cam_iou_thresh)
    radar_match = associate_lidar_radar(lidar_dets, radar_dets, radar_dist_thresh)
    fused, matched_radar = [], set()

    for li, ld in enumerate(lidar_dets):
        box = ld.box.copy()
        lo = _logodds(ld.score)            # existence evidence from LiDAR
        sources, label, vel = {"lidar"}, ld.label, None

        cd = cam_match.get(li)
        if cd is not None:
            sources.add("camera")
            lo += _logodds(cd.score)       # camera = independent existence evidence
            label = cd.label               # camera is the stronger classifier

        rj = radar_match.get(li)
        if rj is not None:
            rd = radar_dets[rj]
            matched_radar.add(rj)
            sources.add("radar")
            lo += _RADAR_EXIST_W * _logodds(rd.score)   # down-weighted noisy evidence
            rng = float(np.linalg.norm(ld.box[:2]))
            box[:2] = _cov_fuse_xy(ld.box[:2], rd.box[:2], rng)   # radar nudges position
            vel = rd.velocity

        obj = FusedObject(box=box, score=float(_sigmoid(lo)), label=label,
                          sources=sources, velocity=vel,
                          camera_confirmed=("camera" in sources),
                          track_id=getattr(ld, "track_id", None))
        fused.append(obj)

    if include_radar_only:
        for rj, rd in enumerate(radar_dets):
            if rj in matched_radar or float(rd.score) < radar_only_min_score:
                continue
            speed = float(np.linalg.norm(rd.velocity)) if rd.velocity is not None else 0.0
            if speed < radar_only_min_speed:
                continue
            fused.append(FusedObject(box=rd.box.copy(),
                                     score=float(_sigmoid(_RADAR_EXIST_W * _logodds(rd.score))),
                                     label=rd.label, sources={"radar"}, velocity=rd.velocity,
                                     track_id=getattr(rd, "track_id", None)))
    return fused


def cov_central(results, **kw):
    """Method C: per-frame covariance-weighted central fusion, then track (like B)."""
    BaseTrack.reset_id_counter()
    tracker = make_tracker("fused")
    per_frame = []
    for r in results:
        sd = r["sample_data"]
        fused = fuse_cov(r["lidar_dets"], r["cam_dets"], r["radar_dets"], sd["cameras"], **kw)
        per_frame.append(_assign_ids_3d(fused, tracker, sd["lidar_to_global"]))
    return per_frame


def _assign_ids_3d(objs: List[FusedObject], tracker, lidar_to_global) -> List[FusedObject]:
    """Track fused boxes (global frame) and tag each with its track id; returns
    only objects matched to an active track (i.e. confirmed this frame)."""
    if objs:
        gboxes = np.stack([transform_box(o.box, lidar_to_global) for o in objs])
        scores = np.array([o.score for o in objs], float)
        labels = np.array([name_to_idx(o.label) for o in objs], int)
    else:
        gboxes, scores, labels = np.zeros((0, 7)), np.zeros(0), np.zeros(0, int)
    tracks = tracker.update(gboxes, scores, labels)
    out, used = [], set()
    if objs and tracks:
        obj_xy = gboxes[:, :2]
        for t in tracks:
            ts = t.get_state()[:2]
            d = np.linalg.norm(obj_xy - ts, axis=1)
            j = int(np.argmin(d))
            if d[j] < 2.0 and j not in used:
                objs[j].track_id = int(t.track_id)
                used.add(j)
                out.append(objs[j])
    return out


def fuse_then_track(results: Sequence[dict], **fuse_kw) -> List[List[FusedObject]]:
    """Architecture B: fuse raw detections per frame, then track the fused output."""
    BaseTrack.reset_id_counter()
    tracker = make_tracker("fused")
    per_frame = []
    for r in results:
        sd = r["sample_data"]
        fused = fuse(r["lidar_dets"], r["cam_dets"], r["radar_dets"], sd["cameras"], **fuse_kw)
        per_frame.append(_assign_ids_3d(fused, tracker, sd["lidar_to_global"]))
    return per_frame


def fuse_then_track_ablation(results: Sequence[dict], use_camera: bool = True,
                             use_radar: bool = True, **fuse_kw) -> List[List[FusedObject]]:
    """Architecture B restricted to a subset of sensors, for ablation.

    Always uses LiDAR as the 3D anchor; ``use_camera``/``use_radar`` toggle whether
    the camera and radar detections are made available to ``fuse``. With both False
    this is the LiDAR-only baseline (each LiDAR detection tracked); enabling them one
    at a time measures each modality's marginal contribution.
    """
    BaseTrack.reset_id_counter()
    tracker = make_tracker("fused")
    per_frame = []
    for r in results:
        sd = r["sample_data"]
        cam = r["cam_dets"] if use_camera else []
        radar = r["radar_dets"] if use_radar else []
        fused = fuse(r["lidar_dets"], cam, radar, sd["cameras"],
                     include_radar_only=use_radar, **fuse_kw)
        per_frame.append(_assign_ids_3d(fused, tracker, sd["lidar_to_global"]))
    return per_frame


def track_then_fuse(results: Sequence[dict], **fuse_kw) -> List[List[FusedObject]]:
    """Architecture A: track each modality, then fuse the per-modality tracks."""
    BaseTrack.reset_id_counter()
    tl, tr = make_tracker("lidar"), make_tracker("radar")
    tc = make_camera_trackers()
    per_frame = []
    for r in results:
        sd = r["sample_data"]
        l2g = sd["lidar_to_global"]
        lidar_t = track_3d(tl, r["lidar_dets"], l2g, "lidar")
        radar_t = track_3d(tr, r["radar_dets"], l2g, "radar")
        attach_velocity(radar_t, r["radar_dets"])
        cam_t = track_2d(tc, r["cam_dets"])
        # Fuse the already-tracked modalities; fused object inherits the anchor's id.
        fused = fuse(lidar_t, cam_t, radar_t, sd["cameras"], **fuse_kw)
        per_frame.append([o for o in fused if o.track_id is not None])
    return per_frame


def confirm_filter(per_frame: List[List[FusedObject]], min_frames: int) -> List[List[FusedObject]]:
    """Drop tracks (by id) that appear in fewer than ``min_frames`` frames."""
    life: Dict[int, int] = {}
    for objs in per_frame:
        for o in objs:
            life[o.track_id] = life.get(o.track_id, 0) + 1
    keep = {tid for tid, n in life.items() if n >= min_frames}
    return [[o for o in objs if o.track_id in keep] for objs in per_frame]


def _as_boxes(gt) -> np.ndarray:
    """Coerce a GT entry to an (N, 7) float array (empty -> (0, 7))."""
    return np.asarray(gt).reshape(-1, 7) if len(gt) else np.zeros((0, 7))


def _match_frame(objs: Sequence[FusedObject], gt: np.ndarray,
                 dist_thresh: float = 2.0):
    """Greedy nearest-neighbour BEV matching (class-agnostic) of predicted objects
    to GT boxes within ``dist_thresh``.

    Returns ``(matched_obj_idx, matched_gt_idx)`` as index sets — a prediction and
    a GT box are paired at most once, in ascending distance order.
    """
    pairs = []
    for i, o in enumerate(objs):
        for g in range(len(gt)):
            d = float(np.linalg.norm(o.box[:2] - gt[g, :2]))
            if d < dist_thresh:
                pairs.append((d, i, g))
    pairs.sort()
    mo, mg = set(), set()
    for _, i, g in pairs:
        if i in mo or g in mg:
            continue
        mo.add(i)
        mg.add(g)
    return mo, mg


def _prf(tp: int, fp: int, fn: int) -> dict:
    """Recall / precision / F1 from raw counts."""
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) else 0.0
    return {"TP": tp, "FP": fp, "FN": fn,
            "recall": recall, "precision": precision, "f1": f1}


def evaluate(per_frame: List[List[FusedObject]],
             gt_per_frame: Sequence[np.ndarray],
             dist_thresh: float = 2.0) -> dict:
    """Recall/precision/F1 (greedy BEV matching to GT) + ID-stability stats."""
    tp = fp = fn = 0
    life: Dict[int, int] = {}
    for objs, gt in zip(per_frame, gt_per_frame):
        for o in objs:
            life[o.track_id] = life.get(o.track_id, 0) + 1
        gt = _as_boxes(gt)
        mo, mg = _match_frame(objs, gt, dist_thresh)
        tp += len(mg)
        fn += len(gt) - len(mg)
        fp += len(objs) - len(mo)
    lengths = list(life.values())
    return {
        **_prf(tp, fp, fn),
        "num_tracks": len(life),
        "mean_track_len": float(np.mean(lengths)) if lengths else 0.0,
    }


def _gt_class_name(label, class_names: Sequence[str]):
    """Map a GT label (int index into ``class_names``, or a name) to a class name,
    or ``None`` if it cannot be resolved."""
    try:
        i = int(label)
        return class_names[i] if 0 <= i < len(class_names) else None
    except (ValueError, TypeError):
        s = str(label)
        return s if s in class_names else None


def per_class_counts(per_frame: List[List[FusedObject]],
                     gt_per_frame: Sequence[np.ndarray],
                     gt_labels_per_frame: Sequence[np.ndarray],
                     class_names: Sequence[str] = NUSCENES_CLASSES,
                     dist_thresh: float = 2.0) -> Dict[str, dict]:
    """Class-aware TP/FP/FN per class (greedy BEV matching *within* each class).

    A prediction is only allowed to match a GT box of the **same** class, so a
    mis-classified-but-well-localised box is both a FP (predicted class) and a FN
    (GT class) — this scores classification, not just localisation.

    Returns ``{class_name: {"TP","FP","FN"}}`` for every class in ``class_names``.
    """
    counts = {c: {"TP": 0, "FP": 0, "FN": 0} for c in class_names}
    for objs, gt, labels in zip(per_frame, gt_per_frame, gt_labels_per_frame):
        gt = _as_boxes(gt)
        labels = np.asarray(labels).reshape(-1)
        gt_names = [_gt_class_name(labels[g], class_names) if g < len(labels) else None
                    for g in range(len(gt))]
        for c in class_names:
            gt_c = gt[[g for g in range(len(gt)) if gt_names[g] == c]] if len(gt) \
                else np.zeros((0, 7))
            objs_c = [o for o in objs if o.label == c]
            mo, mg = _match_frame(objs_c, gt_c, dist_thresh)
            counts[c]["TP"] += len(mg)
            counts[c]["FN"] += len(gt_c) - len(mg)
            counts[c]["FP"] += len(objs_c) - len(mo)
    return counts


# Near / mid / far BEV-range bands (metres from the ego), the autonomous-driving
# concern that an aggregate number hides.
DISTANCE_BINS = [(0.0, 20.0, "0-20m"), (20.0, 35.0, "20-35m"), (35.0, 1e9, "35m+")]


def per_distance_counts(per_frame: List[List[FusedObject]],
                        gt_per_frame: Sequence[np.ndarray],
                        bins=DISTANCE_BINS,
                        dist_thresh: float = 2.0) -> Dict[str, dict]:
    """TP/FP/FN per BEV-range band (class-agnostic matching).

    GT boxes are binned by their range for the recall side (``TP_gt`` / ``FN``);
    predictions by their range for the precision side (``TP_pred`` / ``FP``).

    Returns ``{band_label: {"TP_gt","FN","TP_pred","FP"}}``.
    """
    counts = {lab: {"TP_gt": 0, "FN": 0, "TP_pred": 0, "FP": 0} for _, _, lab in bins}

    def band(r: float):
        for lo, hi, lab in bins:
            if lo <= r < hi:
                return lab
        return None

    for objs, gt in zip(per_frame, gt_per_frame):
        gt = _as_boxes(gt)
        mo, mg = _match_frame(objs, gt, dist_thresh)
        for g in range(len(gt)):
            lab = band(float(np.linalg.norm(gt[g, :2])))
            if lab is None:
                continue
            counts[lab]["TP_gt" if g in mg else "FN"] += 1
        for i, o in enumerate(objs):
            lab = band(float(np.linalg.norm(o.box[:2])))
            if lab is None:
                continue
            counts[lab]["TP_pred" if i in mo else "FP"] += 1
    return counts
