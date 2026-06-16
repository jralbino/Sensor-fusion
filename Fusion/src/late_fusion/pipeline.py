"""End-to-end late-fusion pipeline on NuScenes samples.

Orchestrates the three modality stacks in one process using the repo-root
``module_loader`` (Lidar/Vision each expose a conflicting top-level ``src``; radar
is imported with its ``Radar.`` prefix and does not conflict).

Phases:
  1. (Vision ctx) build the camera detector.
  2. (Lidar  ctx) build the LiDAR model once; per sample load data, run LiDAR 3D
     detection, and run camera 2D detection on each image (the detector object is
     context-independent once built).
  3. (no ctx)     load radar points, run radar detection, transform to LiDAR frame.
  4. adapt + fuse.

All heavy module imports happen inside their phase; the fusion core is imported at
module load time (before any context purges ``src``).
"""
from __future__ import annotations

import os
from typing import List, Optional, Sequence

import numpy as np

# Fusion core — imported up front so it survives the sys.modules purges that
# module_loader performs when switching module contexts.
from .adapters import (
    lidar_results_to_dets,
    radar_dets_to_common,
    yolo_dets_to_dets,
)
from .fusion import fuse
from .geometry import nms_bev
from .types import Detection2D, Detection3D, FusedObject

# module_loader lives at the repo root.
import module_loader


def _make_transform(translation, rotation_quat) -> np.ndarray:
    """4×4 rigid transform from a translation and a (w,x,y,z) quaternion."""
    from pyquaternion import Quaternion

    T = np.eye(4)
    T[:3, :3] = Quaternion(rotation_quat).rotation_matrix
    T[:3, 3] = np.asarray(translation)
    return T


def _radar_to_lidar(nusc, sample) -> np.ndarray:
    """RADAR_FRONT → LIDAR_TOP transform via the shared ego frame."""
    lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    lidar_cs = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
    radar_sd = nusc.get("sample_data", sample["data"]["RADAR_FRONT"])
    radar_cs = nusc.get("calibrated_sensor", radar_sd["calibrated_sensor_token"])
    lidar_to_ego = _make_transform(lidar_cs["translation"], lidar_cs["rotation"])
    radar_to_ego = _make_transform(radar_cs["translation"], radar_cs["rotation"])
    return np.linalg.inv(lidar_to_ego) @ radar_to_ego


# Clean-room LiDAR models use a KITTI-style front-only range (x_min=0), so they
# never detect behind the ego. These are the front-only model types.
_FRONT_ONLY_MODELS = {"pointpillars", "second", "centerpoint"}


def _detect_360(lmain, model, points, device, model_type, score_thresh):
    """Run a front-only detector over the full 360° by also detecting on the
    180°-rotated cloud (rear → front) and un-rotating the results.
    """
    front = lmain.run_detection(model, points, device, model_type, score_thresh=score_thresh)
    fb = np.asarray(front["boxes"], dtype=np.float32).reshape(-1, 7)
    fs = np.asarray(front["scores"], dtype=np.float32).reshape(-1)
    fl = np.asarray(front["labels"]).reshape(-1)

    rot = points.copy()
    rot[:, 0] = -rot[:, 0]
    rot[:, 1] = -rot[:, 1]            # 180° rotation about z (preserves handedness)
    rear = lmain.run_detection(model, rot, device, model_type, score_thresh=score_thresh)
    rb = np.asarray(rear["boxes"], dtype=np.float32).reshape(-1, 7)
    if len(rb) == 0:
        return {"boxes": fb, "scores": fs, "labels": fl}

    rb = rb.copy()
    rb[:, 0] = -rb[:, 0]
    rb[:, 1] = -rb[:, 1]
    rb[:, 6] = rb[:, 6] + np.pi
    rs = np.asarray(rear["scores"], dtype=np.float32).reshape(-1)
    rl = np.asarray(rear["labels"]).reshape(-1)

    # Drop rear boxes that duplicate a front box near the x≈0 boundary.
    if len(fb):
        keep = [i for i in range(len(rb))
                if np.min(np.linalg.norm(fb[:, :2] - rb[i, :2], axis=1)) > 2.0]
        rb, rs, rl = rb[keep], rs[keep], rl[keep]

    return {"boxes": np.concatenate([fb, rb]),
            "scores": np.concatenate([fs, rs]),
            "labels": np.concatenate([fl, rl])}


def _lidar_to_global(nusc, sample) -> np.ndarray:
    """LIDAR_TOP → global transform (for ego-motion-compensated 3D tracking)."""
    sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    ego = nusc.get("ego_pose", sd["ego_pose_token"])
    return (_make_transform(ego["translation"], ego["rotation"])
            @ _make_transform(cs["translation"], cs["rotation"]))


def run_fusion_batch(
    data_root: str,
    indices: Sequence[int],
    version: str = "v1.0-mini",
    lidar_model: str = "centerpoint",
    lidar_checkpoint: Optional[str] = None,
    camera_model_key: str = "yolo26l",
    camera_conf: float = 0.3,
    lidar_score_thresh: float = 0.15,
    radar_conf: float = 0.3,
    radar_kwargs: Optional[dict] = None,
    lidar_360: bool = True,
    device: str = "cuda",
    use_radar: bool = True,
    verbose: bool = True,
) -> List[dict]:
    """Run camera + LiDAR + radar detection on several samples and fuse each.

    Loads NuScenes and both heavy models **once**, then iterates ``indices``.
    Returns a list of per-sample result dicts (see :func:`run_fusion_on_sample`).
    """
    from nuscenes import NuScenes

    def log(*a):
        if verbose:
            print(*a, flush=True)

    nusc = NuScenes(version=version, dataroot=data_root, verbose=False)
    indices = list(indices)

    # --- Phase 1: camera detector (Vision context) ---
    log(f"[1/4] Building camera detector (Vision) for {len(indices)} sample(s)...")
    with module_loader.use_module("Vision"):
        from config.utils.path_manager import path_manager
        from src.detectors.detector_factory import get_object_detector

        cam_model_path = str(path_manager.get_model(camera_model_key, check_exists=True))
        yolo = get_object_detector("yolo", model_path=cam_model_path,
                                   conf=camera_conf, device=device)

    # --- Phase 2: LiDAR model (once) + per-sample LiDAR + camera (Lidar context) ---
    log("[2/4] LiDAR detection + camera inference (Lidar)...")
    import importlib

    per_sample = {}
    with module_loader.use_module("Lidar"):
        import cv2
        from pathlib import Path

        v3 = importlib.import_module("visualize_3d")
        lmain = importlib.import_module("main")
        sweeps = 9 if lidar_model.startswith("mmdet3d") else 0

        model = None
        if lidar_checkpoint and os.path.exists(lidar_checkpoint):
            model = lmain.load_model(lidar_model, lidar_checkpoint, device)
        else:
            log("    (no LiDAR checkpoint -> using GT boxes as 3D anchors)")

        for idx in indices:
            token = nusc.sample[idx]["token"]
            sd = v3.load_sample_data(nusc, token, Path(data_root), sweeps_num=sweeps)

            if model is not None:
                if lidar_360 and lidar_model in _FRONT_ONLY_MODELS:
                    results = _detect_360(lmain, model, sd["points"], device,
                                          lidar_model, lidar_score_thresh)
                else:
                    results = lmain.run_detection(model, sd["points"], device,
                                                  lidar_model, score_thresh=lidar_score_thresh)
            else:
                results = {"boxes": sd["gt_boxes"],
                           "scores": np.ones(len(sd["gt_boxes"])),
                           "labels": sd["gt_labels"]}
            # Class-agnostic BEV NMS removes overlapping duplicates (e.g. a vehicle
            # detected as two classes) that the detector's per-class NMS leaves.
            lidar_dets = nms_bev(lidar_results_to_dets(results), iou_thresh=0.4)

            cam_dets: List[Detection2D] = []
            for cam_name, cam in sd["cameras"].items():
                img = cv2.imread(str(cam["img_path"]))
                if img is None:
                    continue
                try:
                    parsed, _, _ = yolo.detect(img)
                    cam_dets.extend(yolo_dets_to_dets(parsed, cam_name))
                except Exception as e:  # pragma: no cover - robustness
                    log(f"    sample {idx} camera {cam_name} failed: {e}")

            per_sample[idx] = {
                "token": token, "lidar_dets": lidar_dets, "cam_dets": cam_dets,
                "cameras": sd["cameras"], "points": sd["points"],
                "gt_boxes": sd["gt_boxes"], "gt_labels": sd["gt_labels"],
            }

    # --- Phase 3: radar (Radar. prefix, no module_loader context) ---
    radar_by_idx = {idx: [] for idx in indices}
    if use_radar:
        log("[3/4] Radar detection...")
        try:
            from Radar.src.data.radar_utils import load_radar_points, filter_radar_quality
            from Radar.src.detectors.detector_factory import get_radar_detector

            # Adaptive CFAR (rcs_threshold=None) is too strict on sparse mini data.
            rkw = {"rcs_threshold": -50.0} if radar_kwargs is None else radar_kwargs
            rdet = get_radar_detector("cfar_dbscan", **rkw)
            for idx in indices:
                sample = nusc.sample[idx]
                raw = filter_radar_quality(load_radar_points(nusc, sample))
                raw_dets = rdet.detect(raw, conf_threshold=radar_conf)
                dets = radar_dets_to_common(raw_dets, _radar_to_lidar(nusc, sample))
                # Clutter cleanup: drop low-confidence + far-range classical radar hits.
                radar_by_idx[idx] = [d for d in dets if d.score >= radar_conf
                                     and abs(d.box[0]) < 70 and abs(d.box[1]) < 70]
        except Exception as e:  # pragma: no cover - robustness
            log(f"    radar failed ({e}); continuing without radar")
    else:
        log("[3/4] Radar disabled.")

    # --- Phase 4: fuse each sample ---
    log("[4/4] Fusing...")
    out = []
    for idx in indices:
        ps = per_sample[idx]
        radar_dets = radar_by_idx[idx]
        fused = fuse(ps["lidar_dets"], ps["cam_dets"], radar_dets, ps["cameras"])
        out.append({
            "index": idx,
            "fused": fused,
            # Per-sensor raw detections (for single-sensor inspection/videos).
            "lidar_dets": ps["lidar_dets"],
            "cam_dets": ps["cam_dets"],
            "radar_dets": radar_dets,
            "sample_data": {"points": ps["points"], "cameras": ps["cameras"],
                            "token": ps["token"], "gt_boxes": ps["gt_boxes"],
                            "gt_labels": ps["gt_labels"],
                            "lidar_to_global": _lidar_to_global(nusc, nusc.sample[idx])},
            "counts": {"lidar": len(ps["lidar_dets"]), "camera": len(ps["cam_dets"]),
                       "radar": len(radar_dets), "fused": len(fused)},
        })
    return out


def run_fusion_on_sample(data_root: str, sample_idx: int = 0, **kwargs) -> dict:
    """Convenience wrapper: fuse a single sample (see :func:`run_fusion_batch`)."""
    return run_fusion_batch(data_root, [sample_idx], **kwargs)[0]
