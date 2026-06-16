#!/usr/bin/env python3
"""Per-sensor inspection videos for one NuScenes scene, with tracking IDs.

Each sensor's detections are tracked across the scene (3D ByteTrack in the global
frame for LiDAR/radar so ego motion is compensated; 2D ByteTrack per camera) and
the track ID is shown on every box. Each panel is titled with the algorithm used.

  lidar_*.mp4   BEV 3D boxes + 6-camera grid with projected 3D boxes AND
                depth-coloured projected LiDAR points (verifies the projection).
  camera_*.mp4  6-camera grid with 2D detections.
  radar_*.mp4   BEV radar boxes + velocity + 6-camera grid with projected radar boxes.

Run in the fusion container:
    docker compose run --rm fusion python Fusion/make_sensor_videos.py \
        --start-idx 120 --num-frames 41 \
        --lidar-checkpoint Lidar/outputs/centerpoint_run/best.pth

Keep --start-idx / --num-frames within one scene. Mini scene starts:
0, 39, 79, 120, 161, 202, 242, 283, 324, 364.
"""
import argparse
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "Fusion")):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.late_fusion.geometry import (  # noqa: E402
    BOX_EDGES,
    boxes_to_corners_3d,
    project_corners_to_image,
    project_points_to_image,
    transform_box,
)
from src.late_fusion.pipeline import run_fusion_batch  # noqa: E402
from src.late_fusion.tracking_helpers import (  # noqa: E402
    TRACK_CONFIG, attach_velocity, make_camera_trackers, make_tracker,
    track_2d, track_3d)

CAMERA_GRID = [
    ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"],
    ["CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"],
]
LIDAR_MODEL_NAMES = {
    "centerpoint": "CenterPoint", "pointpillars": "PointPillars", "second": "SECOND",
    "mmdet3d_pointpillars": "MMDet3D PointPillars",
    "mmdet3d_centerpoint": "MMDet3D CenterPoint", "mmdet3d_second": "MMDet3D SECOND",
}
CAMERA_MODEL_NAMES = {
    "yolo26l": "YOLO26-L", "yolo26x": "YOLO26-X", "yolo11l": "YOLO11-L",
    "yolo11x": "YOLO11-X", "rtdetr_x": "RT-DETR-X", "rtdetr_l": "RT-DETR-L",
}
RADAR_MODEL_NAME = "CFAR+DBSCAN (classical)"

# Per-sensor tracking config (NuScenes keyframes are 2 Hz → each frame = 0.5 s).
#   max_age:        frames a track coasts (Kalman) through MISSING detections — its
#                   ID survives the gap and re-associates on reappearance.
#   distance_thresh: BEV/center fallback (normalized by box diagonal) for when IoU
#                   drops to 0 between low-framerate frames — robust to gaps/fast motion.
#   confirm:        post-processing — keep only tracks DETECTED in >= this many frames
#                   (false-positive removal). Counts real detections, not coasted frames.
# Tuned per sensor: radar (heavy clutter) is strict on confirm + short coast; camera
# (reliable, occlusions) coasts longer + confirms sooner; LiDAR (weak model) in between.
_CLASS_COLOR_BGR = {
    "car": (0, 200, 0), "truck": (0, 150, 255), "bus": (0, 150, 255),
    "pedestrian": (0, 0, 255), "bicycle": (255, 200, 0), "motorcycle": (255, 200, 0),
    "traffic_cone": (255, 0, 255), "barrier": (200, 200, 0),
}


def _label(d):
    """Display string with track ID when present, e.g. '#7 car 0.83'."""
    tid = f"#{d.track_id} " if getattr(d, "track_id", None) is not None else ""
    return f"{tid}{d.label} {d.score:.2f}"


# ----------------------------- rendering -----------------------------------
def _bev_box(ax, d, color):
    corners = boxes_to_corners_3d(d.box)[[0, 2, 6, 4], :2]
    ax.add_patch(Polygon(corners, closed=True, fill=False, edgecolor=color, lw=1.6))
    ax.text(d.box[0], d.box[1], _label(d), color=color, fontsize=6, ha="center", va="center")


def _draw_cam_projection(ax, cam, dets, box_color, points=None, show_labels=False):
    img = cv2.imread(str(cam["img_path"]))
    if img is None:
        ax.axis("off"); return
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if points is not None and len(points):
        uv, depth = project_points_to_image(points, np.asarray(cam["lidar_to_cam"]),
                                             np.asarray(cam["intrinsic"]),
                                             int(cam["img_w"]), int(cam["img_h"]))
        if len(uv):
            ax.scatter(uv[:, 0], uv[:, 1], c=depth, cmap="jet", s=1.2, alpha=0.45,
                       vmin=0, vmax=50, edgecolors="none")
    for d in dets:
        corners = boxes_to_corners_3d(d.box)
        uv, in_front = project_corners_to_image(corners, np.asarray(cam["lidar_to_cam"]),
                                                np.asarray(cam["intrinsic"]))
        if in_front.sum() < 2:
            continue
        cu, cv_ = uv[in_front, 0].mean(), uv[in_front, 1].mean()
        if cu < -300 or cu > cam["img_w"] + 300 or cv_ < -300 or cv_ > cam["img_h"] + 300:
            continue
        for a, b in BOX_EDGES:
            if in_front[a] and in_front[b]:
                ax.plot([uv[a, 0], uv[b, 0]], [uv[a, 1], uv[b, 1]], color=box_color,
                        lw=1.5, alpha=0.9)
        if show_labels:
            ax.text(cu, uv[in_front, 1].min(), _label(d), color=box_color, fontsize=6,
                    ha="center", va="bottom")
    ax.set_xlim(0, cam["img_w"])
    ax.set_ylim(cam["img_h"], 0)
    ax.axis("off")


def _bev_and_cameras(result, dets, box_color, bev_range, suptitle, out_path,
                     points_overlay=None, show_labels=False, velocities=False):
    pts = result["sample_data"]["points"]
    cameras = result["sample_data"]["cameras"]
    fig = plt.figure(figsize=(22, 9))
    gs = fig.add_gridspec(2, 4)
    ax_bev = fig.add_subplot(gs[:, 0])
    if len(pts):
        m = (np.abs(pts[:, 0]) < bev_range) & (np.abs(pts[:, 1]) < bev_range)
        ax_bev.scatter(pts[m, 0], pts[m, 1], s=0.4, c="#bbb", alpha=0.5)
    for d in dets:
        _bev_box(ax_bev, d, box_color)
        if velocities and d.velocity is not None and np.linalg.norm(d.velocity) > 0.3:
            ax_bev.arrow(d.box[0], d.box[1], d.velocity[0], d.velocity[1],
                         head_width=0.7, color="#d62728", length_includes_head=True)
    ax_bev.set_xlim(-bev_range, bev_range)
    ax_bev.set_ylim(-bev_range, bev_range)
    ax_bev.set_aspect("equal")
    ax_bev.set_title(f"BEV — {len(dets)} tracks")
    ax_bev.set_xlabel("x (m fwd)")
    ax_bev.set_ylabel("y (m left)")
    for r, row in enumerate(CAMERA_GRID):
        for c, cn in enumerate(row):
            ax = fig.add_subplot(gs[r, c + 1])
            cam = cameras.get(cn)
            if cam is None:
                ax.axis("off"); continue
            _draw_cam_projection(ax, cam, dets, box_color, points=points_overlay,
                                 show_labels=show_labels)
            ax.set_title(cn, fontsize=8)
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=80)
    plt.close(fig)


def render_lidar_frame(result, dets, bev_range, algo, out_path, note=""):
    _bev_and_cameras(result, dets, "#1a9850", bev_range,
                     f"LiDAR 3D detection + tracking: {algo}  —  frame {result['index']}{note}  "
                     f"(#id on boxes, jet dots = projected LiDAR points)",
                     out_path, points_overlay=result["sample_data"]["points"])


def render_radar_frame(result, dets, bev_range, algo, out_path, note=""):
    _bev_and_cameras(result, dets, "#1f77b4", bev_range,
                     f"Radar 3D detection + tracking: {algo}  —  frame {result['index']}{note}  "
                     f"(#id + class projected onto cameras, red = velocity)",
                     out_path, show_labels=True, velocities=True)


def render_camera_frame(result, dets, algo, out_path, note=""):
    cameras = result["sample_data"]["cameras"]
    by_cam = {}
    for d in dets:
        by_cam.setdefault(d.camera, []).append(d)
    fig, axes = plt.subplots(2, 3, figsize=(21, 9))
    for r, row in enumerate(CAMERA_GRID):
        for c, cn in enumerate(row):
            ax = axes[r][c]
            cam = cameras.get(cn)
            img = cv2.imread(str(cam["img_path"])) if cam else None
            if img is None:
                ax.axis("off"); continue
            for d in by_cam.get(cn, []):
                x1, y1, x2, y2 = d.bbox.astype(int)
                col = _CLASS_COLOR_BGR.get(d.label, (200, 200, 200))
                cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
                cv2.putText(img, _label(d), (x1, max(0, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title(f"{cn} ({len(by_cam.get(cn, []))})", fontsize=9)
            ax.axis("off")
    fig.suptitle(f"Camera 2D detection + tracking: {algo}  —  frame {result['index']}{note}  "
                 f"({len(dets)} tracks)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=80)
    plt.close(fig)


def _assemble(frame_dir, pattern, out_mp4, fps):
    cmd = ["ffmpeg", "-y", "-framerate", str(fps), "-i", os.path.join(frame_dir, pattern),
           "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p",
           "-c:v", "libx264", out_mp4]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    p = argparse.ArgumentParser(description="Per-sensor inspection videos with tracking")
    p.add_argument("--data-root", default="Fusion/data/sets/nuscenes")
    p.add_argument("--version", default="v1.0-mini")
    p.add_argument("--start-idx", type=int, default=0)
    p.add_argument("--num-frames", type=int, default=39)
    p.add_argument("--lidar-model", default="mmdet3d_pointpillars")
    p.add_argument("--lidar-checkpoint",
                   default="Lidar/models/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d_20210826_225857-f19d00a3.pth")
    p.add_argument("--camera-model", default="yolo26l")
    p.add_argument("--no-lidar-360", action="store_true",
                   help="Disable 360° LiDAR detection (front-only models otherwise "
                        "also detect behind via a 180°-rotated second pass).")
    p.add_argument("--bev-range", type=float, default=60.0)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--min-track-len", type=int, default=0,
                   help="Override the per-sensor 'confirm' frame counts in TRACK_CONFIG "
                        "(0 = use per-sensor defaults). Drops tracks detected in fewer "
                        "than this many frames; also emits a *_filtered video.")
    p.add_argument("--output-dir", default="Fusion/outputs/videos")
    args = p.parse_args()

    lidar_algo = LIDAR_MODEL_NAMES.get(args.lidar_model, args.lidar_model)
    cam_algo = CAMERA_MODEL_NAMES.get(args.camera_model, args.camera_model)

    indices = list(range(args.start_idx, args.start_idx + args.num_frames))
    print(f"Processing {len(indices)} frames (idx {indices[0]}..{indices[-1]})...")
    results = run_fusion_batch(
        data_root=args.data_root, indices=indices, version=args.version,
        lidar_model=args.lidar_model, lidar_checkpoint=args.lidar_checkpoint,
        camera_model_key=args.camera_model, lidar_360=not args.no_lidar_360,
        device="cuda",
    )

    # Trackers from the per-sensor TRACK_CONFIG (gap-tolerant via max_age +
    # distance_thresh; FP-robust via the post-process 'confirm' filter below).
    trk_lidar = make_tracker("lidar")
    trk_radar = make_tracker("radar")
    trk_cam = make_camera_trackers()
    confirm = {s: (args.min_track_len if args.min_track_len > 0 else TRACK_CONFIG[s]["confirm"])
               for s in TRACK_CONFIG}

    # --- Pass 1: track the whole scene, keep per-frame tracked dets + lifespans ---
    print("Pass 1: tracking all frames...")
    per_frame = {"lidar": [], "camera": [], "radar": []}
    life = {"lidar": {}, "camera": {}, "radar": {}}   # track_id -> #frames seen
    for r in results:
        l2g = r["sample_data"]["lidar_to_global"]
        lt = track_3d(trk_lidar, r["lidar_dets"], l2g, "lidar")
        rt = track_3d(trk_radar, r["radar_dets"], l2g, "radar")
        attach_velocity(rt, r["radar_dets"])
        ct = track_2d(trk_cam, r["cam_dets"])
        per_frame["lidar"].append(lt)
        per_frame["radar"].append(rt)
        per_frame["camera"].append(ct)
        for sensor, dets in (("lidar", lt), ("radar", rt), ("camera", ct)):
            for d in dets:
                life[sensor][d.track_id] = life[sensor].get(d.track_id, 0) + 1

    # Valid tracks = DETECTED in >= confirm[sensor] frames (gaps allowed; the count
    # is of real detections, so coasting through a gap doesn't inflate it).
    valid = {s: {tid for tid, n in life[s].items() if n >= confirm[s]} for s in life}
    print("  Track confirmation (gaps allowed; min real detections per sensor):")
    for s in ("lidar", "camera", "radar"):
        print(f"    {s:6s}: confirm>={confirm[s]} max_age={TRACK_CONFIG[s]['max_age']} "
              f"-> {len(life[s]):4d} tracks, {len(valid[s]):4d} kept, "
              f"{len(life[s]) - len(valid[s]):4d} dropped as false positives")

    os.makedirs(args.output_dir, exist_ok=True)
    tag = f"s{args.start_idx}_{args.num_frames}f"

    # --- Pass 2: render two video sets — all tracks, and confirmed-only ---
    modes = [("", None), ("_filtered", valid)]
    for suffix, keep in modes:
        notes = ({s: "" for s in TRACK_CONFIG} if keep is None
                 else {s: f"  [confirmed >= {confirm[s]} frames]" for s in TRACK_CONFIG})
        print(f"Pass 2: rendering '{suffix or 'all'}' videos...")
        with tempfile.TemporaryDirectory() as tmp:
            for sensor in ("lidar", "camera", "radar"):
                os.makedirs(os.path.join(tmp, sensor), exist_ok=True)
            for i, r in enumerate(results):
                lt, ct, rt = per_frame["lidar"][i], per_frame["camera"][i], per_frame["radar"][i]
                if keep is not None:
                    lt = [d for d in lt if d.track_id in keep["lidar"]]
                    rt = [d for d in rt if d.track_id in keep["radar"]]
                    ct = [d for d in ct if d.track_id in keep["camera"]]
                render_lidar_frame(r, lt, args.bev_range, lidar_algo,
                                   os.path.join(tmp, "lidar", f"f{i:04d}.png"), notes["lidar"])
                render_camera_frame(r, ct, cam_algo,
                                    os.path.join(tmp, "camera", f"f{i:04d}.png"), notes["camera"])
                render_radar_frame(r, rt, args.bev_range, RADAR_MODEL_NAME,
                                   os.path.join(tmp, "radar", f"f{i:04d}.png"), notes["radar"])
            for sensor in ("lidar", "camera", "radar"):
                out_mp4 = os.path.join(args.output_dir, f"{sensor}_{tag}{suffix}.mp4")
                _assemble(os.path.join(tmp, sensor), "f%04d.png", out_mp4, args.fps)
                print(f"  {out_mp4}")
    print("Done.")


if __name__ == "__main__":
    main()
