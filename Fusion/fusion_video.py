#!/usr/bin/env python3
"""Full fusion-pipeline videos for one scene (all labels clipped to frame so the
frame size never changes):

  Per-modality fusion (architecture A, stage 1 — each modality tracked on its own):
    modality_camera_*  6-camera grid, 2D tracks (all 6 cameras)
    modality_lidar_*   BEV 3D tracks + projected on 6 cameras
    modality_radar_*   BEV 3D tracks (+velocity) + projected on 6 cameras
  Cross-modality fusion:
    fusionA_final_*    A: fuse the per-modality tracks (all modalities together)
    fusionB_single_*   B: one fusion stage, all raw sensors cooperating, then track

Run in the fusion container:
    docker compose run --rm fusion python Fusion/fusion_video.py --start-idx 120 --num-frames 41
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
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# Black outline around label text for legibility over any background.
_STROKE = [pe.withStroke(linewidth=2.5, foreground="black")]

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "Fusion")):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.late_fusion.geometry import (  # noqa: E402
    BOX_EDGES, boxes_to_corners_3d, project_corners_to_image,
)
from src.late_fusion.multimodal import (  # noqa: E402
    confirm_filter, cov_central, fuse_then_track, track_then_fuse,
)
from src.late_fusion.pipeline import run_fusion_batch  # noqa: E402
from src.late_fusion.tracking_helpers import (  # noqa: E402
    TRACK_CONFIG, attach_velocity, make_camera_trackers, make_tracker, track_2d, track_3d,
)
from tracking.bytetrack import BaseTrack  # noqa: E402

DEFAULT_CKPT = "Lidar/models/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d_20210826_225857-f19d00a3.pth"
CAMERA_GRID = [["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"],
               ["CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]]
_CLASS_COLOR_BGR = {
    "car": (0, 200, 0), "truck": (0, 150, 255), "bus": (0, 150, 255),
    "pedestrian": (0, 0, 255), "bicycle": (255, 200, 0), "motorcycle": (255, 200, 0),
    "traffic_cone": (255, 0, 255), "barrier": (200, 200, 0),
}


def _badge(o):
    order = [("lidar", "L"), ("camera", "C"), ("radar", "R")]
    return "·".join(s for k, s in order if k in getattr(o, "sources", set()))


def _color(o):
    src = getattr(o, "sources", None)
    if src is None:                                  # single-modality detection
        return {"lidar": "#ff7f0e", "radar": "#1f77b4", "camera": "#2ca02c"}.get(
            getattr(o, "source", "lidar"), "#2ca02c")
    if "camera" in src:
        return "#2ca02c"
    if len(src) >= 2:
        return "#17becf"
    if "lidar" in src:
        return "#ff7f0e"
    return "#1f77b4"


def _txt(ax, x, y, s, color, fs=9, va="center"):
    """Bold, black-outlined text, clipped to the axes (so out-of-frame labels
    never change the frame size)."""
    ax.text(x, y, s, color=color, fontsize=fs, ha="center", va=va, clip_on=True,
            fontweight="bold", path_effects=_STROKE)


def _bev_box(ax, box, color, lw=1.8, alpha=1.0, text=None):
    c = boxes_to_corners_3d(box)[[0, 2, 6, 4], :2]
    ax.add_patch(Polygon(c, closed=True, fill=False, edgecolor=color, lw=lw, alpha=alpha,
                         clip_on=True))
    if text:
        _txt(ax, box[0], box[1], text, color)


def _cam_grid(fig, gs, cameras, objs, badge=False, col_start=1):
    for r, row in enumerate(CAMERA_GRID):
        for c, cn in enumerate(row):
            ax = fig.add_subplot(gs[r, c + col_start])
            cam = cameras.get(cn)
            img = cv2.imread(str(cam["img_path"])) if cam else None
            if img is None:
                ax.axis("off"); continue
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            w, h = int(cam["img_w"]), int(cam["img_h"])
            for o in objs:
                uv, infront = project_corners_to_image(
                    boxes_to_corners_3d(o.box), np.asarray(cam["lidar_to_cam"]),
                    np.asarray(cam["intrinsic"]))
                if infront.sum() < 2:
                    continue
                cu, cv_ = uv[infront, 0].mean(), uv[infront, 1].mean()
                if cu < -300 or cu > w + 300 or cv_ < -300 or cv_ > h + 300:
                    continue
                col = _color(o)
                for a, b in BOX_EDGES:
                    if infront[a] and infront[b]:
                        ax.plot([uv[a, 0], uv[b, 0]], [uv[a, 1], uv[b, 1]], color=col, lw=1.4)
                # Only label when the anchor is inside the image (keeps frame fixed).
                ly = uv[infront, 1].min()
                if 0 <= cu <= w and 0 <= ly <= h:
                    lab = f"#{o.track_id} {o.label}" if badge else f"#{o.track_id}"
                    ax.text(cu, ly, lab, color=col, fontsize=8, ha="center", va="bottom",
                            clip_on=True, fontweight="bold", path_effects=_STROKE)
            ax.set_xlim(0, w); ax.set_ylim(h, 0); ax.set_title(cn, fontsize=8); ax.axis("off")


def render_3d(r, objs, suptitle, bev_title, out, velocity=False, badge=False,
              raw_lidar=None, raw_radar=None, rng=60.0):
    sd = r["sample_data"]
    fig = plt.figure(figsize=(22, 9))
    gs = fig.add_gridspec(2, 4)
    ax = fig.add_subplot(gs[:, 0])
    pts = sd["points"]
    if len(pts):
        m = (np.abs(pts[:, 0]) < rng) & (np.abs(pts[:, 1]) < rng)
        ax.scatter(pts[m, 0], pts[m, 1], s=0.3, c="#e3e3e3")
    if raw_lidar:
        for d in raw_lidar:
            _bev_box(ax, d.box, "#ffbb78", lw=0.8, alpha=0.5)
    if raw_radar:
        for d in raw_radar:
            ax.scatter(d.box[0], d.box[1], s=16, c="#9edae5", marker="x", clip_on=True)
    for o in objs:
        col = _color(o)
        if badge:
            t = f"#{o.track_id} {o.label}\n{o.score:.2f} [{_badge(o)}]"
        else:
            t = f"#{o.track_id} {o.label}\n{o.score:.2f}"
        _bev_box(ax, o.box, col, lw=2.0, text=t)
        if velocity and getattr(o, "velocity", None) is not None and np.linalg.norm(o.velocity) > 0.3:
            ax.arrow(o.box[0], o.box[1], o.velocity[0], o.velocity[1], head_width=0.7,
                     color="#d62728", length_includes_head=True, clip_on=True)
    ax.set_xlim(-rng, rng); ax.set_ylim(-rng, rng); ax.set_aspect("equal")
    ax.set_xlabel("x fwd"); ax.set_ylabel("y left"); ax.set_title(bev_title, fontsize=10)
    _cam_grid(fig, gs, sd["cameras"], objs, badge=badge)
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout(); fig.savefig(out, dpi=80); plt.close(fig)


def render_camera(r, cam_objs, suptitle, out):
    cameras = r["sample_data"]["cameras"]
    by_cam = {}
    for d in cam_objs:
        by_cam.setdefault(d.camera, []).append(d)
    fig, axes = plt.subplots(2, 3, figsize=(21, 9))
    for ri, row in enumerate(CAMERA_GRID):
        for ci, cn in enumerate(row):
            ax = axes[ri][ci]
            cam = cameras.get(cn)
            img = cv2.imread(str(cam["img_path"])) if cam else None
            if img is None:
                ax.axis("off"); continue
            for d in by_cam.get(cn, []):
                x1, y1, x2, y2 = d.bbox.astype(int)
                col = _CLASS_COLOR_BGR.get(d.label, (200, 200, 200))
                cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
                # cv2 text is drawn on the array → inherently clipped to the image.
                # Draw a black outline first, then the coloured text, for legibility.
                lab = f"#{d.track_id} {d.label} {d.score:.2f}"
                org = (x1, max(16, y1 - 6))
                cv2.putText(img, lab, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(img, lab, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title(f"{cn} ({len(by_cam.get(cn, []))})", fontsize=9); ax.axis("off")
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout(); fig.savefig(out, dpi=80); plt.close(fig)


def _encode(tmp, out, fps):
    subprocess.run(["ffmpeg", "-y", "-framerate", str(fps), "-i", os.path.join(tmp, "f%04d.png"),
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p",
                    "-c:v", "libx264", out], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _video(results, render_fn, out_path, fps):
    with tempfile.TemporaryDirectory() as tmp:
        for i, r in enumerate(results):
            render_fn(i, r, os.path.join(tmp, f"f{i:04d}.png"))
        _encode(tmp, out_path, fps)
    print(f"  {out_path}")


def main():
    p = argparse.ArgumentParser(description="Full fusion-pipeline videos")
    p.add_argument("--data-root", default="Fusion/data/sets/nuscenes")
    p.add_argument("--version", default="v1.0-mini")
    p.add_argument("--start-idx", type=int, default=120)
    p.add_argument("--num-frames", type=int, default=41)
    p.add_argument("--lidar-model", default="mmdet3d_pointpillars")
    p.add_argument("--lidar-checkpoint", default=DEFAULT_CKPT)
    p.add_argument("--confirm", type=int, default=4)
    p.add_argument("--bev-range", type=float, default=60.0)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--output-dir", default="Fusion/outputs/fusion")
    args = p.parse_args()
    rng = args.bev_range

    indices = list(range(args.start_idx, args.start_idx + args.num_frames))
    print(f"Processing {len(indices)} frames...")
    results = run_fusion_batch(
        data_root=args.data_root, indices=indices, version=args.version,
        lidar_model=args.lidar_model, lidar_checkpoint=args.lidar_checkpoint, device="cuda")

    # --- Per-modality tracking (architecture A, stage 1) ---
    BaseTrack.reset_id_counter()
    tl, tr, tc = make_tracker("lidar"), make_tracker("radar"), make_camera_trackers()
    lidar_pf, radar_pf, cam_pf = [], [], []
    for r in results:
        l2g = r["sample_data"]["lidar_to_global"]
        lidar_pf.append(track_3d(tl, r["lidar_dets"], l2g, "lidar"))
        rt = track_3d(tr, r["radar_dets"], l2g, "radar"); attach_velocity(rt, r["radar_dets"])
        radar_pf.append(rt)
        cam_pf.append(track_2d(tc, r["cam_dets"]))
    lidar_pf = confirm_filter(lidar_pf, TRACK_CONFIG["lidar"]["confirm"])
    radar_pf = confirm_filter(radar_pf, TRACK_CONFIG["radar"]["confirm"])
    cam_pf = confirm_filter(cam_pf, TRACK_CONFIG["camera"]["confirm"])

    # --- Cross-modality fusion (3 methods) ---
    A = confirm_filter(track_then_fuse(results), args.confirm)
    B = confirm_filter(fuse_then_track(results), args.confirm)
    C = confirm_filter(cov_central(results), args.confirm)

    os.makedirs(args.output_dir, exist_ok=True)
    tag = f"s{args.start_idx}_{args.num_frames}f"

    print("Rendering per-modality fusion videos...")
    _video(results, lambda i, r, o: render_camera(
        r, cam_pf[i], f"Modality fusion — CAMERA (YOLO26, all 6 views) — frame {r['index']}", o),
        os.path.join(args.output_dir, f"modality_camera_{tag}.mp4"), args.fps)
    _video(results, lambda i, r, o: render_3d(
        r, lidar_pf[i], f"Modality fusion — LiDAR (PointPillars) — frame {r['index']}",
        f"LiDAR tracks ({len(lidar_pf[i])})", o, rng=rng),
        os.path.join(args.output_dir, f"modality_lidar_{tag}.mp4"), args.fps)
    _video(results, lambda i, r, o: render_3d(
        r, radar_pf[i], f"Modality fusion — RADAR (CFAR+DBSCAN) — frame {r['index']}",
        f"Radar tracks ({len(radar_pf[i])})", o, velocity=True, rng=rng),
        os.path.join(args.output_dir, f"modality_radar_{tag}.mp4"), args.fps)

    print("Rendering cross-modality fusion videos...")
    _video(results, lambda i, r, o: render_3d(
        r, A[i], f"Fusion A (per-modality tracks → fused) — frame {r['index']}",
        f"A fused tracks ({len(A[i])})", o, velocity=True, badge=True, rng=rng),
        os.path.join(args.output_dir, f"fusionA_final_{tag}.mp4"), args.fps)
    _video(results, lambda i, r, o: render_3d(
        r, B[i], f"Fusion B (single stage, all sensors → tracked) — frame {r['index']}",
        f"B fused tracks ({len(B[i])})", o, velocity=True, badge=True,
        raw_lidar=results[i]["lidar_dets"], raw_radar=results[i]["radar_dets"], rng=rng),
        os.path.join(args.output_dir, f"fusionB_single_{tag}.mp4"), args.fps)
    _video(results, lambda i, r, o: render_3d(
        r, C[i], f"Fusion C (covariance-weighted central, researched best) — frame {r['index']}",
        f"C fused tracks ({len(C[i])})", o, velocity=True, badge=True, rng=rng),
        os.path.join(args.output_dir, f"fusionC_covcentral_{tag}.mp4"), args.fps)
    print("Done.")


if __name__ == "__main__":
    main()
