#!/usr/bin/env python3
"""Compare the two multi-sensor fusion+tracking architectures on a scene:

  A) track_then_fuse  — track each modality (camera all 6, LiDAR, radar), then fuse
  B) fuse_then_track  — fuse all sensors per frame, then track

Prints a recall/precision/F1 + ID-stability table vs NuScenes GT, and renders a
side-by-side BEV video (A | B) with GT (gray) and fused tracks (#id, coloured by
the sensors that contributed). Run in the fusion container:

    docker compose run --rm fusion python Fusion/fusion_compare.py \
        --start-idx 120 --num-frames 41
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

_STROKE = [pe.withStroke(linewidth=2.5, foreground="black")]

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "Fusion")):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.late_fusion.geometry import boxes_to_corners_3d  # noqa: E402
from src.late_fusion.multimodal import (  # noqa: E402
    confirm_filter, cov_central, evaluate, fuse_then_track, track_then_fuse,
)
from src.late_fusion.pipeline import run_fusion_batch  # noqa: E402

DEFAULT_CKPT = "Lidar/models/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d_20210826_225857-f19d00a3.pth"


def _obj_color(o):
    if "camera" in o.sources:
        return "#2ca02c"   # camera-confirmed (multi-sensor)
    if "lidar" in o.sources:
        return "#ff7f0e"   # lidar-only
    return "#1f77b4"       # radar-only


def _draw_bev(ax, objs, gt, rng, title):
    # GT as light gray footprints
    for g in np.asarray(gt).reshape(-1, 7):
        c = boxes_to_corners_3d(g)[[0, 2, 6, 4], :2]
        ax.add_patch(Polygon(c, closed=True, fill=False, edgecolor="#bbbbbb", lw=1.0))
    for o in objs:
        c = boxes_to_corners_3d(o.box)[[0, 2, 6, 4], :2]
        col = _obj_color(o)
        ax.add_patch(Polygon(c, closed=True, fill=False, edgecolor=col, lw=1.8, clip_on=True))
        ax.text(o.box[0], o.box[1], f"#{o.track_id} {o.label} {o.score:.2f}", color=col,
                fontsize=8, ha="center", va="center", clip_on=True, fontweight="bold",
                path_effects=_STROKE)
    ax.set_xlim(-rng, rng)
    ax.set_ylim(-rng, rng)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x fwd")
    ax.set_ylabel("y left")


def render_frame(panels, gt, rng, frame_idx, out_path):
    """panels: list of (title, objs)."""
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(9 * n, 9))
    axes = np.atleast_1d(axes)
    for ax, (title, objs) in zip(axes, panels):
        _draw_bev(ax, objs, gt, rng, f"{title}  ({len(objs)} tracks)")
    fig.suptitle(f"Fusion comparison — frame {frame_idx}  "
                 "(gray=GT  green=camera-confirmed  cyan=multi-sensor  "
                 "orange=lidar-only  blue=radar-only)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=85)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Compare fusion architectures A vs B")
    p.add_argument("--data-root", default="Fusion/data/sets/nuscenes")
    p.add_argument("--version", default="v1.0-mini")
    p.add_argument("--start-idx", type=int, default=120)
    p.add_argument("--num-frames", type=int, default=41)
    p.add_argument("--lidar-model", default="mmdet3d_pointpillars")
    p.add_argument("--lidar-checkpoint", default=DEFAULT_CKPT)
    p.add_argument("--confirm", type=int, default=4)
    p.add_argument("--bev-range", type=float, default=60.0)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--output-dir", default="Fusion/outputs/compare")
    args = p.parse_args()

    indices = list(range(args.start_idx, args.start_idx + args.num_frames))
    print(f"Processing {len(indices)} frames (idx {indices[0]}..{indices[-1]})...")
    results = run_fusion_batch(
        data_root=args.data_root, indices=indices, version=args.version,
        lidar_model=args.lidar_model, lidar_checkpoint=args.lidar_checkpoint, device="cuda")
    gt = [r["sample_data"]["gt_boxes"] for r in results]

    A = confirm_filter(track_then_fuse(results), args.confirm)
    B = confirm_filter(fuse_then_track(results), args.confirm)
    C = confirm_filter(cov_central(results), args.confirm)
    runs = [("A track-then-fuse", A), ("B fuse-then-track", B),
            ("C cov-weighted central", C)]
    metrics = [(name, evaluate(objs, gt)) for name, objs in runs]

    print(f"\n=== Fusion method comparison (GT avg {np.mean([len(g) for g in gt]):.0f} obj/frame) ===")
    print(f"{'method':<24} {'recall':>7} {'prec':>6} {'f1':>6} {'tracks':>7} {'meanLen':>8} "
          f"{'TP':>5} {'FP':>5} {'FN':>5}")
    for name, m in metrics:
        print(f"{name:<24} {m['recall']:>7.2f} {m['precision']:>6.2f} {m['f1']:>6.2f} "
              f"{m['num_tracks']:>7d} {m['mean_track_len']:>8.1f} "
              f"{m['TP']:>5d} {m['FP']:>5d} {m['FN']:>5d}")
    best = max(metrics, key=lambda nm: nm[1]["f1"])
    print(f"\nBest by F1: {best[0]}")

    if args.no_video:
        return
    os.makedirs(args.output_dir, exist_ok=True)
    print("Rendering side-by-side video...")
    with tempfile.TemporaryDirectory() as tmp:
        for i, r in enumerate(results):
            render_frame([("A track-then-fuse", A[i]), ("B fuse-then-track", B[i]),
                          ("C cov-central", C[i])], gt[i], args.bev_range, r["index"],
                         os.path.join(tmp, f"f{i:04d}.png"))
        out = os.path.join(args.output_dir, f"fusion_ABC_s{args.start_idx}_{args.num_frames}f.mp4")
        subprocess.run(["ffmpeg", "-y", "-framerate", str(args.fps), "-i",
                        os.path.join(tmp, "f%04d.png"),
                        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p",
                        "-c:v", "libx264", out], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"  {out}")


if __name__ == "__main__":
    main()
