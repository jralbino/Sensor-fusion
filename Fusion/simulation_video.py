#!/usr/bin/env python3
"""Data-driven scene reconstruction / simulation (LiDAR-style, top-down).

Reproduces the scene using ONLY the sensor data: accumulates the LiDAR point cloud
in the global (world) frame to rebuild the static environment, places the ego
vehicle from its pose, and overlays the fused & tracked objects (Method C) with
their trajectory trails. The view follows the ego, so it plays like a simulation /
digital twin replayed from the data.

Run in the fusion container:
    docker compose run --rm fusion python Fusion/simulation_video.py --start-idx 120 --num-frames 41
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

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "Fusion")):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.late_fusion.geometry import boxes_to_corners_3d, transform_box  # noqa: E402
from src.late_fusion.multimodal import confirm_filter, cov_central  # noqa: E402
from src.late_fusion.pipeline import run_fusion_batch  # noqa: E402

DEFAULT_CKPT = "Lidar/models/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d_20210826_225857-f19d00a3.pth"
_STROKE = [pe.withStroke(linewidth=2.5, foreground="black")]


def _color(o):
    if "camera" in o.sources:
        return "#2ca02c"
    if len(o.sources) >= 2:
        return "#17becf"
    if "lidar" in o.sources:
        return "#ff7f0e"
    return "#1f77b4"


def main():
    p = argparse.ArgumentParser(description="Data-driven scene reconstruction simulation")
    p.add_argument("--data-root", default="Fusion/data/sets/nuscenes")
    p.add_argument("--version", default="v1.0-mini")
    p.add_argument("--start-idx", type=int, default=120)
    p.add_argument("--num-frames", type=int, default=41)
    p.add_argument("--lidar-model", default="mmdet3d_pointpillars")
    p.add_argument("--lidar-checkpoint", default=DEFAULT_CKPT)
    p.add_argument("--confirm", type=int, default=4)
    p.add_argument("--view", type=float, default=55.0, help="half-window (m) around ego")
    p.add_argument("--subsample", type=int, default=2500, help="map points kept per frame")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--output-dir", default="Fusion/outputs/simulation")
    args = p.parse_args()

    indices = list(range(args.start_idx, args.start_idx + args.num_frames))
    print(f"Processing {len(indices)} frames...")
    results = run_fusion_batch(
        data_root=args.data_root, indices=indices, version=args.version,
        lidar_model=args.lidar_model, lidar_checkpoint=args.lidar_checkpoint, device="cuda")
    tracks = confirm_filter(cov_central(results), args.confirm)  # researched best method

    os.makedirs(args.output_dir, exist_ok=True)
    map_pts = []                       # accumulated global static map
    ego_traj = []                      # ego global (x, y) trail
    tag = f"s{args.start_idx}_{args.num_frames}f"

    with tempfile.TemporaryDirectory() as tmp:
        for i, r in enumerate(results):
            sd = r["sample_data"]
            l2g = sd["lidar_to_global"]
            pts = sd["points"]

            # Accumulate a subsampled global point map (the reconstructed world).
            if len(pts):
                k = min(args.subsample, len(pts))
                sel = pts[np.random.choice(len(pts), k, replace=False), :3]
                g = (l2g @ np.c_[sel, np.ones(len(sel))].T).T[:, :2]
                map_pts.append(g)
                cur = g
            else:
                cur = np.zeros((0, 2))

            ego = l2g[:2, 3]
            ego_traj.append(ego.copy())

            fig, ax = plt.subplots(figsize=(11, 11))
            ax.set_facecolor("#101418")
            if map_pts:
                M = np.concatenate(map_pts)
                ax.scatter(M[:, 0], M[:, 1], s=0.3, c="#5a6470", alpha=0.5)   # rebuilt map
            if len(cur):
                ax.scatter(cur[:, 0], cur[:, 1], s=0.5, c="#7fd1ff", alpha=0.7)  # live scan
            # Ego trajectory + ego marker
            et = np.array(ego_traj)
            ax.plot(et[:, 0], et[:, 1], "-", c="#ffd000", lw=1.2, alpha=0.8)
            ax.scatter([ego[0]], [ego[1]], s=120, marker="^", c="#ffd000",
                       edgecolors="black", zorder=5)
            # Tracked objects (global): clear box outline + track ID only (decluttered).
            for o in tracks[i]:
                gb = transform_box(o.box, l2g)
                col = _color(o)
                c = boxes_to_corners_3d(gb)[[0, 2, 6, 4], :2]
                ax.add_patch(Polygon(c, closed=True, fill=False, edgecolor=col, lw=2.2,
                                     clip_on=True))
                ax.text(gb[0], gb[1], f"#{o.track_id}", color="white", fontsize=8,
                        ha="center", va="center", clip_on=True, fontweight="bold",
                        path_effects=_STROKE)
            ax.set_xlim(ego[0] - args.view, ego[0] + args.view)
            ax.set_ylim(ego[1] - args.view, ego[1] + args.view)
            ax.set_aspect("equal")
            ax.set_title(f"Data-driven reconstruction simulation — frame {r['index']}  "
                         f"({len(tracks[i])} agents)\n"
                         "blue=live LiDAR  gray=rebuilt map  yellow=ego+path  boxes=fused tracks",
                         fontsize=11, color="white")
            ax.tick_params(colors="#888")
            fig.tight_layout()
            fig.savefig(os.path.join(tmp, f"f{i:04d}.png"), dpi=90, facecolor="#101418")
            plt.close(fig)

        out = os.path.join(args.output_dir, f"reconstruction_sim_{tag}.mp4")
        subprocess.run(["ffmpeg", "-y", "-framerate", str(args.fps), "-i",
                        os.path.join(tmp, "f%04d.png"),
                        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p",
                        "-c:v", "libx264", out], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"  {out}")
    print("Done.")


if __name__ == "__main__":
    main()
