#!/usr/bin/env python3
"""Late-fusion demo: run camera + LiDAR + radar on one NuScenes sample and fuse.

Run inside the fusion container:
    docker compose run --rm fusion python Fusion/late_fusion_demo.py \
        --data-root Fusion/data/sets/nuscenes \
        --lidar-checkpoint Lidar/outputs/centerpoint_run/best.pth \
        --sample-idx 0

Produces a console summary and a BEV PNG of the fused objects
(green = camera-confirmed, orange = LiDAR-only, blue = radar-only;
arrows = radar velocity).
"""
import argparse
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# Repo root on path (for module_loader + src.late_fusion when run as a script).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "Fusion")):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.late_fusion.geometry import boxes_to_corners_3d  # noqa: E402
from src.late_fusion.pipeline import run_fusion_on_sample  # noqa: E402

_SOURCE_COLOR = {"camera": "#2ca02c", "lidar": "#ff7f0e", "radar": "#1f77b4"}


def _object_color(obj) -> str:
    if "camera" in obj.sources:
        return _SOURCE_COLOR["camera"]
    if "lidar" in obj.sources:
        return _SOURCE_COLOR["lidar"]
    return _SOURCE_COLOR["radar"]


def render_bev(result: dict, out_path: str, bev_range: float = 60.0) -> None:
    points = result["sample_data"]["points"]
    fused = result["fused"]

    fig, ax = plt.subplots(figsize=(11, 11))
    if len(points):
        m = (np.abs(points[:, 0]) < bev_range) & (np.abs(points[:, 1]) < bev_range)
        ax.scatter(points[m, 0], points[m, 1], s=0.4, c="#999999", alpha=0.5)

    for obj in fused:
        corners = boxes_to_corners_3d(obj.box)[[0, 2, 6, 4], :2]  # BEV footprint
        color = _object_color(obj)
        ax.add_patch(Polygon(corners, closed=True, fill=False, edgecolor=color, lw=2))
        x, y = obj.box[0], obj.box[1]
        ax.text(x, y, f"{obj.label}\n{obj.score:.2f}", color=color, fontsize=7,
                ha="center", va="center")
        if obj.velocity is not None and np.linalg.norm(obj.velocity) > 0.3:
            ax.arrow(x, y, obj.velocity[0], obj.velocity[1], head_width=0.6,
                     color=_SOURCE_COLOR["radar"], length_includes_head=True)

    ax.set_xlim(-bev_range, bev_range)
    ax.set_ylim(-bev_range, bev_range)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m, forward)")
    ax.set_ylabel("y (m, left)")
    c = result["counts"]
    ax.set_title(f"Late fusion — fused={c['fused']} "
                 f"(lidar={c['lidar']}, camera2D={c['camera']}, radar={c['radar']})\n"
                 "green=camera-confirmed  orange=lidar-only  blue=radar-only")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Late-fusion demo on a NuScenes sample")
    p.add_argument("--data-root", default="Fusion/data/sets/nuscenes")
    p.add_argument("--version", default="v1.0-mini")
    p.add_argument("--sample-idx", type=int, default=0)
    p.add_argument("--lidar-model", default="mmdet3d_pointpillars")
    p.add_argument("--lidar-checkpoint",
                   default="Lidar/models/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d_20210826_225857-f19d00a3.pth")
    p.add_argument("--camera-model", default="yolo26l")
    p.add_argument("--no-radar", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", default="Fusion/outputs")
    args = p.parse_args()

    result = run_fusion_on_sample(
        data_root=args.data_root, version=args.version, sample_idx=args.sample_idx,
        lidar_model=args.lidar_model, lidar_checkpoint=args.lidar_checkpoint,
        camera_model_key=args.camera_model, device=args.device,
        use_radar=not args.no_radar,
    )

    c = result["counts"]
    print("\n=== Late Fusion Result ===")
    print(f"  LiDAR 3D dets : {c['lidar']}")
    print(f"  Camera 2D dets: {c['camera']}")
    print(f"  Radar 3D dets : {c['radar']}")
    print(f"  Fused objects : {c['fused']}")
    print("\n  idx  class           score  sensors                 velocity")
    for i, o in enumerate(result["fused"]):
        srcs = ",".join(sorted(o.sources))
        vel = f"[{o.velocity[0]:+.1f},{o.velocity[1]:+.1f}]" if o.velocity is not None else "-"
        print(f"  {i:3d}  {o.label:14s}  {o.score:4.2f}   {srcs:22s}  {vel}")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"late_fusion_sample{args.sample_idx}.png")
    render_bev(result, out_path)
    print(f"\nBEV saved to {out_path}")


if __name__ == "__main__":
    main()
