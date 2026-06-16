#!/usr/bin/env python3
"""Batch late-fusion test across several NuScenes samples (models loaded once).

Defaults to one representative sample per mini scene. Run in the fusion container:
    docker compose run --rm fusion python Fusion/late_fusion_batch.py \
        --lidar-checkpoint Lidar/outputs/centerpoint_run/best.pth
Produces a summary table + one BEV PNG per sample in Fusion/outputs/batch/.
"""
import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "Fusion")):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.late_fusion.pipeline import run_fusion_batch  # noqa: E402
from late_fusion_demo import render_bev  # noqa: E402

# First sample index of each of the 10 mini scenes.
DEFAULT_INDICES = [0, 39, 79, 120, 161, 202, 242, 283, 324, 364]


def main():
    p = argparse.ArgumentParser(description="Batch late fusion over NuScenes samples")
    p.add_argument("--data-root", default="Fusion/data/sets/nuscenes")
    p.add_argument("--version", default="v1.0-mini")
    p.add_argument("--indices", type=int, nargs="+", default=DEFAULT_INDICES)
    p.add_argument("--lidar-model", default="mmdet3d_pointpillars")
    p.add_argument("--lidar-checkpoint",
                   default="Lidar/models/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d_20210826_225857-f19d00a3.pth")
    p.add_argument("--camera-model", default="yolo26l")
    p.add_argument("--no-radar", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", default="Fusion/outputs/batch")
    args = p.parse_args()

    results = run_fusion_batch(
        data_root=args.data_root, indices=args.indices, version=args.version,
        lidar_model=args.lidar_model, lidar_checkpoint=args.lidar_checkpoint,
        camera_model_key=args.camera_model, device=args.device,
        use_radar=not args.no_radar,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    print("\n=== Batch late fusion ===")
    print(f"{'sample':>7} | {'lidar':>5} {'cam2D':>5} {'radar':>5} | "
          f"{'fused':>5} {'cam+lidar':>9} {'3-sensor':>8} {'radar-only':>10}")
    totals = {"lidar": 0, "camera": 0, "radar": 0, "fused": 0,
              "cam_lidar": 0, "three": 0, "radar_only": 0}
    for r in results:
        c = r["counts"]
        cam_lidar = sum(1 for o in r["fused"] if {"camera", "lidar"} <= o.sources)
        three = sum(1 for o in r["fused"] if len(o.sources) == 3)
        radar_only = sum(1 for o in r["fused"] if o.sources == {"radar"})
        print(f"{r['index']:>7} | {c['lidar']:>5} {c['camera']:>5} {c['radar']:>5} | "
              f"{c['fused']:>5} {cam_lidar:>9} {three:>8} {radar_only:>10}")
        for k, v in (("lidar", c["lidar"]), ("camera", c["camera"]), ("radar", c["radar"]),
                     ("fused", c["fused"]), ("cam_lidar", cam_lidar), ("three", three),
                     ("radar_only", radar_only)):
            totals[k] += v
        render_bev(r, os.path.join(args.output_dir, f"fusion_sample{r['index']}.png"))

    n = len(results)
    print("-" * 70)
    print(f"{'TOTAL':>7} | {totals['lidar']:>5} {totals['camera']:>5} {totals['radar']:>5} | "
          f"{totals['fused']:>5} {totals['cam_lidar']:>9} {totals['three']:>8} "
          f"{totals['radar_only']:>10}   ({n} samples)")
    print(f"\nBEV PNGs saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
