#!/usr/bin/env python3
"""Multi-scene evaluation + sensor ablation for the late-fusion pipeline.

Answers two questions the single-scene `fusion_compare.py` cannot:

  1. **Does fusion actually help?** — an ablation that adds one modality at a time
     (LiDAR-only -> +Camera -> +Camera+Radar), all using the same fuse-then-track
     pipeline, so each row isolates a modality's marginal contribution.
  2. **Which architecture wins, robustly?** — A / B / C scored across *many* scenes,
     reported as mean +/- std (not a single scene that can tie by chance).

Detection (the expensive GPU part) runs **once** over every selected scene; the
fusion + tracking + scoring is cheap CPU and is redone per scene with independent
trackers. Results are printed and saved to a text file.

Run in the fusion container (all 10 NuScenes-mini scenes):
    docker compose run --rm fusion python Fusion/fusion_evaluate.py
    docker compose run --rm fusion python Fusion/fusion_evaluate.py --max-scenes 3   # quick
"""
import argparse
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "Fusion")):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.late_fusion.multimodal import (  # noqa: E402
    confirm_filter, cov_central, evaluate, fuse_then_track, fuse_then_track_ablation,
    track_then_fuse,
)
from src.late_fusion.pipeline import run_fusion_batch  # noqa: E402

DEFAULT_CKPT = "Lidar/models/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d_20210826_225857-f19d00a3.pth"


def scene_index_map(data_root, version):
    """Return [(scene_name, [sample_idx, ...]), ...] in NuScenes order.

    Sample indices are positions in ``nusc.sample`` (what ``run_fusion_batch`` uses).
    """
    from nuscenes import NuScenes
    nusc = NuScenes(version=version, dataroot=data_root, verbose=False)
    tok2idx = {s["token"]: i for i, s in enumerate(nusc.sample)}
    scenes = []
    for sc in nusc.scene:
        idxs, tok = [], sc["first_sample_token"]
        while tok:
            idxs.append(tok2idx[tok])
            tok = nusc.get("sample", tok)["next"]
        scenes.append((sc["name"], idxs))
    return scenes


def _mean_std(values):
    a = np.asarray(values, float)
    return float(a.mean()), float(a.std())


def aggregate(per_scene_metrics, keys):
    """per_scene_metrics: list of metric dicts -> {key: (mean, std)} + pooled micro."""
    agg = {k: _mean_std([m[k] for m in per_scene_metrics]) for k in keys}
    tp = sum(m["TP"] for m in per_scene_metrics)
    fp = sum(m["FP"] for m in per_scene_metrics)
    fn = sum(m["FN"] for m in per_scene_metrics)
    micro_r = tp / (tp + fn) if (tp + fn) else 0.0
    micro_p = tp / (tp + fp) if (tp + fp) else 0.0
    micro_f1 = 2 * micro_r * micro_p / (micro_r + micro_p) if (micro_r + micro_p) else 0.0
    agg["_micro"] = {"recall": micro_r, "precision": micro_p, "f1": micro_f1,
                     "TP": tp, "FP": fp, "FN": fn}
    return agg


def _ms(t):
    return f"{t[0]:.2f}±{t[1]:.2f}"


def main():
    p = argparse.ArgumentParser(description="Multi-scene fusion evaluation + ablation")
    p.add_argument("--data-root", default="Fusion/data/sets/nuscenes")
    p.add_argument("--version", default="v1.0-mini")
    p.add_argument("--scenes", default=None,
                   help="comma list of 0-based scene numbers (default: all)")
    p.add_argument("--max-scenes", type=int, default=None, help="cap number of scenes")
    p.add_argument("--lidar-model", default="mmdet3d_pointpillars")
    p.add_argument("--lidar-checkpoint", default=DEFAULT_CKPT)
    p.add_argument("--confirm", type=int, default=4)
    p.add_argument("--output-dir", default="Fusion/outputs/eval")
    args = p.parse_args()

    scenes = scene_index_map(args.data_root, args.version)
    if args.scenes is not None:
        pick = [int(x) for x in args.scenes.split(",")]
        scenes = [scenes[i] for i in pick]
    if args.max_scenes is not None:
        scenes = scenes[: args.max_scenes]
    all_idx = [i for _, idxs in scenes for i in idxs]
    print(f"Evaluating {len(scenes)} scene(s), {len(all_idx)} frames total.")

    # --- Detection once over every selected scene (the expensive GPU pass) ---
    results = run_fusion_batch(
        data_root=args.data_root, indices=all_idx, version=args.version,
        lidar_model=args.lidar_model, lidar_checkpoint=args.lidar_checkpoint, device="cuda")
    by_index = {r["index"]: r for r in results}
    for r in results:                       # free point clouds; not needed for scoring
        r["sample_data"].pop("points", None)

    # Fusion variants to score (name -> callable(results_scene) -> per-frame objects).
    ablation = [
        ("LiDAR-only",          lambda rs: fuse_then_track_ablation(rs, use_camera=False, use_radar=False)),
        ("+ Camera",            lambda rs: fuse_then_track_ablation(rs, use_camera=True,  use_radar=False)),
        ("+ Camera + Radar",    lambda rs: fuse_then_track_ablation(rs, use_camera=True,  use_radar=True)),
    ]
    archs = [
        ("A track-then-fuse",   track_then_fuse),
        ("B fuse-then-track",   fuse_then_track),
        ("C cov-central",       cov_central),
    ]

    abl_metrics = {name: [] for name, _ in ablation}
    arch_metrics = {name: [] for name, _ in archs}
    for name, idxs in scenes:
        rs = [by_index[i] for i in idxs]
        gt = [r["sample_data"]["gt_boxes"] for r in rs]
        for cname, fn in ablation:
            abl_metrics[cname].append(evaluate(confirm_filter(fn(rs), args.confirm), gt))
        for cname, fn in archs:
            arch_metrics[cname].append(evaluate(confirm_filter(fn(rs), args.confirm), gt))
        gt_avg = np.mean([len(g) for g in gt])
        print(f"  scene {name:<18} ({len(idxs):>2} frames, GT~{gt_avg:.0f}/frame) done")

    keys = ["recall", "precision", "f1"]
    arch_keys = keys + ["num_tracks", "mean_track_len"]
    abl_agg = {n: aggregate(m, keys) for n, m in abl_metrics.items()}
    arch_agg = {n: aggregate(m, arch_keys) for n, m in arch_metrics.items()}

    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out()
    out(f"=== Sensor ablation (fuse-then-track) over {len(scenes)} scene(s) — "
        "mean±std across scenes ===")
    out(f"{'config':<20} {'recall':>11} {'precision':>11} {'f1':>11} {'ΔF1(micro)':>11}")
    base_micro = abl_agg["LiDAR-only"]["_micro"]["f1"]
    for name, _ in ablation:
        a = abl_agg[name]
        d = a["_micro"]["f1"] - base_micro
        dstr = "  base" if name == "LiDAR-only" else f"{d:+.3f}"
        out(f"{name:<20} {_ms(a['recall']):>11} {_ms(a['precision']):>11} "
            f"{_ms(a['f1']):>11} {dstr:>11}")
    out("(ΔF1 is on the pooled micro-F1 vs the LiDAR-only baseline)")

    out()
    out(f"=== Architecture comparison over {len(scenes)} scene(s) — mean±std ===")
    out(f"{'arch':<20} {'recall':>11} {'precision':>11} {'f1':>11} "
        f"{'tracks':>11} {'meanLen':>11} {'microF1':>8}")
    for name, _ in archs:
        a = arch_agg[name]
        out(f"{name:<20} {_ms(a['recall']):>11} {_ms(a['precision']):>11} "
            f"{_ms(a['f1']):>11} {_ms(a['num_tracks']):>11} "
            f"{_ms(a['mean_track_len']):>11} {a['_micro']['f1']:>8.3f}")
    best = max(archs, key=lambda na: arch_agg[na[0]]["_micro"]["f1"])
    out(f"\nBest architecture by pooled micro-F1: {best[0]}")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"ablation_multiscene_{len(scenes)}scenes.txt")
    header = (f"Multi-scene fusion evaluation + ablation\n"
              f"Scenes: {', '.join(n for n, _ in scenes)}\n"
              f"Frames: {len(all_idx)} | confirm_filter min_frames={args.confirm}\n"
              f"LiDAR: {args.lidar_model} | greedy BEV matching to GT, 2 m gate\n\n")
    with open(out_path, "w") as f:
        f.write(header + "\n".join(lines) + "\n")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
