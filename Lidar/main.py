#!/usr/bin/env python3
"""
LiDAR 3D Detection — Single Scene Pipeline

Loads a NuScenes scene, runs model inference, and generates
BEV, 3D interactive, and camera projection visualizations.

Usage:
    venv/bin/python main.py \
        --data-root ../Fusion/data/sets/nuscenes \
        --checkpoint outputs/test_run/best.pth \
        --sample-idx 0 \
        --output-dir outputs/demo
"""

import argparse
import sys
import time
import numpy as np
import torch
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LIDAR_DIR = Path(__file__).resolve().parent
if str(LIDAR_DIR) not in sys.path:
    sys.path.insert(0, str(LIDAR_DIR))

CLASS_NAMES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]


def load_model(model_type, checkpoint_path, device):
    """Load and return model on device."""
    if model_type in ('mmdet3d', 'mmdet3d_pointpillars', 'mmdet3d_second'):
        from src.detectors.mmdet3d_pointpillars import MMDet3DPointPillars
        model = MMDet3DPointPillars(num_classes=10)
        model.load_mmdet3d_checkpoint(checkpoint_path)
    elif model_type == 'mmdet3d_centerpoint':
        from src.detectors.mmdet3d_centerpoint import MMDet3DCenterPoint
        model = MMDet3DCenterPoint()
        model.load_mmdet3d_checkpoint(checkpoint_path)
    elif model_type == 'second':
        from src.detectors.second import SECOND
        model = SECOND(num_classes=10)
        model.load_checkpoint(checkpoint_path)
    elif model_type == 'centerpoint':
        from src.detectors.centerpoint import CenterPoint
        model = CenterPoint(num_classes=10)
        model.load_checkpoint(checkpoint_path)
    else:
        from src.detectors.pointpillars import PointPillars
        model = PointPillars(num_classes=10)
        model.load_checkpoint(checkpoint_path)
    model = model.to(device)
    model.eval()
    return model


def run_detection(model, points, device, model_type='pointpillars',
                  score_thresh=0.15, nms_iou=0.3):
    """Run inference on a raw point cloud, return detections dict."""
    from visualize import voxelize_single

    if model_type in ('mmdet3d', 'mmdet3d_pointpillars', 'mmdet3d_second'):
        from src.detectors.mmdet3d_pointpillars import MMDet3DPointPillars
        batch_dict = voxelize_single(
            points[:, :4],
            voxel_size=MMDet3DPointPillars.VOXEL_SIZE.tolist(),
            point_cloud_range=MMDet3DPointPillars.POINT_CLOUD_RANGE.tolist(),
            max_voxels=MMDet3DPointPillars.MAX_VOXELS,
            max_points_per_voxel=MMDet3DPointPillars.MAX_POINTS_PER_VOXEL,
        )
    elif model_type == 'mmdet3d_centerpoint':
        from src.detectors.mmdet3d_centerpoint import MMDet3DCenterPoint
        batch_dict = voxelize_single(
            points[:, :5] if points.shape[1] >= 5 else points,
            voxel_size=MMDet3DCenterPoint.VOXEL_SIZE.tolist(),
            point_cloud_range=MMDet3DCenterPoint.POINT_CLOUD_RANGE.tolist(),
            max_voxels=MMDet3DCenterPoint.MAX_VOXELS,
            max_points_per_voxel=MMDet3DCenterPoint.MAX_POINTS_PER_VOXEL,
        )
    else:
        batch_dict = voxelize_single(
            points[:, :4],
            voxel_size=(0.16, 0.16, 4.0),
            point_cloud_range=(0, -39.68, -3, 69.12, 39.68, 1),
        )

    for k in batch_dict:
        if isinstance(batch_dict[k], torch.Tensor):
            batch_dict[k] = batch_dict[k].to(device)

    with torch.no_grad():
        pred_dict = model(batch_dict)
        results = model.postprocess(pred_dict, score_thresh=score_thresh,
                                    nms_iou_thresh=nms_iou)
    return results[0]


def parse_args():
    p = argparse.ArgumentParser(description='LiDAR 3D Detection — Single Scene')
    p.add_argument('--data-root', required=True, help='NuScenes data root')
    p.add_argument('--version', default='v1.0-mini', help='NuScenes version')
    p.add_argument('--sample-idx', type=int, default=0, help='Sample index')
    p.add_argument('--checkpoint', default=None, help='Model checkpoint (optional, GT-only if omitted)')
    p.add_argument('--model', choices=['pointpillars', 'second', 'centerpoint',
                                       'mmdet3d', 'mmdet3d_pointpillars',
                                       'mmdet3d_second', 'mmdet3d_centerpoint'],
                   default='pointpillars',
                   help='Model architecture (default: pointpillars)')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--score-thresh', type=float, default=None,
                   help='Score threshold (default: 0.25 for mmdet3d_centerpoint, 0.15 otherwise)')
    p.add_argument('--nms-iou', type=float, default=0.3)
    p.add_argument('--output-dir', default=None,
                   help='Output directory (default: Lidar/outputs/demo/<model>)')
    args = p.parse_args()

    # Model-specific default score threshold
    if args.score_thresh is None:
        if args.model == 'mmdet3d_centerpoint':
            args.score_thresh = 0.25
        else:
            args.score_thresh = 0.15

    return args


def main():
    args = parse_args()
    data_root = Path(args.data_root)

    # Output directory — always inside Lidar/outputs/
    if args.output_dir is not None:
        out_dir = LIDAR_DIR / args.output_dir
    else:
        out_dir = LIDAR_DIR / 'outputs' / 'demo' / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Human-readable model label for titles
    MODEL_LABELS = {
        'pointpillars': 'PointPillars',
        'second': 'SECOND',
        'centerpoint': 'CenterPoint',
        'mmdet3d': 'MMDet3D PointPillars',
        'mmdet3d_pointpillars': 'MMDet3D PointPillars',
        'mmdet3d_second': 'MMDet3D SECOND',
        'mmdet3d_centerpoint': 'MMDet3D CenterPoint',
    }
    model_label = MODEL_LABELS.get(args.model, args.model)

    # --- Load NuScenes ---
    print(f"Loading NuScenes {args.version}...")
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=args.version, dataroot=str(data_root), verbose=False)
    n_samples = len(nusc.sample)

    if args.sample_idx >= n_samples:
        print(f"Error: sample index {args.sample_idx} out of range (max {n_samples - 1})")
        return

    sample_token = nusc.sample[args.sample_idx]['token']
    print(f"Sample {args.sample_idx} (token={sample_token[:8]}...)")

    # --- Load sample data ---
    # Use multi-sweep loading for mmdet3d models (trained with sweeps)
    sweeps_num = 0
    if args.model == 'mmdet3d_centerpoint':
        sweeps_num = 9   # 1 keyframe + 9 sweeps = 10 total
    elif args.model in ('mmdet3d', 'mmdet3d_pointpillars', 'mmdet3d_second'):
        sweeps_num = 9   # PointPillars/SECOND also benefit from sweeps

    sweep_msg = f" ({sweeps_num + 1} sweeps)" if sweeps_num > 0 else ""
    print(f"Loading sample data (points, GT, cameras, calibration){sweep_msg}...")
    from visualize_3d import load_sample_data
    t0 = time.time()
    sample_data = load_sample_data(nusc, sample_token, data_root,
                                   sweeps_num=sweeps_num)
    print(f"  Loaded in {time.time() - t0:.1f}s — {len(sample_data['points'])} points, "
          f"{len(sample_data['gt_boxes'])} GT boxes")

    # --- Auto-detect pretrained checkpoint for mmdet3d models ---
    checkpoint = args.checkpoint
    if checkpoint is None and args.model in ('mmdet3d', 'mmdet3d_pointpillars', 'mmdet3d_second', 'mmdet3d_centerpoint'):
        models_dir = LIDAR_DIR / 'models'
        if args.model in ('mmdet3d', 'mmdet3d_pointpillars', 'mmdet3d_second'):
            auto_ckpt = models_dir / 'hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d_20210826_225857-f19d00a3.pth'
        else:
            auto_ckpt = models_dir / 'centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220811_031844-191a3822.pth'
        if auto_ckpt.exists():
            checkpoint = str(auto_ckpt)
            print(f"Auto-detected pretrained checkpoint: {auto_ckpt.name}")

    # --- Run inference ---
    pred_boxes, pred_labels, pred_scores = None, None, None

    if checkpoint:
        print(f"Loading {args.model} model from {checkpoint}...")
        model = load_model(args.model, checkpoint, device)

        print("Running inference...")
        t0 = time.time()
        det = run_detection(model, sample_data['points'], device,
                            model_type=args.model,
                            score_thresh=args.score_thresh, nms_iou=args.nms_iou)
        infer_time = time.time() - t0
        pred_boxes = det['boxes']
        pred_labels = det['labels']
        pred_scores = det['scores']
        print(f"  {len(pred_boxes)} detections in {infer_time:.2f}s")

        # Print class breakdown
        if len(pred_labels) > 0:
            from collections import Counter
            counts = Counter(CLASS_NAMES[int(l)] for l in pred_labels)
            for cls, cnt in counts.most_common():
                print(f"    {cls}: {cnt}")
    elif args.model in ('mmdet3d', 'mmdet3d_pointpillars', 'mmdet3d_second', 'mmdet3d_centerpoint'):
        print(f"No checkpoint found for {args.model}. Run: bash models/download_pretrained.sh")
    else:
        print("No checkpoint provided — showing GT only")

    # --- 1. BEV Visualization ---
    print("\nGenerating BEV visualization...")
    from visualize import render_bev
    # Use model-appropriate point cloud range for BEV
    if args.model in ('mmdet3d', 'mmdet3d_pointpillars', 'mmdet3d_second'):
        bev_range = [-50, -50, -5, 50, 50, 3]
    elif args.model == 'mmdet3d_centerpoint':
        bev_range = [-51.2, -51.2, -5, 51.2, 51.2, 3]
    else:
        bev_range = None  # default [0, -39.68, -3, 69.12, 39.68, 1]
    fig_bev = render_bev(
        sample_data['points'],
        pred_boxes=pred_boxes,
        pred_labels=pred_labels,
        pred_scores=pred_scores,
        gt_boxes=sample_data['gt_boxes'],
        gt_labels=sample_data['gt_labels'],
        title=f'Sample {args.sample_idx} — {model_label}',
        pc_range=bev_range,
    )
    bev_path = out_dir / f'sample_{args.sample_idx:04d}_{args.model}_bev.png'
    fig_bev.savefig(bev_path, facecolor=fig_bev.get_facecolor())
    plt.close(fig_bev)
    print(f"  Saved: {bev_path}")

    # --- 2. 3D Interactive ---
    print("Generating 3D interactive visualization...")
    from visualize_3d import render_3d_scene
    fig_3d = render_3d_scene(
        sample_data['points'],
        gt_boxes=sample_data['gt_boxes'],
        gt_labels=sample_data['gt_labels'],
        pred_boxes=pred_boxes,
        pred_labels=pred_labels,
        pred_scores=pred_scores,
        max_points=30000,
        title=f'Sample {args.sample_idx} — {model_label} — 3D Scene',
    )
    html_path = out_dir / f'sample_{args.sample_idx:04d}_{args.model}_3d.html'
    fig_3d.write_html(str(html_path))
    print(f"  Saved: {html_path}")

    # --- 3. Camera Projection ---
    print("Generating camera projection visualization...")
    from visualize_3d import render_all_cameras
    fig_cam = render_all_cameras(
        sample_data,
        pred_boxes=pred_boxes,
        pred_labels=pred_labels,
        pred_scores=pred_scores,
        title=f'Sample {args.sample_idx} — {model_label}',
    )
    cam_path = out_dir / f'sample_{args.sample_idx:04d}_{args.model}_cameras.png'
    fig_cam.savefig(cam_path, dpi=150, bbox_inches='tight')
    plt.close(fig_cam)
    print(f"  Saved: {cam_path}")

    # --- Summary ---
    print(f"\nDone! All outputs saved to {out_dir}/")
    print(f"  BEV:     {bev_path.name}")
    print(f"  3D:      {html_path.name}")
    print(f"  Cameras: {cam_path.name}")


if __name__ == '__main__':
    main()
