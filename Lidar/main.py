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
PROJECT_ROOT = LIDAR_DIR.parent
if str(LIDAR_DIR) not in sys.path:
    sys.path.insert(0, str(LIDAR_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
    p.add_argument('--track', action='store_true',
                   help='Enable multi-frame tracking with ByteTrack')
    p.add_argument('--num-frames', type=int, default=20,
                   help='Number of frames for tracking mode (default: 20)')
    args = p.parse_args()

    # Model-specific default score threshold
    if args.score_thresh is None:
        if args.model == 'mmdet3d_centerpoint':
            args.score_thresh = 0.25
        else:
            args.score_thresh = 0.15

    return args


def iterate_scene_samples(nusc, start_idx, num_frames):
    """Yield (sample_idx, sample_token) following the sample['next'] chain."""
    sample = nusc.sample[start_idx]
    for i in range(num_frames):
        yield start_idx + i, sample['token']
        if not sample['next']:
            break
        sample = nusc.get('sample', sample['next'])


def run_tracking(args, nusc, data_root, out_dir, device, model_label, checkpoint):
    """Run multi-frame tracking and save per-frame BEV PNGs."""
    from visualize_3d import load_sample_data, make_transform_matrix
    from visualize import render_bev
    from tracking import ByteTracker3D

    track_dir = out_dir / 'tracking'
    track_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = None
    if checkpoint:
        print(f"Loading {args.model} model from {checkpoint}...")
        model = load_model(args.model, checkpoint, device)

    # Determine sweeps for mmdet3d models
    sweeps_num = 0
    if args.model == 'mmdet3d_centerpoint':
        sweeps_num = 9
    elif args.model in ('mmdet3d', 'mmdet3d_pointpillars', 'mmdet3d_second'):
        sweeps_num = 9

    bev_range = [-60, -60, -5, 60, 60, 3]

    tracker = ByteTracker3D(
        high_thresh=args.score_thresh * 0.8,
        low_thresh=args.score_thresh * 0.3,
        match_thresh=0.2,
        max_age=5,
        min_hits=1,
    )

    # Collect ego poses for ego-motion compensation
    prev_global_to_lidar = None
    track_histories = {}  # track_id -> list of (x, y)
    frame_paths = []

    print(f"\nTracking {args.num_frames} frames starting from sample {args.sample_idx}...")

    for frame_i, (sample_idx, sample_token) in enumerate(
        iterate_scene_samples(nusc, args.sample_idx, args.num_frames)
    ):
        print(f"  Frame {frame_i + 1}/{args.num_frames} (sample {sample_idx})...")

        sample_data = load_sample_data(
            nusc, sample_token, data_root, sweeps_num=sweeps_num,
        )

        # Compute ego pose transform for this frame
        sample_rec = nusc.get('sample', sample_token)
        lidar_sd = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        lidar_cs = nusc.get('calibrated_sensor', lidar_sd['calibrated_sensor_token'])
        ego_pose = nusc.get('ego_pose', lidar_sd['ego_pose_token'])
        lidar_to_ego = make_transform_matrix(lidar_cs['translation'], lidar_cs['rotation'])
        ego_to_global = make_transform_matrix(ego_pose['translation'], ego_pose['rotation'])
        lidar_to_global = ego_to_global @ lidar_to_ego
        global_to_lidar = np.linalg.inv(lidar_to_global)

        # Ego-motion compensation: transform existing track states to current frame
        if prev_global_to_lidar is not None and frame_i > 0:
            for t in tracker.tracks:
                state = t.get_state()
                xyz_hom = np.array([state[0], state[1], state[2], 1.0])
                # prev_lidar -> global -> current_lidar
                xyz_global = np.linalg.inv(prev_global_to_lidar) @ xyz_hom
                xyz_cur = global_to_lidar @ xyz_global
                t.kf.x[0] = xyz_cur[0]
                t.kf.x[1] = xyz_cur[1]
                t.kf.x[2] = xyz_cur[2]

        prev_global_to_lidar = global_to_lidar.copy()

        # Run detection
        if model is not None:
            det = run_detection(
                model, sample_data['points'], device,
                model_type=args.model,
                score_thresh=args.score_thresh, nms_iou=args.nms_iou,
            )
            pred_boxes = det['boxes']
            pred_labels = det['labels']
            pred_scores = det['scores']
        else:
            pred_boxes = np.empty((0, 7))
            pred_labels = np.empty(0, dtype=int)
            pred_scores = np.empty(0)

        # Update tracker
        if len(pred_boxes) > 0:
            active_tracks = tracker.update(pred_boxes, pred_scores, pred_labels)
        else:
            active_tracks = tracker.update(
                np.empty((0, 7)), np.empty(0), np.empty(0, dtype=int),
            )

        # Extract tracked results
        if active_tracks:
            tracked_boxes = np.array([t.get_state() for t in active_tracks])
            tracked_labels = np.array([t.label for t in active_tracks])
            tracked_scores = np.array([t.score for t in active_tracks])
            tracked_ids = np.array([t.track_id for t in active_tracks])

            for t in active_tracks:
                state = t.get_state()
                track_histories.setdefault(t.track_id, []).append(
                    (state[0], state[1])
                )
        else:
            tracked_boxes = np.empty((0, 7))
            tracked_labels = np.empty(0, dtype=int)
            tracked_scores = np.empty(0)
            tracked_ids = np.empty(0, dtype=int)

        # Render BEV with track IDs
        fig_bev = render_bev(
            sample_data['points'],
            pred_boxes=tracked_boxes,
            pred_labels=tracked_labels,
            pred_scores=tracked_scores,
            gt_boxes=sample_data['gt_boxes'],
            gt_labels=sample_data['gt_labels'],
            title=f'{model_label} Tracking — Frame {frame_i + 1}',
            pc_range=bev_range,
            pred_track_ids=tracked_ids if len(tracked_ids) > 0 else None,
            track_histories=track_histories,
        )
        frame_path = track_dir / f'frame_{frame_i:04d}.png'
        fig_bev.savefig(frame_path, facecolor=fig_bev.get_facecolor())
        plt.close(fig_bev)
        frame_paths.append(frame_path)

        n_active = len(active_tracks)
        n_total = len(tracker.tracks)
        print(f"    {n_active} active tracks ({n_total} total)")

    # --- Stitch frames into H.264 MP4 video ---
    import imageio.v3 as iio
    video_path = out_dir / f'tracking_{args.model}.mp4'
    if frame_paths:
        frames = []
        for fp in frame_paths:
            img = iio.imread(fp)
            frames.append(img[:, :, :3])  # ensure RGB, drop alpha if present
        iio.imwrite(
            str(video_path),
            frames,
            fps=2,
            codec="libx264",
            plugin="pyav",
        )
        print(f"\nVideo saved: {video_path}")

    print(f"Tracking complete! {len(frame_paths)} frames in {track_dir}/")


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

    # --- Tracking mode ---
    if args.track:
        run_tracking(args, nusc, data_root, out_dir, device, model_label, checkpoint)
        return

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
    bev_range = [-60, -60, -5, 60, 60, 3]
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
