#!/usr/bin/env python3
"""
Single-file inference for PointPillars.

Reads a LiDAR .bin file, runs detection, and prints a table of results.
"""

import argparse
import numpy as np
import torch
from pathlib import Path

CLASS_NAMES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]


def voxelize_single(points, voxel_size, point_cloud_range, max_points=32, max_voxels=40000):
    """Voxelize a single point cloud and return a batch_dict ready for the model."""
    from src.data.datasets import voxelize_points

    voxels, coords, num_points = voxelize_points(
        points,
        np.array(voxel_size),
        np.array(point_cloud_range),
        max_points_per_voxel=max_points,
        max_voxels=max_voxels,
    )

    # Add batch dimension to coords
    batch_idx = np.zeros((len(coords), 1), dtype=np.int32)
    coords = np.concatenate([batch_idx, coords], axis=1)

    return {
        'voxels': torch.from_numpy(voxels).float(),
        'voxel_coords': torch.from_numpy(coords).int(),
        'voxel_num_points': torch.from_numpy(num_points).int(),
        'batch_size': 1,
    }


def parse_args():
    parser = argparse.ArgumentParser(description='LiDAR 3D Detection Inference')
    parser.add_argument('--checkpoint', required=True, help='Model checkpoint')
    parser.add_argument('--input', required=True, help='Path to .bin point cloud')
    parser.add_argument('--model', choices=['pointpillars', 'second', 'centerpoint'], default='pointpillars',
                        help='Model architecture (default: pointpillars)')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--score-thresh', type=float, default=0.15)
    parser.add_argument('--nms-iou', type=float, default=0.3)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Load model
    print(f"Loading {args.model} model...")
    if args.model == 'second':
        from src.detectors.second import SECOND
        model = SECOND(num_classes=10)
    elif args.model == 'centerpoint':
        from src.detectors.centerpoint import CenterPoint
        model = CenterPoint(num_classes=10)
    else:
        from src.detectors.pointpillars import PointPillars
        model = PointPillars(num_classes=10)
    model.load_checkpoint(args.checkpoint)
    model = model.to(device)
    model.eval()

    # Load point cloud
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: file not found: {input_path}")
        return

    points = np.fromfile(str(input_path), dtype=np.float32).reshape(-1, 5)[:, :4]
    print(f"Loaded {len(points)} points from {input_path.name}")

    # Voxelize
    batch_dict = voxelize_single(
        points,
        voxel_size=(0.16, 0.16, 4.0),
        point_cloud_range=(0, -39.68, -3, 69.12, 39.68, 1),
    )
    for key in batch_dict:
        if isinstance(batch_dict[key], torch.Tensor):
            batch_dict[key] = batch_dict[key].to(device)

    # Inference
    with torch.no_grad():
        pred_dict = model(batch_dict)
        results = model.postprocess(
            pred_dict,
            score_thresh=args.score_thresh,
            nms_iou_thresh=args.nms_iou,
        )

    det = results[0]
    n = len(det['scores'])

    if n == 0:
        print("\nNo detections found.")
        return

    # Sort by score
    order = np.argsort(det['scores'])[::-1]
    boxes = det['boxes'][order]
    scores = det['scores'][order]
    labels = det['labels'][order]

    # Print table
    print(f"\n{'#':<4} {'Class':<22} {'Score':>6} {'X':>7} {'Y':>7} {'Z':>7} {'L':>5} {'W':>5} {'H':>5}")
    print("-" * 75)
    for i in range(n):
        cls = CLASS_NAMES[labels[i]] if labels[i] < len(CLASS_NAMES) else f"cls_{labels[i]}"
        b = boxes[i]
        print(f"{i+1:<4} {cls:<22} {scores[i]:>6.3f} {b[0]:>7.2f} {b[1]:>7.2f} {b[2]:>7.2f} {b[3]:>5.1f} {b[4]:>5.1f} {b[5]:>5.1f}")

    print(f"\nTotal: {n} detections")


if __name__ == '__main__':
    main()
