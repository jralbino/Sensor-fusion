# -*- coding: utf-8 -*-
"""
Radar Detection Pipeline — single-scene detection and visualization.

Usage:
    Radar/venv/bin/python Radar/main.py --data-root Fusion/data/sets/nuscenes --model cfar_dbscan
    Radar/venv/bin/python Radar/main.py --model radar_pillars --checkpoint Radar/outputs/run1/best.pth
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Radar.src.data.radar_utils import (
    ALL_RADAR_SENSORS,
    USEFUL_FEATURE_INDICES,
    load_radar_points,
    filter_radar_quality,
    filter_radar_rcs,
    select_features,
)
from Radar.src.detectors.detector_factory import get_radar_detector

logger = logging.getLogger(__name__)


def load_model(model_type: str, checkpoint_path: str = None, device: str = 'cpu'):
    """Load a radar detector."""
    kwargs = {}
    if model_type == 'cfar_dbscan':
        detector = get_radar_detector(model_type, **kwargs)
    else:
        import torch
        detector = get_radar_detector(model_type, **kwargs)
        if checkpoint_path:
            detector.load_checkpoint(checkpoint_path)
        detector = detector.to(device).eval()
    return detector


def draw_bev(
    points: np.ndarray,
    detections: list,
    bev_range: float = 100.0,
    bev_size: int = 800,
) -> np.ndarray:
    """Draw bird's-eye-view visualization with radar points and detections."""
    canvas = np.zeros((bev_size, bev_size, 3), dtype=np.uint8)

    def to_pixel(x, y):
        px = int((x + bev_range) / (2 * bev_range) * bev_size)
        py = int((-y + bev_range) / (2 * bev_range) * bev_size)
        return px, py

    # Draw radar points (green dots)
    for pt in points:
        px, py = to_pixel(pt[0], pt[1])
        if 0 <= px < bev_size and 0 <= py < bev_size:
            cv2.circle(canvas, (px, py), 2, (0, 200, 0), -1)

    # Draw detection boxes
    colors = {
        'car': (0, 255, 255), 'truck': (255, 165, 0), 'bus': (255, 0, 255),
        'pedestrian': (255, 0, 0), 'motorcycle': (0, 165, 255),
        'bicycle': (255, 255, 0), 'barrier': (128, 128, 128),
        'traffic_cone': (0, 128, 255), 'trailer': (200, 200, 0),
        'construction_vehicle': (128, 0, 128),
    }

    for det in detections:
        box = det.box
        cx, cy, l, w, yaw = box[0], box[1], box[3], box[4], box[6]
        color = colors.get(det.label_name, (255, 255, 255))

        # Corners of rotated box
        corners = np.array([
            [-l/2, -w/2], [l/2, -w/2], [l/2, w/2], [-l/2, w/2]
        ])
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        rot = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
        corners = (rot @ corners.T).T + np.array([cx, cy])

        pts_px = [to_pixel(c[0], c[1]) for c in corners]
        pts_arr = np.array(pts_px, dtype=np.int32)
        cv2.polylines(canvas, [pts_arr], True, color, 2)

        # Label
        px, py = to_pixel(cx, cy)
        label = f"{det.label_name} {det.score:.2f}"
        cv2.putText(canvas, label, (px - 30, py - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        # Velocity arrow
        if det.velocity is not None:
            vx, vy = det.velocity
            end_x, end_y = cx + vx * 2, cy + vy * 2
            epx, epy = to_pixel(end_x, end_y)
            cv2.arrowedLine(canvas, (px, py), (epx, epy), (0, 255, 0), 1, tipLength=0.3)

    # Origin cross
    ox, oy = to_pixel(0, 0)
    cv2.drawMarker(canvas, (ox, oy), (255, 255, 255), cv2.MARKER_CROSS, 15, 2)

    return canvas


def main():
    parser = argparse.ArgumentParser(description="Radar Detection Pipeline")
    parser.add_argument('--data-root', default='Fusion/data/sets/nuscenes')
    parser.add_argument('--version', default='v1.0-mini')
    parser.add_argument('--model', choices=['cfar_dbscan', 'radar_pillars', 'radar_centerpoint'],
                        default='cfar_dbscan')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--sample-idx', type=int, default=0)
    parser.add_argument('--sensors', nargs='+', default=None,
                        help='Radar sensors (default: all 5)')
    parser.add_argument('--nsweeps', type=int, default=6)
    parser.add_argument('--conf-threshold', type=float, default=0.3)
    parser.add_argument('--output-dir', default='Radar/outputs')
    parser.add_argument('--show', action='store_true', help='Show BEV visualization')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Load NuScenes
    from nuscenes.nuscenes import NuScenes
    logger.info("Loading NuScenes %s from %s...", args.version, args.data_root)
    nusc = NuScenes(version=args.version, dataroot=args.data_root, verbose=False)
    sample = nusc.sample[args.sample_idx]
    logger.info("Sample %d, token: %s", args.sample_idx, sample['token'][:8])

    # Load radar points
    sensors = args.sensors or ALL_RADAR_SENSORS
    raw_points = load_radar_points(nusc, sample, sensors=sensors, nsweeps=args.nsweeps)
    raw_points = filter_radar_quality(raw_points)
    logger.info("Radar points after quality filter: %d", len(raw_points))

    # Load detector
    logger.info("Loading model: %s", args.model)
    detector = load_model(args.model, args.checkpoint)

    # Run detection
    t0 = time.time()
    if args.model == 'cfar_dbscan':
        detections = detector.detect(raw_points, conf_threshold=args.conf_threshold)
    else:
        # Select useful features for DL models
        points = select_features(raw_points) if len(raw_points) > 0 else np.zeros((0, 6), dtype=np.float32)
        detections = detector.detect(points, conf_threshold=args.conf_threshold)
    elapsed = (time.time() - t0) * 1000

    logger.info("Detections: %d in %.1f ms", len(detections), elapsed)
    for det in detections:
        vel_str = ""
        if det.velocity is not None:
            vel_str = f", vel=({det.velocity[0]:.1f}, {det.velocity[1]:.1f})"
        logger.info(
            "  %s: score=%.2f, pos=(%.1f, %.1f)%s",
            det.label_name, det.score, det.box[0], det.box[1], vel_str,
        )

    # BEV visualization
    bev_points = raw_points[:, :2] if len(raw_points) > 0 else np.zeros((0, 2))
    bev_img = draw_bev(raw_points[:, :3] if len(raw_points) > 0 else np.zeros((0, 3)),
                       detections)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bev_path = output_dir / f"radar_bev_{args.model}_sample{args.sample_idx}.png"
    cv2.imwrite(str(bev_path), bev_img)
    logger.info("Saved BEV: %s", bev_path)

    if args.show:
        cv2.imshow("Radar BEV", bev_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
