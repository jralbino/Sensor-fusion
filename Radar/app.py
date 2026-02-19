# -*- coding: utf-8 -*-
"""
Radar Detection Studio — Streamlit Application.

Interactive interface for radar-based object detection on NuScenes.
"""
import streamlit as st
import cv2
import numpy as np
import time
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Radar.config.models import RADAR_DETECTORS

st.set_page_config(page_title="Radar Detection Studio", layout="wide", page_icon="📡")
st.title("📡 Radar Detection Studio")


@st.cache_resource
def load_nuscenes(data_root: str, version: str):
    from nuscenes.nuscenes import NuScenes
    return NuScenes(version=version, dataroot=data_root, verbose=False)


@st.cache_resource
def load_detector(model_type: str):
    from Radar.src.detectors.detector_factory import get_radar_detector
    return get_radar_detector(model_type)


def draw_bev(points, detections, bev_range=100.0, bev_size=800):
    """BEV visualization with radar points and detection boxes."""
    canvas = np.zeros((bev_size, bev_size, 3), dtype=np.uint8)

    def to_px(x, y):
        px = int((x + bev_range) / (2 * bev_range) * bev_size)
        py = int((-y + bev_range) / (2 * bev_range) * bev_size)
        return px, py

    # Radar points (green)
    for pt in points:
        px, py = to_px(pt[0], pt[1])
        if 0 <= px < bev_size and 0 <= py < bev_size:
            cv2.circle(canvas, (px, py), 2, (0, 200, 0), -1)

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

        corners = np.array([[-l/2, -w/2], [l/2, -w/2], [l/2, w/2], [-l/2, w/2]])
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        rot = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
        corners = (rot @ corners.T).T + np.array([cx, cy])

        pts_arr = np.array([to_px(c[0], c[1]) for c in corners], dtype=np.int32)
        cv2.polylines(canvas, [pts_arr], True, color, 2)

        px, py = to_px(cx, cy)
        cv2.putText(canvas, f"{det.label_name} {det.score:.2f}",
                    (px - 30, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        if det.velocity is not None:
            vx, vy = det.velocity
            epx, epy = to_px(cx + vx * 2, cy + vy * 2)
            cv2.arrowedLine(canvas, (px, py), (epx, epy), (0, 255, 0), 1, tipLength=0.3)

    ox, oy = to_px(0, 0)
    cv2.drawMarker(canvas, (ox, oy), (255, 255, 255), cv2.MARKER_CROSS, 15, 2)
    return canvas


def run_app():
    with st.sidebar:
        st.header("Configuration")

        # NuScenes data
        st.subheader("Data")
        data_root = st.text_input(
            "NuScenes data root",
            value=str(PROJECT_ROOT / "Fusion" / "data" / "sets" / "nuscenes"),
            key="radar_data_root",
        )
        version = st.selectbox("Version", ["v1.0-mini", "v1.0-trainval"], key="radar_version")
        st.divider()

        # Model selection
        st.subheader("Model")
        model_name = st.selectbox("Radar Detector", list(RADAR_DETECTORS.keys()), key="radar_model")
        model_info = RADAR_DETECTORS[model_name]
        st.divider()

        # Radar options
        st.subheader("Radar Options")
        sensors = st.multiselect(
            "Sensors",
            ["RADAR_FRONT", "RADAR_FRONT_LEFT", "RADAR_FRONT_RIGHT",
             "RADAR_BACK_LEFT", "RADAR_BACK_RIGHT"],
            default=["RADAR_FRONT"],
            key="radar_sensors",
        )
        nsweeps = st.slider("Sweeps", 1, 10, 6, key="radar_nsweeps")
        conf_thres = st.slider("Confidence", 0.1, 1.0, 0.3, 0.05, key="radar_conf")
        bev_range = st.slider("BEV Range (m)", 20, 200, 100, 10, key="radar_bev_range")

    # Load NuScenes
    try:
        nusc = load_nuscenes(data_root, version)
    except Exception as e:
        st.error(f"Failed to load NuScenes: {e}")
        return

    n_samples = len(nusc.sample)
    sample_idx = st.slider("Sample index", 0, max(0, n_samples - 1), 0, key="radar_sample_idx")
    sample = nusc.sample[sample_idx]

    # Load model
    with st.spinner(f"Loading {model_name}..."):
        detector = load_detector(model_info['type'])

    # Load radar points
    from Radar.src.data.radar_utils import (
        load_radar_points, filter_radar_quality, filter_radar_rcs, select_features,
    )

    raw_points = load_radar_points(nusc, sample, sensors=sensors, nsweeps=nsweeps)
    raw_points = filter_radar_quality(raw_points)

    # Metrics
    col_info1, col_info2, col_info3 = st.columns(3)
    col_info1.metric("Raw Radar Points", len(raw_points))

    # Run detection
    t0 = time.time()
    if model_info['type'] == 'cfar_dbscan':
        detections = detector.detect(raw_points, conf_threshold=conf_thres)
    else:
        points = select_features(raw_points) if len(raw_points) > 0 else np.zeros((0, 6), dtype=np.float32)
        detections = detector.detect(points, conf_threshold=conf_thres)
    elapsed_ms = (time.time() - t0) * 1000

    col_info2.metric("Detections", len(detections))
    col_info3.metric("Latency", f"{elapsed_ms:.1f} ms")

    # BEV visualization
    st.subheader("Bird's Eye View")
    pts_3d = raw_points[:, :3] if len(raw_points) > 0 else np.zeros((0, 3))
    bev_img = draw_bev(pts_3d, detections, bev_range=bev_range)
    st.image(cv2.cvtColor(bev_img, cv2.COLOR_BGR2RGB), width=800)

    # Detection details
    col1, col2 = st.columns([2, 1])

    with col1:
        # Camera view with radar projected (if CAM_FRONT available)
        st.subheader("Front Camera")
        cam_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])
        cam_path = Path(data_root) / cam_data['filename']
        if cam_path.exists():
            cam_img = cv2.imread(str(cam_path))
            st.image(cv2.cvtColor(cam_img, cv2.COLOR_BGR2RGB), width="stretch")
        else:
            st.info("Camera image not found")

    with col2:
        st.subheader("Detection Summary")
        if detections:
            from collections import Counter
            counts = Counter(d.label_name for d in detections)
            for name, count in counts.most_common():
                st.success(f"**{name}**: {count}")

            with st.expander("Details"):
                for det in detections:
                    vel_str = ""
                    if det.velocity is not None:
                        vel_str = f" | vel=({det.velocity[0]:.1f}, {det.velocity[1]:.1f})"
                    st.text(
                        f"{det.label_name}: {det.score:.2f} "
                        f"@ ({det.box[0]:.1f}, {det.box[1]:.1f}){vel_str}"
                    )
        else:
            st.info("No detections")

    # Model info
    with st.expander("Model Info"):
        info = detector.get_model_info()
        st.json(info)


run_app()
st.markdown("---")
st.caption(f"Project: {PROJECT_ROOT}")
