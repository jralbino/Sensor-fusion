#!/usr/bin/env python3
"""
LiDAR 3D Detection Studio — Streamlit App

Interactive dashboard to explore NuScenes LiDAR detections
with BEV, 3D, and camera projection views.
Supports custom-trained models and pretrained MMDet3D baselines.
"""

import streamlit as st
import numpy as np
import torch
import time
import sys
from pathlib import Path
from collections import Counter

# Ensure Lidar/ is in sys.path for src imports
LIDAR_DIR = Path(__file__).resolve().parent
if str(LIDAR_DIR) not in sys.path:
    sys.path.insert(0, str(LIDAR_DIR))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CLASS_NAMES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

# Pretrained checkpoint filenames
PRETRAINED_CHECKPOINTS = {
    'mmdet3d_pointpillars': 'hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d_20210826_225857-f19d00a3.pth',
    'mmdet3d_second': 'hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d_20210826_225857-f19d00a3.pth',
    'mmdet3d_centerpoint': 'centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220811_031844-191a3822.pth',
}

# Human-readable model labels
MODEL_LABELS = {
    'pointpillars': 'PointPillars',
    'second': 'SECOND',
    'centerpoint': 'CenterPoint',
    'mmdet3d_pointpillars': 'MMDet3D PointPillars',
    'mmdet3d_second': 'MMDet3D SECOND',
    'mmdet3d_centerpoint': 'MMDet3D CenterPoint',
}

st.set_page_config(
    page_title="LiDAR 3D Detection Studio",
    layout="wide",
    page_icon="🛰️",
)


# ---------------------------------------------------------------------------
# Cached resource loaders
# ---------------------------------------------------------------------------

@st.cache_resource
def load_nuscenes(data_root: str, version: str):
    """Load NuScenes API (cached across reruns)."""
    from nuscenes.nuscenes import NuScenes
    return NuScenes(version=version, dataroot=data_root, verbose=False)


@st.cache_resource
def load_model_cached(checkpoint_path: str, device_str: str, model_type: str):
    """Load detection model (cached across reruns)."""
    from main import load_model
    device = torch.device(device_str)
    model = load_model(model_type, checkpoint_path, device)
    return model, device


@st.cache_data
def load_sample(_nusc, sample_idx: int, data_root: str):
    """Load sample data (cached by sample index)."""
    from visualize_3d import load_sample_data
    sample_token = _nusc.sample[sample_idx]['token']
    return load_sample_data(_nusc, sample_token, Path(data_root))


def find_checkpoints():
    """Find available model checkpoints (trained + pretrained)."""
    results = []

    # Trained checkpoints in models/checkpoints/
    ckpt_dir = LIDAR_DIR / 'models' / 'checkpoints'
    if ckpt_dir.exists():
        for p in sorted(ckpt_dir.glob('*.pth')):
            results.append(('custom', str(p), f"checkpoints/{p.name}"))

    # Training run checkpoints in outputs/ (fallback)
    outputs_dir = LIDAR_DIR / 'outputs'
    if outputs_dir.exists():
        for p in sorted(outputs_dir.glob('**/best.pth')):
            results.append(('custom', str(p), str(p.relative_to(LIDAR_DIR))))
        for p in sorted(outputs_dir.glob('**/latest.pth')):
            results.append(('custom', str(p), str(p.relative_to(LIDAR_DIR))))

    # Pretrained MMDet3D models in models/
    models_dir = LIDAR_DIR / 'models'
    if models_dir.exists():
        for model_key, filename in PRETRAINED_CHECKPOINTS.items():
            ckpt_path = models_dir / filename
            if ckpt_path.exists():
                label = f"{MODEL_LABELS.get(model_key, model_key)} (pretrained)"
                results.append((model_key, str(ckpt_path), label))

    return results


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def run_app():
    st.title("🛰️ LiDAR 3D Detection Studio")

    # --- Sidebar ---
    with st.sidebar:
        st.header("Configuration")

        data_root = st.text_input(
            "NuScenes data root",
            value="../Fusion/data/sets/nuscenes",
        )
        version = st.selectbox("NuScenes version", ["v1.0-mini", "v1.0-trainval"])

        # Load NuScenes
        try:
            nusc = load_nuscenes(data_root, version)
            n_samples = len(nusc.sample)
            st.success(f"Loaded {n_samples} samples")
        except Exception as e:
            st.error(f"Failed to load NuScenes: {e}")
            st.stop()

        st.divider()

        # Sample selection
        sample_idx = st.slider("Sample index", 0, n_samples - 1, 0)

        st.divider()

        # Model selection
        st.subheader("Model")
        ckpt_entries = find_checkpoints()

        if not ckpt_entries:
            st.warning("No checkpoints found. Train a model or run `bash models/download_pretrained.sh`.")
            selected_entry = None
            use_model = False
            model_type = None
        else:
            ckpt_labels = ["None (GT only)"] + [e[2] for e in ckpt_entries]
            selected_label = st.selectbox("Checkpoint", ckpt_labels)

            use_model = selected_label != "None (GT only)"

            if use_model:
                entry_idx = ckpt_labels.index(selected_label) - 1
                entry_kind, ckpt_path, _ = ckpt_entries[entry_idx]

                if entry_kind == 'custom':
                    # Custom model — let user pick architecture
                    model_type = st.selectbox(
                        "Architecture",
                        ["pointpillars", "second", "centerpoint"],
                    )
                else:
                    # Pretrained — architecture is fixed
                    model_type = entry_kind
                    arch_label = MODEL_LABELS.get(model_type, model_type)
                    st.info(f"Architecture: **{arch_label}**")

        if use_model:
            score_thresh = st.slider("Confidence threshold", 0.05, 0.80, 0.15, 0.05)
            nms_iou = st.slider("NMS IoU threshold", 0.1, 0.7, 0.3, 0.05)
            device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
            st.caption(f"Device: {device_str}")

        st.divider()
        st.caption("Built with Streamlit + Plotly")

    # Model label for visualization titles
    if use_model:
        model_label = MODEL_LABELS.get(model_type, model_type)
    else:
        model_label = "GT Only"

    # --- Load sample data ---
    with st.spinner("Loading sample data..."):
        t0 = time.time()
        sample_data = load_sample(nusc, sample_idx, data_root)
        load_time = (time.time() - t0) * 1000

    # --- Run inference ---
    pred_boxes, pred_labels, pred_scores = None, None, None
    infer_time = 0

    if use_model:
        with st.spinner("Running inference..."):
            try:
                model, device = load_model_cached(ckpt_path, device_str, model_type)
                from main import run_detection
                t0 = time.time()
                det = run_detection(
                    model, sample_data['points'], device,
                    model_type=model_type,
                    score_thresh=score_thresh, nms_iou=nms_iou,
                )
                infer_time = (time.time() - t0) * 1000
                pred_boxes = det['boxes']
                pred_labels = det['labels']
                pred_scores = det['scores']
            except Exception as e:
                st.error(f"Inference failed: {e}")

    # --- Metrics row ---
    n_gt = len(sample_data['gt_boxes'])
    n_pred = len(pred_boxes) if pred_boxes is not None else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("GT Boxes", n_gt)
    col2.metric("Predictions", n_pred)
    col3.metric("Load time", f"{load_time:.0f} ms")
    if use_model:
        col4.metric("Inference", f"{infer_time:.0f} ms")
    else:
        col4.metric("Inference", "N/A")

    # --- Class distribution ---
    if n_pred > 0:
        counts = Counter(CLASS_NAMES[int(l)] for l in pred_labels)
        with st.expander("Detection breakdown", expanded=False):
            for cls_name, count in counts.most_common():
                st.write(f"**{cls_name}**: {count}")

    # --- Visualization tabs ---
    tab_bev, tab_3d, tab_cam = st.tabs(["BEV", "3D Interactive", "Camera Projection"])

    # Tab 1: BEV
    with tab_bev:
        from visualize import render_bev
        fig_bev = render_bev(
            sample_data['points'],
            pred_boxes=pred_boxes,
            pred_labels=pred_labels,
            pred_scores=pred_scores,
            gt_boxes=sample_data['gt_boxes'],
            gt_labels=sample_data['gt_labels'],
            title=f"Sample {sample_idx} — {model_label}",
        )
        st.pyplot(fig_bev)
        plt.close(fig_bev)

    # Tab 2: 3D Interactive
    with tab_3d:
        from visualize_3d import render_3d_scene
        fig_3d = render_3d_scene(
            sample_data['points'],
            gt_boxes=sample_data['gt_boxes'],
            gt_labels=sample_data['gt_labels'],
            pred_boxes=pred_boxes,
            pred_labels=pred_labels,
            pred_scores=pred_scores,
            max_points=30000,
            title=f"Sample {sample_idx} — {model_label} — 3D Scene",
        )
        st.plotly_chart(fig_3d, use_container_width=True)

    # Tab 3: Camera projection
    with tab_cam:
        from visualize_3d import render_all_cameras
        fig_cam = render_all_cameras(
            sample_data,
            pred_boxes=pred_boxes,
            pred_labels=pred_labels,
            pred_scores=pred_scores,
            title=f"Sample {sample_idx} — {model_label}",
        )
        st.pyplot(fig_cam)
        plt.close(fig_cam)


run_app()
