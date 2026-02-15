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

# Ensure Lidar/ and project root are in sys.path
LIDAR_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LIDAR_DIR.parent
if str(LIDAR_DIR) not in sys.path:
    sys.path.insert(0, str(LIDAR_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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

        # Tracking controls
        st.subheader("Tracking")
        enable_tracking = st.checkbox("Enable ByteTrack", value=False)
        if enable_tracking:
            num_frames = st.slider("Number of frames", 2, 40, 10)

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
    if enable_tracking and use_model:
        tab_bev, tab_3d, tab_cam, tab_track = st.tabs(
            ["BEV", "3D Interactive", "Camera Projection", "Tracking"]
        )
    else:
        tab_bev, tab_3d, tab_cam = st.tabs(["BEV", "3D Interactive", "Camera Projection"])
        tab_track = None

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

    # Tab 4: Tracking
    if tab_track is not None:
        with tab_track:
            import tempfile
            import os
            import imageio.v3 as iio
            from tracking import ByteTracker3D
            from visualize_3d import load_sample_data as _load_sample, make_transform_matrix
            from main import run_detection, iterate_scene_samples

            tracker = ByteTracker3D(
                high_thresh=score_thresh * 0.8,
                low_thresh=score_thresh * 0.3,
                match_thresh=0.2,
                max_age=5,
                min_hits=1,
            )

            prev_global_to_lidar = None
            track_histories = {}
            frame_images = []  # list of numpy RGB images

            model_obj, device = load_model_cached(ckpt_path, device_str, model_type)

            progress = st.progress(0, text="Running tracking...")
            for frame_i, (sidx, stok) in enumerate(
                iterate_scene_samples(nusc, sample_idx, num_frames)
            ):
                progress.progress(
                    (frame_i + 1) / num_frames,
                    text=f"Frame {frame_i + 1}/{num_frames}",
                )
                sd = _load_sample(nusc, stok, Path(data_root))

                # Ego pose
                sr = nusc.get('sample', stok)
                lsd = nusc.get('sample_data', sr['data']['LIDAR_TOP'])
                lcs = nusc.get('calibrated_sensor', lsd['calibrated_sensor_token'])
                epo = nusc.get('ego_pose', lsd['ego_pose_token'])
                l2e = make_transform_matrix(lcs['translation'], lcs['rotation'])
                e2g = make_transform_matrix(epo['translation'], epo['rotation'])
                l2g = e2g @ l2e
                g2l = np.linalg.inv(l2g)

                if prev_global_to_lidar is not None and frame_i > 0:
                    for t in tracker.tracks:
                        state = t.get_state()
                        xyz_h = np.array([state[0], state[1], state[2], 1.0])
                        xyz_g = np.linalg.inv(prev_global_to_lidar) @ xyz_h
                        xyz_c = g2l @ xyz_g
                        t.kf.x[0] = xyz_c[0]
                        t.kf.x[1] = xyz_c[1]
                        t.kf.x[2] = xyz_c[2]
                prev_global_to_lidar = g2l.copy()

                det = run_detection(
                    model_obj, sd['points'], device,
                    model_type=model_type,
                    score_thresh=score_thresh, nms_iou=nms_iou,
                )
                pb, pl, ps = det['boxes'], det['labels'], det['scores']

                if len(pb) > 0:
                    active = tracker.update(pb, ps, pl)
                else:
                    active = tracker.update(
                        np.empty((0, 7)), np.empty(0), np.empty(0, dtype=int),
                    )

                if active:
                    tb = np.array([t.get_state() for t in active])
                    tl = np.array([t.label for t in active])
                    ts = np.array([t.score for t in active])
                    ti = np.array([t.track_id for t in active])
                    for t in active:
                        s = t.get_state()
                        track_histories.setdefault(t.track_id, []).append(
                            (s[0], s[1])
                        )
                else:
                    tb = np.empty((0, 7))
                    tl = np.empty(0, dtype=int)
                    ts = np.empty(0)
                    ti = None

                fig = render_bev(
                    sd['points'],
                    pred_boxes=tb, pred_labels=tl, pred_scores=ts,
                    gt_boxes=sd['gt_boxes'], gt_labels=sd['gt_labels'],
                    title=f"{model_label} Tracking — Frame {frame_i + 1}",
                    pred_track_ids=ti,
                    track_histories=track_histories,
                )

                # Render figure to numpy array (RGB)
                fig.canvas.draw()
                buf = fig.canvas.buffer_rgba()
                img_rgba = np.asarray(buf).copy()
                img_rgb = img_rgba[:, :, :3]  # drop alpha
                frame_images.append(img_rgb)
                plt.close(fig)

            progress.empty()

            # Build H.264 MP4 video playable in browsers
            if frame_images:
                tmp_path = tempfile.mktemp(suffix='.mp4')
                iio.imwrite(
                    tmp_path,
                    frame_images,
                    fps=2,
                    codec="libx264",
                    plugin="pyav",
                )

                with open(tmp_path, 'rb') as f:
                    video_bytes = f.read()
                os.unlink(tmp_path)

                st.video(video_bytes)
                st.caption(
                    f"{len(frame_images)} frames | "
                    f"{len(track_histories)} unique tracks"
                )


run_app()
