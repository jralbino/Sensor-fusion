# -*- coding: utf-8 -*-
"""
Sensor Fusion Studio - Streamlit Application
Interactive interface to compare object and lane detectors.
"""

import streamlit as st
import cv2
import numpy as np
import time
import json
import pandas as pd
from pathlib import Path
import sys
from collections import Counter
from typing import Optional, List, Dict

# Manually add project root to sys.path for Streamlit
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.utils.path_manager import path_manager
from config.logging_config import setup_logging, get_logger, setup_streamlit_logging

from Vision.src.detectors.object_detector import ObjectDetector
from Vision.src.lanes.lane_factory import get_lane_detector
from Vision.config.models import OBJECT_DETECTORS, LANE_DETECTORS

# --- GLOBAL CONFIGURATION ---
st.set_page_config(
    page_title="Sensor Fusion Studio", 
    layout="wide", 
    page_icon="🚗",
    menu_items={
        'Get Help': 'https://github.com/jralbino/Sensor-fusion',
        'Report a bug': 'https://github.com/jralbino/Sensor-fusion/issues',
        'About': '# Sensor Fusion Studio\nMulti-modal perception for autonomous driving'
    }
)

# Setup logging
logger = setup_logging(
    log_dir=path_manager.get("logs"),
    console=False,  # Don't show in console (Streamlit has its own output)
    file_logging=True
)

# Initialize StreamlitLogHandler and persist it across reruns
if 'st_log_handler' not in st.session_state:
    st.session_state.st_log_handler = setup_streamlit_logging(logger)
st_log_handler = st.session_state.st_log_handler

logger.info("Streamlit app started")


# --- TITLE ---
st.title("🚗 Sensor Fusion: Object & Lane Detection Comparison")

# --- UTILITY FUNCTIONS (imported from testable module) ---
from Vision.src.app_utils import generate_color, get_optimal_font_color, draw_custom_boxes


# --- MODEL CACHE ---
from Vision.src.detectors.detector_factory import get_object_detector

NUSCENES_CAMERAS = [
    "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
]


@st.cache_resource
def load_nuscenes(data_root: str, version: str):
    """Load NuScenes dataset (cached)."""
    from nuscenes.nuscenes import NuScenes
    return NuScenes(version=version, dataroot=data_root, verbose=False)


def iterate_scene_samples(nusc, start_idx: int, num_frames: int):
    """Yield (frame_index, sample_token) for consecutive samples."""
    sample = nusc.sample[start_idx]
    for i in range(num_frames):
        yield i, sample['token']
        if not sample['next']:
            break
        sample = nusc.get('sample', sample['next'])


@st.cache_resource
def load_object_model(name: str) -> Optional[ObjectDetector]:
    """
    Load object detection model (cached).
    
    Args:
        name: Name of the model (without confidence)
        
    Returns:
        ObjectDetector or None if it fails
    """
    if name == "None":
        return None
    
    model_info = OBJECT_DETECTORS.get(name)
    if not model_info:
        st.error(f"❌ Unknown model: {name}")
        return None
    
    try:
        model_path = path_manager.get_model(model_info["key"], check_exists=True)
        logger.info(f"Loading model: {name} from {model_path}")
        
        detector = get_object_detector(model_info["type"], model_path=str(model_path), conf=0.5)
        
        st.success(f"✅ Model {name} loaded successfully")
        return detector
        
    except FileNotFoundError as e:
        st.error(f"❌ Model file not found: {e}")
        logger.error(f"Model not found: {name} - {e}")
        return None
    except Exception as e:
        st.error(f"❌ Error loading {name}: {str(e)}")
        logger.exception(f"Error loading model {name}")
        return None


@st.cache_resource
def load_lane_model(name: str) -> Optional[object]:
    """
    Load lane detection model (cached).
    
    Args:
        name: Name of the model
        
    Returns:
        Lane detector or None
    """
    if name == "None":
        return None
    
    try:
        logger.info(f"Loading lane detector: {name}")
        detector = get_lane_detector(name)
        st.success(f"✅ Lane detector {name} loaded successfully")
        return detector
            
    except (RuntimeError, FileNotFoundError) as e:
        st.error(f"❌ {e}")
        logger.error(f"Failed to load lane detector {name}: {e}")
        return None
    except Exception as e:
        st.error(f"❌ An unexpected error occurred while loading {name}: {e}")
        logger.exception(f"Unexpected error loading lane detector {name}")
        return None


# --- MAIN LOGIC ---
# The entire Streamlit UI logic should be within run_app()
def run_app():
    # --- SIDEBAR ---
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # OPERATION MODE
        app_mode = st.radio("Mode", ["📷 Live Demo", "📡 NuScenes", "📊 View Benchmarks"], key='app_mode_radio')
        st.divider()

        if app_mode == "📷 Live Demo":
            st.subheader("📦 Models")
            
            # Object model selector
            obj_model_type = st.selectbox(
                "Object Detector", 
                list(OBJECT_DETECTORS.keys()) + ["None"],
                key='obj_detector_selector'
            )
            
            # Confidence threshold
            conf_thres = st.slider("Confidence", 0.1, 1.0, 0.5, 0.05, key='conf_slider')
            
            # Filter container (will be filled later)
            filter_container = st.container()
            st.divider()
            
            # Lane model selector
            lane_model_type = st.selectbox(
                "Lane Detector", 
                list(LANE_DETECTORS.values()) + ["None"],
                key='lane_detector_selector'
            )
            
            # Lane visualization options
            lane_viz_options = {}
            if "YOLOP" in lane_model_type:
                with st.expander("Lane Options", expanded=True):
                    lane_viz_options['show_drivable'] = st.checkbox("Drivable Area", value=True, key='show_drivable_checkbox')
                    lane_viz_options['show_lanes'] = st.checkbox("Lane Mask (Red)", value=False, key='show_lanes_checkbox')
                    lane_viz_options['show_lane_points'] = st.checkbox("Vectors", value=True, key='show_lane_points_checkbox')
            else:
                lane_viz_options['show_lines'] = True
            
            st.divider()

            # Tracking toggle
            st.subheader("Tracking")
            enable_tracking = st.checkbox("Enable ByteTrack", value=False, key='tracking_checkbox')

            st.divider()

            # Image source selector
            source_type = st.radio(
                "Input Source",
                ["Sample Image", "Upload Image", "NuScenes Sequence"],
                key='image_source_radio',
            )

            # NuScenes options
            nusc_data_root = None
            nusc_version = None
            nusc_camera = "CAM_FRONT"
            nusc_sample_idx = 0
            nusc_num_frames = 10
            if source_type == "NuScenes Sequence":
                _default_nusc = str(PROJECT_ROOT / "Fusion" / "data" / "sets" / "nuscenes")
                nusc_data_root = st.text_input(
                    "NuScenes data root",
                    value=_default_nusc,
                    key="nusc_data_root",
                )
                nusc_version = st.selectbox(
                    "NuScenes version",
                    ["v1.0-mini", "v1.0-trainval"],
                    key="nusc_version",
                )
                nusc_camera = st.selectbox(
                    "Camera", NUSCENES_CAMERAS, key="nusc_camera"
                )
                # Load NuScenes to know sample count
                try:
                    nusc = load_nuscenes(nusc_data_root, nusc_version)
                    n_samples = len(nusc.sample)
                    nusc_sample_idx = st.slider(
                        "Sample index", 0, max(0, n_samples - 1), 0, key="nusc_sample_idx"
                    )
                    if enable_tracking:
                        nusc_num_frames = st.slider(
                            "Number of frames", 2, 40, 10, key="nusc_num_frames"
                        )
                except Exception as e:
                    st.error(f"Failed to load NuScenes: {e}")
            
            # Logs in sidebar
            with st.expander("📋 System Logs", expanded=False):
                if st.button("Refresh Logs", key='refresh_logs_button'):
                    st.rerun()

                logs = st_log_handler.get_logs(last_n=20)
                if logs:
                    for log in reversed(logs):
                        st.text(f"{log['time'].strftime('%H:%M:%S')} | {log['level']} | {log['message']}")
                else:
                    st.caption("No logs yet")

        elif app_mode == "📡 NuScenes":
            st.subheader("📦 Models")
            nusc_obj_model = st.selectbox(
                "Object Detector",
                list(OBJECT_DETECTORS.keys()) + ["None"],
                key='nusc_obj_detector_selector',
            )
            nusc_conf_thres = st.slider("Confidence", 0.1, 1.0, 0.5, 0.05, key='nusc_conf_slider')
            st.divider()

            nusc_lane_model = st.selectbox(
                "Lane Detector",
                list(LANE_DETECTORS.values()) + ["None"],
                key='nusc_lane_detector_selector',
            )
            nusc_lane_viz = {}
            if "YOLOP" in nusc_lane_model:
                with st.expander("Lane Options", expanded=False):
                    nusc_lane_viz['show_drivable'] = st.checkbox("Drivable Area", value=True, key='nusc_show_drivable')
                    nusc_lane_viz['show_lanes'] = st.checkbox("Lane Mask", value=False, key='nusc_show_lanes')
                    nusc_lane_viz['show_lane_points'] = st.checkbox("Vectors", value=True, key='nusc_show_lane_pts')
            else:
                nusc_lane_viz['show_lines'] = True
            st.divider()

            st.subheader("📂 Dataset")
            _default_nusc = str(PROJECT_ROOT / "Fusion" / "data" / "sets" / "nuscenes")
            nusc_data_root_n = st.text_input(
                "NuScenes data root", value=_default_nusc, key="nusc_n_data_root"
            )
            nusc_version_n = st.selectbox(
                "NuScenes version",
                ["v1.0-mini", "v1.0-trainval"],
                key="nusc_n_version",
            )
            nusc_camera_n = st.selectbox("Camera", NUSCENES_CAMERAS, key="nusc_n_camera")
            nusc_show_all_cams = st.checkbox("Show all 6 cameras", value=False, key="nusc_n_all_cams")

            # Load NuScenes to determine sample count
            nusc_n_sample_idx = 0
            nusc_n = None
            try:
                nusc_n = load_nuscenes(nusc_data_root_n, nusc_version_n)
                n_samples_n = len(nusc_n.sample)
                st.success(f"Loaded {n_samples_n} samples")
                nusc_n_sample_idx = st.slider(
                    "Sample index", 0, max(0, n_samples_n - 1), 0, key="nusc_n_sample_idx"
                )
            except Exception as e:
                st.error(f"Failed to load NuScenes: {e}")
            st.divider()

            nusc_enable_tracking = st.checkbox("Enable ByteTrack", value=False, key='nusc_tracking_checkbox')
            if nusc_enable_tracking:
                nusc_num_frames_n = st.slider("Number of frames", 2, 40, 10, key="nusc_n_num_frames")
            else:
                nusc_num_frames_n = 1


    if app_mode == "📊 View Benchmarks":
        # ========================
        # BENCHMARK MODE
        # ========================
        st.header("📊 Model Performance Benchmarks")
        
        json_path = path_manager.get("output") / "data" / "benchmark_results.json"
        logger.info(f"Benchmark JSON path: {json_path}")
        logger.info(f"Benchmark JSON exists: {json_path.exists()}")
        
        if json_path.exists():
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    bench_data = json.load(f)
                
                st.caption(f"Last updated: {bench_data.get('timestamp', 'Unknown')}")
                
                datasets = bench_data.get("datasets", {})
                
                for ds_name, results in datasets.items():
                    st.subheader(f"📂 Dataset: {ds_name}")
                    
                    if not results:
                        st.warning("No results found for this dataset.")
                        continue
                    
                    # Create DataFrame
                    df = pd.DataFrame(results)
                    
                    # Main table
                    main_cols = ["model", "mAP50-95", "mAP50", "Precision", "Recall", "Inference_Time_ms"]
                    st.dataframe(
                        df[main_cols].style.highlight_max(
                            axis=0, 
                            subset=["mAP50-95", "mAP50", "Precision", "Recall"]
                        ),
                        width="stretch"
                    )
                    
                    # Comparative chart
                    st.write("#### 🏆 mAP Comparison")
                    chart_data = df.set_index("model")[["mAP50-95", "mAP50"]]
                    st.bar_chart(chart_data)
                    
                    # Per-class analysis
                    with st.expander("🔍 Detailed Class Performance"):
                        for item in results:
                            st.write(f"**{item['model']}**")
                            if "per_class" in item:
                                class_df = pd.DataFrame(
                                    list(item["per_class"].items()), 
                                    columns=["Class", "mAP"]
                                )
                                st.dataframe(class_df, width="stretch", hide_index=True)
            
            except Exception as e:
                st.error(f"❌ Error loading benchmarks: {e}")
                logger.exception("Error loading benchmark results")
        
        else:
            st.info("⚠️ No benchmark results found.")
            st.markdown(f"""
            To generate results:
            1. Run `python Vision/run_benchmark.py` on your server
            2. Results will be saved to: `{json_path}`
            """
            )

    elif app_mode == "📡 NuScenes":
        # ========================
        # NUSCENES MODE
        # ========================
        if nusc_n is not None:
            with st.spinner("Loading models..."):
                nusc_obj_detector = load_object_model(nusc_obj_model)
                nusc_lane_detector = load_lane_model(nusc_lane_model)

            if nusc_enable_tracking and nusc_obj_detector:
                _run_nuscenes_tracking(
                    nusc_n, nusc_n_sample_idx, nusc_num_frames_n,
                    nusc_camera_n, nusc_obj_detector, nusc_conf_thres,
                    nusc_lane_detector, nusc_lane_viz,
                    nusc_obj_model, nusc_lane_model,
                )
            else:
                _run_nuscenes_vision_mode(
                    nusc_n, nusc_n_sample_idx, nusc_camera_n,
                    nusc_obj_detector, nusc_conf_thres,
                    nusc_lane_detector, nusc_lane_viz,
                    nusc_obj_model, nusc_lane_model,
                    show_all_cameras=nusc_show_all_cams,
                )
        else:
            st.info("Configure NuScenes dataset in the sidebar to start.")

    elif app_mode == "📷 Live Demo":
        # ========================
        # LIVE DEMO MODE
        # ========================
        
        # Load models
        with st.spinner("Loading models..."):
            obj_detector = load_object_model(obj_model_type)
            lane_detector = load_lane_model(lane_model_type)
        
        # Image selection/upload
        input_image = None

        if source_type == "Sample Image":
            # Search for sample images
            img_dir = path_manager.get("bdd_images_val")

            if img_dir.exists():
                logger.info(f"Sample image directory exists: {img_dir}")
                sample_files = sorted(list(img_dir.glob("*.jpg")))[:50]
                logger.info(f"Found {len(sample_files)} sample .jpg files.")

                if sample_files:
                    selected_sample = st.selectbox(
                        "Select Image",
                        sample_files,
                        format_func=lambda x: x.name,
                        key='sample_image_selector'
                    )

                    if selected_sample:
                        input_image = cv2.imread(str(selected_sample))
                        logger.info(f"Loaded sample image: {selected_sample.name}")
                else:
                    st.warning("No sample images found")
            else:
                st.error(f"Sample directory not found: {img_dir}")

        elif source_type == "Upload Image":
            uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

            if uploaded_file:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                input_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                logger.info(f"Uploaded image: {uploaded_file.name}")

        elif source_type == "NuScenes Sequence":
            # --- NUSCENES SEQUENCE ---
            if nusc_data_root and nusc_version:
                try:
                    nusc = load_nuscenes(nusc_data_root, nusc_version)
                except Exception:
                    nusc = None

                if nusc is not None:
                    if enable_tracking and obj_detector:
                        # Multi-frame tracking mode: process sequence and produce video
                        _run_nuscenes_tracking(
                            nusc, nusc_sample_idx, nusc_num_frames,
                            nusc_camera, obj_detector, conf_thres,
                            lane_detector, lane_viz_options,
                            obj_model_type, lane_model_type,
                        )
                    else:
                        # Single-frame mode: load selected sample image
                        sample = nusc.sample[nusc_sample_idx]
                        cam_data = nusc.get('sample_data', sample['data'][nusc_camera])
                        img_path = Path(nusc_data_root) / cam_data['filename']
                        if img_path.exists():
                            input_image = cv2.imread(str(img_path))
                            logger.info(f"Loaded NuScenes image: {img_path.name}")
                        else:
                            st.error(f"Image not found: {img_path}")

        # Process single image (Sample / Upload / single NuScenes)
        if input_image is not None:
            _process_single_image(
                input_image, obj_detector, lane_detector,
                conf_thres, enable_tracking,
                lane_viz_options, filter_container,
                obj_model_type, lane_model_type,
            )
        elif source_type != "NuScenes Sequence":
            st.info("👆 Please select or upload an image to start")


def _process_single_image(
    input_image, obj_detector, lane_detector,
    conf_thres, enable_tracking,
    lane_viz_options, filter_container,
    obj_model_type, lane_model_type,
):
    """Process and display a single image with detection + optional tracking."""
    process_start = time.time()

    # --- OBJECT DETECTION ---
    raw_detections = []
    obj_latency = 0

    if obj_detector:
        try:
            obj_detector.conf = conf_thres
            raw_detections, _, stats = obj_detector.detect(input_image, classes=None)
            obj_latency = stats['inference_time_ms']
            logger.info(f"Object detection: {len(raw_detections)} objects, {obj_latency:.1f}ms")
        except Exception as e:
            st.error(f"Object detection failed: {e}")
            logger.exception("Object detection error")

    # --- TRACKING ---
    if enable_tracking and raw_detections:
        from tracking import ByteTracker2D
        from Vision.src.app_utils import detections_to_tracker_format, tracker_output_to_detections

        if 'vision_tracker' not in st.session_state:
            st.session_state.vision_tracker = ByteTracker2D(
                high_thresh=conf_thres * 0.8,
                low_thresh=conf_thres * 0.3,
                match_thresh=0.3,
                max_age=10,
                min_hits=1,
            )
        tracker = st.session_state.vision_tracker

        fmt = detections_to_tracker_format(raw_detections)
        if fmt is not None:
            dets_arr, scores_arr, labels_arr, idx_to_name = fmt
            active_tracks = tracker.update(dets_arr, scores_arr, labels_arr)
            raw_detections = tracker_output_to_detections(active_tracks, idx_to_name)

    # --- CLASS FILTER ---
    unique_classes_found = sorted(list(set(d['class_name'] for d in raw_detections)))
    selected_classes_names = []

    with filter_container:
        if unique_classes_found:
            st.divider()
            st.subheader("Active Detections")
            selected_classes_names = st.multiselect(
                f"Filter Visible Objects ({len(unique_classes_found)} types)",
                options=unique_classes_found,
                default=unique_classes_found
            )
        else:
            if obj_model_type != "None":
                st.warning("No objects detected in this image")

    final_detections = [d for d in raw_detections if d['class_name'] in selected_classes_names]

    # --- LANE DETECTION ---
    processed_img = input_image.copy()
    lane_latency = 0

    if lane_detector:
        try:
            try:
                processed_img, lane_latency = lane_detector.detect(processed_img, **lane_viz_options)
            except TypeError:
                processed_img, lane_latency = lane_detector.detect(processed_img)
            logger.info(f"Lane detection: {lane_latency:.1f}ms")
        except Exception as e:
            st.error(f"Lane detection failed: {e}")
            logger.exception("Lane detection error")

    # --- DRAW BOUNDING BOXES ---
    if final_detections:
        processed_img = draw_custom_boxes(processed_img, final_detections)

    # --- SHOW RESULTS ---
    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("Fusion Result")
        st.image(
            cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB),
            width="stretch",
            caption=f"Models: {obj_model_type} + {lane_model_type}"
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("Lane Latency", f"{lane_latency:.1f} ms")
        m2.metric("Object Latency", f"{obj_latency:.1f} ms")
        m3.metric("Objects Visible", len(final_detections))

    with col2:
        st.subheader("Summary")

        if final_detections:
            counts = Counter([d['class_name'] for d in final_detections])
            for name, count in counts.most_common():
                st.success(f"**{name}**: {count}")

            with st.expander("JSON Details"):
                st.json(final_detections)
        else:
            st.info("No objects detected or filtered")

    total_time = (time.time() - process_start) * 1000
    logger.info(f"Total processing time: {total_time:.1f}ms")


def _run_nuscenes_vision_mode(
    nusc, sample_idx, camera,
    obj_detector, conf_thres,
    lane_detector, lane_viz_options,
    obj_model_type, lane_model_type,
    show_all_cameras=False,
):
    """Show NuScenes sample with object + lane detection on selected camera (and optionally all 6)."""
    sample = nusc.sample[sample_idx]

    def _process_camera(cam_name):
        """Load + detect on a single camera image, return annotated BGR image."""
        cam_data = nusc.get('sample_data', sample['data'][cam_name])
        img_path = Path(nusc.dataroot) / cam_data['filename']
        if not img_path.exists():
            return None, 0, 0, []
        frame = cv2.imread(str(img_path))
        if frame is None:
            return None, 0, 0, []

        raw_dets = []
        obj_lat = 0
        if obj_detector:
            obj_detector.conf = conf_thres
            try:
                raw_dets, _, stats = obj_detector.detect(frame, classes=None)
                obj_lat = stats['inference_time_ms']
            except Exception as e:
                logger.warning(f"Detection failed on {cam_name}: {e}")

        canvas = frame.copy()
        lane_lat = 0
        if lane_detector:
            try:
                try:
                    canvas, lane_lat = lane_detector.detect(canvas, **lane_viz_options)
                except TypeError:
                    canvas, lane_lat = lane_detector.detect(canvas)
            except Exception as e:
                logger.warning(f"Lane detection failed on {cam_name}: {e}")

        if raw_dets:
            canvas = draw_custom_boxes(canvas, raw_dets)

        return canvas, obj_lat, lane_lat, raw_dets

    st.subheader(f"NuScenes — Sample {sample_idx}")

    if show_all_cameras:
        # Show all 6 cameras in a 2x3 grid (2 rows, 3 cols)
        tab_selected, tab_all = st.tabs([f"Selected: {camera}", "All Cameras"])
    else:
        tab_selected = st.container()
        tab_all = None

    with tab_selected:
        with st.spinner(f"Running detection on {camera}..."):
            annotated, obj_lat, lane_lat, dets = _process_camera(camera)

        if annotated is not None:
            m1, m2, m3 = st.columns(3)
            m1.metric("Objects", len(dets))
            m2.metric("Object Latency", f"{obj_lat:.1f} ms")
            m3.metric("Lane Latency", f"{lane_lat:.1f} ms")

            st.image(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                caption=f"{camera} | Models: {obj_model_type} + {lane_model_type}",
                width="stretch",
            )

            if dets:
                with st.expander("Detection details"):
                    from collections import Counter
                    counts = Counter(d['class_name'] for d in dets)
                    for cls, cnt in counts.most_common():
                        st.write(f"**{cls}**: {cnt}")
        else:
            st.error(f"Could not load image for {camera}")

    if tab_all is not None:
        with tab_all:
            st.write("Running detection on all 6 cameras...")
            row1 = st.columns(3)
            row2 = st.columns(3)
            cols = row1 + row2
            for col, cam_name in zip(cols, NUSCENES_CAMERAS):
                with col:
                    annotated_cam, _, _, dets_cam = _process_camera(cam_name)
                    if annotated_cam is not None:
                        col.image(
                            cv2.cvtColor(annotated_cam, cv2.COLOR_BGR2RGB),
                            caption=f"{cam_name} ({len(dets_cam)} objs)",
                            use_container_width=True,
                        )
                    else:
                        col.warning(f"{cam_name}: no image")


def _run_nuscenes_tracking(
    nusc, start_idx, num_frames, camera,
    obj_detector, conf_thres,
    lane_detector, lane_viz_options,
    obj_model_type, lane_model_type,
):
    """Run multi-frame tracking on a NuScenes camera sequence and display video."""
    import tempfile
    import imageio.v3 as iio
    from tracking import ByteTracker2D
    from Vision.src.app_utils import detections_to_tracker_format, tracker_output_to_detections

    st.subheader(f"NuScenes Tracking: {camera}")

    tracker = ByteTracker2D(
        high_thresh=conf_thres * 0.8,
        low_thresh=conf_thres * 0.3,
        match_thresh=0.2,
        max_age=10,
        min_hits=1,
        distance_thresh=1.5,
    )

    obj_detector.conf = conf_thres
    frames_rgb = []
    all_track_ids = set()
    progress = st.progress(0, text="Processing frames...")

    for i, token in iterate_scene_samples(nusc, start_idx, num_frames):
        sample = nusc.get('sample', token)
        cam_data = nusc.get('sample_data', sample['data'][camera])
        img_path = Path(nusc.dataroot) / cam_data['filename']

        if not img_path.exists():
            logger.warning(f"Missing image: {img_path}")
            continue

        frame = cv2.imread(str(img_path))

        # Object detection
        try:
            raw_dets, _, _ = obj_detector.detect(frame, classes=None)
        except Exception as e:
            logger.warning(f"Detection failed on frame {i}: {e}")
            raw_dets = []

        # Tracking
        fmt = detections_to_tracker_format(raw_dets)
        if fmt is not None:
            dets_arr, scores_arr, labels_arr, idx_to_name = fmt
            active_tracks = tracker.update(dets_arr, scores_arr, labels_arr)
            tracked_dets = tracker_output_to_detections(active_tracks, idx_to_name)
            for d in tracked_dets:
                all_track_ids.add(d['track_id'])
        else:
            tracked_dets = []
            tracker.update(
                np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=int)
            )

        # Lane detection
        canvas = frame.copy()
        if lane_detector:
            try:
                try:
                    canvas, _ = lane_detector.detect(canvas, **lane_viz_options)
                except TypeError:
                    canvas, _ = lane_detector.detect(canvas)
            except Exception:
                pass

        # Draw tracked boxes
        if tracked_dets:
            canvas = draw_custom_boxes(canvas, tracked_dets)

        frames_rgb.append(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        progress.progress((i + 1) / num_frames, text=f"Frame {i + 1}/{num_frames}")

    progress.empty()

    if not frames_rgb:
        st.warning("No frames were processed.")
        return

    # Encode to MP4 video
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    iio.imwrite(
        tmp_path,
        frames_rgb,
        fps=2,
        codec="libx264",
        plugin="pyav",
    )

    st.video(tmp_path)
    st.caption(
        f"{len(frames_rgb)} frames | {len(all_track_ids)} unique tracks | "
        f"Models: {obj_model_type} + {lane_model_type}"
    )

    # Frame slider to inspect individual frames
    if len(frames_rgb) > 1:
        frame_idx = st.slider(
            "Inspect frame", 0, len(frames_rgb) - 1, 0, key="nuscenes_frame_slider"
        )
        st.image(frames_rgb[frame_idx], caption=f"Frame {frame_idx}", width="stretch")


# Call the app's main logic
run_app()

# --- FOOTER ---
st.markdown("---")
st.caption(f"🗂️ Project Base: `{path_manager.BASE_DIR}`")
