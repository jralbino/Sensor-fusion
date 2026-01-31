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

# --- UTILITY FUNCTIONS ---
def generate_color(class_name: str) -> tuple:
    """Generate a unique color based on the hash of the class name."""
    hash_val = hash(class_name)
    r = (hash_val & 0xFF0000) >> 16
    g = (hash_val & 0x00FF00) >> 8
    b = hash_val & 0x0000FF
    return (b, g, r)


def get_optimal_font_color(bg_color_bgr: tuple) -> tuple:
    """Calculate the optimal text color based on the background luminance."""
    b, g, r = bg_color_bgr
    luminance = (0.299 * r + 0.587 * g + 0.114 * b)
    return (0, 0, 0) if luminance > 140 else (255, 255, 255)


def draw_custom_boxes(img: np.ndarray, detections: List[Dict]) -> np.ndarray:
    """Draw bounding boxes with labels on the image."""
    canvas = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2
    
    for det in detections:
        x1, y1, x2, y2 = map(int, det['bbox'])
        name = det['class_name']
        conf = det['confidence']
        label = f"{name} {conf:.2f}"
        box_color = generate_color(name)
        text_color = get_optimal_font_color(box_color)
        
        # Draw rectangle
        cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, 2)
        
        # Draw label with background
        (w, h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
        top_pos = y1 - h - 10 if y1 - h - 10 > 0 else y1 + h + 5
        text_pos_y = y1 - 5 if y1 - h - 10 > 0 else y1 + h + 5
        
        cv2.rectangle(canvas, (x1, top_pos), (x1 + w + 5, top_pos + h + 10), box_color, -1)
        cv2.putText(canvas, label, (x1 + 2, text_pos_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    
    return canvas


# --- MODEL CACHE ---
from Vision.src.detectors.detector_factory import get_object_detector

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
        app_mode = st.radio("Mode", ["📷 Live Demo", "📊 View Benchmarks"], key='app_mode_radio')
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
            
            # Image source selector
            source_type = st.radio("Input Source", ["Sample Image", "Upload Image"], key='image_source_radio')
            
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

    else:
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
        
        else:  # Upload Image
            uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
            
            if uploaded_file:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                input_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                logger.info(f"Uploaded image: {uploaded_file.name}")
        
        # Process image if available
        if input_image is not None:
            process_start = time.time()
            
            # --- OBJECT DETECTION ---
            raw_detections = []
            obj_latency = 0
            
            if obj_detector:
                try:
                    # Update confidence threshold (without reloading model)
                    obj_detector.conf = conf_thres
                    
                    raw_detections, _, stats = obj_detector.detect(input_image, classes=None)
                    obj_latency = stats['inference_time_ms']
                    
                    logger.info(f"Object detection: {len(raw_detections)} objects, {obj_latency:.1f}ms")
                
                except Exception as e:
                    st.error(f"❌ Object detection failed: {e}")
                    logger.exception("Object detection error")
            
            # --- CLASS FILTER ---
            unique_classes_found = sorted(list(set(d['class_name'] for d in raw_detections)))
            selected_classes_names = []
            
            with filter_container:
                if unique_classes_found:
                    st.divider()
                    st.subheader("🎯 Active Detections")
                    selected_classes_names = st.multiselect(
                        f"Filter Visible Objects ({len(unique_classes_found)} types)",
                        options=unique_classes_found,
                        default=unique_classes_found
                    )
                else:
                    if obj_model_type != "None":
                        st.warning("No objects detected in this image")
            
            # Apply filter
            final_detections = [d for d in raw_detections if d['class_name'] in selected_classes_names]
            
            # --- LANE DETECTION ---
            processed_img = input_image.copy()
            lane_latency = 0
            
            if lane_detector:
                try:
                    # Try with options (YOLOP)
                    try:
                        processed_img, lane_latency = lane_detector.detect(processed_img, **lane_viz_options)
                    except TypeError:
                        # Fallback without options (other detectors)
                        processed_img, lane_latency = lane_detector.detect(processed_img)
                    
                    logger.info(f"Lane detection: {lane_latency:.1f}ms")
                
                except Exception as e:
                    st.error(f"❌ Lane detection failed: {e}")
                    logger.exception("Lane detection error")
            
            # --- DRAW BOUNDING BOXES ---
            if final_detections:
                processed_img = draw_custom_boxes(processed_img, final_detections)
            
            # --- SHOW RESULTS ---
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.subheader("🎬 Fusion Result")
                st.image(
                    cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), 
                    width="stretch",
                    caption=f"Models: {obj_model_type} + {lane_model_type}"
                )
                
                # Metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("Lane Latency", f"{lane_latency:.1f} ms")
                m2.metric("Object Latency", f"{obj_latency:.1f} ms")
                m3.metric("Objects Visible", len(final_detections))
            
            with col2:
                st.subheader("📊 Summary")
                
                if final_detections:
                    # Count objects by class
                    counts = Counter([d['class_name'] for d in final_detections])
                    
                    for name, count in counts.most_common():
                        st.success(f"**{name}**: {count}")
                    
                    # JSON details
                    with st.expander("🔍 JSON Details"):
                        st.json(final_detections)
                else:
                    st.info("No objects detected or filtered")
            
            total_time = (time.time() - process_start) * 1000
            logger.info(f"Total processing time: {total_time:.1f}ms")
        
        else:
            st.info("👆 Please select or upload an image to start")


# Call the app's main logic
run_app()

# --- FOOTER ---
st.markdown("---")
st.caption(f"🗂️ Project Base: `{path_manager.BASE_DIR}`")
