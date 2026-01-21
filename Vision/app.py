import streamlit as st
import cv2
import numpy as np
import tempfile
import time
import json
import pandas as pd
from pathlib import Path
import sys
from collections import Counter

# Añadir src al path
sys.path.append(str(Path(__file__).parent / "src"))

from utils.paths import PathManager
from detectors.object_detector import ObjectDetector
from lanes.yolop_detector import YOLOPDetector
from lanes.ufld_detector import UFLDDetector
from lanes.polylanenet_detector import PolyLaneNetDetector
from lanes.deeplab_detector import DeepLabDetector
from lanes.segformer_detector import SegFormerDetector

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Sensor Fusion Studio", layout="wide", page_icon="🚗")
st.title("🚗 Sensor Fusion: Object & Lane Detection Comparison")

# --- FUNCIONES DE UTILIDAD ---
def generate_color(class_name):
    hash_val = hash(class_name)
    r = (hash_val & 0xFF0000) >> 16
    g = (hash_val & 0x00FF00) >> 8
    b = hash_val & 0x0000FF
    return (b, g, r)

def get_optimal_font_color(bg_color_bgr):
    b, g, r = bg_color_bgr
    luminance = (0.299 * r + 0.587 * g + 0.114 * b)
    return (0, 0, 0) if luminance > 140 else (255, 255, 255)

def draw_custom_boxes(img, detections):
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
        
        cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, 2)
        (w, h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
        top_pos = y1 - h - 10 if y1 - h - 10 > 0 else y1 + h + 10
        text_pos_y = y1 - 5 if y1 - h - 10 > 0 else y1 + h + 5
        cv2.rectangle(canvas, (x1, top_pos), (x1 + w + 5, top_pos + h + 10), box_color, -1)
        cv2.putText(canvas, label, (x1 + 2, text_pos_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    return canvas

# --- CACHÉ DE MODELOS ---
@st.cache_resource
def load_object_model(name, conf):
    if name == "None": return None
    files = {
        "YOLO11-L": "yolo11l",
        "YOLO11-X": "yolo11x",
        "RTDETR-L (Original)": "rtdetr_l",
        "RTDETR-BDD (Finetuned)": "rtdetr_bdd"
    }
    
    path = PathManager.get_model_path(files.get(name, "yolo11l"))
    if not path.exists():
        st.error(f"❌ Model not found: {path}")
        return None
    return ObjectDetector(model_path=path, conf=conf)

@st.cache_resource
def load_lane_model(name):
    if name == "None": return None
    try:
        if "YOLOP" in name: return YOLOPDetector()
        elif "UFLD" in name: return UFLDDetector(model_path=str(PathManager.get_path("models", "tusimple_18.pth")))
        elif "PolyLaneNet" in name: return PolyLaneNetDetector(model_path=str(PathManager.get_path("models", "model_2305.pt")))
        elif "DeepLab" in name: return DeepLabDetector()
        elif "SegFormer" in name: return SegFormerDetector()
    except Exception as e: return None
    return None

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # MODO DE OPERACIÓN
    app_mode = st.radio("Mode", ["📷 Live Demo", "📊 View Benchmarks"])
    st.divider()

    if app_mode == "📷 Live Demo":
        st.subheader("📦 Models")
        obj_model_type = st.selectbox("Object Detector", ["YOLO11-L", "YOLO11-X", "RTDETR-L (Original)", "RTDETR-BDD (Finetuned)", "None"])
        conf_thres = st.slider("Confidence", 0.1, 1.0, 0.5)
        
        filter_container = st.container()
        st.divider()
        
        lane_model_type = st.selectbox("Lane Detector", ["YOLOP (Panoptic)", "UFLD", "PolyLaneNet", "DeepLabV3", "SegFormer", "None"])
        lane_viz_options = {}
        if "YOLOP" in lane_model_type:
            with st.expander("Lane Options", expanded=True):
                lane_viz_options['show_drivable'] = st.checkbox("Drivable Area", value=True)
                lane_viz_options['show_lanes'] = st.checkbox("Lane Mask (Red)", value=False)
                lane_viz_options['show_lane_points'] = st.checkbox("Vectors", value=True)
        else:
            lane_viz_options['show_lines'] = True
        
        st.divider()
        source_type = st.radio("Input Source", ["Sample Image", "Upload Image"])

# --- LÓGICA PRINCIPAL ---

if app_mode == "📊 View Benchmarks":
    # MODO BENCHMARK
    st.header("📊 Model Performance Benchmarks")
    
    json_path = PathManager.get_data_path("output_vision") / "data/benchmark_results.json"
    if json_path.exists():
        with open(json_path, 'r') as f:
            bench_data = json.load(f)
        
        st.caption(f"Last updated: {bench_data.get('timestamp', 'Unknown')}")
        
        datasets = bench_data.get("datasets", {})
        
        for ds_name, results in datasets.items():
            st.subheader(f"📂 Dataset: {ds_name}")
            if not results:
                st.warning("No results found for this dataset.")
                continue
                
            # Crear DataFrame para tabla bonita
            df = pd.DataFrame(results)
            # Seleccionar columnas principales
            main_cols = ["model", "mAP50-95", "mAP50", "Precision", "Recall", "Inference_Time_ms"]
            st.dataframe(df[main_cols].style.highlight_max(axis=0, subset=["mAP50-95", "mAP50", "Precision", "Recall"]), width="stretch")
            
            # Gráfica Comparativa
            st.write("#### 🏆 mAP Comparison")
            chart_data = df.set_index("model")[["mAP50-95", "mAP50"]]
            st.bar_chart(chart_data)
            
            # Análisis Detallado (Clases)
            with st.expander("🔍 Detailed Class Performance "):
                for item in results:
                    st.write(f"**{item['model']}**")
                    if "per_class" in item:
                        class_df = pd.DataFrame(list(item["per_class"].items()), columns=["Class", "mAP"])
                        st.dataframe(class_df, width="stretch", hide_index=True)
                
    else:
        st.info("⚠️ No benchmark results found.")
        st.markdown("""
        To generate results:
        1. Run `python Vision/run_benchmark.py` on your server.
        2. Copy `Vision/data/benchmark_results.json` to this machine.
        """)

else:
    # MODO LIVE DEMO (Código original)
    obj_detector = load_object_model(obj_model_type, conf_thres)
    lane_detector = load_lane_model(lane_model_type)

    input_image = None
    if source_type == "Sample Image":
        img_dir = PathManager.get_data_path("bdd100k") / "images/100k/val"
        if not img_dir.exists(): img_dir = PathManager.get_path("data", "images")
        if img_dir.exists():
            sample_files = sorted(list(img_dir.glob("*.jpg")))[:50]
            selected_sample = st.selectbox("Select Image", sample_files, format_func=lambda x: x.name)
            if selected_sample: input_image = cv2.imread(str(selected_sample))
    else:
        uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'png'])
        if uploaded_file:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            input_image = cv2.imdecode(file_bytes, 1)

    if input_image is not None:
        process_start = time.time()
        
        raw_detections = []
        obj_latency = 0
        if obj_detector:
            obj_detector.conf = conf_thres
            raw_detections, _, stats = obj_detector.detect(input_image, classes=None)
            obj_latency = stats['inference_time_ms']

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
                if obj_model_type != "None": st.warning("No objects detected.")

        final_detections = [d for d in raw_detections if d['class_name'] in selected_classes_names]

        processed_img = input_image.copy()
        lane_latency = 0
        if lane_detector:
            try: processed_img, lane_latency = lane_detector.detect(processed_img, **lane_viz_options)
            except TypeError: processed_img, lane_latency = lane_detector.detect(processed_img)

        if final_detections:
            processed_img = draw_custom_boxes(processed_img, final_detections)

        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("Fusion Result")
            st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), width="stretch")
            m1, m2, m3 = st.columns(3)
            m1.metric("Lane Latency", f"{lane_latency:.1f} ms")
            m2.metric("Object Latency", f"{obj_latency:.1f} ms")
            m3.metric("Objects Visible", len(final_detections))

        with col2:
            st.subheader("Summary")
            if final_detections:
                counts = Counter([d['class_name'] for d in final_detections])
                for name, count in counts.items():
                    st.success(f"**{name}**: {count}")
                with st.expander("JSON Details"): st.json(final_detections)
            else:
                st.write("No objects found.")
    else:
        st.info("Please select or upload an image to start.")

# Footer
st.markdown("---")
st.caption(f"Project Base: `{PathManager.BASE_DIR}`")