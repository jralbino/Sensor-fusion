import streamlit as st
import cv2
import numpy as np
import tempfile
import time
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
    """Genera un color único y consistente basado en el nombre de la clase."""
    hash_val = hash(class_name)
    r = (hash_val & 0xFF0000) >> 16
    g = (hash_val & 0x00FF00) >> 8
    b = hash_val & 0x0000FF
    return (b, g, r) # BGR para OpenCV

def draw_custom_boxes(img, detections):
    """
    Dibuja las cajas manualmente. 
    Necesario porque el plot() de YOLO dibuja todo lo que detecta, 
    y nosotros queremos dibujar solo lo que el usuario filtra.
    """
    canvas = img.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, det['bbox'])
        name = det['class_name']
        conf = det['confidence']
        label = f"{name} {conf:.2f}"
        color = generate_color(name)
        
        # Caja
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        
        # Etiqueta
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(canvas, (x1, y1 - 20), (x1 + w, y1), color, -1)
        cv2.putText(canvas, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return canvas

# --- CACHÉ DE MODELOS ---
@st.cache_resource
def load_object_model(name, conf):
    if name == "None": return None
    files = {"YOLO11-L": "yolo11l.pt", "YOLO11-X": "yolo11x.pt", "RTDETR-L": "rtdetr-l.pt"}
    path = PathManager.get_path("models", files.get(name, "yolo11l.pt"))
    if not path.exists():
        st.error(f"❌ Model not found: {path}")
        return None
    # Importante: Inicializamos el modelo. La confianza se actualiza dinámicamente luego.
    return ObjectDetector(model_path=str(path), conf=conf)

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

# --- SIDEBAR: CONFIGURACIÓN INICIAL ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Modelos
    st.subheader("📦 Models")
    obj_model_type = st.selectbox("Object Detector", ["YOLO11-L", "YOLO11-X", "RTDETR-L", "None"])
    conf_thres = st.slider("Confidence", 0.1, 1.0, 0.5)
    
    st.divider()
    lane_model_type = st.selectbox("Lane Detector", ["YOLOP (Panoptic)", "UFLD", "PolyLaneNet", "DeepLabV3", "SegFormer", "None"])
    
    # Opciones de visualización de Carriles
    lane_viz_options = {}
    if "YOLOP" in lane_model_type:
        with st.expander("Lane Options", expanded=True):
            lane_viz_options['show_drivable'] = st.checkbox("Drivable Area", value=True)
            lane_viz_options['show_lanes'] = st.checkbox("Lane Mask (Red)", value=False)
            lane_viz_options['show_lane_points'] = st.checkbox("Vectors", value=True)
    else:
        lane_viz_options['show_lines'] = True # Default para otros

    st.divider()
    source_type = st.radio("Input Source", ["Sample Image", "Upload Image"])

# --- CARGA DE MODELOS ---
obj_detector = load_object_model(obj_model_type, conf_thres)
lane_detector = load_lane_model(lane_model_type)

# --- 1. OBTENER IMAGEN DE ENTRADA ---
# Movemos esto al principio para poder procesar ANTES de dibujar el resto del sidebar
input_image = None

if source_type == "Sample Image":
    img_dir = PathManager.get_path("data", "raw", "bdd100k", "images", "100k", "val")
    if not img_dir.exists(): img_dir = PathManager.get_path("data", "images")
    
    if img_dir.exists():
        sample_files = sorted(list(img_dir.glob("*.jpg")))[:50]
        selected_sample = st.selectbox("Select Image", sample_files, format_func=lambda x: x.name)
        if selected_sample:
            input_image = cv2.imread(str(selected_sample))
else:
    uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'png'])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        input_image = cv2.imdecode(file_bytes, 1)

# --- 2. PIPELINE DE PROCESAMIENTO ---
if input_image is not None:
    process_start = time.time()
    
    # A) DETECCIÓN DE OBJETOS (RAW)
    # Detectamos TODO primero para saber qué hay
    raw_detections = []
    obj_latency = 0
    
    if obj_detector:
        obj_detector.conf = conf_thres
        # classes=None para detectar TODO lo que el modelo sepa
        raw_detections, _, stats = obj_detector.detect(input_image, classes=None)
        obj_latency = stats['inference_time_ms']

    # B) FILTRO DINÁMICO (LADO IZQUIERDO)
    # Identificamos qué clases ÚNICAS hay en esta imagen específica
    unique_classes_found = sorted(list(set(d['class_name'] for d in raw_detections)))
    
    selected_classes_names = []
    if unique_classes_found:
        st.sidebar.divider()
        st.sidebar.subheader("🎯 Active Detections")
        st.sidebar.info(f"Detected: {', '.join(unique_classes_found)}")
        
        # El multiselect se llena SOLO con lo encontrado
        selected_classes_names = st.sidebar.multiselect(
            "Filter Visible Objects",
            options=unique_classes_found,
            default=unique_classes_found # Por defecto mostramos todo lo encontrado
        )
    else:
        st.sidebar.warning("No objects detected.")

    # C) APLICAR FILTRO A LOS RESULTADOS
    final_detections = [d for d in raw_detections if d['class_name'] in selected_classes_names]

    # D) DETECCIÓN DE CARRILES
    # Hacemos esto antes de dibujar cajas para que los carriles queden "al fondo"
    processed_img = input_image.copy()
    lane_latency = 0
    
    if lane_detector:
        try:
            processed_img, lane_latency = lane_detector.detect(processed_img, **lane_viz_options)
        except TypeError:
            processed_img, lane_latency = lane_detector.detect(processed_img)

    # E) DIBUJAR CAJAS (MANUALMENTE)
    # Pintamos sobre la imagen que ya tiene carriles
    if final_detections:
        processed_img = draw_custom_boxes(processed_img, final_detections)

    # --- 3. VISUALIZACIÓN ---
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Fusion Result")
        st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), use_container_width=True)
        
        # Métricas
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
            
            with st.expander("JSON Details"):
                st.json(final_detections)
        else:
            if raw_detections:
                st.info("Objects detected but hidden by filter.")
            else:
                st.write("No objects found.")

else:
    st.info("Please select or upload an image to start.")

# Footer
st.markdown("---")
st.caption(f"Project Base: `{PathManager.BASE_DIR}`")