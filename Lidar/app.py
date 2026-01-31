import streamlit as st
import numpy as np
import plotly.graph_objects as go
import cv2
import sys
import os
from pathlib import Path

# Import the consolidated path manager
from config.utils.path_manager import path_manager

try:
    from Lidar.src.lidar_models import ModelManager
    from Lidar.src.lidar_utils import DataLoader
    from Lidar.src.lidar_vis import LidarVisualizer
except ImportError as e:
    st.error(f"❌ Critical error importing modules: {e}")
    st.stop()

st.set_page_config(layout="wide", page_title="LiDAR Fusion Pro", page_icon="🚗")

# --- 2. AUXILIARY FUNCTIONS ---
def get_box_lines(boxes, model_type="pp"):
    x_lines, y_lines, z_lines = [], [], []
    
    for box in boxes:
        if len(box) < 7: continue 
        x, y, z, dx, dy, dz, yaw = box[:7]
        
        # --- FIX: ROTATION CORRECTION (Only for CenterPoint) ---
        if model_type == 'cp':
            dx, dy = dy, dx
            yaw += np.pi / 2

        corners = np.array([
            [dx/2, dy/2, dz/2],  [dx/2, dy/2, -dz/2],
            [dx/2, -dy/2, dz/2], [dx/2, -dy/2, -dz/2],
            [-dx/2, dy/2, dz/2], [-dx/2, dy/2, -dz/2],
            [-dx/2, -dy/2, dz/2],[-dx/2, -dy/2, -dz/2]
        ])
        
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        corners_rotated = np.dot(corners, R.T)
        corners_global = corners_rotated + np.array([x, y, z])
        
        lines_idx = [0, 1, 3, 2, 0, 4, 5, 1, 5, 7, 3, 7, 6, 2, 6, 4]
        for idx in lines_idx:
            x_lines.append(corners_global[idx, 0])
            y_lines.append(corners_global[idx, 1])
            z_lines.append(corners_global[idx, 2])
        
        x_lines.append(None); y_lines.append(None); z_lines.append(None)
        
    return x_lines, y_lines, z_lines

@st.cache_resource
def get_loader(root, ver):
    return DataLoader(root=root, version=ver)

@st.cache_resource
def get_manager():
    return ModelManager() # No base_dir needed in ModelManager.__init__ anymore

# --- 3. UI SIDEBAR ---
with st.sidebar:
    st.header("📂 Dataset & Model")
    
    # Smart paths
    # Use path_manager for robust path handling
    nusc_base_path = path_manager.get_data_detail("nuscenes_base")
        
    dataroot_input = st.text_input("NuScenes Path", value=str(nusc_base_path))
    dataroot = dataroot_input.strip()
    version = st.selectbox("Version", ["v1.0-mini", "v1.0-trainval"], index=0 if path_manager.get_user_setting("nuscenes_version") == "v1.0-mini" else 1)
    
    st.divider()
    
    st.subheader("🧠 3D Detector")
    # ✅ ONLY KEEP POINTPILLARS AND CENTERPOINT
    model_map = {
        "PointPillars": "pp", 
        "CenterPoint": "cp"
    }
    selected_model_name = st.radio("Architecture", list(model_map.keys()))
    model_key = model_map[selected_model_name]
    
    conf_thres = st.slider("Confidence", 0.0, 1.0, float(path_manager.get_user_setting("default_conf")))
    run_inference = st.checkbox("Execute Detection", value=True)

# --- 4. MAIN LOGIC ---
st.title("🛰️ LiDAR & Camera Fusion Viewer")

if not os.path.exists(dataroot):
    st.warning(f"⚠️ Path not found: `{dataroot}`")
    st.stop()

try:
    with st.spinner("Loading NuScenes..."):
        loader = get_loader(root=dataroot, ver=version) # Pass dataroot and version
        manager = get_manager()
        viz = LidarVisualizer()
        
        scene = loader.nusc.scene[0]
        token = scene['first_sample_token']
        frames = []
        for _ in range(10): # Process first 10 frames
            frames.append(token)
            if not loader.nusc.get('sample', token)['next']: break
            token = loader.nusc.get('sample', token)['next']
            
    curr_frame_idx = st.slider("Timeline", 0, len(frames)-1, 0, format="Frame %d")
    curr_token = frames[curr_frame_idx]
    data = loader.get_sample_data(curr_token)
    
    # --- INFERENCE ---
    final_dets = []
    if run_inference:
        if model_key not in manager.models:
            with st.spinner(f"Loading {selected_model_name} weights..."):
                try:
                    manager.load_model(selected_model_name)
                except FileNotFoundError as fnf:
                    st.error("🛑 **Missing Files**")
                    st.code(str(fnf))
                    st.stop()
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.stop()
        
        raw_dets = manager.predict(model_key, data['lidar_path'])
        final_dets = [d for d in raw_dets if d['score'] >= conf_thres]
        st.caption(f"✅ Detections: {len(final_dets)} ({selected_model_name})")

    # --- VISUALIZATION ---
    col1, col2 = st.columns([1,1])

    with col1:
        st.subheader("1. Point Cloud (3D)")
        points = np.fromfile(data['lidar_path'], dtype=np.float32).reshape(-1, 5)
        points = points[::5] 
        
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2],
            mode='markers',
            marker=dict(size=1.2, color=points[:, 3], colorscale='Jet', opacity=0.8),
            name='LiDAR'
        ))
        
        if final_dets:
            boxes = [d['box'] for d in final_dets]
            lx, ly, lz = get_box_lines(boxes, model_type=model_key)
            fig.add_trace(go.Scatter3d(
                x=lx, y=ly, z=lz, mode='lines', line=dict(color='#00FF00', width=4), name='BBoxes'
            ))

        fig.update_layout(scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), aspectmode='data', bgcolor="black"), margin=dict(l=0, r=0, t=0, b=0), height=500)
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.subheader("2. Camera Projection")
        if os.path.exists(data['cam_path']):
            img_proj = viz.render_camera(
                data['cam_path'], data['lidar_path'], final_dets, data,
                title=f"Frame {curr_frame_idx} | {selected_model_name}"
            )
            st.image(cv2.cvtColor(img_proj, cv2.COLOR_BGR2RGB), width="stretch")
        else:
            st.warning("Image not found.")

except Exception as e:
    st.error(f"Unexpected error: {e}")