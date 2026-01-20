import streamlit as st
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(layout="wide", page_title="LiDAR Viewer")

st.title("📡 LiDAR 3D Viewer")

# Sidebar para controles
st.sidebar.header("Configuración")
uploaded_file = st.sidebar.file_uploader("Cargar archivo .bin (LiDAR)", type=["bin", "pcd"])

def read_bin_file(file_buffer):
    """Lee archivos binarios de nuScenes (x, y, z, intensity, ring_index)"""
    points = np.frombuffer(file_buffer.read(), dtype=np.float32)
    # nuScenes usa 5 dimensiones por punto
    points = points.reshape((-1, 5)) 
    return points[:, :3], points[:, 3] # Retorna XYZ e Intensidad

if uploaded_file is not None:
    try:
        points_xyz, intensity = read_bin_file(uploaded_file)
        
        # Filtro simple para visualización (reducir puntos para que no se congele el navegador)
        # Tomamos 1 de cada 10 puntos
        skip = 10 
        
        st.write(f"Mostrando {len(points_xyz)//skip} puntos (downsampled).")

        fig = go.Figure(data=[go.Scatter3d(
            x=points_xyz[::skip, 0],
            y=points_xyz[::skip, 1],
            z=points_xyz[::skip, 2],
            mode='markers',
            marker=dict(
                size=1,
                color=intensity[::skip], # Colorear por intensidad
                colorscale='Viridis',
                opacity=0.8
            )
        )])

        fig.update_layout(
            scene=dict(
                xaxis_title='X (Metros)',
                yaxis_title='Y (Metros)',
                zaxis_title='Z (Altura)',
                aspectmode='data' # Mantiene la proporción real 1:1:1
            ),
            margin=dict(r=0, b=0, l=0, t=0),
            height=600
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error leyendo el archivo: {e}")
else:
    st.info("Sube un archivo .bin de nuScenes para visualizarlo.")