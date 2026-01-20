import os
from pathlib import Path
from nuscenes.nuscenes import NuScenes

# Inicializamos nuScenes (ajusta dataroot según tu configuración)
nusc = NuScenes(version='v1.0-mini', dataroot='Fusion/data/sets/nuscenes', verbose=True)

def demo_lidar_projection(scene_index=0):
    """
    Proyecta LiDAR en cámara y guarda la imagen en Fusion/runs/test/,
    creando la carpeta automáticamente si no existe.
    """
    
    # --- 1. CONFIGURACIÓN DE RUTAS ---
    # Obtenemos la ruta absoluta de este script (Fusion/src/lidar_to_camera.py)
    CURRENT_FILE = Path(__file__).resolve()
    
    # Definimos la raíz del módulo Fusion (subimos un nivel desde src/)
    FUSION_ROOT = CURRENT_FILE.parent.parent 
    
    # Definimos la carpeta de destino: Fusion/runs/test
    OUTPUT_DIR = FUSION_ROOT / "runs" / "test"
    
    # --- 2. CREACIÓN AUTOMÁTICA DE CARPETA ---
    # exist_ok=True: no da error si ya existe.
    # parents=True: crea carpetas intermedias (ej. crea 'runs' si falta).
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Directorio de salida verificado: {OUTPUT_DIR}")

    # --- 3. LÓGICA DE NUSCENES ---
    my_scene = nusc.scene[scene_index]
    first_sample_token = my_scene['first_sample_token']
    my_sample = nusc.get('sample', first_sample_token)

    # Definimos el nombre final del archivo
    output_path = OUTPUT_DIR / f"fusion_sample_{first_sample_token}.jpg"

    print(f"Renderizando proyección para el token: {first_sample_token}")
    
    nusc.render_pointcloud_in_image(
        my_sample['token'],
        pointsensor_channel='LIDAR_TOP',
        camera_channel='CAM_FRONT',
        render_intensity=True,
        show_lidarseg=False,
        out_path=str(output_path)  # Convertimos Path a string para la librería
    )
    
    print(f"✅ Imagen guardada exitosamente en:\n{output_path}")

if __name__ == "__main__":
    demo_lidar_projection()