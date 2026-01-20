import sys
import cv2
import numpy as np
from pathlib import Path

# Configuración de rutas
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR / 'src'))

from lidar_utils import DataLoader
from lidar_models import ModelManager
from lidar_vis import LidarVisualizer

def main():
    print("🚀 Iniciando diagnóstico: SECUENCIA DE 5 FRAMES...")
    
    # 1. Inicializar
    data_loader = DataLoader()
    model_mgr = ModelManager(base_dir=BASE_DIR)
    visualizer = LidarVisualizer()
    
    # Cargar modelos
    model_mgr.load_model('pointpillars')
    model_mgr.load_model('centerpoint')
    
    # Obtener primer token
    scene = data_loader.nusc.scene[0]
    token = scene['first_sample_token']
    
    # Directorio de salida
    output_dir = BASE_DIR / "debug_sequence"
    output_dir.mkdir(exist_ok=True)
    
    # 2. Bucle de 5 Frames
    for i in range(5):
        print(f"\n📸 Procesando Frame {i+1}/5 [Token: {token}]")
        calib = data_loader.get_sample_data(token)
        
        # --- A. PointPillars ---
        dets_pp = model_mgr.predict('pp', calib['lidar_path'])
        print(f"   🔹 PointPillars: {len(dets_pp)} detecciones")
        
        img_pp = visualizer.project_lidar_to_cam(
            calib['cam_path'], calib['lidar_path'], dets_pp, calib,
            title=f"Frame {i+1}: PointPillars"
        )
        cv2.imwrite(str(output_dir / f"frame_{i+1}_pointpillars.jpg"), img_pp)
        
        # --- B. CenterPoint ---
        dets_cp = model_mgr.predict('cp', calib['lidar_path'])
        print(f"   🔹 CenterPoint: {len(dets_cp)} detecciones")
        
        img_cp = visualizer.project_lidar_to_cam(
            calib['cam_path'], calib['lidar_path'], dets_cp, calib,
            title=f"Frame {i+1}: CenterPoint"
        )
        cv2.imwrite(str(output_dir / f"frame_{i+1}_centerpoint.jpg"), img_cp)
        
        # --- DEBUG: Imprimir coordenadas de la detección más confiable ---
        # Esto nos ayudará a ver si la Z o el tamaño tienen sentido
        if dets_cp:
            # Ordenar por score
            top_det = sorted(dets_cp, key=lambda x: x['score'], reverse=True)[0]
            print(f"   🔍 Top Box CenterPoint: Pos={np.round(top_det['box'][:3], 2)} | Dim={np.round(top_det['box'][3:6], 2)}")

        # Avanzar al siguiente frame
        sample_record = data_loader.nusc.get('sample', token)
        if sample_record['next']:
            token = sample_record['next']
        else:
            print("⚠️ Fin de la escena alcanzado antes de los 5 frames.")
            break

    print(f"\n✅ Proceso completado. Revisa la carpeta: {output_dir}")

if __name__ == "__main__":
    main()