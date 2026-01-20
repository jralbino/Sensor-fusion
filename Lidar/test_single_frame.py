import sys
import cv2
from pathlib import Path

# Añadir src al path
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR / 'src'))

from lidar_utils import DataLoader
from lidar_models import ModelManager
from lidar_vis import LidarVisualizer

def main():
    print("🚀 Iniciando prueba COMPARATIVA (PointPillars vs CenterPoint)...")
    
    # 1. Inicializar Módulos
    data_loader = DataLoader()
    model_mgr = ModelManager(base_dir=BASE_DIR)
    visualizer = LidarVisualizer()
    
    # 2. Cargar AMBOS Modelos
    model_mgr.load_model('pointpillars')
    model_mgr.load_model('centerpoint')
    
    # 3. Obtener Datos del Primer Frame
    scene = data_loader.nusc.scene[0]
    token = scene['first_sample_token']
    calib_data = data_loader.get_sample_data(token)
    
    print(f"📸 Procesando frame: {token}")
    
    # --- MODELO 1: POINTPILLARS ---
    print("\n--- 1. Ejecutando PointPillars ---")
    dets_pp = model_mgr.predict('pp', calib_data['lidar_path'])
    print(f"✅ Detecciones: {len(dets_pp)}")
    
    img_pp = visualizer.project_lidar_to_cam(
        calib_data['cam_path'],
        calib_data['lidar_path'],
        dets_pp,
        calib_data,
        title="PointPillars (Undistorted)"
    )
    cv2.imwrite(str(BASE_DIR / "debug_pointpillars.jpg"), img_pp)
    print("💾 Guardado: debug_pointpillars.jpg")

    # --- MODELO 2: CENTERPOINT ---
    print("\n--- 2. Ejecutando CenterPoint ---")
    dets_cp = model_mgr.predict('cp', calib_data['lidar_path'])
    print(f"✅ Detecciones: {len(dets_cp)}")
    
    img_cp = visualizer.project_lidar_to_cam(
        calib_data['cam_path'],
        calib_data['lidar_path'],
        dets_cp,
        calib_data,
        title="CenterPoint (Undistorted)"
    )
    cv2.imwrite(str(BASE_DIR / "debug_centerpoint.jpg"), img_cp)
    print("💾 Guardado: debug_centerpoint.jpg")

    print("\n🏁 Prueba finalizada. Compara las dos imágenes generadas.")

if __name__ == "__main__":
    main()