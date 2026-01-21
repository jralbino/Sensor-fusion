import sys
import json
import cv2
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Setup de rutas
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR / 'src'))

from lidar_utils import DataLoader
from lidar_models import ModelManager
from lidar_vis import LidarVisualizer

# --- CONFIGURACIÓN DE MODELOS ---
AVAILABLE_MODELS = {
    'pointpillars': 'pp',
    'centerpoint': 'cp'
}

def parse_args():
    parser = argparse.ArgumentParser(description="LiDAR Fusion Benchmark Pipeline")
    parser.add_argument('--frames', type=int, default=20, help="Número de imágenes a procesar")
    parser.add_argument('--skip_infer', action='store_true', help="Saltar inferencia y usar JSONs existentes")
    return parser.parse_args()

def run_inference(data_loader, model_mgr, frames_limit, output_dir):
    """Fase 1: Ejecutar modelos y guardar JSONs."""
    print(f"\n🧠 --- FASE 1: INFERENCIA ({frames_limit} frames) ---")
    
    scene = data_loader.nusc.scene[0]
    first_token = scene['first_sample_token']
    
    # Procesar cada modelo secuencialmente para ahorrar memoria VRAM
    for model_name, model_key in AVAILABLE_MODELS.items():
        print(f"\n🚀 Procesando modelo: {model_name.upper()}...")
        
        # Cargar modelo en memoria
        try:
            model_mgr.load_model(model_name)
        except Exception as e:
            print(f"❌ Error cargando {model_name}: {e}")
            continue

        results = []
        token = first_token
        
        for i in tqdm(range(frames_limit), desc=f"Infering {model_name}"):
            # Obtener datos
            try:
                calib = data_loader.get_sample_data(token)
            except:
                break # Fin de escena
            
            # Predecir
            detections = model_mgr.predict(model_key, calib['lidar_path'])
            
            # Guardar resultado
            results.append({
                "token": token,
                "timestamp": i, # Dummy timestamp for order
                "detections": detections
            })
            
            # Siguiente token
            sample = data_loader.nusc.get('sample', token)
            if not sample['next']: break
            token = sample['next']
            
        # Guardar JSON
        json_path = output_dir / f"detections_{model_name}.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"✅ Resultados guardados en: {json_path}")
        
        # Limpiar modelo de memoria (opcional, depende de tu GPU)
        model_mgr.models = {} 

def generate_videos(data_loader, output_dir, limit=50):
    """Fase 2: Generar videos combinados (Cam + BEV) desde JSONs."""
    print(f"\n🎥 --- FASE 2: GENERACIÓN DE VIDEOS ---")
    
    viz = LidarVisualizer()
    
    for model_name in AVAILABLE_MODELS.keys():
        json_path = output_dir / f"detections_{model_name}.json"
        if not json_path.exists():
            print(f"⚠️ No se encontró JSON para {model_name}, saltando video.")
            continue
            
        print(f"🎬 Renderizando video para: {model_name}...")
        
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        # Configurar Video Writer
        # Cam: 1600x900. BEV: haremos 900x900 para que coincida en altura
        # Total: 2500x900
        video_path = str(output_dir / f"video_{model_name}_combined.mp4")
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 5.0, (2500, 900))
        
        for frame_data in tqdm(data[:limit], desc="Rendering"):
            token = frame_data['token']
            dets = frame_data['detections']
            calib = data_loader.get_sample_data(token)
            
            # 1. Render Cámara Frontal (1600 x 900)
            img_cam = viz.render_camera(
                calib['cam_path'], 
                calib['lidar_path'], 
                dets, 
                calib, 
                title=f"{model_name.upper()} - Cam Projection"
            )
            
            # 2. Render BEV (Redimensionado a 900x900 para encajar)
            img_bev_raw = viz.render_bev(
                calib['lidar_path'], 
                dets, 
                title="Eagle Eye View"
            )
            img_bev = cv2.resize(img_bev_raw, (900, 900))
            
            # 3. Concatenar (Lado a Lado)
            combined = np.hstack((img_cam, img_bev))
            
            out.write(combined)
            
        out.release()
        print(f"🎉 Video listo: {video_path}")

def main():
    args = parse_args()
    
    # Directorios
    output_dir = BASE_DIR / "runs" / "final_benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Inicializar componentes comunes
    data_loader = DataLoader() # Carga NuScenes una sola vez
    model_mgr = ModelManager(base_dir=BASE_DIR)
    
    # FASE 1: Inferencia
    if not args.skip_infer:
        run_inference(data_loader, model_mgr, args.frames, output_dir)
    else:
        print("⏩ Saltando inferencia, usando JSONs existentes...")
        
    # FASE 2: Video
    generate_videos(data_loader, output_dir)
    
    print("\n✅ ¡Pipeline completado exitosamente!")
    print(f"📂 Archivos en: {output_dir}")

if __name__ == "__main__":
    main()