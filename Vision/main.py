import sys
from pathlib import Path

# Añadimos 'src' al path
sys.path.append('Vision/src')

from predictor import BatchPredictor
from visualizer import ResultVisualizer

def main():
    # --- 1. CONFIGURACIÓN DEL ENTORNO ---
    # Rutas relativas desde la carpeta 'Vision/'
    IMAGES_DIR = "Vision/data/raw/bdd100k/images/100k/val"
    PREDICTIONS_DIR = "Vision/output/predictions"
    VIDEOS_DIR = "Vision/output/videos"
    MODELS_DIR = "Vision/models"  # Carpeta donde moviste los .pt

    # --- 2. CONFIGURACIÓN DE LOS MODELOS A COMPARAR ---
    # Aquí defines tus contendientes. 
    # Formato: (Nombre_Para_Video, Nombre_Archivo_Modelo)
    models_to_run = [
        ("YOLO11-X", f"{MODELS_DIR}/yolo11x.pt"),
        ("RTDETR-L", f"{MODELS_DIR}/rtdetr-l.pt")
    ]

    # --- 3. FASE DE INFERENCIA (GENERAR JSONs) ---
    print("--- FASE 1: GENERACIÓN DE PREDICCIONES ---")
    
    predictor = BatchPredictor(images_dir=IMAGES_DIR, output_dir=PREDICTIONS_DIR)
    
    generated_jsons = []
    
    for name, model_file in models_to_run:
        # Aquí controlas el 'limit'. 
        # Pon limit=50 para probar rápido, o limit=None para todo el set.
        json_path = predictor.run_inference(
            model_name=name,
            model_path=model_file,
            conf=0.50,    # Umbral de confianza
            limit=50      # <--- ¡CAMBIA ESTO A None CUANDO ESTÉS LISTO!
        )
        
        if json_path:
            generated_jsons.append((name, json_path))

    # --- 4. FASE DE VISUALIZACIÓN (GENERAR VIDEO) ---
    if not generated_jsons:
        print("No se generaron predicciones. Abortando video.")
        return

    print("\n--- FASE 2: VISUALIZACIÓN Y VIDEO ---")
    
    viz = ResultVisualizer(images_dir=IMAGES_DIR, output_dir=VIDEOS_DIR)
    
    # Nombre del video automático basado en los modelos
    video_name = "vs".join([m[0] for m in generated_jsons]) + "_comparison.mp4"
    
    viz.generate_video(
        json_files=generated_jsons, 
        output_filename=video_name,
        fps=5
    )

if __name__ == "__main__":
    main()