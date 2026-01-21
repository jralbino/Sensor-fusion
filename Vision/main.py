import sys
import yaml
from pathlib import Path

# Añadimos 'src' al path
sys.path.append('Vision/src')

from predictor import BatchPredictor
from visualizer import ResultVisualizer
from benchmark import ModelBenchmark
from lanes.yolop_detector import YOLOPDetector
#from lanes.deeplab_detector import DeepLabDetector 
#from lanes.segformer_detector import SegFormerDetector
from lanes.polylanenet_detector import PolyLaneNetDetector
from lanes.ufld_detector import UFLDDetector
from utils.paths import PathManager
from config.global_config import USER_SETTINGS, DATA_PATHS, MODEL_PATHS


# Vision/main.py

def create_subset_yaml(base_yaml, images_dir, limit, output_yaml_path):
    print(f"\n✂️ Creando configuración temporal para {limit} imágenes...")
    
    # 1. Obtener imágenes con rutas absolutas dinámicas del servidor actual
    img_paths = sorted(list(Path(images_dir).glob("*.jpg")))[:limit]
    img_paths = [str(p.resolve()) for p in img_paths] # Resuelve la ruta real en el disco
    
    # 2. Guardar .txt en la carpeta de configuración centralizada
    subset_txt_path = PathManager.BASE_DIR / "Vision/config/temp_val_subset.txt"
    with open(subset_txt_path, 'w') as f:
        f.write('\n'.join(img_paths))
        
    # 3. Configurar el YAML dinámicamente
    with open(base_yaml, 'r') as f:
        config = yaml.safe_load(f) or {}
        
    # Inyectar rutas basadas en la configuración global
    config['path'] = str(PathManager.get_data_path("bdd100k").resolve())
    config['val'] = str(subset_txt_path.resolve())
    config['train'] = str(subset_txt_path.resolve())
    
    with open(output_yaml_path, 'w') as f:
        yaml.dump(config, f)
        
    return output_yaml_path

def main():
    # --- 1. CONFIGURACIÓN ---
    IMAGES_DIR = DATA_PATHS["bdd100k"] / "images/100k/val"
    PREDICTIONS_DIR = DATA_PATHS["output_vision"] / "predictions"
    VIDEOS_DIR = DATA_PATHS["output_vision"] / "videos"
   
    
    # YAML Original (Dataset Completo)
    BASE_YAML = "Vision/config/bdd_coco_val.yaml"
    
    # Límite de prueba (Cámbialo a None para correr todo)
    LIMIT = USER_SETTINGS["video_limit"]
    models_to_run = [
        ("YOLO11-X", MODEL_PATHS["yolo11x"]),
        ("RTDETR-L", MODEL_PATHS["rtdetr_l"])
    ]
    
    # --- FASE 1: INFERENCIA (JSONs) ---
    print(f"\n🎬 --- FASE 1: PREDICCIONES (Límite: {LIMIT}) ---")
    predictor = BatchPredictor(images_dir=IMAGES_DIR, output_dir=PREDICTIONS_DIR)
    generated_jsons = []
    
    for name, model_file in models_to_run:
        json_path = predictor.run_inference(
            model_name=name, model_path=model_file,
            conf=0.50, limit=LIMIT
        )
        if json_path: generated_jsons.append((name, json_path))

    # --- FASE 2: VIDEO COMPARATIVO (SENSOR FUSION) ---
    if generated_jsons:
        print("\n🎥 --- FASE 2: VIDEO COMPARATIVO (SENSOR FUSION) ---")
        
        # --- OPCIONES DE CONFIGURACIÓN (Solo afectan a YOLOP, DeepLab las ignora) ---
        LANE_OPTIONS = {
            'show_drivable': True,
            'show_lanes': False,
            'show_lane_points': True
        }
        
        # ### SELECCIÓN DE MODELO DE CARRILES (Visualización de fondo) ###
        
        # OPCIÓN A: YOLOP (Especializado en Carriles/Road) -> DESCOMENTAR PARA USAR
        print(f"   Iniciando YOLOP... Config: {LANE_OPTIONS}")
        lane_model = YOLOPDetector()
        model_suffix = "YOLOP"

        # OPCIÓN B: DeepLabV3 (Generalista COCO) -> ACTIVO AHORA
        #print(f"   Iniciando DeepLabV3 (General Segmentation)...")
        #lane_model = DeepLabDetector()
        #model_suffix = "DeepLab"

        # ### SELECCIÓN DE MODELO ###
        #print(f"   Iniciando NVIDIA SegFormer (Cityscapes SOTA)...")
        #lane_model = SegFormerDetector() # Instanciamos SegFormer
        #model_suffix = "NVIDIA_SegFormer"

        #print(f"   Iniciando UFLD (TuSimple Benchmark Winner)...")
        #lane_model = UFLDDetector() 
        #model_suffix = "UFLD"
        
        #print(f"   Iniciando PolyLaneNet (TuSimple Challenge - Regresión Polinomial)...")
        #lane_model = PolyLaneNetDetector(model_path="models/model_2305.pt")
        #model_suffix = "PolyLaneNet"

        viz = ResultVisualizer(images_dir=IMAGES_DIR, output_dir=VIDEOS_DIR)
        
        video_name = f"fusion_comparison_{model_suffix}.mp4"
        
        viz.generate_video(
            generated_jsons, 
            video_name, 
            fps=5,
            lane_detector=lane_model,
            lane_config=LANE_OPTIONS 
        )

        # ------------------------------------------------------------
        
        

    # --- FASE 3: BENCHMARK (Con el mismo límite) ---
    print(f"\n📊 --- FASE 3: BENCHMARK (Sobre {LIMIT if LIMIT else 'todas'} las imágenes) ---")
    
    # AQUI ESTÁ EL CAMBIO CLAVE:
    # Generamos un YAML que apunte solo a las imágenes del LIMIT
    temp_yaml = "Vision/config/temp_benchmark.yaml"
    active_yaml = create_subset_yaml(BASE_YAML, IMAGES_DIR, LIMIT, temp_yaml)
    
    benchmarker = ModelBenchmark(data_yaml=active_yaml)
    benchmarker.run(models_to_run)
    
    # Limpieza opcional (borrar temp)
    # import os; os.remove(temp_yaml)

if __name__ == "__main__":
    main()