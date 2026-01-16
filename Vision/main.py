import sys
import yaml
from pathlib import Path

# Añadimos 'src' al path
sys.path.append('src')

from predictor import BatchPredictor
from visualizer import ResultVisualizer
from benchmark import ModelBenchmark
from lanes.yolop_detector import YOLOPDetector
#from lanes.deeplab_detector import DeepLabDetector 
#from lanes.segformer_detector import SegFormerDetector
from lanes.polylanenet_detector import PolyLaneNetDetector
from lanes.ufld_detector import UFLDDetector

def create_subset_yaml(base_yaml, images_dir, limit, output_yaml_path):
    """
    Crea un archivo YAML temporal que apunta solo a las primeras 'limit' imágenes.
    CORREGIDO: Asegura que exista la key 'train' para evitar error de Ultralytics.
    """
    if limit is None:
        # Si usamos el yaml base, asegurarnos que tenga train. 
        # Si falla aquí, edita bdd_coco_val.yaml y descomenta 'train'.
        return base_yaml 

    print(f"\n✂️ Creando configuración temporal para {limit} imágenes...")
    
    # 1. Obtener la lista de archivos
    img_paths = sorted(list(Path(images_dir).glob("*.jpg")))[:limit]
    
    # Convertir a rutas absolutas (CRÍTICO para que YOLO no se pierda)
    img_paths = [str(p.resolve()) for p in img_paths]
    
    # 2. Crear archivo .txt con la lista de imágenes
    subset_txt_path = Path("config/temp_val_subset.txt")
    # Asegurar que el directorio config existe
    subset_txt_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(subset_txt_path, 'w') as f:
        f.write('\n'.join(img_paths))
        
    # 3. Leer el YAML base
    with open(base_yaml, 'r') as f:
        config = yaml.safe_load(f) or {} # Manejar si está vacío
        
    # --- FIX: INYECTAR CLAVES OBLIGATORIAS ---
    abs_txt_path = str(subset_txt_path.absolute())
    
    config['val'] = abs_txt_path
    
    # Si 'train' no existe, lo rellenamos con la misma ruta para que YOLO no se queje
    if 'train' not in config:
        config['train'] = abs_txt_path 
    
    # 4. Guardar el nuevo YAML temporal
    with open(output_yaml_path, 'w') as f:
        yaml.dump(config, f)
        
    return output_yaml_path

def main():
    # --- 1. CONFIGURACIÓN ---
    IMAGES_DIR = "data/raw/bdd100k/images/100k/val"
    PREDICTIONS_DIR = "output/predictions"
    VIDEOS_DIR = "output/videos"
    MODELS_DIR = "models"
    
    # YAML Original (Dataset Completo)
    BASE_YAML = "config/bdd_coco_val.yaml"
    
    # Límite de prueba (Cámbialo a None para correr todo)
    LIMIT = 50 

    models_to_run = [
        ("YOLO11-X", f"{MODELS_DIR}/yolo11x.pt"),
        ("RTDETR-L", f"{MODELS_DIR}/rtdetr-l.pt")
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
    temp_yaml = "config/temp_benchmark.yaml"
    active_yaml = create_subset_yaml(BASE_YAML, IMAGES_DIR, LIMIT, temp_yaml)
    
    benchmarker = ModelBenchmark(data_yaml=active_yaml)
    benchmarker.run(models_to_run)
    
    # Limpieza opcional (borrar temp)
    # import os; os.remove(temp_yaml)

if __name__ == "__main__":
    main()