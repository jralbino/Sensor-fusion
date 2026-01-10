import cv2
import os
from pathlib import Path
from detectors.object_detector import ObjectDetector

def run_comparison(image_folder, output_folder):
    # Definir los modelos a comparar
    # 'n' es nano (rápido), 's' small, 'm' medium, 'l' large, 'x' extra large
    models_to_test = ['yolo11l.pt', 'rtdetr-l.pt'] 
    
    detectors = [ObjectDetector(m) for m in models_to_test]
    
    # Obtener algunas imágenes
    img_paths = list(Path(image_folder).glob("*.jpg"))[:3] # Probamos solo con 3
    
    if not img_paths:
        print(f"No se encontraron imágenes en {image_folder}")
        return

    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    for img_p in img_paths:
        print(f"\nProcesando: {img_p.name}")
        original_img = cv2.imread(str(img_p))
        
        for i, detector in enumerate(detectors):
            model_name = models_to_test[i].split('.')[0]
            
            _, result_img, info = detector.detect(original_img)
            
            print(f" -> Modelo: {model_name} | Latencia: {info['latency_ms']}ms | Objetos: {info['detections_count']}")
            
            # Guardar resultado
            save_name = output_path / f"{img_p.stem}_{model_name}.jpg"
            cv2.imwrite(str(save_name), result_img)

if __name__ == "__main__":
    # AJUSTA LA RUTA A DONDE TENGAS TUS IMAGENES (TEST O VAL)
    imgs_dir = "data/raw/bdd100k/images/100k/val" 
    out_dir = "output/comparison"
    
    run_comparison(imgs_dir, out_dir)