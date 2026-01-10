import cv2
import sys
from pathlib import Path
sys.path.append('src')

from lanes.lane_detector import LaneDetector

def test_lane_detection():
    # Cargar una imagen de prueba
    images_dir = Path("data/raw/bdd100k/images/100k/val")
    img_paths = list(images_dir.glob("*.jpg"))[:5] # Probar con 5 imágenes
    
    if not img_paths:
        print("No hay imágenes.")
        return

    detector = LaneDetector(debug=True)
    output_dir = Path("output/lanes_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("🛣️ Probando detección de carriles...")
    
    for p in img_paths:
        img = cv2.imread(str(p))
        if img is None: continue
        
        # Detección
        result = detector.detect(img)
        
        # Guardar
        save_path = output_dir / p.name
        cv2.imwrite(str(save_path), result)
        print(f"   Guardado: {save_path}")
    
    print("✅ Prueba terminada. Revisa la carpeta output/lanes_test")

if __name__ == "__main__":
    test_lane_detection()