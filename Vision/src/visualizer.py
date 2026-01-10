import json
import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

class ResultVisualizer:
    def __init__(self, images_dir, output_dir):
        """
        Inicializa el visualizador con las rutas base.
        """
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuración de Clases (COCO)
        self.classes = {
            0: "Person", 1: "Bicycle", 2: "Car", 3: "Motorcycle",
            5: "Bus", 7: "Truck", 9: "Traffic Light", 11: "Stop Sign"
        }
        
        # Paleta de Colores
        self.colors = {
            0: (0, 0, 255),      # Person -> Rojo
            1: (0, 165, 255),    # Bicycle -> Naranja
            2: (255, 0, 0),      # Car -> Azul
            3: (0, 255, 255),    # Motorcycle -> Amarillo
            5: (255, 0, 255),    # Bus -> Magenta
            7: (255, 255, 0),    # Truck -> Cyan
            9: (0, 255, 0),      # Traffic Light -> Verde
            11: (200, 200, 200)  # Stop Sign -> Gris
        }

    def _load_predictions(self, json_path):
        """Método privado para cargar y parsear un JSON."""
        print(f"Cargando: {json_path}")
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        lookup = {item['image_name']: item for item in data['results']}
        image_names = set(lookup.keys())
        meta = data.get('meta', {})
        return lookup, image_names, meta

    def _draw_frame(self, img, result_entry, model_name):
        """Dibuja las detecciones sobre un frame."""
        canvas = img.copy()
        
        # Etiqueta "No Data" si no hay resultados
        if result_entry is None:
            cv2.putText(canvas, f"{model_name} (No Data)", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return canvas

        # Dibujar Detecciones
        for det in result_entry['detections']:
            cls_id = det['class_id']
            # --- AQUÍ MOSTRAMOS TODAS LAS CLASES (Sin filtros) ---
            if cls_id in self.classes:
                x1, y1, x2, y2 = map(int, det['bbox'])
                conf = det['confidence']
                label = f"{self.classes[cls_id]} {conf:.2f}"
                color = self.colors.get(cls_id, (128,128,128))
                
                # Caja y Texto
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(canvas, (x1, y1 - 20), (x1 + w, y1), color, -1)
                cv2.putText(canvas, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

        # Info del Modelo (Overlay)
        latency = result_entry.get('inference_ms', 0)
        info = f"{model_name} | {latency}ms"
        cv2.rectangle(canvas, (0, 0), (300, 40), (0, 0, 0), -1)
        cv2.putText(canvas, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return canvas

    def generate_video(self, json_files, output_filename, fps=5):
        """
        Genera el video comparativo.
        
        Args:
            json_files: Lista de tuplas [('NombreModelo', 'ruta.json')]
            output_filename: Nombre del archivo de salida (ej: 'video_final.mp4')
            fps: Frames por segundo.
        """
        # 1. Cargar datos
        model_data = []
        all_images = set()
        
        for name, path in json_files:
            if not os.path.exists(path):
                print(f"⚠️ Archivo no encontrado: {path}")
                continue
            lookup, names, meta = self._load_predictions(path)
            model_data.append({"name": name, "lookup": lookup})
            all_images.update(names)
            
        sorted_images = sorted(list(all_images))
        if not sorted_images:
            print("❌ No hay imágenes para procesar.")
            return

        # 2. Configurar Video Writer
        first_path = self.images_dir / sorted_images[0]
        if not first_path.exists():
            print(f"❌ Error: No encuentro imágenes en {self.images_dir}")
            return
            
        sample_img = cv2.imread(str(first_path))
        h, w, _ = sample_img.shape
        sbs_width = w * len(model_data)
        
        save_path = self.output_dir / output_filename
        writer = cv2.VideoWriter(str(save_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (sbs_width, h))
        
        print(f"🎥 Generando video: {save_path}")
        print(f"Frames a procesar: {len(sorted_images)}")

        # 3. Loop de generación
        for img_name in tqdm(sorted_images):
            full_path = self.images_dir / img_name
            if not full_path.exists(): continue
            
            original = cv2.imread(str(full_path))
            if original is None: continue
            
            frames = []
            for m in model_data:
                entry = m["lookup"].get(img_name)
                frames.append(self._draw_frame(original, entry, m["name"]))
                
            sbs_frame = np.hstack(frames)
            writer.write(sbs_frame)
            
        writer.release()
        print("✅ Video generado exitosamente.")