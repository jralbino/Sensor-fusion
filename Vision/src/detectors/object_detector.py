from ultralytics import YOLO, RTDETR
import time
import torch
from pathlib import Path
from utils.paths import PathManager

class ObjectDetector:
    def __init__(self, model_path='models/yolo11l.pt', conf=0.5, iou=0.45, device=None):
        """
        Args:
            model_path (str): Ruta al modelo .pt
            conf (float): Confidence Threshold.
            iou (float): NMS Threshold.
            device (str): 'cuda' o 'cpu'.
        """
        self.model_path = model_path
        self.conf = conf
        self.iou = iou
        
        if device:
            self.device = device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        print(f"Cargando {model_path} en {self.device} (Conf: {self.conf}, IOU: {self.iou})...")
        
        if 'rtdetr' in model_path.stem:
            self.model = RTDETR(model_path)
        else:
            self.model = YOLO(model_path)
            
    def get_parameters(self):
        return {
            "model_architecture": self.model_path,
            "confidence_threshold": self.conf,
            "iou_threshold": self.iou,
            "device": self.device
        }

    def detect(self, image, classes=None):
        """
        Retorna:
            detections (list): Lista de dicts
            plot (numpy array): Imagen pintada
            stats (dict): Tiempos de inferencia
        """
        start_time = time.time()
        
        # Calculamos la ruta segura: Vision/runs/detect
        # Así evitamos que cree 'runs' en la raíz del proyecto
        project_path = PathManager.get_data_path("output_vision")
        
        # Inferencia con redirección de carpeta 'runs'
        results = self.model.predict(
            source=image, 
            conf=self.conf, 
            iou=self.iou,
            device=self.device, 
            verbose=False,
            classes=classes,
            
            # --- CORRECCIÓN DE RUTAS ---
            save=False,               # No guardar imágenes en disco (lo hacemos en RAM)
            project=str(project_path), # Forzar carpeta Vision/runs/detect
            name="inference",
            exist_ok=True                    # Subcarpeta 'inference' (se reusa si existe)
            # ---------------------------
        )[0]
        
        end_time = time.time()
        
        # Procesar resultados
        parsed_detections = []
        names = results.names 
        
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = names[cls_id]
            
            parsed_detections.append({
                "class_id": cls_id,
                "class_name": cls_name,
                "confidence": round(confidence, 4),
                "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]
            })

        stats = {
            "inference_time_ms": round((end_time - start_time) * 1000, 2),
            "total_objects": len(parsed_detections)
        }
        
        return parsed_detections, results.plot(), stats