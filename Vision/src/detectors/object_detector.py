from ultralytics import YOLO, RTDETR
import time
import torch

class ObjectDetector:
    def __init__(self, model_path='yolo11l.pt', conf=0.5, iou=0.45, device=None):
        """
        Args:
            model_path (str): Ruta al modelo .pt
            conf (float): Confidence Threshold. Detecciones abajo de esto se ignoran.
            iou (float): NMS Threshold. Controla qué tanto se pueden solapar las cajas.
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
        
        if 'rtdetr' in model_path.lower():
            self.model = RTDETR(model_path)
        else:
            self.model = YOLO(model_path)
            
    def get_parameters(self):
        """Retorna los metadatos de configuración para guardarlos en el JSON."""
        return {
            "model_architecture": self.model_path,
            "confidence_threshold": self.conf,
            "iou_threshold": self.iou,
            "device": self.device
        }

    def detect(self, image):
        """
        Retorna:
            detections (list): Lista de dicts con formato {'class': str, 'conf': float, 'bbox': [x1,y1,x2,y2]}
            plot (numpy array): Imagen pintada
            stats (dict): Tiempos de inferencia
        """
        start_time = time.time()
        
        # Inferencia con parámetros explícitos
        results = self.model.predict(
            source=image, 
            conf=self.conf, 
            iou=self.iou,
            device=self.device, 
            verbose=False
        )[0] # Tomamos el primer resultado (batch size 1)
        
        end_time = time.time()
        
        # Procesar resultados para JSON
        parsed_detections = []
        names = results.names # Diccionario de nombres de clases {0: 'person', ...}
        
        for box in results.boxes:
            # Coordenadas xyxy (pixels absolutos)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = names[cls_id]
            
            parsed_detections.append({
                "class_id": cls_id,
                "class_name": cls_name,
                "confidence": round(confidence, 4),
                "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)] # x1, y1, x2, y2
            })

        stats = {
            "inference_time_ms": round((end_time - start_time) * 1000, 2),
            "total_objects": len(parsed_detections)
        }
        
        return parsed_detections, results.plot(), stats