from ultralytics import YOLO, RTDETR
import time
import torch
from pathlib import Path
from .object_detector_base import ObjectDetectorBase
from config.utils.path_manager import path_manager

class ObjectDetector(ObjectDetectorBase):
    def __init__(self, model_path='models/yolo11l.pt', conf=0.5, iou=0.45, device=None):
        """
        Args:
            model_path (str): Path to the .pt model
            conf (float): Confidence Threshold.
            iou (float): NMS Threshold.
            device (str): 'cuda' or 'cpu'.
        """
        self.model_path = model_path
        self.conf = conf
        self.iou = iou
        
        if device:
            self.device = device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        print(f"Loading {model_path} on {self.device} (Conf: {self.conf}, IOU: {self.iou})...")
        
        if 'rtdetr' in Path(model_path).stem:
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
        Returns:
            detections (list): List of dicts
            plot (numpy array): Plotted image
            stats (dict): Inference times
        """
        start_time = time.time()
        
        # Calculate the safe path: Vision/runs/detect
        # This avoids creating 'runs' in the project root
        project_path = path_manager.get("output")
        
        # Inference with 'runs' folder redirection
        results = self.model.predict(
            source=image, 
            conf=self.conf, 
            iou=self.iou,
            device=self.device, 
            verbose=False,
            classes=classes,
            
            # --- PATH CORRECTION ---
            save=False,               # Do not save images to disk (we do it in RAM)
            project=str(project_path), # Force Vision/runs/detect folder
            name="inference",
            exist_ok=True                    # 'inference' subfolder (reused if it exists)
            # ---------------------------
        )[0]
        
        end_time = time.time()
        
        # Process results
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