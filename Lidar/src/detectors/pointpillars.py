from mmdet3d.apis import init_model, inference_detector
from .base import BaseLidarDetector
import numpy as np

class PointPillarsDetector(BaseLidarDetector):
    def load_model(self):
        print(f"🏗️ Loading PointPillars from {self.checkpoint_path}...")
        return init_model(self.config_path, self.checkpoint_path, device=self.device)

    def detect(self, pcd_path):
        # MMDetection3D handles reading the bin internally in 'inference_detector'
        result, data = inference_detector(self.model, pcd_path)
        
        # Process output (pred_instances_3d)
        pred_instances = result.pred_instances_3d
        bboxes_3d = pred_instances.bboxes_3d.tensor.cpu().numpy() # [x, y, z, l, w, h, rot]
        scores = pred_instances.scores_3d.cpu().numpy()
        labels = pred_instances.labels_3d.cpu().numpy()
        
        detections = []
        for box, score, label in zip(bboxes_3d, scores, labels):
            if score > 0.3: # Basic confidence filter
                detections.append({
                    "box_3d": box.tolist(), # [x, y, z, dx, dy, dz, rot]
                    "score": float(score),
                    "label": int(label),
                    "model": "PointPillars"
                })
        return detections