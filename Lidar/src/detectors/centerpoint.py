from mmdet3d.apis import init_model, inference_detector
from .base import BaseLidarDetector

class CenterPointDetector(BaseLidarDetector):
    def load_model(self):
        print(f"🎯 Loading CenterPoint from {self.checkpoint_path}...")
        return init_model(self.config_path, self.checkpoint_path, device=self.device)

    def detect(self, pcd_path):
        result, data = inference_detector(self.model, pcd_path)
        
        pred_instances = result.pred_instances_3d
        bboxes_3d = pred_instances.bboxes_3d.tensor.cpu().numpy()
        scores = pred_instances.scores_3d.cpu().numpy()
        labels = pred_instances.labels_3d.cpu().numpy()
        
        detections = []
        for box, score, label in zip(bboxes_3d, scores, labels):
            if score > 0.3:
                detections.append({
                    "box_3d": box.tolist(),
                    "score": float(score),
                    "label": int(label),
                    "model": "CenterPoint"
                })
        return detections