import torch
import cv2
import numpy as np
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
import torchvision.transforms as transforms
import time

class DeepLabDetector:
    def __init__(self, device=None):
        """
        Generic Semantic Segmentation Detector (DeepLabV3-ResNet50).
        Trained on COCO (21 standard Pascal VOC classes).
        Useful for comparing "dense" vs "light" segmentation of YOLOP.
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🧠 Loading DeepLabV3 (ResNet50) on {self.device}...")
        
        # Load default weights (COCO/Pascal VOC)
        weights = DeepLabV3_ResNet50_Weights.DEFAULT
        self.model = deeplabv3_resnet50(weights=weights)
        self.model.to(self.device)
        self.model.eval()

        # Standard transformation recommended by the weights
        self.transform = weights.transforms()
        
        # Color mapping for the 21 classes (for nice visualization)
        # 0=background, 15=person, 7=car, etc.
        self.colors = np.random.randint(0, 255, (21, 3), dtype=np.uint8)
        self.colors[0] = [0, 0, 0] # Black transparent background

    def detect(self, img_bgr, **kwargs):
        """
        Performs semantic segmentation.
        Note: **kwargs is used to absorb extra arguments like 'show_lanes'
        that main.py might send (for YOLOP compatibility), although not used here.
        """
        t_start = time.time()
        
        # 1. Preprocessing
        # DeepLab expects RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # Convert to tensor using the official transformation
        input_tensor = self.transform(torch.from_numpy(img_rgb).permute(2, 0, 1)).unsqueeze(0).to(self.device)
        
        # 2. Inference
        with torch.no_grad():
            output = self.model(input_tensor)['out'][0]
        
        # 3. Post-processing
        # output shape: (21, H, W). We do argmax to see which class wins in each pixel.
        predictions = output.argmax(0).byte().cpu().numpy()
        
        # Resize mask to original image size if necessary
        # (DeepLab sometimes resizes internally, but torchvision usually keeps it)
        if predictions.shape != img_bgr.shape[:2]:
            predictions = cv2.resize(predictions, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

        t_end = time.time()
        latency = (t_end - t_start) * 1000

        # 4. Visualization
        # Create a color image based on classes
        seg_color = self.colors[predictions]
        
        # Blend with original
        # Alpha 0.5 to see the video under the mask
        result = cv2.addWeighted(img_bgr, 0.5, seg_color, 0.5, 0)
        
        return result, latency