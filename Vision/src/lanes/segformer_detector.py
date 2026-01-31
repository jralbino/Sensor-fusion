import torch
import cv2
import numpy as np
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import time

class SegFormerDetector:
    def __init__(self, device=None):
        """
        Segmentation detector based on NVIDIA SegFormer.
        Model: nvidia/segformer-b0-finetuned-cityscapes-1024-1024
        
        This model is SOTA (State-of-the-Art) in scene segmentation.
        We will use it to find the 'Drivable Area' (Road) with NVIDIA precision,
        and then extract the lines within that area.
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🐉 Loading NVIDIA SegFormer on {self.device}...")
        
        model_name = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
        
        try:
            self.processor = SegformerImageProcessor.from_pretrained(model_name)
            self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"❌ Error loading SegFormer: {e}")
            print("Try: pip install transformers")
            raise e

        # In Cityscapes, the 'Road' class is index 0.
        self.road_class_index = 0

    def detect(self, img_bgr, **kwargs):
        """
        Returns:
            result: Fused image (Green Drivable Area + Cyan Lines).
            latency: Time in ms.
        """
        t_start = time.time()
        
        h_orig, w_orig, _ = img_bgr.shape
        
        # 1. Preprocessing (HuggingFace Processor handles resize and norm)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=img_rgb, return_tensors="pt").to(self.device)

        # 2. Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # 3. Post-processing (Upsampling to original size)
        logits = outputs.logits
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=(h_orig, w_orig), # Restore original size
            mode="bilinear",
            align_corners=False,
        )
        
        # Argmax to get the winning class for each pixel
        pred_seg = upsampled_logits.argmax(dim=1)[0] # Shape: (H, W)
        
        # Extract road mask (Class 0)
        road_mask = (pred_seg == self.road_class_index).byte().cpu().numpy()

        t_end = time.time()
        latency = (t_end - t_start) * 1000

        # 4. Advanced Visualization
        # A) Paint Drivable Area (NVIDIA Green)
        overlay = np.zeros_like(img_bgr, dtype=np.uint8)
        overlay[road_mask == 1] = [0, 255, 0]
        
        # B) Trick: Extract lines INSIDE the road using edges
        # If we already know where the road is, white/yellow lines within it are lanes.
        if road_mask.sum() > 0:
            # Crop only the road from the original image
            road_area = cv2.bitwise_and(img_bgr, img_bgr, mask=road_mask)
            
            # Convert to grayscale and increase contrast
            gray_road = cv2.cvtColor(road_area, cv2.COLOR_BGR2GRAY)
            
            # Bright line filter (Adaptive or fixed high threshold)
            # Lines are usually much brighter than asphalt
            _, lines_mask = cv2.threshold(gray_road, 180, 255, cv2.THRESH_BINARY)
            
            # Clean noise (Erosion + Dilation)
            kernel = np.ones((3,3), np.uint8)
            lines_mask = cv2.morphologyEx(lines_mask, cv2.MORPH_OPEN, kernel)
            
            # Paint lines in Red/Cyan on the overlay
            overlay[lines_mask == 255] = [255, 255, 0] # Cyan

        # Blend everything
        result = cv2.addWeighted(img_bgr, 0.8, overlay, 0.4, 0)
        
        return result, latency