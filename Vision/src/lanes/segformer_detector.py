import torch
import cv2
import numpy as np
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import time

class SegFormerDetector:
    def __init__(self, device=None):
        """
        Detector de Segmentación basado en NVIDIA SegFormer.
        Modelo: nvidia/segformer-b0-finetuned-cityscapes-1024-1024
        
        Este modelo es SOTA (State-of-the-Art) en segmentación de escenas.
        Lo usaremos para encontrar el 'Drivable Area' (Carretera) con precisión NVIDIA,
        y luego extraeremos las líneas dentro de esa área.
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🐉 Cargando NVIDIA SegFormer en {self.device}...")
        
        model_name = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
        
        try:
            self.processor = SegformerImageProcessor.from_pretrained(model_name)
            self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"❌ Error cargando SegFormer: {e}")
            print("Intenta: pip install transformers")
            raise e

        # En Cityscapes, la clase 'Road' (Carretera) es el índice 0.
        self.road_class_index = 0

    def detect(self, img_bgr, **kwargs):
        """
        Retorna:
            result: Imagen fusionada (Drivable Area verde + Líneas Cyan).
            latency: Tiempo en ms.
        """
        t_start = time.time()
        
        h_orig, w_orig, _ = img_bgr.shape
        
        # 1. Preproceso (HuggingFace Processor se encarga del resize y norm)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=img_rgb, return_tensors="pt").to(self.device)

        # 2. Inferencia
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # 3. Post-proceso (Upsampling a tamaño original)
        logits = outputs.logits
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=(h_orig, w_orig), # Restaurar tamaño original
            mode="bilinear",
            align_corners=False,
        )
        
        # Argmax para obtener la clase ganadora en cada pixel
        pred_seg = upsampled_logits.argmax(dim=1)[0] # Shape: (H, W)
        
        # Extraer máscara de carretera (Clase 0)
        road_mask = (pred_seg == self.road_class_index).byte().cpu().numpy()

        t_end = time.time()
        latency = (t_end - t_start) * 1000

        # 4. Visualización Avanzada
        # A) Pinta el Área de Conducción (Verde NVIDIA)
        overlay = np.zeros_like(img_bgr, dtype=np.uint8)
        overlay[road_mask == 1] = [0, 255, 0]
        
        # B) Truco: Extraer líneas DENTRO de la carretera usando bordes
        # Si ya sabemos dónde está la carretera, las líneas blancas/amarillas dentro de ella son carriles.
        if road_mask.sum() > 0:
            # Recortamos solo la carretera de la imagen original
            road_area = cv2.bitwise_and(img_bgr, img_bgr, mask=road_mask)
            
            # Convertimos a escala de grises y aumentamos contraste
            gray_road = cv2.cvtColor(road_area, cv2.COLOR_BGR2GRAY)
            
            # Filtro de líneas brillantes (Umbral adaptativo o fijo alto)
            # Las líneas suelen ser mucho más brillantes que el asfalto
            _, lines_mask = cv2.threshold(gray_road, 180, 255, cv2.THRESH_BINARY)
            
            # Limpiar ruido (Erosión + Dilatación)
            kernel = np.ones((3,3), np.uint8)
            lines_mask = cv2.morphologyEx(lines_mask, cv2.MORPH_OPEN, kernel)
            
            # Pintar líneas en Rojo/Cyan sobre el overlay
            overlay[lines_mask == 255] = [255, 255, 0] # Cyan

        # Mezclar todo
        result = cv2.addWeighted(img_bgr, 0.8, overlay, 0.4, 0)
        
        return result, latency