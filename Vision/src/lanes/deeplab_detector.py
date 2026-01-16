import torch
import cv2
import numpy as np
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
import torchvision.transforms as transforms
import time

class DeepLabDetector:
    def __init__(self, device=None):
        """
        Detector de Segmentación Semántica Generalista (DeepLabV3-ResNet50).
        Entrenado en COCO (21 clases estándar de Pascal VOC).
        Útil para comparar segmentación "densa" vs la "ligera" de YOLOP.
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🧠 Cargando DeepLabV3 (ResNet50) en {self.device}...")
        
        # Cargar pesos por defecto (COCO/Pascal VOC)
        weights = DeepLabV3_ResNet50_Weights.DEFAULT
        self.model = deeplabv3_resnet50(weights=weights)
        self.model.to(self.device)
        self.model.eval()

        # Transformación estándar recomendada por los pesos
        self.transform = weights.transforms()
        
        # Mapeo de colores para las 21 clases (para que se vea bonito)
        # 0=background, 15=person, 7=car, etc.
        self.colors = np.random.randint(0, 255, (21, 3), dtype=np.uint8)
        self.colors[0] = [0, 0, 0] # Background negro transparente

    def detect(self, img_bgr, **kwargs):
        """
        Realiza segmentación semántica.
        Nota: **kwargs se usa para absorber argumentos extra como 'show_lanes' 
        que main.py podría enviar (para compatibilidad con YOLOP), aunque aquí no se usen.
        """
        t_start = time.time()
        
        # 1. Preproceso
        # DeepLab espera RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # Convertir a tensor usando la transformación oficial
        input_tensor = self.transform(torch.from_numpy(img_rgb).permute(2, 0, 1)).unsqueeze(0).to(self.device)
        
        # 2. Inferencia
        with torch.no_grad():
            output = self.model(input_tensor)['out'][0]
        
        # 3. Post-proceso
        # output shape: (21, H, W). Hacemos argmax para ver qué clase gana en cada pixel.
        predictions = output.argmax(0).byte().cpu().numpy()
        
        # Redimensionar la máscara al tamaño original de la imagen si es necesario
        # (DeepLab a veces cambia el tamaño internamente, pero torchvision suele mantenerlo)
        if predictions.shape != img_bgr.shape[:2]:
            predictions = cv2.resize(predictions, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

        t_end = time.time()
        latency = (t_end - t_start) * 1000

        # 4. Visualización
        # Crear una imagen de color basada en las clases
        seg_color = self.colors[predictions]
        
        # Mezclar con la original
        # Alpha 0.5 para ver el video debajo de la máscara
        result = cv2.addWeighted(img_bgr, 0.5, seg_color, 0.5, 0)
        
        return result, latency