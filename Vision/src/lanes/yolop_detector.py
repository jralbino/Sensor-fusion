import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
import time

class YOLOPDetector:
    def __init__(self, device=None):
        """
        Detector YOLOP con vectorización de líneas (Polígonos simplificados).
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🛣️ Cargando YOLOP en {self.device}...")
        
        try:
            self.model = torch.hub.load('hustvl/yolop', 'yolop', pretrained=True, trust_repo=True)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"❌ Error cargando YOLOP: {e}")
            raise e

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.img_size = 640

    def detect(self, img_bgr, show_drivable=True, show_lanes=False, show_lane_points=True):
        """
        Retorna la imagen procesada y la latencia.
        """
        t_start = time.time()

        h_orig, w_orig, _ = img_bgr.shape
        
        # 1. Preproceso
        img_resized = cv2.resize(img_bgr, (self.img_size, self.img_size))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)

        # 2. Inferencia
        with torch.no_grad():
            _, da_seg_out, ll_seg_out = self.model(input_tensor)

        # 3. Post-proceso (Interpolación)
        da_seg_mask = torch.nn.functional.interpolate(da_seg_out, size=(h_orig, w_orig), mode='bilinear')
        ll_seg_mask = torch.nn.functional.interpolate(ll_seg_out, size=(h_orig, w_orig), mode='bilinear')
        
        da_mask = torch.max(da_seg_mask, 1)[1].byte().squeeze().cpu().numpy()
        ll_mask = torch.max(ll_seg_mask, 1)[1].byte().squeeze().cpu().numpy()

        t_end = time.time()
        latency = (t_end - t_start) * 1000

        # 4. Visualización
        # Usamos una copia limpia o el overlay según se desee
        result = img_bgr.copy()
        overlay = np.zeros_like(img_bgr, dtype=np.uint8)
        
        # A) Área Verde (Drivable)
        if show_drivable:
            overlay[da_mask == 1] = [0, 255, 0] 
            result = cv2.addWeighted(result, 1.0, overlay, 0.4, 0)
        
        # B) Máscara Sólida (Opcional, el usuario pidió apagarla generalmente)
        if show_lanes:
            overlay_lanes = np.zeros_like(img_bgr, dtype=np.uint8)
            overlay_lanes[ll_mask == 1] = [0, 0, 255]
            result = cv2.addWeighted(result, 1.0, overlay_lanes, 0.4, 0)

        # C) VECTORIZACIÓN: Líneas simplificadas con pocos puntos
        if show_lane_points:
            # Encontrar contornos en la máscara binaria
            contours, _ = cv2.findContours(ll_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                # Filtrar ruido: Si el área es muy pequeña, ignorar
                if cv2.contourArea(cnt) < 100:
                    continue
                
                # --- MAGIA MATEMÁTICA: SIMPLIFICACIÓN ---
                # epsilon determina la "tolerancia". 
                # 0.02 (2%) del perímetro es un buen balance para carreteras.
                # Si quieres líneas AÚN más rectas/simples, sube a 0.03 o 0.04.
                epsilon = 0.02 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                
                # Dibujar la polilínea simplificada (Cyan, gruesa)
                cv2.drawContours(result, [approx], -1, (255, 255, 0), 3)
                
                # Dibujar los vértices/puntos de anclaje (Rojos)
                for point in approx:
                    x, y = point[0]
                    cv2.circle(result, (x, y), 5, (0, 0, 255), -1)

        return result, latency