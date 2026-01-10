import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
import time  # <--- NUEVO

class YOLOPDetector:
    def __init__(self, device=None):
        """
        Detector de carriles YOLOP con medición de tiempo.
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

    def detect(self, img_bgr, show_drivable=True, show_lanes=True, show_lane_points=False):
        """
        Retorna:
            result (numpy array): Imagen con visualización.
            latency (float): Tiempo de inferencia en milisegundos.
        """
        t_start = time.time()  # <--- INICIO CRONÓMETRO

        h_orig, w_orig, _ = img_bgr.shape
        
        # 1. Preproceso
        img_resized = cv2.resize(img_bgr, (self.img_size, self.img_size))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)

        # 2. Inferencia
        with torch.no_grad():
            _, da_seg_out, ll_seg_out = self.model(input_tensor)

        # 3. Post-proceso
        da_seg_mask = torch.nn.functional.interpolate(da_seg_out, size=(h_orig, w_orig), mode='bilinear')
        ll_seg_mask = torch.nn.functional.interpolate(ll_seg_out, size=(h_orig, w_orig), mode='bilinear')
        
        da_mask = torch.max(da_seg_mask, 1)[1].byte().squeeze().cpu().numpy()
        ll_mask = torch.max(ll_seg_mask, 1)[1].byte().squeeze().cpu().numpy()

        t_end = time.time()  # <--- FIN CRONÓMETRO
        latency = (t_end - t_start) * 1000  # ms

        # 4. Visualización
        overlay = np.zeros_like(img_bgr, dtype=np.uint8)
        
        if show_drivable:
            overlay[da_mask == 1] = [0, 255, 0] 
        if show_lanes:
            overlay[ll_mask == 1] = [0, 0, 255]

        result = cv2.addWeighted(img_bgr, 1.0, overlay, 0.4, 0)

        if show_lane_points:
            contours, _ = cv2.findContours(ll_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(result, contours, -1, (255, 255, 0), 2)
            for cnt in contours:
                epsilon = 0.005 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                for point in approx:
                    x, y = point[0]
                    cv2.circle(result, (x, y), 3, (0, 255, 255), -1)

        return result, latency  # <--- RETORNAMOS LA LATENCIA