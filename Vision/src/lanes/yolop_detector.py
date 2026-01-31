import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
import time
import torch.nn.functional as F

class YOLOPDetector:
    def __init__(self, device=None):
        """
        Detector YOLOP con lógica corregida (Argmax + Filtrado de Bordes).
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

    def detect(self, img_bgr, show_drivable=True, show_lanes=True, show_lane_points=True):
        t_start = time.time()
        h_orig, w_orig, _ = img_bgr.shape
        
        # 1. Preproceso
        img_resized = cv2.resize(img_bgr, (self.img_size, self.img_size))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)

        # 2. Inferencia
        with torch.no_grad():
            _, da_seg_out, ll_seg_out = self.model(input_tensor)

        # 3. Post-proceso
        # Interpolamos primero los logits al tamaño original
        da_seg_resized = F.interpolate(da_seg_out, size=(h_orig, w_orig), mode='bilinear', align_corners=True)
        ll_seg_resized = F.interpolate(ll_seg_out, size=(h_orig, w_orig), mode='bilinear', align_corners=True)
        
        # A) Área Conducible: Argmax estándar
        da_mask = torch.max(da_seg_resized, 1)[1].byte().squeeze().cpu().numpy()
        
        # B) Líneas de Carril: Argmax (Volvemos a la lógica original que funcionaba mejor geométricamente)
        ll_mask = torch.max(ll_seg_resized, 1)[1].byte().squeeze().cpu().numpy()

        t_end = time.time()
        latency = (t_end - t_start) * 1000

        # 4. Visualización
        result = img_bgr.copy()
        
        # Capa de Área Conducible (Verde)
        if show_drivable:
            overlay_da = np.zeros_like(result)
            overlay_da[da_mask == 1] = [0, 255, 0]
            # Usamos una mezcla suave solo donde hay detección
            mask_bool = da_mask == 1
            if np.any(mask_bool): # Only apply if there are actual pixels to modify
                # Apply addWeighted to the full images and then use np.where to blend
                blended_segment = cv2.addWeighted(result, 0.6, overlay_da, 0.4, 0)
                result = np.where(mask_bool[:, :, None], blended_segment, result)

        # Capa de Líneas (Roja - Opcional)
        if show_lanes:
            overlay_ll = np.zeros_like(result)
            # Pintamos rojo solo donde ll_mask es 1 Y NO es da_mask (evitar solapamiento feo)
            # A veces ayuda pintar solo lo que no es drivable, pero YOLOP suele separar bien.
            overlay_ll[ll_mask == 1] = [0, 0, 255]
            
            mask_bool = ll_mask == 1
            result[mask_bool] = cv2.addWeighted(result[mask_bool], 0.6, overlay_ll[mask_bool], 0.4, 0)

        # Capa de Vectores (Cyan - Limpia)
        if show_lane_points:
            contours, _ = cv2.findContours(ll_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                try:
                    area = cv2.contourArea(cnt)
                    
                    # FILTRO 1: Ruido pequeño
                    if area < 400: continue
                    
                    # FILTRO 2: El "Marco" de la imagen
                    # Si el contorno es casi tan grande como la imagen entera, es un error del modelo
                    x, y, w, h = cv2.boundingRect(cnt)
                    if w > w_orig * 0.95 and h > h_orig * 0.95:
                        continue # Ignoramos el marco completo
                    
                    epsilon = 0.02 * cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, epsilon, True)
                    
                    # Dibujar línea cyan
                    cv2.drawContours(result, [approx], -1, (255, 255, 0), 3)
                    
                    # Dibujar puntos
                    for point in approx:
                        px, py = point[0]
                        cv2.circle(result, (px, py), 5, (0, 0, 255), -1)
                except Exception as e:
                    print(f"DEBUG YOLOP ERROR in contour processing: {e}, contour: {cnt}")
                    continue # Skip this problematic contour

        return result, latency