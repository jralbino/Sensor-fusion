import torch
import torch.nn as nn
import cv2
import numpy as np
import time
from pathlib import Path
from efficientnet_pytorch import EfficientNet

# ... (Mantén las clases CustomHead y PolyRegression igual que antes) ...
# ... (Solo cambiaremos la clase PolyLaneNetDetector abajo) ...

class CustomHead(nn.Module):
    def __init__(self, in_features, out_features):
        super(CustomHead, self).__init__()
        self.regular_outputs_layer = nn.Linear(in_features, out_features)
    def forward(self, x):
        return self.regular_outputs_layer(x)

class PolyRegression(nn.Module):
    def __init__(self, num_outputs=35, backbone='efficientnet-b1', pretrained=False):
        super(PolyRegression, self).__init__()
        if pretrained: self.model = EfficientNet.from_pretrained(backbone)
        else: self.model = EfficientNet.from_name(backbone)
        feature_dim = self.model._fc.in_features
        self.model._fc = CustomHead(feature_dim, num_outputs)
    def forward(self, x):
        return self.model(x)

class PolyLaneNetDetector:
    def __init__(self, model_path=None, device='cuda'):
        # FIX DE PATH
        if model_path is None:
             model_path = "models/model_2305.pt"
             
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"📉 Cargando PolyLaneNet desde {model_path}...")        
        
        self.input_width = 640
        self.input_height = 360
        self.num_lanes = 5
        
        self.model = PolyRegression(num_outputs=35, backbone='efficientnet-b1', pretrained=False)
        self.model.to(self.device)
        
        path = Path(model_path)
        if not path.exists():
            print("Error path model")
            #path = Path("Vision/models") / path.name
        
        checkpoint = torch.load(path, map_location=self.device)
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
        else:
            state_dict = checkpoint
            
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        self.model.load_state_dict(new_state_dict, strict=False)
        self.model.eval()

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        
        # Variable para imprimir debug solo una vez
        self.debug_printed = False

    def detect(self, img_bgr, **kwargs):
        t_start = time.time()
        h_orig, w_orig, _ = img_bgr.shape

        # Preproceso
        img = cv2.resize(img_bgr, (self.input_width, self.input_height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(self.device)
        tensor = (tensor - self.mean) / self.std

        with torch.no_grad():
            output = self.model(tensor)

        t_end = time.time()
        latency = (t_end - t_start) * 1000

        pred = output[0].cpu().numpy()
        pred = pred.reshape(self.num_lanes, 7) 
        
        result = img_bgr.copy()
        # Colores (B, G, R, C, M)
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]

        for i in range(self.num_lanes):
            lane_params = pred[i]
            
            # 1. Confianza
            raw_conf = lane_params[0]
            conf = 1 / (1 + np.exp(-raw_conf))
            
            # Filtro de confianza (puedes bajarlo a 0.3 si detecta poco)
            if conf < 0.5: continue 

            # 2. Rango Vertical
            # El modelo devuelve y_min/y_max normalizados
            y_min_norm = lane_params[1] 
            y_max_norm = lane_params[2] 
            
            # 3. Coeficientes
            # IMPORTANTE: PolyLaneNet oficial usa [c0, c1, c2, c3] o [c3, c2, c1, c0]
            # Basado en que la línea amarilla (REPO) funcionó, la lógica de esa prueba
            # usaba x = c0 + ...
            # Así que asumimos: [c3, c2, c1, c0] -> x = c3*y^3 + ... + c0
            coeffs = lane_params[3:] 
            c3, c2, c1, c0 = coeffs[0], coeffs[1], coeffs[2], coeffs[3]

            # Convertir rangos a pixeles
            y_start = int(y_min_norm * h_orig)
            y_end = int(y_max_norm * h_orig)
            
            y_start = max(0, min(h_orig, y_start))
            y_end = max(0, min(h_orig, y_end))
            
            if abs(y_end - y_start) < 20: continue

            # Generar puntos
            # Importante: Iteramos de abajo (y_end) hacia arriba (y_start) o viceversa
            plot_y = np.linspace(y_start, y_end, num=50)
            points = []
            
            for y in plot_y:
                # --- LA FÓRMULA GANADORA (AMARILLA/REPO) ---
                # y_norm mide distancia desde el borde inferior
                # y=h -> y_norm=0 (Coche)
                # y=0 -> y_norm=1 (Cielo)
                y_norm = (h_orig - y) / h_orig
                
                # Ecuación Polinómica
                # x = c3*y^3 + c2*y^2 + c1*y + c0
                x_norm = (c3 * (y_norm**3) + 
                          c2 * (y_norm**2) + 
                          c1 * y_norm + 
                          c0)
                
                x = int(x_norm * w_orig)
                
                # Validar que esté en pantalla
                if -50 < x < w_orig + 50:
                    points.append((x, int(y)))

            if len(points) > 2:
                for k in range(len(points) - 1):
                    cv2.line(result, points[k], points[k+1], colors[i], 3)
                    
        return result, latency

    def _draw_test_line(self, img, color, label, coeffs, normalization):
        h, w, _ = img.shape
        # Simulamos un rango vertical típico de carril (desde el horizonte hasta el coche)
        y_start = int(h * 0.45) # Horizonte
        y_end = int(h * 0.95)   # Coche
        
        c3, c2, c1, c0 = coeffs # 0, 0, 0, 0.5
        points = []
        
        plot_y = np.linspace(y_start, y_end, num=50)
        
        for y in plot_y:
            # APLICAR DIFERENTES NORMALIZACIONES DE Y
            if normalization == "standard":
                y_norm = y / h
            elif normalization == "inverted_y":
                y_norm = 1.0 - (y / h)
            elif normalization == "relative":
                y_norm = (y - y_start) / (y_end - y_start)
            elif normalization == "repo_style":
                 y_norm = (h - y) / h

            # Ecuación simple: x = c0 (0.5)
            # Para probar curvas, añadiremos una curva artificial ligera:
            # x = 0.5 + 0.1 * y_norm^2 (para ver hacia dónde se curva)
            x_norm = c0 + 0.1 * (y_norm**2) 
            
            x = int(x_norm * w)
            points.append((x, int(y)))
            
        # Dibujar
        if len(points) > 2:
            for k in range(len(points) - 1):
                cv2.line(img, points[k], points[k+1], color, 4)
            # Poner etiqueta al inicio de la línea
            cv2.putText(img, label, points[0], cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)