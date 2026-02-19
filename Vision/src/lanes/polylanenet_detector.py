import torch
import torch.nn as nn
import cv2
import numpy as np
import time
from pathlib import Path
from efficientnet_pytorch import EfficientNet
from config.utils.path_manager import path_manager

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
             model_path = path_manager.get_model("polylanenet")
             
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
            
            # Filtro de confianza
            if conf < 0.3:
                continue

            # 2. Rango Vertical
            # El modelo usa coordenadas estándar: 0=arriba(cielo), 1=abajo(coche)
            # (confirmado del código oficial: ys = y_samples / img_h)
            y_min_norm = lane_params[1]  # límite superior del carril
            y_max_norm = lane_params[2]  # límite inferior del carril

            # Filtro de horizonte: TuSimple entrenó con y_min >= 160/720 ≈ 0.22
            # Predicciones con y_min_norm < 0.20 son ruido del modelo (líneas fantasma
            # que empiezan en el cielo y cruzan las líneas reales).
            if y_min_norm < 0.20:
                continue

            # Conversión directa a píxeles (espacio estándar de imagen)
            y_start = int(y_min_norm * h_orig)  # fila superior del carril
            y_end   = int(y_max_norm * h_orig)  # fila inferior del carril

            y_start = max(0, min(h_orig - 1, y_start))
            y_end   = max(0, min(h_orig - 1, y_end))

            if y_end <= y_start or (y_end - y_start) < 20:
                continue

            # 3. Coeficientes polinómicos: [c3, c2, c1, c0] (mayor grado primero)
            # Evaluación oficial: np.polyval(lane[3:], y_norm) donde y_norm=y/h
            # x_norm = c3*y^3 + c2*y^2 + c1*y + c0
            coeffs = lane_params[3:]  # [c3, c2, c1, c0]

            # Generar puntos usando normalización ESTÁNDAR (igual que el entrenamiento)
            plot_y = np.linspace(y_start, y_end, num=50)
            points = []

            for y in plot_y:
                # Normalización estándar: 0=arriba(cielo), 1=abajo(coche)
                # Igual que el código oficial: ys = y_pixel / img_h
                y_norm = y / h_orig

                # Evaluación polinómica (numpy polyval: coefs de mayor a menor grado)
                x_norm = np.polyval(coeffs, y_norm)
                x = int(x_norm * w_orig)

                if 0 <= x < w_orig:  # Solo puntos dentro del ancho de la imagen
                    points.append((x, int(y)))

            # Rechazar si menos del 30% de puntos están dentro de la imagen
            # (evita líneas casi completamente fuera del frame)
            if len(points) < max(3, int(0.30 * len(plot_y))):
                continue

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