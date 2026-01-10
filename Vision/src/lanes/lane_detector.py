import cv2
import numpy as np

class LaneDetector:
    def __init__(self, debug=False):
        """
        Detector de líneas clásico usando Ventanas Deslizantes y Polinomios.
        """
        self.debug = debug
        
        # Estado previo para suavizar detecciones entre frames (video estable)
        self.left_fit_prev = None
        self.right_fit_prev = None
        self.sanity_check_passes = 0

    def preprocess(self, img):
        """Aplica umbrales de color y gradiente para aislar líneas."""
        # 1. Convertir a HLS (Hue, Lightness, Saturation)
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        l_channel = hls[:, :, 1]
        s_channel = hls[:, :, 2]

        # 2. Filtrar blancos (Alta luminosidad)
        white_mask = cv2.inRange(l_channel, 200, 255)
        
        # 3. Filtrar amarillos (Hue específico + Alta saturación)
        # Hue del amarillo en OpenCV es aprox 20-30
        yellow_lower = np.array([15, 30, 115])
        yellow_upper = np.array([35, 204, 255])
        yellow_mask = cv2.inRange(hls, yellow_lower, yellow_upper)
        
        # 4. Combinar
        combined_binary = cv2.bitwise_or(white_mask, yellow_mask)
        
        # Limpieza morfológica (quitar ruido)
        kernel = np.ones((5,5), np.uint8)
        combined_binary = cv2.morphologyEx(combined_binary, cv2.MORPH_CLOSE, kernel)
        
        return combined_binary

    def perspective_transform(self, img_binary):
        """Transforma la vista frontal a 'Bird's Eye View'."""
        h, w = img_binary.shape
        
        # Puntos definidos a mano para BDD100K (1280x720)
        # Ajustados para enfocarse en el carril frente al coche
        src = np.float32([
            [200, 720],  # Bottom Left
            [1100, 720], # Bottom Right
            [595, 450],  # Top Left (Horizonte)
            [685, 450]   # Top Right (Horizonte)
        ])
        
        # Destino: Rectángulo plano
        dst = np.float32([
            [300, h],       # Bottom Left
            [980, h],       # Bottom Right
            [300, 0],       # Top Left
            [980, 0]        # Top Right
        ])
        
        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src) # Para volver atrás
        
        warped = cv2.warpPerspective(img_binary, M, (w, h), flags=cv2.INTER_LINEAR)
        return warped, Minv

    def find_lines_sliding_window(self, binary_warped):
        """Encuentra los pixeles de las líneas usando histogramas."""
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
        midpoint = int(histogram.shape[0]//2)
        
        # Picos del histograma = base de las líneas
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        
        # Hiperparámetros
        nwindows = 9
        margin = 100
        minpix = 50
        window_height = int(binary_warped.shape[0]//nwindows)
        
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        left_lane_inds = []
        right_lane_inds = []
        
        leftx_current = leftx_base
        rightx_current = rightx_base
        
        for window in range(nwindows):
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            
            # Límites X
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            
            # Identificar pixeles dentro de la ventana
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            # Re-centrar si encontramos suficientes pixeles
            if len(good_left_inds) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))
                
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        
        # Extraer coordenadas
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        
        # Ajuste Polinomial de 2do grado (Parábola): x = Ay^2 + By + C
        # Si no detecta nada, usa el previo (suavizado simple)
        if len(leftx) == 0 or len(rightx) == 0:
            return self.left_fit_prev, self.right_fit_prev

        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        
        # Guardar para el siguiente frame
        self.left_fit_prev = left_fit
        self.right_fit_prev = right_fit
        
        return left_fit, right_fit

    def draw_lane(self, original_img, binary_warped, left_fit, right_fit, Minv):
        """Pinta el área del carril sobre la imagen original."""
        h, w = binary_warped.shape
        ploty = np.linspace(0, h-1, h)
        
        # Generar puntos X a partir del polinomio
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        
        # Crear imagen vacía para pintar
        warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        
        # Puntos para polígono
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        
        # Dibujar carril verde
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
        
        # Deshacer perspectiva (volver a vista frontal)
        newwarp = cv2.warpPerspective(color_warp, Minv, (w, h))
        
        # Combinar
        result = cv2.addWeighted(original_img, 1, newwarp, 0.3, 0)
        return result

    def detect(self, img):
        """
        Pipeline principal.
        Retorna: Imagen con carril pintado.
        """
        # Copia para no alterar original
        out_img = np.copy(img)
        
        # 1. Preproceso (Color -> Binario)
        binary = self.preprocess(out_img)
        
        # 2. Perspectiva (Frontal -> Aérea)
        warped, Minv = self.perspective_transform(binary)
        
        # 3. Encontrar Líneas (Matemáticas)
        left_fit, right_fit = self.find_lines_sliding_window(warped)
        
        # 4. Dibujar y regresar a perspectiva original
        if left_fit is not None and right_fit is not None:
            result = self.draw_lane(out_img, warped, left_fit, right_fit, Minv)
            return result
        else:
            return out_img # Si falla, retorna original