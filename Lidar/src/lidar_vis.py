import cv2
import numpy as np
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import LidarPointCloud, Box
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility

class LidarVisualizer:
    def __init__(self):
        self.colors = {
            0: (0, 0, 255),    # Coche (Rojo)
            1: (0, 255, 255),  # Camión (Amarillo)
            'dots': (0, 255, 0) # Puntos (Verde)
        }
        # Configuración BEV
        self.bev_res = 0.1  # Metros por pixel
        self.bev_range = 50 # Metros a la redonda
        
        # Correcciones Manuales (Heredadas de tus pruebas)
        self.SHIFT_X = 0.5

    def render_bev(self, points_path, detections, title="BEV"):
        """Genera la vista aérea (Eagle Eye)."""
        # Dimensión de imagen
        img_size = int((self.bev_range * 2) / self.bev_res)
        img = np.zeros((img_size, img_size, 3), dtype=np.uint8)

        # 1. PUNTOS
        points = np.fromfile(points_path, dtype=np.float32, count=-1).reshape([-1, 5])
        x_pts, y_pts = points[:, 0], points[:, 1]
        
        # Mapeo: X mundo -> -Y imagen (Arriba), Y mundo -> -X imagen (Izquierda)
        # Centramos sumando img_size / 2
        x_img = ((-y_pts / self.bev_res) + img_size / 2).astype(np.int32)
        y_img = ((-x_pts / self.bev_res) + img_size / 2).astype(np.int32)
        
        # Filtrar puntos dentro del rango
        mask = (x_img >= 0) & (x_img < img_size) & (y_img >= 0) & (y_img < img_size)
        img[y_img[mask], x_img[mask]] = (200, 200, 200)

        # 2. CAJAS
        for det in detections:
            if det['score'] < 0.2: continue
            v = det['box']
            
            # Dimensiones (Intercambiadas para corregir giro)
            w, l, h = v[4], v[3], v[5]
            x, y = v[0] + self.SHIFT_X, v[1]
            yaw = v[6]

            # Rectángulo rotado para OpenCV
            # Center (x,y) en pixels
            center_x = (-y / self.bev_res) + img_size / 2
            center_y = (-x / self.bev_res) + img_size / 2
            size_x = w / self.bev_res
            size_y = l / self.bev_res
            
            # OpenCV BoxPoints
            rect = ((center_x, center_y), (size_x, size_y), -np.degrees(yaw))
            box = cv2.boxPoints(rect).astype(np.int32) # Fix para numpy nuevo
            
            c = self.colors.get(det['label'], (0, 255, 255))
            cv2.drawContours(img, [box], 0, c, 2)
            
            # Indicador de frente
            front = (box[0] + box[1]) // 2
            cv2.line(img, tuple(box.mean(axis=0).astype(int)), tuple(front), (0, 255, 255), 1)

        cv2.putText(img, title, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return img

    def render_camera(self, img_path, points_path, detections, calib, title=""):
        """Proyección Frontal con correcciones."""
        img = cv2.imread(img_path)
        if img is None: return np.zeros((900, 1600, 3), dtype=np.uint8)
        
        # 1. Undistort
        intrinsic = np.array(calib['intrinsic'])
        dist_coeffs = np.array(calib['distortion'])
        if np.any(dist_coeffs):
            img = cv2.undistort(img, intrinsic, dist_coeffs)

        # 2. Puntos
        points = np.fromfile(points_path, dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]
        pc = LidarPointCloud(points.T.copy())
        self._apply_transforms(pc, calib)
        
        depths = pc.points[2, :]
        points_img = view_points(pc.points[:3, :], intrinsic, normalize=True)
        
        mask = depths > 1.0
        for i in range(points_img.shape[1]):
            if mask[i]:
                x, y = int(points_img[0, i]), int(points_img[1, i])
                if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                    d = min(depths[i], 60.0) / 60.0
                    c = (int(255*(1-d)), int(255*d), 0)
                    cv2.circle(img, (x, y), 2, c, -1)

        # 3. Cajas
        for det in detections:
            if det['score'] < 0.2: continue
            v = det['box']
            
            w, l, h = v[4], v[3], v[5]
            center = list(v[:3])
            center[0] += self.SHIFT_X
            # Z-Shift Dinámico: Subir la mitad de la altura + un poco
            center[2] += (h / 2.0) + 0.1 
            
            box = Box(center=center, size=[w, l, h], orientation=Quaternion(axis=[0,0,1], radians=v[6]))
            self._apply_transforms_box(box, calib)
            
            if box_in_image(box, intrinsic, (img.shape[1], img.shape[0]), vis_level=BoxVisibility.ANY):
                self._draw_box_cv2(img, box, intrinsic, det['label'])
                
        cv2.putText(img, title, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        return img

    def _draw_box_cv2(self, img, box, intrinsic, label):
        corners = view_points(box.corners(), intrinsic, normalize=True)[:2, :]
        corners = corners.astype(np.int32).T
        c = self.colors.get(label, (255, 0, 255))
        for i in range(4):
            cv2.line(img, tuple(corners[i]), tuple(corners[(i+1)%4]), c, 2)
            cv2.line(img, tuple(corners[i+4]), tuple(corners[(i+1)%4 + 4]), c, 2)
            cv2.line(img, tuple(corners[i]), tuple(corners[i+4]), c, 2)

    def _apply_transforms(self, pc, calib):
        pc.rotate(Quaternion(calib['cs_lidar']['rotation']).rotation_matrix)
        pc.translate(np.array(calib['cs_lidar']['translation']))
        pc.rotate(Quaternion(calib['pose_lidar']['rotation']).rotation_matrix)
        pc.translate(np.array(calib['pose_lidar']['translation']))
        pc.translate(-np.array(calib['pose_cam']['translation']))
        pc.rotate(Quaternion(calib['pose_cam']['rotation']).rotation_matrix.T)
        pc.translate(-np.array(calib['cs_cam']['translation']))
        pc.rotate(Quaternion(calib['cs_cam']['rotation']).rotation_matrix.T)

    def _apply_transforms_box(self, box, calib):
        box.rotate(Quaternion(calib['cs_lidar']['rotation']))
        box.translate(np.array(calib['cs_lidar']['translation']))
        box.rotate(Quaternion(calib['pose_lidar']['rotation']))
        box.translate(np.array(calib['pose_lidar']['translation']))
        box.translate(-np.array(calib['pose_cam']['translation']))
        box.rotate(Quaternion(calib['pose_cam']['rotation']).inverse)
        box.translate(-np.array(calib['cs_cam']['translation']))
        box.rotate(Quaternion(calib['cs_cam']['rotation']).inverse)