import sys
import time
import json
import cv2
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock
from pyquaternion import Quaternion

# --- 1. MOCKS Y CONFIGURACIÓN (Necesario para que mmdet3d no falle) ---
MOCK_MODULES = [
    'lyft_dataset_sdk', 'lyft_dataset_sdk.lyftdataset', 'lyft_dataset_sdk.utils',
    'lyft_dataset_sdk.utils.data_classes', 'lyft_dataset_sdk.eval',
    'lyft_dataset_sdk.eval.detection', 'lyft_dataset_sdk.eval.detection.mAP_evaluation'
]
for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = MagicMock()

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR / 'src'))

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, Box
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility
from mmdet3d.utils import register_all_modules
from detectors.pointpillars import PointPillarsDetector
from detectors.centerpoint import CenterPointDetector

# Evitar logs ruidosos
try: register_all_modules(init_default_scope=False)
except: pass


# ==========================================
# CLASE 1: VISUALIZADOR (PINTOR)
# ==========================================
class LidarVisualizer:
    """Clase pura para renderizar. No sabe de modelos, solo recibe datos y pinta."""
    
    def __init__(self):
        self.colors = {
            0: (0, 0, 255),   # Car -> Rojo
            1: (0, 255, 255), # Truck/Otro -> Amarillo
            'dots': (0, 255, 0) # Puntos -> Verde
        }

    def render_bev(self, points_path, detections, title="BEV"):
        """Genera vista Eagle Eye (BEV) desde cero."""
        resolution = 0.1
        side_range = (-40, 40)
        fwd_range = (-40, 40)
        width = int((side_range[1] - side_range[0]) / resolution)
        height = int((fwd_range[1] - fwd_range[0]) / resolution)
        
        img = np.zeros((height, width, 3), dtype=np.uint8)

        # 1. Puntos
        points = np.fromfile(points_path, dtype=np.float32, count=-1).reshape([-1, 5])
        x_pts, y_pts = points[:, 0], points[:, 1]
        
        # Mapping Lidar -> Imagen
        x_img = ((-y_pts - side_range[0]) / resolution).astype(np.int32)
        y_img = ((-x_pts - fwd_range[0]) / resolution).astype(np.int32)
        
        mask = (x_img >= 0) & (x_img < width) & (y_img >= 0) & (y_img < height)
        img[y_img[mask], x_img[mask]] = (200, 200, 200)

        # 2. Cajas
        for det in detections:
            if det['score'] < 0.2: continue
            
            v = det['box'] # [x, y, z, dx, dy, dz, yaw]
            x, y, dx, dy, yaw = v[0], v[1], v[3], v[4], v[6]
            
            # Coordenadas rect rotado
            center = ((-y - side_range[0]) / resolution, (-x - fwd_range[0]) / resolution)
            size = (dy / resolution, dx / resolution)
            
            rect = (center, size, -np.degrees(yaw))
            box = cv2.boxPoints(rect)
            # --- CORRECCIÓN NUMPY ---
            box = box.astype(np.int32) # Reemplaza np.int0
            
            color = self.colors.get(det['label'], (255, 255, 255))
            cv2.drawContours(img, [box], 0, color, 2)
            
        cv2.putText(img, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return img

    def render_camera_projection(self, img_path, points_path, detections, calib, title="CAM"):
        """Proyecta Puntos y Cajas sobre imagen frontal."""
        img = cv2.imread(img_path)
        if img is None: return np.zeros((900, 1600, 3), dtype=np.uint8)
        
        # --- A. Proyectar Puntos ---
        points = np.fromfile(points_path, dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]
        pc = LidarPointCloud(points.T.copy()) # Copia crítica
        
        # Transformaciones Lidar -> Cam
        self._transform_pc(pc, calib)
        
        depths = pc.points[2, :]
        intrinsic = np.array(calib['cs_cam']['camera_intrinsic'])
        points_img = view_points(pc.points[:3, :], intrinsic, normalize=True)
        
        # Pintar puntos válidos
        mask = depths > 1.0
        points_img = points_img[:, mask]
        depths = depths[mask]
        
        for i in range(points_img.shape[1]):
            x, y = int(points_img[0, i]), int(points_img[1, i])
            if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                d = min(depths[i], 60.0) / 60.0
                # Gradiente Verde -> Amarillo
                color = (int(255 * (1-d)), int(255 * d), 0)
                cv2.circle(img, (x, y), 1, color, -1)

        # --- B. Proyectar Cajas ---
        for det in detections:
            if det['score'] < 0.15: continue
            
            v = det['box']
            # Crear Box NuScenes
            box = Box(center=v[:3], size=v[3:6], orientation=Quaternion(axis=[0, 0, 1], radians=v[6]))
            
            # Aplicar mismas transformaciones
            self._transform_box(box, calib)
            
            if box_in_image(box, intrinsic, (img.shape[1], img.shape[0]), vis_level=BoxVisibility.ANY):
                c = self.colors.get(det['label'], (255, 0, 255))
                # box.render usa matplotlib internamente o cv2 dependiendo de la versión
                # Usaremos render_cv2 personalizado para control total
                box.render(img, view=intrinsic, normalize=True, colors=(c, c, c), linewidth=2)

        cv2.putText(img, title, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        return img

    def _transform_pc(self, pc, calib):
        """Aplica la cadena de transformación al PointCloud in-place."""
        pc.rotate(Quaternion(calib['cs_lidar']['rotation']).rotation_matrix)
        pc.translate(np.array(calib['cs_lidar']['translation']))
        pc.rotate(Quaternion(calib['pose_lidar']['rotation']).rotation_matrix)
        pc.translate(np.array(calib['pose_lidar']['translation']))
        pc.translate(-np.array(calib['pose_cam']['translation']))
        pc.rotate(Quaternion(calib['pose_cam']['rotation']).rotation_matrix.T)
        pc.translate(-np.array(calib['cs_cam']['translation']))
        pc.rotate(Quaternion(calib['cs_cam']['rotation']).rotation_matrix.T)

    def _transform_box(self, box, calib):
        """Aplica la cadena de transformación a la Box in-place."""
        box.rotate(Quaternion(calib['cs_lidar']['rotation']))
        box.translate(np.array(calib['cs_lidar']['translation']))
        box.rotate(Quaternion(calib['pose_lidar']['rotation']))
        box.translate(np.array(calib['pose_lidar']['translation']))
        box.translate(-np.array(calib['pose_cam']['translation']))
        box.rotate(Quaternion(calib['pose_cam']['rotation']).inverse)
        box.translate(-np.array(calib['cs_cam']['translation']))
        box.rotate(Quaternion(calib['cs_cam']['rotation']).inverse)


# ==========================================
# CLASE 2: INFERENCIA (CEREBRO)
# ==========================================
class ModelInferencer:
    """Maneja la carga de modelos y ejecución."""
    
    def __init__(self, base_dir, ckpt_dir):
        self.base_dir = base_dir
        self.ckpt_dir = ckpt_dir
        self.models = {}

    def load_pointpillars(self):
        print("🏗️ Cargando PointPillars...")
        self.models['pp'] = PointPillarsDetector(
            config_path=str(self.base_dir / "configs/pointpillars/pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d.py"),
            checkpoint_path=str(self.ckpt_dir / "pointpillars_nus.pth")
        )

    def load_centerpoint(self):
        print("🎯 Cargando CenterPoint...")
        self.models['cp'] = CenterPointDetector(
            config_path=str(self.base_dir / "configs/centerpoint/centerpoint_voxel0075_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py"),
            checkpoint_path=str(self.ckpt_dir / "centerpoint_nus.pth")
        )

    def predict(self, model_key, lidar_path):
        if model_key not in self.models:
            raise ValueError(f"Modelo {model_key} no cargado.")
        
        # Ejecutar detección
        raw_dets = self.models[model_key].detect(lidar_path)
        
        # Sanitizar salida para JSON
        clean_dets = []
        for d in raw_dets:
            clean_dets.append({
                "box": d['box_3d'], # Asumimos que ya es lista por tu main.py
                "score": float(d['score']),
                "label": int(d['label'])
            })
        return clean_dets


# ==========================================
# CLASE 3: ORQUESTADOR (CONTROLLER)
# ==========================================
class DemoOrchestrator:
    def __init__(self):
        self.base_dir = BASE_DIR
        self.output_dir = self.base_dir / "runs" / "video_demo_modular"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_root = 'Fusion/data/sets/nuscenes'
        
        print("🌍 Inicializando NuScenes...")
        self.nusc = NuScenes(version='v1.0-mini', dataroot=self.data_root, verbose=False)
        
        # Instanciar componentes
        self.inferencer = ModelInferencer(self.base_dir, self.base_dir / "checkpoints")
        self.visualizer = LidarVisualizer()

    def get_calib(self, sample_token):
        """Helper para extraer toda la calibración de un token."""
        sample = self.nusc.get('sample', sample_token)
        sd_lidar = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        sd_cam = self.nusc.get('sample_data', sample['data']['CAM_FRONT'])
        
        return {
            'lidar_path': str(self.nusc.get_sample_data_path(sample['data']['LIDAR_TOP'])),
            'cam_path': str(self.nusc.get_sample_data_path(sample['data']['CAM_FRONT'])),
            'cs_lidar': self.nusc.get('calibrated_sensor', sd_lidar['calibrated_sensor_token']),
            'pose_lidar': self.nusc.get('ego_pose', sd_lidar['ego_pose_token']),
            'cs_cam': self.nusc.get('calibrated_sensor', sd_cam['calibrated_sensor_token']),
            'pose_cam': self.nusc.get('ego_pose', sd_cam['ego_pose_token'])
        }

    def run_pipeline(self, num_frames=20):
        # 1. Cargar Modelos
        self.inferencer.load_pointpillars()
        self.inferencer.load_centerpoint()
        
        scene = self.nusc.scene[0]
        token = scene['first_sample_token']
        
        # Estructura para guardar resultados en disco
        history = []

        print(f"\n⚡ Procesando {num_frames} frames...")
        for i in range(num_frames):
            print(f"   Frame {i+1}/{num_frames}...")
            
            calib = self.get_calib(token)
            
            # A. Inferencia
            dets_pp = self.inferencer.predict('pp', calib['lidar_path'])
            dets_cp = self.inferencer.predict('cp', calib['lidar_path'])
            
            # Guardar datos para depuración
            frame_data = {
                "token": token,
                "calib": calib, # Pasamos paths y datos para el paso de video
                "pp": dets_pp,
                "cp": dets_cp
            }
            history.append(frame_data)
            
            if self.nusc.get('sample', token)['next']:
                token = self.nusc.get('sample', token)['next']
            else:
                break

        # 2. Generar Videos (Separamos la lógica de generación)
        self.create_videos(history)

    def create_videos(self, history):
        print("\n🎬 Renderizando videos...")
        
        # Definir Writers
        w_bev_pp = cv2.VideoWriter(str(self.output_dir / "pp_bev.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), 5.0, (800, 800))
        w_cam_pp = cv2.VideoWriter(str(self.output_dir / "pp_cam.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), 5.0, (1600, 900))
        
        w_bev_cp = cv2.VideoWriter(str(self.output_dir / "cp_bev.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), 5.0, (800, 800))
        w_cam_cp = cv2.VideoWriter(str(self.output_dir / "cp_cam.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), 5.0, (1600, 900))

        for frame in history:
            calib = frame['calib']
            
            # Render PointPillars
            img_bev = self.visualizer.render_bev(calib['lidar_path'], frame['pp'], "PP BEV")
            img_cam = self.visualizer.render_camera_projection(calib['cam_path'], calib['lidar_path'], frame['pp'], calib, "PP CAM")
            w_bev_pp.write(img_bev)
            w_cam_pp.write(img_cam)
            
            # Render CenterPoint
            img_bev = self.visualizer.render_bev(calib['lidar_path'], frame['cp'], "CP BEV")
            img_cam = self.visualizer.render_camera_projection(calib['cam_path'], calib['lidar_path'], frame['cp'], calib, "CP CAM")
            w_bev_cp.write(img_bev)
            w_cam_cp.write(img_cam)

        w_bev_pp.release(); w_cam_pp.release()
        w_bev_cp.release(); w_cam_cp.release()
        
        print(f"✅ Videos individuales guardados en {self.output_dir}")
        self.combine_videos()

    def combine_videos(self):
        print("🔗 Combinando videos finales...")
        # Lógica simple de combinación usando OpenCV
        def stack_vids(n1, n2, out_n, w, h):
            p1 = str(self.output_dir / n1)
            p2 = str(self.output_dir / n2)
            out_p = str(self.output_dir / out_n)
            
            c1 = cv2.VideoCapture(p1)
            c2 = cv2.VideoCapture(p2)
            out = cv2.VideoWriter(out_p, cv2.VideoWriter_fourcc(*'mp4v'), 5.0, (w*2, h))
            
            while True:
                r1, f1 = c1.read()
                r2, f2 = c2.read()
                if not r1 or not r2: break
                out.write(np.hstack((f1, f2)))
            c1.release(); c2.release(); out.release()
            
        stack_vids("pp_bev.mp4", "cp_bev.mp4", "COMPARE_BEV.mp4", 800, 800)
        stack_vids("pp_cam.mp4", "cp_cam.mp4", "COMPARE_CAM.mp4", 1600, 900)
        print("🎉 ¡Proceso finalizado!")


if __name__ == "__main__":
    app = DemoOrchestrator()
    app.run_pipeline(num_frames=20)