import sys
from unittest.mock import MagicMock
from pathlib import Path
from nuscenes.nuscenes import NuScenes

def setup_mocks():
    MOCK_MODULES = [
        'lyft_dataset_sdk', 'lyft_dataset_sdk.lyftdataset', 'lyft_dataset_sdk.utils',
        'lyft_dataset_sdk.utils.data_classes', 'lyft_dataset_sdk.eval',
        'lyft_dataset_sdk.eval.detection', 'lyft_dataset_sdk.eval.detection.mAP_evaluation'
    ]
    for mod_name in MOCK_MODULES:
        sys.modules[mod_name] = MagicMock()

class DataLoader:
    def __init__(self, root='Fusion/data/sets/nuscenes', version='v1.0-mini'):
        print("🌍 Inicializando NuScenes...")
        self.nusc = NuScenes(version=version, dataroot=root, verbose=False)

    def get_sample_data(self, sample_token):
        sample = self.nusc.get('sample', sample_token)
        
        sd_lidar = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        sd_cam = self.nusc.get('sample_data', sample['data']['CAM_FRONT'])
        
        cs_cam = self.nusc.get('calibrated_sensor', sd_cam['calibrated_sensor_token'])
        
        return {
            'lidar_path': str(self.nusc.get_sample_data_path(sample['data']['LIDAR_TOP'])),
            'cam_path': str(self.nusc.get_sample_data_path(sample['data']['CAM_FRONT'])),
            'cs_lidar': self.nusc.get('calibrated_sensor', sd_lidar['calibrated_sensor_token']),
            'pose_lidar': self.nusc.get('ego_pose', sd_lidar['ego_pose_token']),
            'cs_cam': cs_cam,
            'pose_cam': self.nusc.get('ego_pose', sd_cam['ego_pose_token']),
            'intrinsic': cs_cam['camera_intrinsic'],
            # --- NUEVO: Extraer coeficientes de distorsión ---
            'distortion': cs_cam.get('camera_distortion', [0,0,0,0,0]) 
        }