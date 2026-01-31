import pickle
import numpy as np
import os
from pathlib import Path
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from pyquaternion import Quaternion

# Import the consolidated path manager
from config.utils.path_manager import path_manager

# Standard NuScenes classes
CLASSES = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 
    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]

def get_matrix(translation, rotation):
    """
    Converts translation and rotation to a 4x4 matrix.
    IMPORTANT: Returns a LIST, not a numpy array, to avoid shape errors in validation.
    """
    mat = np.eye(4)
    mat[:3, :3] = Quaternion(rotation).rotation_matrix
    mat[:3, 3] = translation
    return mat.tolist() # <--- SOLUTION TO MATRIX ERROR

def get_box_geometry(box):
    """Returns geometry [x, y, z, l, w, h, yaw] (7 dims)"""
    v = np.dot(box.rotation_matrix, np.array([1, 0, 0]))
    yaw = np.arctan2(v[1], v[0])
    l, w, h = box.wlh[1], box.wlh[0], box.wlh[2]
    return [box.center[0], box.center[1], box.center[2], l, w, h, yaw]

def get_velocity(box):
    """Returns [vx, vy]"""
    if not np.isnan(box.velocity).any():
        return [box.velocity[0], box.velocity[1]]
    return [0.0, 0.0]

def generate_infos(nusc, scenes, out_path, version, data_root):
    data_list = []
    print(f"🔄 Processing {len(scenes)} scenes for {out_path.name}...")
    
    for scene in nusc.scene:
        if scene['name'] not in scenes:
            continue
            
        token = scene['first_sample_token']
        while token:
            sample = nusc.get('sample', token)
            lidar_token = sample['data']['LIDAR_TOP']
            lidar_path_abs = nusc.get_sample_data_path(lidar_token)
            
            try:
                lidar_path_rel = Path(lidar_path_abs).relative_to(data_root)
            except ValueError:
                lidar_path_rel = Path(lidar_path_abs).name 

            # Basic Info
            lidar_rec = nusc.get('sample_data', lidar_token)
            cs_record = nusc.get('calibrated_sensor', lidar_rec['calibrated_sensor_token'])
            pose_record = nusc.get('ego_pose', lidar_rec['ego_pose_token'])

            # 4x4 Matrices (now as lists)
            lidar2ego_mat = get_matrix(cs_record['translation'], cs_record['rotation'])
            ego2global_mat = get_matrix(pose_record['translation'], pose_record['rotation'])

            info = {
                'sample_idx': sample['token'],
                'token': sample['token'],
                'timestamp': sample['timestamp'],
                'lidar_points': {
                    'lidar_path': str(lidar_path_rel),
                    'num_pts_feats': 5,
                    'lidar2ego': lidar2ego_mat, 
                    'ego2global': ego2global_mat
                },
                # Keep legacy for compatibility
                'lidar2ego': {
                    'translation': cs_record['translation'],
                    'rotation': cs_record['rotation'],
                },
                'ego2global': {
                    'translation': pose_record['translation'],
                    'rotation': pose_record['rotation'],
                },
                'images': {},
                'instances': [] 
            }
            
            # Process Annotations
            for ann_token in sample['anns']:
                ann = nusc.get('sample_annotation', ann_token)
                cat_name = ann['category_name']
                
                mapped_label = -1
                for idx, cls_name in enumerate(CLASSES):
                    if cls_name in cat_name:
                        mapped_label = idx
                        break
                
                if mapped_label == -1: continue

                # Transformations
                box = nusc.get_box(ann['token'])
                box.translate(-np.array(pose_record['translation']))
                box.rotate(Quaternion(pose_record['rotation']).inverse)
                box.translate(-np.array(cs_record['translation']))
                box.rotate(Quaternion(cs_record['rotation']).inverse)
                
                box_geo = get_box_geometry(box) 
                velocity = get_velocity(box)
                
                instance = {
                    'bbox_3d': box_geo,   
                    'bbox_label_3d': mapped_label,
                    'velocity': velocity, 
                    'num_lidar_pts': ann['num_lidar_pts'],
                    'num_radar_pts': ann['num_radar_pts'],
                    'bbox_3d_isvalid': True, 
                    'ignore_flag': 0
                }
                info['instances'].append(instance)

            data_list.append(info)
            token = sample['next']
    
    # Final Structure
    final_dict = {
        'metainfo': {
            'categories': {k: i for i, k in enumerate(CLASSES)},
            'dataset': 'nuscenes',
            'version': version
        },
        'data_list': data_list
    }

    with open(out_path, 'wb') as f:
        pickle.dump(final_dict, f)
    print(f"✅ Saved: {out_path} ({len(data_list)} samples)")

def main():
    data_root = path_manager.get_data_detail("nuscenes_base")
    
    if not data_root.exists():
        print(f"❌ Data root not found: {data_root}")
        return
        
    version = path_manager.get_user_setting("nuscenes_version")
    print(f"📂 Generating data (List-format Matrices) for: {version}")
    
    try:
        nusc = NuScenes(version=version, dataroot=str(data_root), verbose=False)
    except Exception as e:
        print(f"❌ Error loading NuScenes: {e}")
        return

    splits = create_splits_scenes()
    train_scenes = splits['mini_train'] if version == 'v1.0-mini' else splits['train']
    val_scenes = splits['mini_val'] if version == 'v1.0-mini' else splits['val']
    
    generate_infos(nusc, train_scenes, path_manager.get_data_detail("nuscenes_infos_train_pkl"), version, data_root)
    generate_infos(nusc, val_scenes, path_manager.get_data_detail("nuscenes_infos_val_pkl"), version, data_root)
    
    print("\n🎉 Data regenerated successfully!")

if __name__ == '__main__':
    main()