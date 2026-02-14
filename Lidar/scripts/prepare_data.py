#!/usr/bin/env python3
"""
Prepare NuScenes dataset for training.

Creates info files containing:
- Point cloud paths
- Ground truth annotations
- Calibration data
"""

import argparse
import pickle
from pathlib import Path
import numpy as np
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from pyquaternion import Quaternion


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare NuScenes data')
    parser.add_argument('--data-root', required=True, help='NuScenes root directory')
    parser.add_argument('--version', default='v1.0-mini', help='NuScenes version')
    parser.add_argument('--output-dir', default=None, help='Output directory (default: data_root)')
    return parser.parse_args()


def get_box_from_annotation(nusc, ann_token, cs_record, pose_record):
    """
    Get box parameters from annotation in LiDAR coordinates.
    
    Returns:
        box: [7] array (x, y, z, l, w, h, yaw) in LiDAR frame
    """
    ann = nusc.get('sample_annotation', ann_token)
    
    # Get box in global frame
    box_global = nusc.get_box(ann_token)
    
    # Transform to ego frame
    box_global.translate(-np.array(pose_record['translation']))
    box_global.rotate(Quaternion(pose_record['rotation']).inverse)
    
    # Transform to LiDAR frame
    box_global.translate(-np.array(cs_record['translation']))
    box_global.rotate(Quaternion(cs_record['rotation']).inverse)
    
    # Get box parameters
    center = box_global.center
    size = box_global.wlh[[1, 0, 2]]  # Convert to l, w, h
    
    # Get yaw angle
    yaw = box_global.orientation.yaw_pitch_roll[0]
    
    # Combine
    box = np.array([
        center[0], center[1], center[2],
        size[0], size[1], size[2],
        yaw
    ], dtype=np.float32)
    
    return box


def create_nuscenes_infos(
    nusc: NuScenes,
    scenes: list,
    output_path: Path
):
    """
    Create info file for a split.
    
    Args:
        nusc: NuScenes instance
        scenes: List of scene names
        output_path: Where to save info file
    """
    print(f"Creating info file for {len(scenes)} scenes...")
    
    # Class name mapping
    CLASS_NAMES = [
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
        'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    ]
    
    name_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    
    infos = []
    
    for scene in tqdm(nusc.scene, desc="Processing scenes"):
        if scene['name'] not in scenes:
            continue
        
        # Get first sample in scene
        sample_token = scene['first_sample_token']
        
        while sample_token:
            sample = nusc.get('sample', sample_token)
            
            # Get LiDAR data
            lidar_token = sample['data']['LIDAR_TOP']
            sd_lidar = nusc.get('sample_data', lidar_token)
            cs_record = nusc.get('calibrated_sensor', sd_lidar['calibrated_sensor_token'])
            pose_record = nusc.get('ego_pose', sd_lidar['ego_pose_token'])
            
            # LiDAR path (relative to data root)
            lidar_path = Path(sd_lidar['filename'])
            
            # Process annotations
            instances = []
            
            for ann_token in sample['anns']:
                ann = nusc.get('sample_annotation', ann_token)
                
                # Get category
                category = ann['category_name']
                
                # Map to our classes
                label = -1
                for class_name in CLASS_NAMES:
                    if class_name in category:
                        label = name_to_idx[class_name]
                        break
                
                if label == -1:
                    continue  # Skip unknown classes
                
                # Get box in LiDAR frame
                try:
                    box = get_box_from_annotation(
                        nusc, ann_token, cs_record, pose_record
                    )
                except:
                    continue  # Skip if transformation fails
                
                # Create instance
                instance = {
                    'bbox_3d': box.tolist(),
                    'bbox_label_3d': label,
                    'bbox_3d_isvalid': True,
                    'num_lidar_pts': ann.get('num_lidar_pts', 0),
                    'num_radar_pts': ann.get('num_radar_pts', 0)
                }
                
                instances.append(instance)
            
            # Create info dict
            info = {
                'sample_idx': sample['token'],
                'timestamp': sample['timestamp'],
                'lidar_points': {
                    'lidar_path': str(lidar_path),
                    'num_pts_feats': 5
                },
                'instances': instances
            }
            
            infos.append(info)
            
            # Next sample
            sample_token = sample['next']
    
    # Create final data structure
    data = {
        'metainfo': {
            'dataset': 'nuscenes',
            'version': nusc.version,
            'categories': {name: idx for idx, name in enumerate(CLASS_NAMES)}
        },
        'data_list': infos
    }
    
    # Save
    print(f"Saving {len(infos)} samples to {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"✅ Created {output_path}")


def main():
    args = parse_args()
    
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir) if args.output_dir else data_root
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading NuScenes {args.version} from {data_root}")
    
    # Load NuScenes
    nusc = NuScenes(
        version=args.version,
        dataroot=str(data_root),
        verbose=False
    )
    
    # Get splits
    splits = create_splits_scenes()
    
    if args.version == 'v1.0-mini':
        train_scenes = splits['mini_train']
        val_scenes = splits['mini_val']
    elif args.version == 'v1.0-trainval':
        train_scenes = splits['train']
        val_scenes = splits['val']
    else:
        print(f"Unknown version: {args.version}")
        return
    
    # Create train info
    train_path = output_dir / 'nuscenes_infos_train.pkl'
    create_nuscenes_infos(nusc, train_scenes, train_path)
    
    # Create val info
    val_path = output_dir / 'nuscenes_infos_val.pkl'
    create_nuscenes_infos(nusc, val_scenes, val_path)
    
    print("\n✅ Data preparation complete!")
    print(f"   Train: {train_path}")
    print(f"   Val: {val_path}")


if __name__ == '__main__':
    main()
