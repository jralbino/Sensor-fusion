import argparse
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Import the consolidated path manager
from config.utils.path_manager import path_manager

# --- 2. MOCKING (Lyft SDK) ---
try:
    from Lidar.src.lidar_utils import setup_mocks # Use absolute import
    setup_mocks()
except ImportError:
    MOCK_MODULES = [
        'lyft_dataset_sdk', 'lyft_dataset_sdk.lyftdataset', 'lyft_dataset_sdk.utils',
        'lyft_dataset_sdk.utils.data_classes', 'lyft_dataset_sdk.eval',
        'lyft_dataset_sdk.eval.detection', 'lyft_dataset_sdk.eval.detection.mAP_evaluation'
    ]
    for mod_name in MOCK_MODULES:
        sys.modules[mod_name] = MagicMock()

from mmengine.config import Config
from mmengine.runner import Runner
from mmdet3d.utils import register_all_modules

def parse_args():
    parser = argparse.ArgumentParser(description='Train CenterPoint')
    default_config = str(path_manager.get_model_detail("centerpoint_cfg")) # Use path_manager
    parser.add_argument('--config', default=default_config, help='Path to config file')
    parser.add_argument('--work-dir', help='Output directory')
    return parser.parse_args()

def remove_db_sampler(pipeline):
    """Removes 'ObjectSample' from the pipeline."""
    if not isinstance(pipeline, list): return pipeline
    new_pipeline = []
    for transform in pipeline:
        if transform['type'] == 'ObjectSample':
            pass 
        else:
            new_pipeline.append(transform)
    return new_pipeline

def update_dataset_config(dataset_cfg, data_root, ann_file, use_db_sampler=True):
    """Updates dataloaders (image/lidar paths)."""
    if 'dataset' in dataset_cfg and isinstance(dataset_cfg['dataset'], dict):
        update_dataset_config(dataset_cfg['dataset'], data_root, ann_file, use_db_sampler)
    else:
        dataset_cfg['data_root'] = data_root
        dataset_cfg['ann_file'] = ann_file
        dataset_cfg['data_prefix'] = dict(pts='', img='', sweeps='')
        
        if 'pipeline' in dataset_cfg:
            if not use_db_sampler:
                dataset_cfg['pipeline'] = remove_db_sampler(dataset_cfg['pipeline'])
            else:
                for t in dataset_cfg['pipeline']:
                    if t['type'] == 'ObjectSample' and 'db_sampler' in t:
                        t['db_sampler']['data_root'] = data_root
                        t['db_sampler']['info_path'] = str(path_manager.get_data_detail("nuscenes_dbinfos_train_pkl")) # Use path_manager

def strip_cbgs_wrapper(dataloader_cfg):
    """Removes CBGSDataset in Mini mode."""
    dataset = dataloader_cfg.get('dataset', {})
    if dataset.get('type') == 'CBGSDataset':
        dataloader_cfg['dataset'] = dataset['dataset']

def update_evaluator_config(evaluator_cfg, data_root, ann_file):
    """
    NEW CORRECTION: Updates the .pkl path in the Evaluator (Metric).
    This avoids FileNotFoundError during validation.
    """
    if evaluator_cfg is None:
        return

    # The evaluator can be a dict or a list of dicts
    evaluators = [evaluator_cfg] if isinstance(evaluator_cfg, dict) else evaluator_cfg
    
    for eval_item in evaluators:
        # NuScenesMetric uses 'ann_file' and 'data_root'
        if 'ann_file' in eval_item:
            eval_item['ann_file'] = ann_file
        if 'data_root' in eval_item:
            eval_item['data_root'] = data_root

def main():
    args = parse_args()
    register_all_modules(init_default_scope=True)

    if not os.path.exists(args.config):
        print(f"❌ Config not found: {args.config}")
        return

    cfg = Config.fromfile(args.config)
    data_root = str(path_manager.get_data_detail("nuscenes_base")) # Use path_manager
    
    # Files
    db_info_path = str(path_manager.get_data_detail("nuscenes_dbinfos_train_pkl")) # Use path_manager
    has_db_info = os.path.exists(db_info_path)
    
    if not has_db_info:
        print("\n⚠️  Info: Cut&Paste Augmentation disabled.")

    # --- APPLY CORRECTIONS ---
    cfg.data_root = data_root
    
    # 1. Dataloaders (Data reading)
    strip_cbgs_wrapper(cfg.train_dataloader)
    update_dataset_config(cfg.train_dataloader.dataset, data_root, str(path_manager.get_data_detail("nuscenes_infos_train_pkl")), use_db_sampler=has_db_info) # Use path_manager
    update_dataset_config(cfg.val_dataloader.dataset, data_root, str(path_manager.get_data_detail("nuscenes_infos_val_pkl")), use_db_sampler=False) # Use path_manager
    update_dataset_config(cfg.test_dataloader.dataset, data_root, str(path_manager.get_data_detail("nuscenes_infos_val_pkl")), use_db_sampler=False) # Use path_manager
    
    # 2. Evaluators (Metric calculation) - HERE WAS THE BUG
    update_evaluator_config(cfg.val_evaluator, data_root, str(path_manager.get_data_detail("nuscenes_infos_val_pkl"))) # Use path_manager
    update_evaluator_config(cfg.test_evaluator, data_root, str(path_manager.get_data_detail("nuscenes_infos_val_pkl"))) # Use path_manager

    # Output
    if args.work_dir:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = str(path_manager.get("lidar_centerpoint_train_output")) # Use path_manager

    # Settings
    cfg.train_dataloader.batch_size = 2
    cfg.train_dataloader.num_workers = 2
    cfg.train_cfg.val_interval = 1 
    cfg.train_cfg.max_epochs = 1

    print(f"\n🚀 Starting CenterPoint training...")
    print(f"   Data: {data_root}")
    
    runner = Runner.from_cfg(cfg)
    
    # Resume if it crashed (Optional, mmdet3d usually handles it if you set resume=True)
    runner.train()

if __name__ == '__main__':
    main()