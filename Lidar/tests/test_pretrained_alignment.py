#!/usr/bin/env python3
"""
Test Pretrained Alignment — MMDet3D models on NuScenes mini val.

Downloads official MMDet3D checkpoints, loads them via our wrapper models,
runs inference on the NuScenes mini validation set, and checks that
detections align with GT using AP and NDS metrics.

Supports both PointPillars and CenterPoint pretrained models.

Usage:
    cd Lidar
    # Test PointPillars only
    venv/bin/python tests/test_pretrained_alignment.py \
        --data-root ../Fusion/data/sets/nuscenes --model pointpillars

    # Test CenterPoint only
    venv/bin/python tests/test_pretrained_alignment.py \
        --data-root ../Fusion/data/sets/nuscenes --model centerpoint

    # Test both
    venv/bin/python tests/test_pretrained_alignment.py \
        --data-root ../Fusion/data/sets/nuscenes --model all
"""

import argparse
import sys
import os
import urllib.request
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import logging

# Ensure Lidar/ is on the path
LIDAR_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LIDAR_ROOT))

from src.data.datasets import voxelize_points
from evaluate import (
    evaluate, compute_nds_metrics, compute_nds,
    compute_iou_bev_batch, CLASS_NAMES,
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Checkpoint configs
CHECKPOINT_DIR = LIDAR_ROOT / "models"

MODELS = {
    'pointpillars': {
        'url': (
            "https://download.openmmlab.com/mmdetection3d/v1.0.0_models/pointpillars/"
            "hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d/"
            "hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d_20210826_225857-f19d00a3.pth"
        ),
        'filename': 'hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d_20210826_225857-f19d00a3.pth',
        'voxel_size': np.array([0.25, 0.25, 8.0]),
        'point_cloud_range': np.array([-50, -50, -5, 50, 50, 3]),
        'max_points_per_voxel': 64,
        'max_voxels': 40000,
        'display_name': 'MMDet3D PointPillars',
        'num_point_features': 4,
        'sweeps_num': 9,
    },
    'second': {
        'url': (
            "https://download.openmmlab.com/mmdetection3d/v1.0.0_models/pointpillars/"
            "hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d/"
            "hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d_20210826_225857-f19d00a3.pth"
        ),
        'filename': 'hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d_20210826_225857-f19d00a3.pth',
        'voxel_size': np.array([0.25, 0.25, 8.0]),
        'point_cloud_range': np.array([-50, -50, -5, 50, 50, 3]),
        'max_points_per_voxel': 64,
        'max_voxels': 40000,
        'display_name': 'MMDet3D SECOND (PointPillars backbone)',
        'num_point_features': 4,
        'sweeps_num': 9,
    },
    'centerpoint': {
        'url': (
            "https://download.openmmlab.com/mmdetection3d/v1.0.0_models/centerpoint/"
            "centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus/"
            "centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220811_031844-191a3822.pth"
        ),
        'filename': 'centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220811_031844-191a3822.pth',
        'voxel_size': np.array([0.2, 0.2, 8.0]),
        'point_cloud_range': np.array([-51.2, -51.2, -5, 51.2, 51.2, 3]),
        'max_points_per_voxel': 20,
        'max_voxels': 40000,
        'display_name': 'MMDet3D CenterPoint (Pillar)',
        'num_point_features': 5,
        'sweeps_num': 9,
    },
}


def download_checkpoint(model_key: str):
    """Download checkpoint if not already present."""
    cfg = MODELS[model_key]
    ckpt_path = CHECKPOINT_DIR / cfg['filename']

    if ckpt_path.exists():
        logger.info(f"Checkpoint already exists: {ckpt_path.name}")
        return ckpt_path

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading {cfg['display_name']} checkpoint...")
    logger.info(f"  URL: {cfg['url']}")

    def _progress(count, block_size, total_size):
        pct = count * block_size * 100 / total_size
        sys.stdout.write(f"\r  {pct:.1f}%")
        sys.stdout.flush()

    urllib.request.urlretrieve(cfg['url'], str(ckpt_path), _progress)
    print()
    logger.info(f"Saved to {ckpt_path}")
    return ckpt_path


def create_model(model_key: str, ckpt_path: Path):
    """Create and load the appropriate model."""
    if model_key in ('pointpillars', 'second'):
        # Both use the same MMDet3D architecture (SECOND backbone + SECONDFPN)
        from src.detectors.mmdet3d_second import MMDet3DSECOND
        model = MMDet3DSECOND(num_classes=10)
    elif model_key == 'centerpoint':
        from src.detectors.mmdet3d_centerpoint import MMDet3DCenterPoint
        model = MMDet3DCenterPoint()
    else:
        raise ValueError(f"Unknown model: {model_key}")

    model.load_mmdet3d_checkpoint(str(ckpt_path))
    return model


class MMDet3DValDataset(torch.utils.data.Dataset):
    """Minimal val dataset with configurable grid settings."""

    def __init__(self, data_root: str, info_path: str, voxel_size, point_cloud_range,
                 max_points_per_voxel=64, max_voxels=40000, num_point_features=4,
                 sweeps_num=0):
        import pickle
        self.data_root = Path(data_root)
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.max_points_per_voxel = max_points_per_voxel
        self.max_voxels = max_voxels
        self.num_point_features = num_point_features
        self.sweeps_num = sweeps_num
        self.nusc = None

        with open(info_path, 'rb') as f:
            data = pickle.load(f)

        if isinstance(data, dict):
            self.infos = data.get('data_list', data.get('infos', []))
        else:
            self.infos = data

        # Initialize NuScenes API if multi-sweep loading is needed
        if self.sweeps_num > 0:
            from nuscenes.nuscenes import NuScenes
            version = 'v1.0-mini' if 'mini' in str(data_root) or len(self.infos) < 200 else 'v1.0-trainval'
            self.nusc = NuScenes(version=version, dataroot=str(data_root), verbose=False)
            logger.info(f"Multi-sweep loading enabled: {sweeps_num} sweeps")

        self.class_names = CLASS_NAMES
        logger.info(f"Loaded {len(self.infos)} val samples")

    def __len__(self):
        return len(self.infos)

    def __getitem__(self, idx):
        info = self.infos[idx]

        # Load points — multi-sweep or single
        if self.sweeps_num > 0 and self.nusc is not None:
            sample_token = info['sample_idx']
            from visualize_3d import load_multi_sweep_points
            points = load_multi_sweep_points(
                self.nusc, sample_token, self.data_root,
                sweeps_num=self.sweeps_num,
            )
            points = points[:, :self.num_point_features]
        else:
            if 'lidar_points' in info:
                lidar_path = info['lidar_points']['lidar_path']
            elif 'lidar_path' in info:
                lidar_path = info['lidar_path']
            else:
                raise KeyError("Cannot find lidar path")

            full_path = self.data_root / lidar_path
            if not full_path.exists():
                full_path = Path(lidar_path)

            points = np.fromfile(str(full_path), dtype=np.float32).reshape(-1, 5)[:, :self.num_point_features]

        # Load GT
        gt_boxes, gt_labels = self._load_annotations(info)

        # Voxelize
        voxels, coords, num_points = voxelize_points(
            points,
            self.voxel_size,
            self.point_cloud_range,
            self.max_points_per_voxel,
            self.max_voxels,
        )

        return {
            'voxels': torch.from_numpy(voxels).float(),
            'voxel_coords': torch.from_numpy(coords).int(),
            'voxel_num_points': torch.from_numpy(num_points).int(),
            'gt_boxes': torch.from_numpy(gt_boxes).float(),
            'gt_labels': torch.from_numpy(gt_labels).long(),
        }

    def _load_annotations(self, info):
        gt_boxes, gt_labels = [], []

        if 'instances' in info:
            for inst in info['instances']:
                if not inst.get('bbox_3d_isvalid', True):
                    continue
                box = inst['bbox_3d']
                label = inst['bbox_label_3d']
                if label < len(self.class_names):
                    gt_boxes.append(box)
                    gt_labels.append(label)
        elif 'gt_boxes' in info:
            gt_boxes = info['gt_boxes']
            gt_labels = info.get('gt_labels', np.zeros(len(gt_boxes)))

        if len(gt_boxes) == 0:
            return np.zeros((0, 7), dtype=np.float32), np.zeros(0, dtype=np.int64)

        return np.array(gt_boxes, dtype=np.float32), np.array(gt_labels, dtype=np.int64)

    @staticmethod
    def collate_fn(batch):
        voxels_list, coords_list, npts_list = [], [], []
        gt_boxes_list, gt_labels_list = [], []

        for i, sample in enumerate(batch):
            voxels_list.append(sample['voxels'])
            coords = sample['voxel_coords']
            batch_idx = torch.full((len(coords), 1), i, dtype=coords.dtype)
            coords_list.append(torch.cat([batch_idx, coords], dim=1))
            npts_list.append(sample['voxel_num_points'])
            gt_boxes_list.append(sample['gt_boxes'])
            gt_labels_list.append(sample['gt_labels'])

        return {
            'voxels': torch.cat(voxels_list, dim=0),
            'voxel_coords': torch.cat(coords_list, dim=0),
            'voxel_num_points': torch.cat(npts_list, dim=0),
            'batch_size': len(batch),
            'gt_boxes': gt_boxes_list,
            'gt_labels': gt_labels_list,
        }


def compute_gt_recall(det_results, gt_results, iou_thresh=0.25):
    """Compute % of GT boxes that have at least 1 detection with IoU > threshold."""
    total_gt = 0
    matched_gt = 0

    for det, gt in zip(det_results, gt_results):
        gt_boxes = gt['boxes']
        det_boxes = det['boxes']

        total_gt += len(gt_boxes)

        if len(gt_boxes) == 0 or len(det_boxes) == 0:
            continue

        iou = compute_iou_bev_batch(det_boxes, gt_boxes)
        max_iou_per_gt = iou.max(axis=0)
        matched_gt += (max_iou_per_gt >= iou_thresh).sum()

    return matched_gt / max(total_gt, 1), matched_gt, total_gt


def run_single_model(model_key: str, args):
    """Run evaluation for a single model. Returns True if all checks pass."""
    cfg = MODELS[model_key]
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*80}")
    print(f"  {cfg['display_name']} — Alignment Test")
    print(f"{'='*80}")

    # 1. Download checkpoint
    ckpt_path = download_checkpoint(model_key)

    # 2. Create model
    logger.info(f"Creating {cfg['display_name']} wrapper...")
    model = create_model(model_key, ckpt_path)
    model = model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params / 1e6:.2f}M")

    # 3. Load validation data
    data_root = Path(args.data_root)
    val_info = data_root / 'nuscenes_infos_val.pkl'
    if not val_info.exists():
        logger.error(f"Val info not found: {val_info}")
        logger.error("Run NuScenes info generation first.")
        return False

    sweeps_num = cfg.get('sweeps_num', 0)
    dataset = MMDet3DValDataset(
        str(data_root), str(val_info),
        voxel_size=cfg['voxel_size'],
        point_cloud_range=cfg['point_cloud_range'],
        max_points_per_voxel=cfg['max_points_per_voxel'],
        max_voxels=cfg['max_voxels'],
        num_point_features=cfg.get('num_point_features', 4),
        sweeps_num=sweeps_num,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=MMDet3DValDataset.collate_fn,
        pin_memory=True,
    )

    # 4. Run inference
    det_results = []
    gt_results = []

    logger.info(f"Running inference on {len(val_loader)} batches...")
    with torch.no_grad():
        for batch_dict in tqdm(val_loader, desc=f'{model_key} inference'):
            for key in batch_dict:
                if isinstance(batch_dict[key], torch.Tensor):
                    batch_dict[key] = batch_dict[key].to(device)

            pred_dict = model(batch_dict)
            dets = model.postprocess(
                pred_dict,
                score_thresh=args.score_thresh,
                nms_iou_thresh=args.nms_iou,
            )
            det_results.extend(dets)

            for b in range(batch_dict['batch_size']):
                gt_results.append({
                    'boxes': batch_dict['gt_boxes'][b].cpu().numpy(),
                    'labels': batch_dict['gt_labels'][b].cpu().numpy(),
                })

    # 5. Evaluate
    logger.info("Computing metrics...")

    ap_dict = evaluate(det_results, gt_results, iou_thresholds=(0.25, 0.5))
    nds_metrics = compute_nds_metrics(det_results, gt_results, iou_thresh=0.5)
    recall_025, n_matched, n_total = compute_gt_recall(det_results, gt_results, iou_thresh=0.25)

    # 6. Print results
    print(f"\n{'='*80}")
    print(f"  {cfg['display_name']} — Results")
    print(f"{'='*80}")

    iou_thresholds = [0.25, 0.5]
    header = f"{'Class':<25}" + "".join(f"{'AP@'+str(t):<12}" for t in iou_thresholds)
    print(f"\n{header}")
    print("-" * 50)

    maps = {t: [] for t in iou_thresholds}
    for cls_name in CLASS_NAMES:
        row = f"{cls_name:<25}"
        for t in iou_thresholds:
            ap = ap_dict.get((cls_name, t), 0.0)
            row += f"{ap*100:>8.2f}    "
            maps[t].append(ap)
        print(row)

    print("-" * 50)
    mAP_values = {}
    row = f"{'mAP':<25}"
    for t in iou_thresholds:
        m = np.mean(maps[t])
        mAP_values[t] = m
        row += f"{m*100:>8.2f}    "
    print(row)

    # NDS table
    print(f"\n{'Class':<25}{'ATE (m)':<12}{'ASE':<12}{'AOE (rad)':<12}{'Matches':<10}")
    print("-" * 70)
    for cls_name in CLASS_NAMES:
        m = nds_metrics[cls_name]
        ate_s = f"{m['ate']:.3f}" if not np.isnan(m['ate']) else "N/A"
        ase_s = f"{m['ase']:.3f}" if not np.isnan(m['ase']) else "N/A"
        aoe_s = f"{m['aoe']:.3f}" if not np.isnan(m['aoe']) else "N/A"
        print(f"{cls_name:<25}{ate_s:<12}{ase_s:<12}{aoe_s:<12}{m['count']:<10}")

    mean_m = nds_metrics['mean']
    ate_s = f"{mean_m['ate']:.3f}" if not np.isnan(mean_m['ate']) else "N/A"
    ase_s = f"{mean_m['ase']:.3f}" if not np.isnan(mean_m['ase']) else "N/A"
    aoe_s = f"{mean_m['aoe']:.3f}" if not np.isnan(mean_m['aoe']) else "N/A"
    print("-" * 70)
    print(f"{'Mean':<25}{ate_s:<12}{ase_s:<12}{aoe_s:<12}{mean_m['count']:<10}")

    mAP_05 = mAP_values.get(0.5, 0.0)
    nds = compute_nds(mAP_05, nds_metrics)

    # Summary metrics
    total_dets = sum(len(d['boxes']) for d in det_results)
    total_gt = sum(len(g['boxes']) for g in gt_results)

    print(f"\n{'Metric':<35}{'Value':<15}")
    print("-" * 50)
    print(f"{'GT Recall @ IoU=0.25':<35}{recall_025*100:>8.2f}%    ({n_matched}/{n_total})")
    nds_s = f"{nds*100:.2f}" if not np.isnan(nds) else "N/A"
    print(f"{'NDS (approx)':<35}{nds_s:>8s}%")
    print(f"{'Total detections':<35}{total_dets:>8d}")
    print(f"{'Total GT boxes':<35}{total_gt:>8d}")

    # 7. PASS/FAIL checks
    print(f"\n{'='*80}")
    print(f"  PASS / FAIL Checks — {cfg['display_name']}")
    print(f"{'='*80}")

    checks = []

    map025 = mAP_values.get(0.25, 0.0)
    passed = map025 > 0.10
    checks.append(passed)
    print(f"  [{'PASS' if passed else 'FAIL'}] mAP@0.25 > 10%: {map025*100:.2f}%")

    passed = mAP_05 > 0.15
    checks.append(passed)
    print(f"  [{'PASS' if passed else 'FAIL'}] mAP@0.5 > 15%: {mAP_05*100:.2f}%")

    passed = recall_025 > 0.30
    checks.append(passed)
    print(f"  [{'PASS' if passed else 'FAIL'}] GT Recall@0.25 > 30%: {recall_025*100:.2f}%")

    mean_ate = mean_m.get('ate', float('nan'))
    passed = not np.isnan(mean_ate) and mean_ate < 2.0
    checks.append(passed)
    print(f"  [{'PASS' if passed else 'FAIL'}] Mean ATE < 2.0m: {f'{mean_ate:.3f}m' if not np.isnan(mean_ate) else 'N/A'}")

    passed = total_dets > 0
    checks.append(passed)
    print(f"  [{'PASS' if passed else 'FAIL'}] Total detections > 0: {total_dets}")

    all_passed = all(checks)
    print(f"\n  {'ALL CHECKS PASSED' if all_passed else 'SOME CHECKS FAILED'} — {cfg['display_name']}")

    return all_passed


def run_test(args):
    """Main test pipeline."""
    models_to_test = list(MODELS.keys()) if args.model == 'all' else [args.model]

    results = {}
    for model_key in models_to_test:
        results[model_key] = run_single_model(model_key, args)

    # Final summary
    print(f"\n{'='*80}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*80}")
    for model_key, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {MODELS[model_key]['display_name']}")

    all_passed = all(results.values())
    print(f"\n  {'ALL MODELS PASSED' if all_passed else 'SOME MODELS FAILED'}")
    print(f"{'='*80}")

    return all_passed


def parse_args():
    parser = argparse.ArgumentParser(description='Test pretrained MMDet3D alignment')
    parser.add_argument('--data-root', required=True, help='NuScenes data root')
    parser.add_argument('--model', choices=['pointpillars', 'second', 'centerpoint', 'all'],
                        default='all', help='Which model to test (default: all)')
    parser.add_argument('--device', default='cuda:0', help='Device (default: cuda:0)')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--score-thresh', type=float, default=0.1)
    parser.add_argument('--nms-iou', type=float, default=0.3)
    parser.add_argument('--inspect', action='store_true', help='Print checkpoint keys')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    success = run_test(args)
    sys.exit(0 if success else 1)
