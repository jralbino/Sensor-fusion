#!/usr/bin/env python3
"""
Evaluate PointPillars on NuScenes validation set.

Computes per-class AP at IoU thresholds [0.25, 0.5] and reports mAP.
Also computes NDS-style metrics: ATE, ASE, AOE per class.
"""

import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CLASS_NAMES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]


def compute_ap(scores, matched, n_gt):
    """Compute Average Precision using all-point interpolation.

    Args:
        scores: (N,) detection scores (sorted descending)
        matched: (N,) bool, True if detection matched a GT
        n_gt: total number of ground-truth objects for this class

    Returns:
        AP value (float)
    """
    if n_gt == 0:
        return 0.0

    tp = np.cumsum(matched).astype(np.float64)
    fp = np.cumsum(~matched).astype(np.float64)

    recall = tp / n_gt
    precision = tp / (tp + fp)

    # Prepend sentinel values
    recall = np.concatenate([[0.0], recall, [1.0]])
    precision = np.concatenate([[1.0], precision, [0.0]])

    # Make precision monotonically decreasing
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    # Find recall change points
    idx = np.where(recall[1:] != recall[:-1])[0]
    ap = np.sum((recall[idx + 1] - recall[idx]) * precision[idx + 1])
    return float(ap)


def compute_iou_bev_batch(boxes_a, boxes_b):
    """Axis-aligned BEV IoU between two sets of boxes (numpy).

    Args:
        boxes_a: (N, 7)
        boxes_b: (M, 7)

    Returns:
        iou: (N, M)
    """
    from src.training.losses import _corners_bev, _axis_aligned_bbox
    a = torch.from_numpy(boxes_a).float()
    b = torch.from_numpy(boxes_b).float()
    aa = _axis_aligned_bbox(_corners_bev(a))
    bb = _axis_aligned_bbox(_corners_bev(b))

    x1 = torch.max(aa[:, 0:1], bb[:, 0:1].T)
    y1 = torch.max(aa[:, 1:2], bb[:, 1:2].T)
    x2 = torch.min(aa[:, 2:3], bb[:, 2:3].T)
    y2 = torch.min(aa[:, 3:4], bb[:, 3:4].T)

    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area_a = (aa[:, 2] - aa[:, 0]) * (aa[:, 3] - aa[:, 1])
    area_b = (bb[:, 2] - bb[:, 0]) * (bb[:, 3] - bb[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return (inter / union.clamp(min=1e-6)).numpy()


def evaluate(det_results, gt_results, iou_thresholds=(0.25, 0.5)):
    """Compute per-class AP at given IoU thresholds.

    Args:
        det_results: list of dicts with 'boxes', 'scores', 'labels'
        gt_results: list of dicts with 'boxes', 'labels'
        iou_thresholds: tuple of IoU thresholds

    Returns:
        dict mapping (class_name, iou_thresh) -> AP
    """
    n_classes = len(CLASS_NAMES)
    ap_dict = {}

    for iou_thresh in iou_thresholds:
        for cls_idx in range(n_classes):
            all_scores = []
            all_matched = []
            n_gt_total = 0

            for det, gt in zip(det_results, gt_results):
                # Filter detections for this class
                det_mask = det['labels'] == cls_idx
                det_boxes = det['boxes'][det_mask]
                det_scores = det['scores'][det_mask]

                # Filter GT for this class
                gt_mask = gt['labels'] == cls_idx
                gt_boxes = gt['boxes'][gt_mask]

                n_gt_total += len(gt_boxes)

                if len(det_boxes) == 0:
                    continue

                # Sort by score descending
                order = np.argsort(det_scores)[::-1]
                det_boxes = det_boxes[order]
                det_scores = det_scores[order]

                matched = np.zeros(len(det_boxes), dtype=bool)

                if len(gt_boxes) > 0:
                    iou_matrix = compute_iou_bev_batch(det_boxes, gt_boxes)
                    gt_used = np.zeros(len(gt_boxes), dtype=bool)

                    for d_idx in range(len(det_boxes)):
                        best_gt = iou_matrix[d_idx].argmax()
                        if iou_matrix[d_idx, best_gt] >= iou_thresh and not gt_used[best_gt]:
                            matched[d_idx] = True
                            gt_used[best_gt] = True

                all_scores.append(det_scores)
                all_matched.append(matched)

            if n_gt_total == 0:
                ap_dict[(CLASS_NAMES[cls_idx], iou_thresh)] = 0.0
                continue

            all_scores = np.concatenate(all_scores) if all_scores else np.array([])
            all_matched = np.concatenate(all_matched) if all_matched else np.array([], dtype=bool)

            # Global sort by score
            order = np.argsort(all_scores)[::-1]
            all_matched = all_matched[order]
            all_scores = all_scores[order]

            ap = compute_ap(all_scores, all_matched, n_gt_total)
            ap_dict[(CLASS_NAMES[cls_idx], iou_thresh)] = ap

    return ap_dict


def match_detections(det_results, gt_results, iou_thresh=0.5):
    """Match detections to GT boxes per class using BEV IoU.

    Returns:
        matches: list of (pred_box, gt_box, cls_idx) tuples for TP matches
    """
    matches = []

    for det, gt in zip(det_results, gt_results):
        for cls_idx in range(len(CLASS_NAMES)):
            det_mask = det['labels'] == cls_idx
            gt_mask = gt['labels'] == cls_idx

            det_boxes = det['boxes'][det_mask]
            det_scores = det['scores'][det_mask]
            gt_boxes = gt['boxes'][gt_mask]

            if len(det_boxes) == 0 or len(gt_boxes) == 0:
                continue

            # Sort by score
            order = np.argsort(det_scores)[::-1]
            det_boxes = det_boxes[order]

            iou_matrix = compute_iou_bev_batch(det_boxes, gt_boxes)
            gt_used = np.zeros(len(gt_boxes), dtype=bool)

            for d_idx in range(len(det_boxes)):
                best_gt = iou_matrix[d_idx].argmax()
                if iou_matrix[d_idx, best_gt] >= iou_thresh and not gt_used[best_gt]:
                    matches.append((det_boxes[d_idx], gt_boxes[best_gt], cls_idx))
                    gt_used[best_gt] = True

    return matches


def compute_nds_metrics(det_results, gt_results, iou_thresh=0.5):
    """Compute NuScenes Detection Score components: ATE, ASE, AOE.

    ATE: Average Translation Error (Euclidean center distance in BEV)
    ASE: Average Scale Error (1 - IoU_3D of matched boxes, approximated as
         1 - volume_intersection / volume_union using axis-aligned volumes)
    AOE: Average Orientation Error (smallest angle difference in yaw)

    Returns:
        dict mapping class_name -> {'ate': float, 'ase': float, 'aoe': float, 'count': int}
        and 'mean' -> averaged across classes with matches
    """
    matches = match_detections(det_results, gt_results, iou_thresh)

    # Accumulate per class
    per_class = {cls: {'ate': [], 'ase': [], 'aoe': []} for cls in range(len(CLASS_NAMES))}

    for pred_box, gt_box, cls_idx in matches:
        # ATE: Euclidean distance in BEV (x, y)
        ate = np.sqrt((pred_box[0] - gt_box[0])**2 + (pred_box[1] - gt_box[1])**2)
        per_class[cls_idx]['ate'].append(ate)

        # ASE: 1 - IoU_3D (approximated with axis-aligned volume overlap)
        pred_vol = pred_box[3] * pred_box[4] * pred_box[5]  # l * w * h
        gt_vol = gt_box[3] * gt_box[4] * gt_box[5]
        # Approximate scale similarity as min/max ratio per dimension
        scale_iou = 1.0
        for d in range(3, 6):  # l, w, h
            scale_iou *= min(pred_box[d], gt_box[d]) / max(pred_box[d], gt_box[d], 1e-6)
        per_class[cls_idx]['ase'].append(1.0 - scale_iou)

        # AOE: Smallest absolute yaw difference (handle wraparound)
        yaw_diff = pred_box[6] - gt_box[6]
        # Normalize to [-pi, pi]
        yaw_diff = (yaw_diff + np.pi) % (2 * np.pi) - np.pi
        per_class[cls_idx]['aoe'].append(abs(yaw_diff))

    # Aggregate
    results = {}
    valid_ates, valid_ases, valid_aoes = [], [], []

    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        data = per_class[cls_idx]
        count = len(data['ate'])
        if count == 0:
            results[cls_name] = {'ate': float('nan'), 'ase': float('nan'),
                                 'aoe': float('nan'), 'count': 0}
            continue

        ate_mean = float(np.mean(data['ate']))
        ase_mean = float(np.mean(data['ase']))
        aoe_mean = float(np.mean(data['aoe']))

        results[cls_name] = {'ate': ate_mean, 'ase': ase_mean,
                             'aoe': aoe_mean, 'count': count}
        valid_ates.append(ate_mean)
        valid_ases.append(ase_mean)
        valid_aoes.append(aoe_mean)

    # Mean across classes that have matches
    results['mean'] = {
        'ate': float(np.mean(valid_ates)) if valid_ates else float('nan'),
        'ase': float(np.mean(valid_ases)) if valid_ases else float('nan'),
        'aoe': float(np.mean(valid_aoes)) if valid_aoes else float('nan'),
        'count': sum(r['count'] for r in results.values() if isinstance(r.get('count'), int)),
    }

    return results


def compute_nds(mAP, nds_metrics):
    """Compute NuScenes Detection Score.

    NDS = 1/10 * [5*mAP + sum(max(1-TP_metric, 0) for TP_metric in [ATE, ASE, AOE, AVE, AAE])]

    We only have ATE, ASE, AOE (no velocity/attribute), so we scale accordingly:
    NDS = 1/6 * [3*mAP + (1-ATE_norm) + (1-ASE) + (1-AOE_norm)]
    where ATE_norm = min(ATE/2, 1), AOE_norm = min(AOE/pi, 1).
    """
    mean = nds_metrics.get('mean', {})
    ate = mean.get('ate', float('nan'))
    ase = mean.get('ase', float('nan'))
    aoe = mean.get('aoe', float('nan'))

    if np.isnan(ate) or np.isnan(ase) or np.isnan(aoe):
        return float('nan')

    tp_ate = max(1.0 - min(ate / 2.0, 1.0), 0.0)
    tp_ase = max(1.0 - min(ase, 1.0), 0.0)
    tp_aoe = max(1.0 - min(aoe / np.pi, 1.0), 0.0)

    nds = (3.0 * mAP + tp_ate + tp_ase + tp_aoe) / 6.0
    return nds


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate 3D Object Detection')
    parser.add_argument('--data-root', required=True, help='NuScenes data root')
    parser.add_argument('--checkpoint', required=True, help='Model checkpoint path')
    parser.add_argument('--model', choices=['pointpillars', 'second', 'centerpoint'], default='pointpillars',
                        help='Model architecture (default: pointpillars)')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--score-thresh', type=float, default=0.1)
    parser.add_argument('--nms-iou', type=float, default=0.3)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    from src.data.datasets import create_dataloader

    # Model
    logger.info(f"Loading {args.model} model...")
    if args.model == 'second':
        from src.detectors.second import SECOND
        model = SECOND(num_classes=10)
    elif args.model == 'centerpoint':
        from src.detectors.centerpoint import CenterPoint
        model = CenterPoint(num_classes=10)
    else:
        from src.detectors.pointpillars import PointPillars
        model = PointPillars(num_classes=10)
    model.load_checkpoint(args.checkpoint)
    model = model.to(device)
    model.eval()

    # Validation loader
    data_root = Path(args.data_root)
    val_info = data_root / 'nuscenes_infos_val.pkl'
    if not val_info.exists():
        logger.error(f"Val info not found: {val_info}")
        return

    val_loader = create_dataloader(
        data_root=str(data_root),
        info_path=str(val_info),
        batch_size=args.batch_size,
        num_workers=2,
        split='val',
    )

    # Run inference
    det_results = []
    gt_results = []

    logger.info(f"Running inference on {len(val_loader)} batches...")
    with torch.no_grad():
        for batch_dict in tqdm(val_loader, desc='Evaluating'):
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

            # Collect GT
            for b in range(batch_dict['batch_size']):
                gt_results.append({
                    'boxes': batch_dict['gt_boxes'][b].cpu().numpy(),
                    'labels': batch_dict['gt_labels'][b].cpu().numpy(),
                })

    # Evaluate
    logger.info("Computing metrics...")
    ap_dict = evaluate(det_results, gt_results)

    # Print table
    iou_thresholds = [0.25, 0.5]
    header = f"{'Class':<25}" + "".join(f"{'AP@'+str(t):<12}" for t in iou_thresholds)
    print("\n" + "=" * 50)
    print(header)
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
    print("=" * 50)

    # --- NDS Metrics (ATE, ASE, AOE) ---
    logger.info("Computing NDS metrics (ATE, ASE, AOE)...")
    nds_metrics = compute_nds_metrics(det_results, gt_results, iou_thresh=0.5)

    print(f"\n{'='*70}")
    print(f"{'Class':<25}{'ATE (m)':<12}{'ASE':<12}{'AOE (rad)':<12}{'Matches':<10}")
    print(f"{'-'*70}")

    for cls_name in CLASS_NAMES:
        m = nds_metrics[cls_name]
        ate_str = f"{m['ate']:.3f}" if not np.isnan(m['ate']) else "N/A"
        ase_str = f"{m['ase']:.3f}" if not np.isnan(m['ase']) else "N/A"
        aoe_str = f"{m['aoe']:.3f}" if not np.isnan(m['aoe']) else "N/A"
        print(f"{cls_name:<25}{ate_str:<12}{ase_str:<12}{aoe_str:<12}{m['count']:<10}")

    print(f"{'-'*70}")
    mean_m = nds_metrics['mean']
    ate_str = f"{mean_m['ate']:.3f}" if not np.isnan(mean_m['ate']) else "N/A"
    ase_str = f"{mean_m['ase']:.3f}" if not np.isnan(mean_m['ase']) else "N/A"
    aoe_str = f"{mean_m['aoe']:.3f}" if not np.isnan(mean_m['aoe']) else "N/A"
    print(f"{'Mean':<25}{ate_str:<12}{ase_str:<12}{aoe_str:<12}{mean_m['count']:<10}")

    # NDS
    mAP_05 = mAP_values.get(0.5, 0.0)
    nds = compute_nds(mAP_05, nds_metrics)
    nds_str = f"{nds*100:.2f}" if not np.isnan(nds) else "N/A"
    print(f"\nNDS (approx): {nds_str}%  (mAP@0.5={mAP_05*100:.2f}%)")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
