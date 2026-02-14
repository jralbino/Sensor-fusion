# LiDAR 3D Object Detection — PointPillars, SECOND & CenterPoint

3D object detection system using PointPillars, SECOND, and CenterPoint on NuScenes mini dataset.
Detects 10 classes: car, truck, construction_vehicle, bus, trailer, barrier, motorcycle, bicycle, pedestrian, traffic_cone.

## Project Structure

```
Lidar/
├── app.py                  # Streamlit interactive dashboard
├── train_simple.py         # Training with TensorBoard logging
├── evaluate.py             # Evaluation (mAP + NDS metrics)
├── infer.py                # Inference on single .bin files
├── visualize.py            # BEV static visualization (PNG)
├── visualize_3d.py         # Interactive 3D + camera projection
├── configs/
│   └── pointpillars_nuscenes.yaml
├── scripts/
│   ├── prepare_data.py     # NuScenes → pickle info files
│   └── train.py            # Advanced training (multi-GPU, AMP)
├── src/
│   ├── core/
│   │   ├── base_detector.py
│   │   └── geometry.py     # 3D IoU, NMS, box corners
│   ├── data/
│   │   └── datasets.py     # NuScenes dataset + voxelization
│   ├── detectors/
│   │   ├── pointpillars.py  # PointPillars model
│   │   ├── second.py       # SECOND model (sparse convolutions)
│   │   └── centerpoint.py  # CenterPoint model (heatmap-based)
│   ├── training/
│   │   ├── losses.py              # Anchor-based loss (PointPillars/SECOND)
│   │   └── centerpoint_loss.py    # Heatmap-based loss (CenterPoint)
│   └── utils/
│       └── voxel_generator.py
├── outputs/                # Checkpoints, TensorBoard logs, visualizations
├── requirements.txt
└── README.md
```

## Setup

```bash
cd Lidar

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Data**: Download [NuScenes mini](https://www.nuscenes.org/nuscenes#download) and extract to `../Fusion/data/sets/nuscenes/`.

## 1. Data Preparation

Generate pickle info files from raw NuScenes data:

```bash
venv/bin/python scripts/prepare_data.py \
    --data-root ../Fusion/data/sets/nuscenes \
    --version v1.0-mini
```

Creates `nuscenes_infos_train.pkl` and `nuscenes_infos_val.pkl` in the data root.

## 2. Training

```bash
# PointPillars (default)
venv/bin/python train_simple.py \
    --data-root ../Fusion/data/sets/nuscenes \
    --output-dir outputs/my_run \
    --num-epochs 20 \
    --batch-size 2 \
    --lr 0.001

# SECOND (sparse convolutions)
venv/bin/python train_simple.py \
    --model second \
    --data-root ../Fusion/data/sets/nuscenes \
    --output-dir outputs/second_run \
    --num-epochs 20 \
    --batch-size 2 \
    --lr 0.001

# CenterPoint (center-based heatmap)
venv/bin/python train_simple.py \
    --model centerpoint \
    --data-root ../Fusion/data/sets/nuscenes \
    --output-dir outputs/centerpoint_run \
    --num-epochs 20 \
    --batch-size 2 \
    --lr 0.001
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-root` | required | Path to NuScenes data |
| `--output-dir` | `outputs/test_run` | Checkpoints and logs |
| `--model` | `pointpillars` | Model architecture (`pointpillars`, `second`, or `centerpoint`) |
| `--num-epochs` | 2 | Number of training epochs |
| `--batch-size` | 2 | Batch size |
| `--lr` | 0.001 | Learning rate |
| `--device` | `cuda:0` | Device (`cuda:0` or `cpu`) |

**TensorBoard**: Logs are saved to `<output-dir>/tensorboard/`. Monitor with:

```bash
tensorboard --logdir outputs/my_run/tensorboard
```

Logged metrics (per step): `loss`, `cls_loss`, `box_loss`, `dir_loss`, `grad_norm`, `lr`.
Logged metrics (per epoch): `train_loss`, `val_loss`, `lr`.

Saves `best.pth` (lowest val loss) and `latest.pth` after each epoch.

## 3. Evaluation

```bash
venv/bin/python evaluate.py \
    --data-root ../Fusion/data/sets/nuscenes \
    --checkpoint outputs/my_run/best.pth \
    --device cuda:0

# For SECOND model
venv/bin/python evaluate.py \
    --model second \
    --data-root ../Fusion/data/sets/nuscenes \
    --checkpoint outputs/second_run/best.pth
```

Reports:
- **mAP** at IoU thresholds 0.25 and 0.5 (per class + mean)
- **NDS metrics**: ATE (translation error), ASE (scale error), AOE (orientation error)
- **NDS score**: Combined detection quality metric

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-root` | required | NuScenes data root |
| `--checkpoint` | required | Model checkpoint |
| `--score-thresh` | 0.1 | Min confidence for detections |
| `--nms-iou` | 0.3 | NMS IoU threshold |
| `--batch-size` | 2 | Batch size |

## 4. Inference

Run detection on a single LiDAR point cloud:

```bash
venv/bin/python infer.py \
    --checkpoint outputs/my_run/best.pth \
    --input path/to/pointcloud.bin
```

## 5. Visualization

### BEV (Bird's Eye View) — Static PNG

```bash
# Single file
venv/bin/python visualize.py \
    --checkpoint outputs/my_run/best.pth \
    --input path/to/pointcloud.bin

# Full validation set with GT comparison
venv/bin/python visualize.py \
    --checkpoint outputs/my_run/best.pth \
    --dataset --data-root ../Fusion/data/sets/nuscenes \
    --max-samples 10
```

### 3D Interactive + Camera Projection

```bash
# GT only (no model needed)
venv/bin/python visualize_3d.py \
    --data-root ../Fusion/data/sets/nuscenes \
    --sample-idx 0 1 2

# With model predictions
venv/bin/python visualize_3d.py \
    --data-root ../Fusion/data/sets/nuscenes \
    --checkpoint outputs/my_run/best.pth \
    --sample-idx 0 1 2
```

Outputs per sample:
- `sample_XXXX_3d.html` — Interactive 3D scene (Plotly, opens in browser)
- `sample_XXXX_cameras.png` — 6-camera grid with projected LiDAR points and boxes

## 6. Streamlit App

Interactive dashboard for exploring detections:

```bash
cd Lidar
venv/bin/streamlit run app.py
```

Features:
- Browse NuScenes samples with slider
- Toggle model predictions on/off
- Three visualization tabs: BEV, 3D Interactive, Camera Projection
- Per-class detection counts and metrics

## Model Architectures

### PointPillars

Converts 3D point clouds into a 2D pseudo-image via pillar-based voxelization, then uses a 2D CNN backbone with SSD-style detection head.

- **Grid**: 432 x 496 pillars, voxel size 0.16m x 0.16m x 4.0m
- **Range**: x=[0, 69.12], y=[-39.68, 39.68], z=[-3, 1] meters
- **Backbone**: 3-block dense 2D CNN encoder with transpose-conv decoder (64 → 128 → 256 channels)
- **Head**: 2 anchors per location, outputs class scores + 7-DOF boxes + direction
- **Loss**: Focal (classification) + SmoothL1 (regression) + BCE (direction)

### SECOND (Sparsely Embedded Convolutional Detection)

Same pillar-based voxelization and detection head as PointPillars, but replaces the dense 2D CNN backbone with sparse 2D convolutions (spconv) for more efficient feature extraction.

- **Sparse Backbone**: 3 blocks of SparseConv2d + SubMConv2d (64 → 128 → 256 channels)
- **Decoder**: Multi-scale feature fusion via transposed convolutions → 384-channel BEV features
- **Output**: Same (248, 216) feature map as PointPillars → same anchor grid and loss
- **Requires**: `spconv-cu120` (CUDA 12.x) or `spconv-cu118` (CUDA 11.8)
- **Reference**: Yan et al., "SECOND: Sparsely Embedded Convolutional Detection", Sensors 2018

### CenterPoint (Center-based 3D Detection)

Anchor-free detector that predicts object centers via per-class heatmaps, then regresses box attributes at each peak. Uses the same pillar-based voxelization and 2D CNN backbone as PointPillars.

- **Head**: Shared 2-layer conv + 5 parallel branches (heatmap, center offset, dimensions, height, rotation)
- **Decoding**: Top-K heatmap peaks + sub-pixel offset → world coordinates
- **Loss**: Penalty-reduced focal loss (heatmap) + L1 (regression)
- **No anchors**: Simpler than anchor-based methods, no IoU matching needed
- **Reference**: Yin et al., "Center-based 3D Object Detection and Tracking", CVPR 2021
