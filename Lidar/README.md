# LiDAR 3D Object Detection & Tracking

3D object detection and multi-object tracking on NuScenes mini dataset using pillar-based voxelization with multiple backbone architectures.

Detects 10 classes: car, truck, construction_vehicle, bus, trailer, barrier, motorcycle, bicycle, pedestrian, traffic_cone.

## How It Works

### Detection Pipeline

1. **Voxelization**: Raw LiDAR points (x, y, z, intensity) are grouped into vertical pillars on a 2D grid (0.16m x 0.16m cells, 4m height).
2. **Voxel Feature Encoder (VFE)**: A PointNet-style MLP processes each pillar's points into a fixed-size feature vector. Optionally uses augmented features (cluster center offsets, voxel center offsets, padding mask) for 10-feature input compatible with MMDet3D pretrained models.
3. **Scatter**: Pillar features are placed back onto the 2D grid, forming a pseudo-image (C x H x W).
4. **Backbone**: A 2D CNN extracts multi-scale features:
   - **PointPillars**: Dense 2D CNN (3 blocks: 64 → 128 → 256 channels)
   - **SECOND**: Sparse 2D CNN via spconv (same channel progression, fewer FLOPs)
   - **CenterPoint**: Same dense backbone, different detection head
5. **Detection Head**:
   - **Anchor-based** (PointPillars/SECOND): 18 multi-class anchors per location, predicts class scores + 7-DOF box (x,y,z,l,w,h,yaw) + direction
   - **Anchor-free** (CenterPoint): Per-class heatmaps with regression branches (center offset, dimensions, height, rotation)
6. **Post-processing**: Fast axis-aligned BEV NMS filters redundant detections.

### Tracking (ByteTrack 3D)

Multi-frame tracking adds consistent object IDs across time:

1. **Two-threshold association**: High-confidence detections match first (Hungarian algorithm on BEV IoU), then low-confidence detections match remaining tracks.
2. **3D Kalman filter**: 10-dimensional state [x, y, z, l, w, h, yaw, vx, vy, vz] predicts object motion between frames.
3. **Ego-motion compensation**: Track states are transformed between LiDAR frames using NuScenes ego-pose chain (lidar → ego → global → ego → lidar).
4. **Track lifecycle**: NEW → TRACKED → LOST → REMOVED. Lost tracks survive `max_age` frames before removal.

### Pretrained MMDet3D Models

Wrappers for official MMDetection3D checkpoints allow evaluation without training:
- `mmdet3d_pointpillars`: PointPillars with SECOND backbone + SECONDFPN (NDS ~49%)
- `mmdet3d_second`: Same architecture, different weight initialization
- `mmdet3d_centerpoint`: CenterPoint with pillar-based VFE (NDS ~51%)

These models use 10-feature VFE input, 400x400 or 500x500 grids, 360-degree detection range, and multi-sweep LiDAR loading (10 sweeps).

## Setup

```bash
cd Lidar
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Data**: Download [NuScenes mini](https://www.nuscenes.org/nuscenes#download) and extract to `../Fusion/data/sets/nuscenes/`.

**Pretrained models**: Run `bash models/download_pretrained.sh` or manually place checkpoints in `models/`.

## Usage

### 1. Data Preparation

Generate pickle info files from raw NuScenes data:

```bash
venv/bin/python scripts/prepare_data.py \
    --data-root ../Fusion/data/sets/nuscenes \
    --version v1.0-mini
```

### 2. Training

```bash
# PointPillars (default)
venv/bin/python train_simple.py \
    --data-root ../Fusion/data/sets/nuscenes \
    --output-dir outputs/my_run \
    --num-epochs 20 --batch-size 2 --lr 0.001

# SECOND (sparse convolutions)
venv/bin/python train_simple.py \
    --model second \
    --data-root ../Fusion/data/sets/nuscenes \
    --output-dir outputs/second_run \
    --num-epochs 20 --batch-size 2

# CenterPoint (heatmap-based)
venv/bin/python train_simple.py \
    --model centerpoint \
    --data-root ../Fusion/data/sets/nuscenes \
    --output-dir outputs/centerpoint_run \
    --num-epochs 20 --batch-size 2
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `pointpillars` | Architecture: `pointpillars`, `second`, `centerpoint` |
| `--data-root` | required | NuScenes data root |
| `--output-dir` | `outputs/test_run` | Checkpoints + TensorBoard logs |
| `--num-epochs` | 2 | Training epochs |
| `--batch-size` | 2 | Batch size |
| `--lr` | 0.001 | Learning rate (cosine annealing + 1-epoch warmup) |

**TensorBoard**: `tensorboard --logdir outputs/my_run/tensorboard`

Logged: loss, cls_loss, box_loss, dir_loss, grad_norm, lr (per step) + train/val loss (per epoch).

### 3. Evaluation

```bash
venv/bin/python evaluate.py \
    --data-root ../Fusion/data/sets/nuscenes \
    --checkpoint outputs/my_run/best.pth

# Pretrained model (auto-detects checkpoint)
venv/bin/python evaluate.py \
    --model mmdet3d_pointpillars \
    --data-root ../Fusion/data/sets/nuscenes
```

Reports: mAP@0.25 and mAP@0.5 per class, NDS metrics (ATE, ASE, AOE), NDS score.

### 4. Single-Scene Detection + Visualization

```bash
# With pretrained model (auto-detects checkpoint)
venv/bin/python main.py \
    --data-root ../Fusion/data/sets/nuscenes \
    --model mmdet3d_pointpillars \
    --sample-idx 0

# GT only (no checkpoint needed)
venv/bin/python main.py \
    --data-root ../Fusion/data/sets/nuscenes \
    --sample-idx 0
```

Outputs per sample:
- `*_bev.png` — Bird's Eye View with predictions + ground truth
- `*_3d.html` — Interactive 3D scene (Plotly, opens in browser)
- `*_cameras.png` — 6-camera grid with projected boxes

### 5. Multi-Frame Tracking

```bash
venv/bin/python main.py \
    --data-root ../Fusion/data/sets/nuscenes \
    --model mmdet3d_pointpillars \
    --sample-idx 0 \
    --track --num-frames 20
```

Outputs per-frame BEV PNGs to `outputs/demo/<model>/tracking/` with:
- Color-coded track IDs on each box ("ID:X class score")
- Trajectory polylines showing track history
- Ego-motion compensated tracking across frames

| Argument | Default | Description |
|----------|---------|-------------|
| `--track` | off | Enable multi-frame tracking mode |
| `--num-frames` | 20 | Number of sequential frames to track |
| `--score-thresh` | 0.15 | Detection confidence threshold |
| `--nms-iou` | 0.3 | NMS IoU threshold |

### 6. Streamlit Dashboard

```bash
cd Lidar
venv/bin/streamlit run app.py
```

Features:
- Browse NuScenes samples with slider
- Select model + checkpoint (trained or pretrained)
- Four visualization tabs: BEV, 3D Interactive, Camera Projection, Tracking
- Tracking tab: enable ByteTrack, set number of frames, frame-by-frame playback with slider
- Per-class detection counts and latency metrics

## Project Structure

```
Lidar/
├── app.py                  # Streamlit interactive dashboard
├── main.py                 # Detection + tracking pipeline
├── train_simple.py         # Training with TensorBoard + cosine LR
├── evaluate.py             # mAP + NDS evaluation
├── visualize.py            # BEV visualization (supports track IDs + trajectories)
├── visualize_3d.py         # 3D interactive + camera projection + multi-sweep loading
├── infer.py                # Single .bin file inference
├── scripts/
│   ├── prepare_data.py     # NuScenes → pickle info files
│   └── create_gt_database.py  # GT-sampling database generation
├── src/
│   ├── core/
│   │   ├── base_detector.py
│   │   └── geometry.py     # 3D IoU, fast BEV NMS, box corners
│   ├── data/
│   │   └── datasets.py     # NuScenes dataset + voxelization + GT-sampling
│   ├── detectors/
│   │   ├── pointpillars.py          # PointPillars (augmented VFE support)
│   │   ├── second.py                # SECOND (spconv backbone)
│   │   ├── centerpoint.py           # CenterPoint (heatmap head)
│   │   ├── mmdet3d_pointpillars.py  # MMDet3D PointPillars wrapper
│   │   ├── mmdet3d_second.py        # MMDet3D SECOND wrapper
│   │   └── mmdet3d_centerpoint.py   # MMDet3D CenterPoint wrapper
│   └── training/
│       ├── losses.py              # Focal + SmoothL1 + Dir loss, multi-class anchors
│       └── centerpoint_loss.py    # Heatmap focal + L1 regression loss
├── models/                 # Pretrained MMDet3D checkpoints
├── outputs/                # Training runs, visualizations, tracking frames
├── venv/                   # Dedicated Python 3.11 venv (CUDA + spconv)
├── requirements.txt        # Pinned dependencies
└── tests/                  # Unit tests
```

## Tests

```bash
# From project root
Lidar/venv/bin/python -m pytest Lidar/tests/ -v

# Tracking module tests
Lidar/venv/bin/python -m pytest tracking/tests/ -v
```

## Model Architectures

| Model | Backbone | Head | Grid | Params |
|-------|----------|------|------|--------|
| PointPillars | Dense 2D CNN (64→128→256) | 18 anchors/loc, 7-DOF + dir | 432 x 496 | 4.0M |
| SECOND | Sparse 2D CNN (spconv) | Same anchor-based head | 432 x 496 | 2.1M |
| CenterPoint | Dense 2D CNN | Per-class heatmaps + regression | 432 x 496 | 4.4M |
| SECOND 360 | Sparse 2D CNN | Anchor-based, augmented VFE | 500 x 500 | 2.25M |
