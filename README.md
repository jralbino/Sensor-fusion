# Multi-Modal Sensor Fusion for Autonomous Driving

Multi-modal autonomous driving perception system with three modules: **LiDAR** (3D detection + tracking), **Vision** (2D detection + lane segmentation + tracking), and **Fusion** (LiDAR-to-camera projection). Uses NuScenes mini for 3D and BDD100K for 2D.

## Modules

### LiDAR — 3D Object Detection & Tracking

Detects 10 object classes in 3D LiDAR point clouds using pillar-based voxelization and 2D/sparse CNN backbones.

| Model | Description | Params |
|-------|-------------|--------|
| **PointPillars** | Dense 2D CNN backbone, anchor-based head | 4.0M |
| **SECOND** | Sparse 2D CNN (spconv), same head | 2.1M |
| **CenterPoint** | Dense backbone, anchor-free heatmap head | 4.4M |
| **MMDet3D wrappers** | Pretrained PointPillars/SECOND/CenterPoint from MMDetection3D | — |

Pipeline: Raw points → Voxelization (0.16m pillars) → VFE → Backbone → Detection head → NMS → Boxes (x,y,z,l,w,h,yaw).

Multi-frame **ByteTrack** tracking with ego-motion compensation and Kalman-filtered 3D states.

Visualizations: BEV (PNG), interactive 3D (Plotly HTML), 6-camera projection (PNG), and Streamlit dashboard.

See [Lidar/README.md](Lidar/README.md) for full details.

### Vision — 2D Object Detection & Lane Segmentation

Compares multiple SOTA detectors and lane models on BDD100K driving images.

| Task | Models |
|------|--------|
| **Object Detection** | YOLO11 (L/X), RT-DETR (L, BDD-finetuned, people-finetuned) |
| **Lane Detection** | YOLOP (BDD100K pretrained), UFLD (CULane / TuSimple), PolyLaneNet |

UFLD improvements: min-points filter + polynomial smoothing suppress spurious detections. UFLD (CULane) is recommended for BDD100K's diverse urban scenes.

Multi-frame **ByteTrack** tracking for consistent object IDs across video sequences.

App supports three modes: **Live Demo** (BDD100K images), **NuScenes** (6-camera grid with detections), and **Benchmarks**.

See [Vision/README.md](Vision/README.md) for full details.

### Radar — Radar Point Cloud Detection

Detects objects in NuScenes radar point clouds (5 radars fused, 6 features per point: x, y, z, rcs, vx_comp, vy_comp).

| Model | Approach | Description |
|-------|----------|-------------|
| **CFAR + DBSCAN** | Classical signal processing | Range-ordered CFAR thresholding + DBSCAN clustering + heuristic classification |
| **RadarPillars** | Deep learning | PointPillars-style pillar VFE adapted for radar (0.5m voxels, 200m range) |
| **RadarCenterPoint** | Deep learning | Heatmap-based anchor-free detection for radar |

Includes Streamlit app (`app.py`), CLI pipeline (`main.py`), NuScenes dataset loader, and 59 unit tests.

See [Radar/README.md](Radar/README.md) for full details.

### Fusion — LiDAR-Camera Projection

Projects LiDAR 3D points onto camera images using NuScenes calibration matrices (extrinsics + intrinsics). Demonstrates the spatial alignment between sensors.

### Tracking — ByteTrack Multi-Object Tracking

Shared `tracking/` module used by both LiDAR (3D) and Vision (2D) pipelines.

- Two-threshold association (high/low confidence) with Hungarian matching
- 2D Kalman filter (8-dim state: cx, cy, aspect_ratio, h + velocities)
- 3D Kalman filter (10-dim state: x, y, z, l, w, h, yaw + velocities)
- Axis-aligned BEV IoU for 3D, standard IoU for 2D
- No external dependencies beyond numpy + scipy

## Installation

Each module has its own Python 3.11 virtual environment with pinned dependencies in `requirements.txt`.

### Vision Module

```bash
cd Vision
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### LiDAR Module (needs spconv, nuscenes-devkit, CUDA)

```bash
cd Lidar
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Radar Module

```bash
cd Radar
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Data Setup

- **NuScenes mini**: Download from [nuscenes.org](https://www.nuscenes.org/nuscenes#download) → extract to `Fusion/data/sets/nuscenes/`
- **BDD100K**: Validation images → `Vision/data/raw/bdd100k/images/100k/val/`

## Quick Start

```bash
# --- LiDAR ---
# Prepare data (generates pickle info files)
Lidar/venv/bin/python Lidar/scripts/prepare_data.py \
    --data-root Fusion/data/sets/nuscenes --version v1.0-mini

# Single-scene detection + visualization
Lidar/venv/bin/python Lidar/main.py \
    --data-root Fusion/data/sets/nuscenes \
    --model mmdet3d_pointpillars --sample-idx 0

# Multi-frame tracking (BEV PNGs with track IDs + trajectories)
Lidar/venv/bin/python Lidar/main.py \
    --data-root Fusion/data/sets/nuscenes \
    --model mmdet3d_pointpillars --sample-idx 0 \
    --track --num-frames 10

# Training
Lidar/venv/bin/python Lidar/train_simple.py \
    --data-root Fusion/data/sets/nuscenes \
    --model pointpillars --num-epochs 20 --batch-size 2

# Streamlit dashboard (BEV, 3D, cameras, tracking)
cd Lidar && venv/bin/streamlit run app.py

# --- Vision ---
# Interactive Streamlit app (detection + lanes + tracking)
Vision/venv/bin/streamlit run Vision/app.py

# Batch processing with videos
Vision/venv/bin/python Vision/main.py

# Batch tracking (outputs tracked JSON)
Vision/venv/bin/python Vision/main.py --track --limit 50

# --- Radar ---
# Classical detector (no GPU needed)
Radar/venv/bin/python Radar/main.py \
    --data-root Fusion/data/sets/nuscenes --model cfar_dbscan

# Radar Streamlit app
Radar/venv/bin/streamlit run Radar/app.py

# --- Fusion ---
Lidar/venv/bin/python Fusion/src/lidar_to_camera.py

# --- Tests ---
Lidar/venv/bin/python -m pytest tracking/tests/ -v   # 22 tracking tests
Lidar/venv/bin/python -m pytest Lidar/tests/ -v
Vision/venv/bin/python -m pytest Vision/tests/ -v    # 32 unit + 14 integration
Radar/venv/bin/python -m pytest Radar/tests/ -v      # 59 tests
```

## Configuration

All paths are managed through `config/config.yaml` → `config/utils/path_manager.py` (PathManager singleton). Paths are relative to the project root.

## Project Structure

```
Sensor-fusion/
├── Lidar/                  # 3D LiDAR detection module (own venv)
│   ├── app.py              # Streamlit dashboard (BEV / 3D / cameras / tracking)
│   ├── main.py             # Detection + tracking pipeline
│   ├── train_simple.py     # Training with TensorBoard + cosine LR
│   ├── evaluate.py         # mAP + NDS evaluation
│   ├── visualize.py        # BEV visualization (track IDs + trajectories)
│   ├── visualize_3d.py     # 3D interactive + 6-camera projection + multi-sweep
│   └── src/                # Models, losses, data loading
├── Vision/                 # 2D detection module (own venv)
│   ├── app.py              # Streamlit app (Live Demo / NuScenes / Benchmarks)
│   ├── main.py             # Batch inference + tracking
│   ├── dashboard_app.py    # Benchmark analysis dashboard
│   ├── run_benchmark.py    # Performance benchmarks
│   ├── debug_polylanenet.py # GT vs PolyLaneNet diagnostic tool
│   └── src/                # Detectors, lane models, utilities
├── Radar/                  # Radar detection module (own venv)
│   ├── app.py              # Streamlit app
│   ├── main.py             # CLI detection pipeline
│   ├── src/                # CFAR+DBSCAN, RadarPillars, RadarCenterPoint
│   └── tests/              # 59 unit tests
├── Fusion/                 # LiDAR-camera projection
│   └── src/lidar_to_camera.py
├── tracking/               # ByteTrack MOT (shared by Lidar + Vision)
│   ├── bytetrack.py        # Core two-threshold association + Hungarian matching
│   ├── kalman_2d.py        # 2D Kalman filter (8-dim state)
│   ├── kalman_3d.py        # 3D Kalman filter (10-dim state)
│   ├── tracker_2d.py       # ByteTracker2D (xyxy boxes)
│   ├── tracker_3d.py       # ByteTracker3D (7-param boxes, BEV IoU)
│   └── tests/              # 22 unit tests
├── config/                 # Centralized path management
│   ├── config.yaml         # All paths + model filenames
│   └── utils/path_manager.py  # PathManager singleton
└── README.md
```
