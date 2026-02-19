# Radar — Radar Point Cloud Detection

Object detection on NuScenes radar point clouds using classical signal-processing and deep learning approaches.

## How It Works

NuScenes provides five radar sensors fused into a single point cloud. Each radar point has six features:

| Feature | Description |
|---------|-------------|
| x, y, z | 3D position in vehicle frame (meters) |
| rcs | Radar cross-section — reflection strength (dBsm) |
| vx_comp, vy_comp | Ego-motion compensated velocity (m/s) |

### Detection Pipeline

Three detectors are available via a factory pattern (`src/detectors/detector_factory.py`):

| Model | Approach | Description |
|-------|----------|-------------|
| **CFAR + DBSCAN** | Classical | Range-ordered CFAR thresholding → DBSCAN clustering → heuristic classification by geometry + RCS |
| **RadarPillars** | Deep learning | PointPillars-style VFE adapted for radar: 0.5 m voxels, 200 m detection range, 64-channel uniform backbone |
| **RadarCenterPoint** | Deep learning | Anchor-free heatmap head (same backbone), velocity prediction branch |

**Deep learning models** input: pillared radar point cloud (6 features per point).
**CFAR + DBSCAN**: no learned weights, runs on CPU with no CUDA requirement.

### Data Loading

`src/data/radar_dataset.py` reads NuScenes radar samples:
- Fuses all 5 radar sensors into the ego frame
- Applies quality filtering (`invalid_state`, `is_quality_valid` flags)
- Applies RCS threshold to remove low-confidence returns
- Selects 6 useful features per point

## Setup

```bash
cd Radar
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Data**: NuScenes mini → `../Fusion/data/sets/nuscenes/`

## Usage

### 1. CLI Detection Pipeline

```bash
# Classical detector (no GPU, no checkpoint needed)
Radar/venv/bin/python Radar/main.py \
    --data-root Fusion/data/sets/nuscenes \
    --model cfar_dbscan \
    --sample-idx 0

# Deep learning detector (loads checkpoint if provided)
Radar/venv/bin/python Radar/main.py \
    --data-root Fusion/data/sets/nuscenes \
    --model radar_pillars \
    --checkpoint Radar/outputs/run1/best.pth \
    --sample-idx 0
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `cfar_dbscan` | `cfar_dbscan`, `radar_pillars`, `radar_centerpoint` |
| `--data-root` | required | NuScenes data root |
| `--sample-idx` | `0` | Sample index to run on |
| `--checkpoint` | none | Path to trained checkpoint (deep learning only) |

### 2. Streamlit App

```bash
Radar/venv/bin/streamlit run Radar/app.py
```

Interactive dashboard to visualize radar detections on NuScenes samples.

## Project Structure

```
Radar/
├── app.py                  # Streamlit interactive dashboard
├── main.py                 # CLI detection pipeline
├── requirements.txt        # Pinned dependencies
├── config/
│   └── models.py           # Radar model registry
├── src/
│   ├── core/
│   │   └── base_radar_detector.py  # Detection3D dataclass + base class
│   ├── data/
│   │   ├── radar_dataset.py        # NuScenes radar dataset loader
│   │   └── radar_utils.py          # Feature selection, filtering, sensor constants
│   ├── detectors/
│   │   ├── detector_factory.py     # Factory: get_radar_detector()
│   │   ├── cfar_dbscan.py          # CFAR thresholding + DBSCAN clustering
│   │   ├── radar_pillars.py        # RadarPillars (deep learning)
│   │   └── radar_centerpoint.py    # RadarCenterPoint (deep learning)
│   └── utils/
│       └── voxel_generator.py      # Radar-specific pillar voxelization
└── tests/
    ├── test_radar.py               # Core unit tests
    ├── test_radar_extended.py      # Extended edge-case tests
    └── test_nuscenes_integration.py # NuScenes integration tests
```

## Tests

```bash
# From project root
Radar/venv/bin/python -m pytest Radar/tests/ -v   # 59 tests
```

## Key Differences vs LiDAR

| | LiDAR | Radar |
|---|---|---|
| Points/frame | ~30,000 | ~100–500 (sparse) |
| Features | x, y, z, intensity | x, y, z, rcs, vx_comp, vy_comp |
| Voxel size | 0.16 m | 0.5 m |
| Range | 70 m × 80 m | 200 m × 200 m |
| Backbone channels | 64 → 128 → 256 | 64 (uniform, matches sparsity) |
| Velocity | Not directly | Measured (compensated) |
