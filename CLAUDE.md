# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-modal autonomous driving perception system with three modules: **Lidar** (3D detection), **Vision** (2D detection + lanes), and **Fusion** (LiDAR-to-camera projection). Uses NuScenes mini for 3D and BDD100K for 2D.

## Key Commands

### Lidar Module (uses its own venv)
```bash
# Setup
cd Lidar && python3.11 -m venv venv && source venv/bin/activate && pip install -r requirements.txt

# Data preparation (must run first)
Lidar/venv/bin/python Lidar/scripts/prepare_data.py --data-root Fusion/data/sets/nuscenes --version v1.0-mini

# Training (supports --model pointpillars|second|centerpoint)
Lidar/venv/bin/python Lidar/train_simple.py --data-root Fusion/data/sets/nuscenes --output-dir Lidar/outputs/run1 --num-epochs 20 --batch-size 2

# Evaluation
Lidar/venv/bin/python Lidar/evaluate.py --checkpoint <path> --data-root Fusion/data/sets/nuscenes

# Tests
Lidar/venv/bin/python -m pytest Lidar/tests/
```

### Vision Module (uses its own venv)
```bash
# Setup
cd Vision && python3.11 -m venv venv && source venv/bin/activate && pip install -r requirements.txt

# Streamlit apps
Vision/venv/bin/streamlit run Vision/app.py
Vision/venv/bin/streamlit run Vision/dashboard_app.py

# Batch processing
Vision/venv/bin/python Vision/main.py

# Tests (unit + integration)
Vision/venv/bin/python -m pytest Vision/tests/
Vision/venv/bin/python -m pytest Vision/tests/test_app.py -v          # 32 unit tests
Vision/venv/bin/python -m pytest Vision/tests/test_integration.py -v  # 14 integration tests
```

### Fusion Module
```bash
python Fusion/src/lidar_to_camera.py
```

## Architecture

### Three-Module Structure
- **Lidar/** — 3D object detection (PointPillars, SECOND w/ spconv, CenterPoint w/ heatmaps). 10 NuScenes classes.
- **Vision/** — 2D detection (YOLO11, RT-DETR) and lane detection (YOLOP, PolyLaneNet, UFLD)
- **Fusion/** — Projects LiDAR points onto camera images
- **config/** — Centralized path management via `PathManager` singleton loading `config/config.yaml`

### Lidar Detection Pipeline
1. **Voxelization**: Points → pillars (0.16m×0.16m×4.0m), grid 432×496×1, range [0, -39.68, -3] to [69.12, 39.68, 1]
2. **VFE** (Voxel Feature Encoder) → Pillar Feature Net → Scatter to BEV pseudo-image
3. **Backbone**: Dense 2D CNN (PointPillars), Sparse 3D CNN via spconv (SECOND), or same backbone with heatmap head (CenterPoint)
4. **Head**: Anchor-based (18 anchors/loc, multi-class) for PP/SECOND; anchor-free heatmap for CenterPoint
5. **Loss**: Focal (cls) + SmoothL1 (box) + Dir BCE (direction) in `src/training/losses.py`; CenterPoint uses `centerpoint_loss.py`
6. **Post-processing**: Fast axis-aligned BEV NMS in `src/core/geometry.py`

### Vision Detection Pipeline
- Factory pattern: `Vision/src/detectors/detector_factory.py` and `Vision/src/lanes/lane_factory.py`
- Models auto-download or load from `Vision/models/`

### Configuration
All paths flow through `config/config.yaml` → `config/utils/path_manager.py` (PathManager singleton). Paths are relative to project root.

### Data Locations
- NuScenes: `Fusion/data/sets/nuscenes/`
- BDD100K: `Vision/data/raw/bdd100k/`
- Lidar outputs/checkpoints: `Lidar/outputs/`, `Lidar/checkpoints/`

## Formatting & Linting
```bash
black <file>
flake8 <file>
mypy <file>
```
