# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-modal autonomous driving perception system: **Lidar** (3D detection), **Vision** (2D detection + lanes), **Radar** (3D detection), and **Fusion** (LiDAR-to-camera projection + decision-level late fusion). Uses NuScenes mini for 3D and BDD100K for 2D.

**Two ways to run:** per-module host venvs (`Lidar/venv`, `Vision/venv`, `Radar/venv`) for quick edits, or **Docker** (recommended — one GPU container per stack, reproducible). See `docker/README.md`. Note the host venvs have internally inconsistent pins (e.g. numpy vs nuscenes-devkit); the containers pin a consistent set.

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
# LiDAR→camera projection
python Fusion/src/lidar_to_camera.py

# Late (decision-level) fusion — see Fusion/README.md
docker compose run --rm fusion python -m pytest Fusion/tests/test_late_fusion.py -v
docker compose run --rm fusion python Fusion/late_fusion_demo.py \
    --sample-idx 0 --lidar-checkpoint Lidar/outputs/centerpoint_run/best.pth
```

### Docker (recommended)
```bash
docker compose build vision lidar radar fusion   # light stacks (GPU)
docker compose build mmdet3d                      # heavy: official BEVFusion/CMT/DSVT zoo
docker compose run --rm vision python Vision/main.py
docker compose run --rm --service-ports vision streamlit run Vision/app.py --server.address 0.0.0.0
```
One container per dependency stack; repo + datasets are bind-mounted. Full reference in `docker/README.md`.

## Architecture

### Module Structure
- **Lidar/** — 3D object detection (PointPillars, SECOND w/ spconv, CenterPoint w/ heatmaps). 10 NuScenes classes.
- **Vision/** — 2D detection (YOLO26, YOLO11, RT-DETR-X/L) and lane detection (YOLOP, PolyLaneNet, UFLD)
- **Radar/** — 3D detection (CFAR+DBSCAN classical, RadarPillars, RadarCenterPoint)
- **Fusion/** — LiDAR→camera projection + decision-level late fusion (`src/late_fusion/`)
- **tracking/** — ByteTrack MOT (2D + 3D trackers) at the repo root
- **config/** — Centralized path management via `PathManager` singleton loading `config/config.yaml`
- **module_loader.py** — repo-root helper to import the conflicting per-module `src` packages one at a time (used by the fusion pipeline)

### Lidar Detection Pipeline
1. **Voxelization**: Points → pillars (0.16m×0.16m×4.0m), grid 432×496×1, range [0, -39.68, -3] to [69.12, 39.68, 1]
2. **VFE** (Voxel Feature Encoder) → Pillar Feature Net → Scatter to BEV pseudo-image
3. **Backbone**: Dense 2D CNN (PointPillars), Sparse 3D CNN via spconv (SECOND), or same backbone with heatmap head (CenterPoint)
4. **Head**: Anchor-based (18 anchors/loc, multi-class) for PP/SECOND; anchor-free heatmap for CenterPoint
5. **Loss**: Focal (cls) + SmoothL1 (box) + Dir BCE (direction) in `src/training/losses.py`; CenterPoint uses `centerpoint_loss.py`
6. **Post-processing**: Fast axis-aligned BEV NMS in `src/core/geometry.py`

### Vision Detection Pipeline
- Factory pattern: `Vision/src/detectors/detector_factory.py` and `Vision/src/lanes/lane_factory.py`
- Models registered in `Vision/config/models.py` (`OBJECT_DETECTORS`) + `config/config.yaml`; weights in `Vision/models/`
- Latest detectors: **YOLO26** (L/X) and **RT-DETR-X** (ultralytics 8.4.x)

### Late Fusion Pipeline (`Fusion/src/late_fusion/`)
- LiDAR-anchored decision-level fusion: LiDAR 3D boxes + camera 2D (projection-IoU association, class refinement) + radar (velocity / moving objects)
- Pure-NumPy core (types/geometry/association/fusion + the A/B/C `multimodal` architectures) with 29 unit tests (`tests/test_late_fusion.py` + `tests/test_multimodal.py`); `pipeline.py` orchestrates the real detectors via `module_loader`

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
