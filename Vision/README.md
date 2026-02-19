# Vision — 2D Object Detection, Lane Segmentation & Tracking

Comparative analysis of SOTA object detectors (YOLO11, RT-DETR) and lane detection models (YOLOP, PolyLaneNet, UFLD) on BDD100K driving images, with optional ByteTrack multi-object tracking. Also supports NuScenes camera browsing with object and lane detection.

## How It Works

### Object Detection

Uses a factory pattern (`src/detectors/detector_factory.py`) to load different detectors with a unified interface:

| Model | Type | Description |
|-------|------|-------------|
| **YOLO11-L** | YOLO | Ultralytics YOLO v11 Large — fast, general-purpose |
| **YOLO11-X** | YOLO | Ultralytics YOLO v11 Extra-Large — highest accuracy |
| **RT-DETR-L** | RT-DETR | Real-Time DETR transformer — COCO pretrained |
| **RT-DETR-BDD** | RT-DETR | Fine-tuned on BDD100K — better for driving scenes |
| **RT-DETR-people** | RT-DETR | Fine-tuned for pedestrian detection |

Each detector receives an image, runs inference, and returns a list of detections with bounding boxes (x1, y1, x2, y2), class names, and confidence scores.

### Lane Detection

Uses a factory pattern (`src/lanes/lane_factory.py`) with four lane models:

| Model | Output | Training Data | Description |
|-------|--------|--------------|-------------|
| **YOLOP** | Drivable area mask + lane lines | BDD100K | Multi-task panoptic model — best for BDD100K |
| **UFLD (CULane)** | Smoothed lane point coordinates | CULane (urban) | Ultra-Fast Lane Detection, 18 row anchors, ~70 ms — recommended for diverse scenes |
| **UFLD (TuSimple)** | Smoothed lane point coordinates | TuSimple (highway) | Same architecture, 56 row anchors, ~230 ms |
| **PolyLaneNet** | Polynomial lane curves | TuSimple | Polynomial regression; weaker generalization outside highway scenes |

`_process_output` improvements applied to UFLD: minimum-points filter (`min_points=4`) discards spurious partial lanes; degree-2 polynomial smoothing produces continuous curves instead of jagged segments.

### Tracking (ByteTrack 2D)

When enabled, detections across sequential images are associated using ByteTrack:

1. **Two-threshold association**: High-confidence detections are matched first to existing tracks using IoU-based Hungarian assignment, then low-confidence detections match remaining tracks.
2. **2D Kalman filter**: 8-dimensional state [cx, cy, aspect_ratio, h, vx, vy, va, vh] predicts box motion between frames.
3. **Track IDs**: Each tracked object gets a persistent ID displayed on its bounding box ("ID:X class score").
4. **Track colors**: Deterministic color palette (20 colors) based on track ID for visual consistency.

### Benchmarking

`run_benchmark.py` evaluates detectors against BDD100K annotations:
- mAP@0.5 and mAP@0.5:0.95 per class and mean
- Precision, Recall
- Inference time (ms)

Results are stored as JSON and visualized in the dashboard app.

## Setup

```bash
# Create dedicated venv (from project root)
cd Vision
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Data**: Place BDD100K validation images in `Vision/data/raw/bdd100k/images/100k/val/`.

**Models**: Model weights are auto-downloaded on first use or can be placed manually in `Vision/models/` as configured in `config/config.yaml`.

## Usage

### 1. Interactive Streamlit App

```bash
# From project root (using Vision venv)
Vision/venv/bin/streamlit run Vision/app.py
```

Three operating modes selectable from the sidebar:

**Live Demo (BDD100K)**
- Select object detector + lane detector from sidebar
- Adjust confidence threshold
- Enable ByteTrack tracking for persistent object IDs across image changes
- Filter visible classes, view detection JSON details
- Sample images from BDD100K validation set or upload custom images

**NuScenes**
- Browse NuScenes camera samples with a frame slider
- Select any of the 6 NuScenes cameras (FRONT, FRONT_LEFT, FRONT_RIGHT, BACK, BACK_LEFT, BACK_RIGHT)
- Apply the same object + lane detectors as in Live Demo
- "Show all 6 cameras" checkbox renders a 2×3 camera grid
- Per-frame metrics: detection count, inference latency

**Benchmark Dashboard**
- View stored benchmark results (mAP, precision, recall, latency) for all models

### 2. Batch Processing + Video Generation

```bash
# Standard mode — runs all configured models, generates individual + 2x2 comparison videos
Vision/venv/bin/python Vision/main.py
```

Processes all validation images through each model, saves prediction JSONs to `output/predictions/`, and generates annotated videos with lane overlay to `output/videos/`.

### 3. Batch Tracking

```bash
# Track objects across a sequence of images
Vision/venv/bin/python Vision/main.py --track --limit 50

# Full options
Vision/venv/bin/python Vision/main.py --track \
    --model YOLO11-X \
    --conf 0.5 \
    --images-dir Vision/data/raw/bdd100k/images/100k/val \
    --output output/predictions/YOLO11-X_tracked.json \
    --limit 100
```

Images are sorted by filename and processed sequentially. Output JSON maps each filename to a list of tracked detections with `track_id` fields.

| Argument | Default | Description |
|----------|---------|-------------|
| `--track` | off | Enable tracking mode |
| `--model` | `YOLO11-X` | Object detector name |
| `--conf` | 0.5 | Confidence threshold |
| `--images-dir` | BDD100K val | Input image directory |
| `--output` | auto | Output JSON path |
| `--limit` | all | Max images to process |

### 4. Benchmarking

```bash
Vision/venv/bin/python Vision/run_benchmark.py
```

Generates `output/data/benchmark_results.json` with per-model metrics.

### 5. Analysis Dashboard

```bash
Vision/venv/bin/streamlit run Vision/dashboard_app.py
```

Displays benchmark charts, per-class performance, model comparisons, and generated videos.

### 6. PolyLaneNet Diagnostic Tool

```bash
# Compare PolyLaneNet predictions against BDD100K Ground Truth
Vision/venv/bin/python Vision/debug_polylanenet.py --idx 0

# Options
--idx N        Index of validation image (default: 0)
--conf FLOAT   Confidence threshold (default: 0.3)
--out PATH     Output comparison image path
```

Generates a side-by-side image (GT polylines left, PolyLaneNet predictions right) plus a per-lane console table showing confidence, vertical range, and filter reason.

## Project Structure

```
Vision/
├── app.py                  # Streamlit app (Live Demo / NuScenes / Benchmarks)
├── main.py                 # Batch inference + tracking pipeline
├── dashboard_app.py        # Benchmark analysis dashboard
├── run_benchmark.py        # Performance benchmark runner
├── debug_polylanenet.py    # GT vs PolyLaneNet comparison diagnostic
├── config/
│   └── models.py           # Model registry (OBJECT_DETECTORS, LANE_DETECTORS)
├── src/
│   ├── detectors/
│   │   ├── detector_factory.py    # Factory: get_object_detector()
│   │   ├── object_detector.py     # Base class
│   │   ├── yolo_detector.py       # YOLO11 wrapper
│   │   └── rtdetr_detector.py     # RT-DETR wrapper
│   ├── lanes/
│   │   ├── lane_factory.py        # Factory: routes UFLD/PolyLaneNet/YOLOP
│   │   ├── yolop_detector.py      # YOLOP (multi-task, BDD100K pretrained)
│   │   ├── polylanenet_detector.py # PolyLaneNet (EfficientNet-B1, TuSimple)
│   │   └── ufld_detector.py       # UFLD (TuSimple/CULane, poly smoothing)
│   ├── predictor.py        # BatchPredictor (JSON output)
│   └── visualizer.py       # ResultVisualizer (video generation)
├── models/                 # Model weights (.pt, .pth)
│   ├── tusimple_18.pth     # UFLD TuSimple checkpoint
│   ├── culane_18.pth       # UFLD CULane checkpoint
│   └── model_2305.pt       # PolyLaneNet checkpoint (EfficientNet-B1)
├── data/raw/bdd100k/       # BDD100K dataset
├── venv/                   # Dedicated Python 3.11 venv
├── requirements.txt        # Pinned dependencies
├── tests/
│   ├── test_app.py         # 32 unit tests (app_utils, tracking adapters)
│   └── test_integration.py # 14 integration tests (real images + models)
└── README.md
```

## Tests

```bash
# Unit tests (fast, no model/images needed)
Vision/venv/bin/python -m pytest Vision/tests/test_app.py -v

# Integration tests (requires BDD100K images + model weights)
Vision/venv/bin/python -m pytest Vision/tests/test_integration.py -v

# All Vision tests
Vision/venv/bin/python -m pytest Vision/tests/ -v
```
