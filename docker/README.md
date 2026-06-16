# Docker stacks

Each sensor/model family runs in its own container because their dependency
stacks conflict (different `numpy`, `torch`, `mmcv` versions). The repo and the
datasets are **bind-mounted** (`./:/workspace`), so you edit code on the host and
run it in any container — only the Python environment differs.

| Service   | Base / key deps                              | Use for |
|-----------|----------------------------------------------|---------|
| `vision`  | torch cu128 + ultralytics 8.4.14, numpy<2    | YOLO26, RT-DETR, lane models, `Vision/app.py` |
| `lidar`   | torch cu128 + spconv-cu120, numpy 1.26       | In-repo PointPillars/SECOND/CenterPoint |
| `radar`   | torch cu128 + scikit-learn, numpy<2          | CFAR+DBSCAN, RadarPillars, RadarCenterPoint |
| `fusion`  | torch cu128 + spconv + ultralytics + sklearn | **Decision-level (late) fusion**: LiDAR 3D + camera 2D + radar in BEV + ByteTrack3D |
| `mmdet3d` | torch 2.1 cu121 + mmcv/mmdet/mmdet3d 1.4     | **Official** BEVFusion/CMT/CenterPoint/DSVT checkpoints + NuScenes data prep |

`fusion` is the unified stack that runs all three light modalities in one process
(possible because every container is standardised on numpy<2). The heavy
feature-level fusion model BEVFusion stays in `mmdet3d` (different torch/mmcv).

## Prerequisites
- Docker + Compose v2, NVIDIA Container Toolkit (verify: `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`).

## Build
```bash
docker compose build vision lidar     # light stacks (minutes)
docker compose build mmdet3d          # heavy: compiles CUDA ops (~15-30 min)
```

## Run things
```bash
# One-off command in a container (auto-removed):
docker compose run --rm vision python Vision/main.py
docker compose run --rm lidar  python Lidar/main.py --model centerpoint

# Streamlit app (note --service-ports to publish the port):
docker compose run --rm --service-ports vision \
    streamlit run Vision/app.py --server.address 0.0.0.0
# then open http://localhost:8501

# Interactive shell:
docker compose run --rm mmdet3d bash
```

## GPU notes
- This is the same physical GPU inside the container — **Docker does not add VRAM.**
  On a 6 GB card, BEVFusion-full inference will still OOM on GPU; run it on CPU
  (slow, offline) or on a larger GPU. LiDAR-only models (CenterPoint, DSVT,
  VoxelNeXt) fit in 6-8 GB.
- `TORCH_CUDA_ARCH_LIST=8.6` in the mmdet3d image targets Ampere
  (RTX 3050 / 3060Ti / 3090). Change it if you build for a different GPU.

## Fusion container: importing multiple modules
`Lidar/`, `Vision/`, and `Radar/` each expose a **top-level `src` package**, so they
cannot all be on `PYTHONPATH` simultaneously — they'd shadow each other (and so
does `Fusion/src`). Use the repo-root helper **`module_loader.py`**, which isolates
one module root at a time and purges the `src`/`config` import cache between
switches:

```python
from module_loader import use_module, load

with use_module("Vision"):
    from config.utils.path_manager import path_manager
    from src.detectors.detector_factory import get_object_detector
    yolo = get_object_detector("yolo", model_path=..., device="cuda")

centerpoint = load("Lidar", "src.detectors.centerpoint")
radar_pillars = load("Radar", "src.detectors.radar_pillars")
```

Validated end-to-end in the `fusion` container: YOLO26 (camera, GPU) + LiDAR +
radar detectors loaded in one process. For full isolation (separate CUDA
contexts), run each sensor's inference as its own subprocess/container and have
`fusion` consume the emitted detections.

## Host vs container
Host venvs (`Vision/venv`, `Lidar/venv`) still work for quick edits, but their
dependency sets are internally inconsistent (e.g. `numpy==2.4.2` pinned next to
`nuscenes-devkit` which needs `numpy<2`). The containers pin a consistent,
reproducible set — prefer them for anything you want to reproduce on the server.
