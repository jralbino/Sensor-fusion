# Fusion module

Two capabilities live here:

1. **Geometric projection** — `src/lidar_to_camera.py` projects LiDAR points onto
   camera images using NuScenes calibration (extrinsics + intrinsics).
2. **Decision-level (late) sensor fusion** — `src/late_fusion/` fuses the
   *detections* of LiDAR (3D), camera (2D) and radar into unified objects.

NuScenes data lives under `Fusion/data/sets/nuscenes/` (shared with the Lidar module).

---

## 1. LiDAR → camera projection

`src/lidar_to_camera.py` loads a NuScenes sample, transforms the LiDAR point cloud
into the camera frame (lidar→ego→global→camera), applies the pinhole projection,
and overlays the visible points colored by depth.

```bash
Lidar/venv/bin/python Fusion/src/lidar_to_camera.py     # host venv
```

---

## 2. Late (decision-level) sensor fusion

### Why late fusion?
The laptop GPU is 6 GB, where feature-level fusion (BEVFusion) does not fit. Late
fusion runs each modality's own detector and combines the *outputs* in BEV — it
fits in 6 GB, reuses every existing detector, and is the pragmatic path to a
working multi-sensor system. (Feature-level BEVFusion stays in the `mmdet3d`
container for a bigger GPU.)

### Strategy (LiDAR-anchored)
- **LiDAR** detections provide the 3D boxes (the anchors).
- **Camera** 2D detections confirm/refine class: each 3D box is projected into
  every camera and matched by image-space IoU. Class agreement boosts confidence
  (noisy-OR); on disagreement the camera class (stronger classifier) wins.
- **Radar** detections supply velocity to matched objects; confident **moving**
  radar detections with no LiDAR match are emitted as radar-only objects (radar's
  unique strength). Static radar clutter is gated out by a minimum speed.

### Layout
```
src/late_fusion/
  types.py        # Detection3D, Detection2D, FusedObject + class maps
  geometry.py     # 3D box corners, LiDAR->image projection, 2D IoU, BEV distance
  association.py  # LiDAR<->camera (projection IoU), LiDAR<->radar (BEV dist), Hungarian
  fusion.py       # fuse(): LiDAR-anchored combination
  adapters.py     # raw detector outputs -> common types (+ radar->LiDAR transform)
  lanes.py        # camera-only YOLOP lane / drivable-area overlay (not associated)
  pipeline.py     # end-to-end orchestration across the three module stacks
../late_fusion_demo.py        # CLI demo on one NuScenes sample (+ BEV PNG)
../tests/test_late_fusion.py  # 15 pure-CPU unit tests (the guaranteed small test)
```
The core (`types`/`geometry`/`association`/`fusion`) is pure NumPy and fully
unit-tested with no GPU or data dependencies.

### Cross-module orchestration
The pipeline drives the LiDAR, Vision and Radar stacks from one process, but
`Lidar/`, `Vision/`, `Radar/` and `Fusion/` each expose a conflicting top-level
`src` package. The repo-root helper [`module_loader.py`](../module_loader.py)
(`use_module()` / `load()`) isolates one module root at a time and purges the
`src`/`config` import cache between switches. Radar uses its `Radar.` prefix and
does not conflict. See `docker/README.md`.

### Run it (in the `fusion` container)
```bash
# Unit tests — run anywhere with numpy + scipy + pytest:
docker compose run --rm fusion python -m pytest Fusion/tests/test_late_fusion.py -v

# End-to-end demo on one NuScenes mini sample (camera + LiDAR + radar):
docker compose run --rm fusion python Fusion/late_fusion_demo.py \
    --sample-idx 0 \
    --lidar-checkpoint Lidar/outputs/centerpoint_run/best.pth
# -> console summary + Fusion/outputs/late_fusion_sample0.png
#    (green=camera-confirmed  orange=lidar-only  blue=radar-only  arrows=radar velocity)
```
Without `--lidar-checkpoint` the demo falls back to GT boxes as 3D anchors, so it
still runs end-to-end. Use `--no-radar` to disable radar.

### Per-sensor inspection videos
Before trusting fusion, review each sensor (and the 3D→image projection)
independently over a scene:
```bash
docker compose run --rm fusion python Fusion/make_sensor_videos.py \
    --start-idx 120 --num-frames 41 \
    --lidar-checkpoint Lidar/outputs/centerpoint_run/best.pth
```
Each box shows its **track ID** (`#id`): per-sensor ByteTrack, 3D tracked in the
global frame (ego-motion compensated), 2D tracked per camera.

Outputs in `Fusion/outputs/videos/` (`*` = `<sensor>_s<start>_<n>f`):
- `lidar_*`  — BEV 3D boxes + **6-camera grid with projected 3D wireframes** and
  depth-coloured projected LiDAR points (verifies the projection geometry).
- `camera_*` — 6-camera grid with 2D detections.
- `radar_*`  — BEV radar boxes + velocity + projected onto cameras.
- `*_filtered` — **false-positive removal**: a second set keeping only tracks
  *detected* in enough frames. Tracking is **gap-tolerant** (per-sensor
  `TRACK_CONFIG`): `max_age` lets a track coast through missing detections (the ID
  survives the gap and re-associates on reappearance) and `distance_thresh` is the
  low-framerate fallback (NuScenes keyframes are 2 Hz). The `confirm` threshold
  (min real detections) is per sensor — lidar 4, camera 3, radar 5 — overridable
  with `--min-track-len`. The console prints kept vs dropped per sensor.

Keep `--start-idx`/`--num-frames` within one scene (mini scene starts:
0, 39, 79, 120, 161, 202, 242, 283, 324, 364).

### Multi-sensor fusion architectures (A / B / C)
`src/late_fusion/multimodal.py` implements three architectures and `fusion_compare.py`
scores them vs NuScenes GT + renders a side-by-side BEV video:
- **A `track_then_fuse`** (hierarchical): track each modality (camera across all 6
  views, LiDAR, radar) independently, then fuse the *tracks*.
- **B `fuse_then_track`** (flat/central): fuse all sensors' raw detections per
  frame, then track the fused result.
- **C `cov_central`** (covariance-weighted central, the researched best practice):
  like B but the per-frame fusion (`fuse_cov`) is principled — radar position fused
  inverse-variance with range-dependent noise, existence combined in Bayesian
  log-odds with radar evidence down-weighted. Drives `simulation_video.py`.
```bash
docker compose run --rm fusion python Fusion/fusion_compare.py --start-idx 120 --num-frames 41
```
On scene 120 (41 frames) the three tie at **F1 0.63**; they differ in the trade-off:
**B** has the highest precision (0.85) and fewest false positives (fusing first removes
spurious detections before tracking), while **C** gives the best ID stability (longest
mean track length) at near-B precision. Full table in
`outputs/compare/metrics_s120_41f.txt`; `fusion_compare.py` prints it and picks the
best by F1. LiDAR uses the official MMDet3D PointPillars checkpoint; radar stays
classical (no pretrained radar model exists) with clutter gating.

### Programmatic use
```python
from src.late_fusion import fuse, Detection3D, Detection2D
fused = fuse(lidar_dets, cam_dets, radar_dets, cameras)  # -> List[FusedObject]
```

### Conventions
- 3D quantities are in the **LiDAR frame** (x forward, y left, z up).
- 3D box = `[x, y, z, l, w, h, yaw]`; 2D box = `[x1, y1, x2, y2]` pixels.

### Next steps
- Multi-frame tracking of fused objects with `tracking/ByteTracker3D`.
- Evaluate on the full NuScenes `val` split (needs `v1.0-trainval`).
- A Streamlit fusion app (port 8504) over the pipeline.

---

## Structure
```
Fusion/
├── src/
│   ├── lidar_to_camera.py     # LiDAR→camera projection
│   └── late_fusion/           # decision-level fusion package
├── late_fusion_demo.py        # late-fusion CLI demo
├── tests/                     # fusion unit tests
└── data/sets/nuscenes/        # NuScenes dataset (shared with Lidar)
```
