# Sensor Fusion — Method

Decision-level (late) multi-sensor fusion of **LiDAR + camera + radar** on NuScenes,
with per-modality tracking and three cross-modality fusion architectures (A, B and C).
This document describes each stage of the method. Code lives in
`Fusion/src/late_fusion/`; the per-modality and final-fusion videos are produced by
`Fusion/fusion_video.py`, and the A/B/C comparison by `Fusion/fusion_compare.py`.

---

## 0. Why decision-level fusion

The target GPU has 6 GB, where feature-level fusion (e.g. BEVFusion) does not fit.
**Late fusion** runs each sensor's own detector and combines the *detections* in the
bird's-eye-view (BEV) frame. It fits in 6 GB, reuses every existing detector, and is
robust: each modality contributes its strength (LiDAR = accurate 3D geometry, camera
= classification, radar = velocity / moving objects).

All 3D quantities live in the **LiDAR frame** of the keyframe (x forward, y left,
z up). A 3D box is `[x, y, z, l, w, h, yaw]`; a 2D box is `[x1, y1, x2, y2]` pixels.

---

## 1. Per-sensor detection (`pipeline.run_fusion_batch`)

For each NuScenes keyframe the pipeline loads the synchronized sensor data and runs
one detector per modality:

| Modality | Model | Output |
|---|---|---|
| **LiDAR** | Official **MMDet3D PointPillars** (NuScenes checkpoint), 9-sweep, 360° | 3D boxes `[x,y,z,l,w,h,yaw]` + score + class |
| **Camera** | **YOLO26-L** on each of the 6 cameras | 2D boxes per camera + score + COCO class |
| **Radar** | Classical **CFAR + DBSCAN** (no pretrained radar model exists) | 3D cluster boxes + score + **velocity** |

Then each raw output is converted to a common type (`adapters.py`):
- `lidar_results_to_dets` → `Detection3D` (source="lidar"), then **class-agnostic BEV
  NMS** (`geometry.nms_bev`) removes overlapping duplicates (e.g. one vehicle
  detected as two classes — per-class NMS inside the detector leaves these).
- `yolo_dets_to_dets` → `Detection2D` (COCO→NuScenes class map; non-traffic classes
  dropped).
- `radar_dets_to_common` → `Detection3D` (source="radar"), **transformed from the
  RADAR_FRONT frame into the LiDAR frame** and with velocity rotated accordingly;
  then a score + range gate trims classical-radar clutter.

The pipeline returns, per frame: the per-sensor detections, the 6 cameras'
calibration (`intrinsic`, `lidar_to_cam`), `lidar_to_global` (for tracking),
the LiDAR point cloud, and the GT boxes (for evaluation).

---

## 2. Common representation (`types.py`)

- `Detection3D` — `box(7)`, `score`, `label`, `source`, optional `velocity`, `track_id`.
- `Detection2D` — `bbox(4)`, `score`, `label`, `camera`, `track_id`.
- `FusedObject` — `box(7)`, `score`, `label`, `sources` (which sensors agreed),
  `velocity`, `camera_confirmed`, `per_source_score`, `track_id`.

Keeping every modality in these types makes the association/fusion logic
modality-agnostic.

---

## 3. Per-modality fusion (tracking) — `tracking_helpers.py`

Each modality is tracked **independently** with gap-tolerant ByteTrack. This is the
"fusion of each modality" stage and the inputs to architecture A.

- **Camera** — `track_2d`: one ByteTracker2D **per camera** (all 6 considered), IoU +
  center-distance association in image space.
- **LiDAR / Radar** — `track_3d`: ByteTrack3D in the **global frame** (boxes are
  transformed LiDAR→global with `lidar_to_global`, tracked, then mapped back). Tracking
  in world coordinates **compensates ego motion**, so static objects keep stable IDs.
  Radar additionally has its velocity attached from the nearest raw radar detection.

Per-sensor tracking config (`TRACK_CONFIG`, NuScenes keyframes are 2 Hz = 0.5 s/frame):

| sensor | max_age (gap tolerance) | distance_thresh | confirm (min real detections) |
|---|---|---|---|
| lidar  | 4 frames | 3.0 | 4 |
| camera | 5 frames | 3.0 | 3 |
| radar  | 3 frames | 2.5 | 5 |

- **`max_age`** lets a track coast (Kalman) through *missing* detections — its ID
  survives the gap and re-associates on reappearance (the tracker matches LOST tracks).
- **`distance_thresh`** is a center-distance fallback (normalized by box diagonal) for
  when IoU drops to 0 between low-framerate frames.
- **`confirm`** is the post-processing gate (`confirm_filter`): drop any track *detected*
  in fewer than N frames → removes false positives (flickering clutter). It counts real
  detections, not coasted frames, so gaps are allowed.

---

## 4. Cross-modality association (`association.py`)

- **LiDAR ↔ camera**: each 3D box is projected into every camera (`project_box_to_image`)
  and matched to that camera's 2D detections by image-space IoU (Hungarian, per camera).
  Best match across cameras is kept.
- **LiDAR ↔ radar**: matched by BEV center distance (Hungarian, gate 3 m).

---

## 5. Fusion logic (`fusion.fuse`) — LiDAR-anchored

1. **LiDAR** detections provide the 3D boxes (anchors).
2. A matched **camera** detection confirms the object and refines its class
   (vision is the stronger classifier). On class agreement the confidence is boosted
   (noisy-OR `1-(1-a)(1-b)`); on disagreement the camera class wins.
3. A matched **radar** detection supplies **velocity**.
4. Confident **moving** radar detections with no LiDAR match are emitted as
   radar-only objects (radar's unique value); static radar clutter is gated out by a
   minimum speed.

Each `FusedObject` records `sources` (e.g. `{lidar, camera, radar}`), shown in the
videos as an `L·C·R` badge and a colour (green = camera-confirmed, cyan = multi-sensor
without camera, orange = LiDAR-only, blue = radar-only).

---

## 6. Fusion architectures (`multimodal.py`)

Three methods are implemented and compared. **C is the literature-recommended method
for this exact setup** (2D camera + 3D LiDAR + noisy radar): central-level fusion with
per-sensor, range-dependent measurement uncertainty (radar down-weighted), Bayesian
existence combination, LiDAR as the 3D seed, camera for semantics, radar for velocity
(see CLOCs for camera-LiDAR late fusion and covariance-intersection / multi-sensor
Kalman tracking for the radar weighting).


### A — `track_then_fuse` (hierarchical / late)
Track each modality independently (Section 3), **then fuse the tracks**: the
already-temporally-smoothed per-modality tracks are passed to `fuse`. Each fused
object inherits the anchoring modality's track ID.
→ video `fusionA_final_*` ("all modalities together").

### B — `fuse_then_track` (flat / central)
**Fuse all sensors' raw detections per frame** (Section 5), then track the fused
result in 3D. A single fusion stage where every sensor cooperates at once.
→ video `fusionB_single_*` ("one fusion stage, all sensors cooperating").

### C — `cov_central` (covariance-weighted central, the researched best method)
Like B (single fusion stage, then track) but the fusion is principled (`fuse_cov`):
- **Position**: LiDAR provides the box; a matched radar position is fused
  inverse-variance weighted with a **range-dependent radar noise**
  (`σ_radar = 0.6 + 0.03·range` m vs `σ_lidar = 0.3` m), so the noisy radar barely
  moves the accurate LiDAR estimate, and even less far away.
- **Existence/confidence**: combined in **Bayesian log-odds** (independent evidence)
  instead of noisy-OR, with radar evidence **down-weighted** (×0.5) because it is
  unreliable. Camera and LiDAR contribute full weight.
- **Velocity** from radar; **class** from camera.
→ video panel `C cov-central` in `fusion_compare.py`.

All three pass through the `confirm_filter` (Section 3) for false-positive removal.

### Does fusion help? (ablation)
`fusion_evaluate.py` adds one modality at a time to the same fuse-then-track
pipeline, scored over **all 10 NuScenes-mini scenes (404 frames)**, mean±std across
scenes (greedy BEV matching, 2 m; results in `outputs/eval/ablation_multiscene_10scenes.txt`):

| config | recall | precision | F1 | ΔF1 (pooled, vs LiDAR-only) |
|---|---|---|---|---|
| LiDAR-only | 0.42±0.13 | 0.72±0.15 | 0.51±0.12 | base |
| + Camera | 0.43±0.12 | 0.72±0.14 | 0.52±0.11 | +0.007 |
| + Camera + Radar | 0.44±0.11 | 0.70±0.15 | 0.52±0.10 | +0.006 |

**Honest finding:** the official LiDAR detector is already strong, so on F1 the gains
are **small**. The camera adds a little **recall** (and refines class/score) at no
precision cost; radar adds a touch more recall but costs a little precision (classical
CFAR+DBSCAN is noisy), so net F1 is flat. Fusion's real value here is the **extra
recall, class refinement, and velocity** it contributes — not captured by a single
class-agnostic F1. A learned radar detector and per-class metrics (see §9) would likely
widen the camera/radar margins.

### Which architecture is better?
Same 10-scene protocol, mean±std (+ pooled micro-F1):

| arch | recall | precision | F1 | tracks | meanLen | micro-F1 |
|---|---|---|---|---|---|---|
| **A track-then-fuse** | **0.49±0.12** | 0.63±0.16 | 0.53±0.10 | 102±38 | 13.0±3.6 | **0.551** |
| B fuse-then-track | 0.44±0.11 | **0.70±0.15** | 0.52±0.10 | **74±28** | 14.7±4.5 | 0.528 |
| C cov-weighted central | 0.47±0.13 | 0.68±0.15 | **0.54±0.10** | 80±28 | **14.3±3.9** | 0.541 |

**This reverses the single-scene impression** (where B looked best) — a key reason to
evaluate on many scenes. Across all 10 it is a **precision/recall trade-off**, not a
clear winner:
- **A** maximises **recall / micro-F1** but emits the most tracks (noisier, more FP).
- **B** maximises **precision** with the fewest, cleanest tracks (fusing before
  tracking suppresses spurious detections), at the cost of recall.
- **C** is the best **balance**: top macro-F1 with both good recall and precision, and
  the principled covariance/Bayesian fusion — a sensible default; it drives
  `simulation_video.py`.

Single-scene tables (scene 120) live in `outputs/compare/metrics_s120_41f.txt` via
`fusion_compare.py`; the robust multi-scene numbers above come from `fusion_evaluate.py`.

---

## 7. Key coordinate transforms

- **radar → lidar**: `inv(lidar_to_ego) @ radar_to_ego` (from calibrated_sensor
  extrinsics); applied to box center, yaw, and the velocity vector.
- **lidar → global** (for tracking): `ego_to_global @ lidar_to_ego` (calibrated_sensor
  + ego_pose); tracking in global compensates ego motion.
- **lidar → image** (camera association + drawing): `cam_ego_to_sensor @
  global_to_cam_ego @ ego_to_global @ lidar_to_ego`, then the pinhole intrinsic.
  NuScenes images are already undistorted (pinhole), so no distortion model is needed.

---

## 8. Outputs & how to run

```bash
# Per-modality fusion + final + single-stage videos (5 mp4s):
docker compose run --rm fusion python Fusion/fusion_video.py --start-idx 120 --num-frames 41
#   modality_camera_*, modality_lidar_*, modality_radar_*   (Section 3)
#   fusionA_final_*  (A, Section 6)   fusionB_single_*  (B, Section 6)

# A/B/C comparison on ONE scene (metrics table + 3-panel BEV video):
docker compose run --rm fusion python Fusion/fusion_compare.py --start-idx 120 --num-frames 41

# Multi-scene evaluation + sensor ablation (ALL 10 mini scenes; no video):
docker compose run --rm fusion python Fusion/fusion_evaluate.py
#   --max-scenes 3   quick subset   |   --scenes 3,4   pick scenes by number

# Data-driven scene reconstruction / simulation (global map + ego + agents, LiDAR-style):
docker compose run --rm fusion python Fusion/simulation_video.py --start-idx 120 --num-frames 41
```
`fusion_video.py` also emits a standalone video per fusion option
(`fusionA_final_*`, `fusionB_single_*`, `fusionC_covcentral_*`). The simulation
accumulates the LiDAR cloud in the world frame to rebuild the static environment and
replays the ego + fused-tracked agents through it — reproducing the scene from data
alone.
Keep `--start-idx`/`--num-frames` within one scene (mini scene starts: 0, 39, 79, 120,
161, 202, 242, 283, 324, 364). All label text is black-outlined and clipped to the
frame so frame size stays constant.

---

## 9. Limitations & next steps

- **Radar** is the noisiest modality: classical CFAR+DBSCAN gives unreliable classes
  (no pretrained radar model exists). It contributes velocity / moving objects, not
  classification.
- **Camera** tracks are per-camera (a car in two overlapping views can get two IDs);
  true cross-camera identity fusion would need 3D lifting or overlap association.
- Evaluated on all 10 NuScenes-**mini** scenes (404 frames, `fusion_evaluate.py`); the
  full `v1.0-trainval` val split is still needed for benchmark-grade numbers.
- The ablation shows camera/radar add little to a single class-agnostic F1. **Per-class
  + distance-stratified metrics and a velocity (AVE) metric** would expose where each
  modality actually helps (small/distant objects, moving objects).
- A **learned radar detector** (replacing CFAR+DBSCAN) would likely turn radar's small
  net contribution positive.
- Next: a Streamlit fusion app, and BEVFusion (feature-level) in the `mmdet3d` container
  on a larger GPU as a late-vs-feature-level comparison point.
