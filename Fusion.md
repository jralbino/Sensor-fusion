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

### Which is better?
`fusion_compare.py` runs all three and `evaluate` scores each against NuScenes GT
(greedy BEV matching, 2 m): recall / precision / F1 + ID-stability (track count, mean
track length). On scene 120 (41 frames, GT avg 58 obj/frame; results saved to
`outputs/compare/metrics_s120_41f.txt`):

| arch | recall | precision | F1 | tracks | meanLen | FP |
|---|---|---|---|---|---|---|
| A track-then-fuse | **0.51** | 0.82 | 0.63 | 133 | 11.1 | 260 |
| B fuse-then-track | 0.50 | **0.85** | 0.63 | **118** | 11.9 | **216** |
| C cov-weighted central | **0.51** | 0.84 | 0.63 | **118** | **12.2** | 233 |

**All three tie at F1 = 0.63**; they differ in the precision/recall trade-off and ID
stability. **B** (fuse-then-track) gives the highest precision and fewest false
positives: fusing first removes spurious detections (via multi-sensor confirmation)
*before* tracking, so the tracker sees cleaner input, whereas A tracks each noisy
modality independently and propagates its false positives (most FP, most tracks).
**C** keeps B's fuse-then-track structure but replaces the heuristic fusion with
covariance-weighted position fusion and Bayesian-log-odds existence (radar
down-weighted) — the literature-recommended setup — and gives the best ID stability
(longest mean track length) at near-B precision; it is the method driven in
`simulation_video.py`. All three are kept in `fusion_compare.py` (which prints the
full TP/FP/FN + F1 table and picks the best by F1) for testing on more sequences.

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

# A/B/C comparison (metrics table + 3-panel BEV video):
docker compose run --rm fusion python Fusion/fusion_compare.py --start-idx 120 --num-frames 41

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
- Evaluated only on NuScenes mini; full `v1.0-trainval` val needed for benchmark-grade
  numbers.
- Next: consolidate B as the main pipeline, a Streamlit fusion app, and BEVFusion
  (feature-level) in the `mmdet3d` container on a larger GPU.
