# Fusion — LiDAR-Camera Projection

Projects LiDAR 3D point clouds onto camera images using NuScenes calibration data (extrinsics + intrinsics), demonstrating the spatial alignment between sensors.

## How It Works

`src/lidar_to_camera.py` loads a NuScenes sample, reads:
- The LiDAR point cloud (`.pcd.bin`)
- The camera image for a chosen camera
- The sensor calibration (lidar-to-ego, ego-to-global, camera extrinsics + intrinsics)

It then transforms LiDAR points into the camera frame, applies the pinhole projection, and overlays the visible points on the image colored by depth.

## Setup

No separate virtual environment is required. Use the Lidar or Vision venv, both of which include `nuscenes-devkit`.

**Data**: NuScenes mini → `Fusion/data/sets/nuscenes/`

## Usage

```bash
# From the project root
Lidar/venv/bin/python Fusion/src/lidar_to_camera.py
```

## Project Structure

```
Fusion/
├── src/
│   └── lidar_to_camera.py   # LiDAR-to-camera projection script
└── data/
    └── sets/nuscenes/       # NuScenes dataset (shared with Lidar module)
```
