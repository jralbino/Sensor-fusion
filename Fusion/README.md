# 융 Fusion Module

This module is responsible for fusing data from different sensors, primarily focusing on projecting LiDAR point clouds onto camera images to combine 3D and 2D information. This integration is crucial for a comprehensive understanding of the environment in autonomous driving.

## 📂 Structure
*   `src/lidar_to_camera.py`: Script to demonstrate the projection of LiDAR points onto a camera image.
*   `data/`: Contains data used for fusion demonstrations (e.g., NuScenes data).

## 🚀 Setup & Execution

### 1. Prerequisites
Ensure you have Python 3.9+ and a virtual environment set up. For a full project setup, please refer to the main `README.md` in the project root.

### 2. Install Dependencies
All project dependencies, including those specific to the Fusion module, are installed by running the following command from the **project root directory** (after activating your virtual environment):
```bash
pip install -e ".[vision,lidar,fusion]"
```
This ensures all necessary packages for Fusion (and other modules) are available.

### 3. Data Setup (NuScenes)
The Fusion module often utilizes data that includes both camera images and LiDAR point clouds, such as the NuScenes dataset. Ensure your data is prepared and located as specified in the main project configuration (`Fusion/data/sets/nuscenes` for NuScenes).

### 4. Running the LiDAR to Camera Projection
To run the demonstration script that projects LiDAR points onto a camera image:
```bash
# From the project root directory:
python Fusion/src/lidar_to_camera.py
```
This script will visualize the fused data, allowing you to see how 3D LiDAR information aligns with 2D camera imagery.
