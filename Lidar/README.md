#  LiDAR Module

This module focuses on 3D object detection using LiDAR point cloud data. It integrates state-of-the-art models like PointPillars and CenterPoint, and is designed for processing datasets such as NuScenes.

## 📂 Structure
*   `main.py`: Main script for running the LiDAR processing pipeline, including data preparation, inference, and visualization.
*   `train.py`: Script for training 3D object detection models (e.g., CenterPoint) using `mmdet3d`.
*   `app.py`: Streamlit application for interactive visualization of LiDAR data and detection results.
*   `src/detectors/`: Contains implementations of various 3D object detectors.
*   `configs/`: Configuration files for `mmdet3d` models.
*   `checkpoints/`: Directory for storing pre-trained model weights.

## 🚀 Setup & Execution

### 1. Prerequisites
Ensure you have Python 3.9+ and a virtual environment set up. For a full project setup, please refer to the main `README.md` in the project root.

### 2. Install Dependencies
All project dependencies, including those specific to the LiDAR module, are installed by running the following command from the **project root directory** (after activating your virtual environment):
```bash
pip install -e ".[vision,lidar,fusion]"
```
This ensures all necessary packages for LiDAR (and other modules) are available.

### 3. Data Setup (NuScenes)
The LiDAR module primarily uses the NuScenes dataset. Ensure your NuScenes data is prepared and located at the path specified by `path_manager.get_data_detail("nuscenes_base")` (default: `Fusion/data/sets/nuscenes`). This includes the necessary info files (`nuscenes_infos_train.pkl`, `nuscenes_infos_val.pkl`) and the database info file (`nuscenes_dbinfos_train.pkl`).

### 4. Model Downloads
This module relies on several pre-trained models. Please refer to the **main project's `README.md`** for a comprehensive list of required LiDAR models, their download links, and their designated locations within the `Lidar/checkpoints/` directory. The `PathManager` uses these configurations to locate your models.

### 5. Running the Applications

**Important:** Always activate your virtual environment before running any scripts. Streamlit applications are best run from the project root directory.

#### a) LiDAR Processing Pipeline
To run the main LiDAR processing pipeline (e.g., data generation, inference, visualization):
```bash
# From the project root directory:
python Lidar/main.py
```

#### b) Training 3D Object Detection Models
To train a 3D object detection model (e.g., CenterPoint):
```bash
# From the project root directory:
python Lidar/train.py
```

#### c) Interactive LiDAR Viewer (Streamlit App)
To launch the interactive Streamlit application for visualizing LiDAR data and detections:
```bash
# From the project root directory:
streamlit run Lidar/app.py
```
