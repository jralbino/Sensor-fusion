# 🚗 Multi-Modal Sensor Fusion for Autonomous Driving

![Project Banner](assets/banner_demo.png)
*(Insert an impactful image combining vision and data here)*

A comprehensive repository for autonomous driving perception. This project implements **Computer Vision**, **Lidar/Radar Processing**, and **Sensor Fusion** pipelines for robust object and lane detection in complex environments (BDD100K, NuScenes). The codebase has been refactored for improved modularity, maintainability, and consistent path management using a centralized `PathManager`.

## 🌟 Key Features
*   **Vision:** State-of-the-Art (SOTA) comparative analysis (YOLO11, RT-DETR, YOLOP, PolyLaneNet).
*   **Lidar:** 3D object detection with PointPillars and CenterPoint.
*   **Fusion:** Projection of Lidar point clouds onto camera images.
*   **Training:** Fine-tuning scripts to adapt models to driving datasets.
*   **Benchmarks:** Automated tools for measuring mAP and Latency.
*   **Interactive Interface:** Streamlit-based application for visualization and comparison.

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/jralbino/Sensor-fusion.git
    cd Sensor-fusion
    ```

2.  **Set up a virtual environment:**
    It is highly recommended to use a virtual environment to manage dependencies.
    ```bash
    python -m venv venv
    
    # On Windows
    venv\Scripts\activate
    # On Linux/macOS
    source venv/bin/activate
    ```

3.  **Install project dependencies (including sub-module specific ones):**
    Install the project in editable mode, along with all extra dependencies for Vision, Lidar, and Fusion modules. This command will install everything defined in `setup.py` and the respective `requirements.txt` files.
    ```bash
    pip install -e ".[vision,lidar,fusion]"
    ```
    *(Note: If you only need specific modules, you can install them individually, e.g., `pip install -e ".[vision]"`)*

## 📦 Required Models
For full project functionality, download the following pre-trained weights and place them in the `Vision/models/` and `Lidar/checkpoints/` directories as specified by the `config/config.yaml` file. The `PathManager` will locate them automatically.

### Vision Models (Place in `Vision/models/`)
| Model       | Description                 | Files                                      |
|-------------|-----------------------------|--------------------------------------------|
| **YOLOv11** | General Object Detection    | `yolo11l.pt`, `yolo11x.pt`                 |
| **RT-DETR** | Transformer-based Detector  | `rtdetr-l.pt`, `rtdetr-bdd-best.pt`        |
| **UFLD**    | Fast Lane Detection         | `tusimple_18.pth`                          |
| **PolyLaneNet** | Lane Regression Model     | `model_2305.pt`                            |

### Lidar Models (Place in `Lidar/checkpoints/`)
| Model       | Description                 | Files                                      |
|-------------|-----------------------------|--------------------------------------------|
| **PointPillars** | 3D Object Detector (Lidar) | `pointpillars_nus.pth`                     |
| **CenterPoint** | 3D Object Detector (Lidar) | `centerpoint_nus.pth`                      |

## 🚀 Quick Start & Usage

### Running the Vision Application
The Vision component includes a Streamlit application for interactive object and lane detection. **Always run Streamlit apps from the project root directory.**
```bash
# Ensure your virtual environment is active
# From the project root directory:
streamlit run Vision/app.py
```

### Running the Lidar Pipeline
The Lidar component has a pipeline for processing Lidar data, running inference, and generating visualizations.
```bash
# Ensure your virtual environment is active
# From the project root directory:
python Lidar/main.py
```

### Running the Fusion Demonstration
The Fusion component includes a script to demonstrate the projection of Lidar points onto a camera image.
```bash
# Ensure your virtual environment is active
# From the project root directory:
python Fusion/src/lidar_to_camera.py
```
