# Gemini Code Assistant Context

## Project Overview

This repository is a comprehensive multi-modal sensor fusion project for autonomous driving perception. It integrates computer vision, Lidar, and sensor fusion pipelines to achieve robust object and lane detection in complex environments, utilizing datasets like BDD100K and NuScenes.

The project is structured into three main components:

*   **Vision**: This component focuses on 2D object and lane detection using various deep learning models like YOLO, RT-DETR, and lane detection models like YOLOP and PolyLaneNet. It includes a Streamlit-based application for interactive visualization and model comparison.
*   **Lidar**: This component handles 3D object detection from Lidar point clouds. It uses models like PointPillars and CenterPoint and includes scripts for training, evaluation, and visualization.
*   **Fusion**: This component is responsible for fusing data from different sensors, primarily projecting Lidar point clouds onto camera images to combine 3D and 2D data.

The project is well-structured, with clear separation of concerns between the different sensor modalities. It also includes a comprehensive set of scripts for training, evaluation, and visualization.

## Building and Running

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/jralbino/Sensor-fusion.git
    cd Sensor-fusion
    ```

2.  **Set up a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Vision Application

The `Vision` component includes a Streamlit application for interactive object and lane detection.

1.  **Ensure `PYTHONPATH` is set to the project root:**
    ```bash
    export PYTHONPATH=$(pwd)
    ```

2.  **Run the Streamlit app:**
    ```bash
    streamlit run Vision/app.py
    ```

### Running the Lidar Pipeline

The `Lidar` component has a pipeline for processing Lidar data, running inference, and generating visualizations.

1.  **Ensure `PYTHONPATH` is set to the project root:**
    ```bash
    export PYTHONPATH=$(pwd)
    ```

2.  **Run the main Lidar pipeline:**
    ```bash
    python Lidar/main.py
    ```

### Running the Fusion Demonstration

The `Fusion` component includes a script to demonstrate the projection of Lidar points onto a camera image.

1.  **Ensure `PYTHONPATH` is set to the project root:**
    ```bash
    export PYTHONPATH=$(pwd)
    ```

2.  **Run the Lidar to camera projection script:**
    ```bash
    python Fusion/src/lidar_to_camera.py
    ```

## Development Conventions

*   **Path Management**: The project now uses a consolidated `PathManager` in `config/utils/path_manager.py` to manage all file and directory paths and global configurations loaded from `config/config.yaml`. This ensures consistency and makes it easy to reconfigure the project.
*   **Configuration**: All global configurations are now centralized in `config/config.yaml`, managed by the `PathManager`.
*   **Logging**: The project uses a custom logging setup from `config/logging_config.py` to provide structured and informative logs.
*   **Jupyter Notebooks**: The project includes notebooks for experimentation and analysis, which can be found in `Vision/notebooks`.
*   **Modular Design**: The code is organized into modules with clear responsibilities, such as detectors, data loaders, and visualizers. This promotes code reuse and maintainability.
*   **Dependencies**: Project dependencies are managed in the `requirements.txt` file.
*   **Pre-trained Models**: The project relies on several pre-trained models, which need to be downloaded and placed in the appropriate directories. Refer to the `README.md` for more details.
*   **Benchmarking**: The `Vision` component includes a benchmarking script (`run_benchmark.py`) to evaluate model performance.

## Achievements Summary

During this session, the following key improvements and fixes were implemented across the project:

*   **Dependency Management**: Updated `requirements.txt` to include all necessary packages (`efficientnet_pytorch`, `transformers`, `prefetch_generator`, `yacs`), ensuring a smoother setup process.
*   **Path Management Consolidation**:
    *   Consolidated `config/utils/paths.py`, `config/utils/path_manager.py` (old), and `config/global_config.py` into a single, robust `PathManager` class located at `config/utils/path_manager.py`.
    *   All global configurations (`DATA_PATHS`, `MODEL_PATHS`, `USER_SETTINGS`) are now loaded from `config/config.yaml`, promoting a centralized and flexible configuration system.
    *   Removed redundant files: `config/utils/paths.py` and `config/global_config.py`.
*   **Codebase Modernization & Fixes**:
    *   **`PYTHONPATH` Usage**: Ensured consistent `PYTHONPATH` setup for running scripts from the project root.
    *   **Import Statements**: Corrected numerous `ModuleNotFoundError` and `NameError` issues by updating import statements across `Vision/main.py`, `Vision/src/lanes/lane_factory.py`, `Vision/src/lanes/polylanenet_detector.py`, `Vision/src/detectors/object_detector.py`, `Vision/src/utils/nuscenes_to_yolo.py` to correctly reference the consolidated `PathManager`.
    *   **Streamlit `app.py` Robustness**:
        *   Fixed `SyntaxError` and `StreamlitDuplicateElementKey` errors in `Vision/app.py` by correcting syntax, ensuring unique keys for widgets, and properly structuring the Streamlit app logic.
        *   Implemented persistent Streamlit logging using `st.session_state` to ensure logs are displayed correctly across reruns.
    *   **YOLOP Detector Fix**: Resolved a `TypeError` in `Vision/src/lanes/yolop_detector.py` related to `cv2.addWeighted` by ensuring correct image array handling, which was causing issues during lane detection visualization.
    *   **Dataset Path Correction**: Fixed `FileNotFoundError` in `Vision/train_balanced.py` by updating the dataset `path` in `Vision/config/bdd_balanced.yaml` to use an absolute path, allowing `ultralytics` to correctly locate the training data.
    *   **CLI Cleanup**: Removed `Vision/cli.py` as it was identified as redundant for the project's general use.
*   **Documentation & Accessibility**:
    *   Translated `Vision/dashboard_app.py` to English for broader accessibility.
    *   Updated `Vision/README.md` with comprehensive, English instructions for setup and running various components, improving clarity and maintainability.
*   **Functionality Confirmation**: All major Vision components (`app.py`, `main.py`, `run_benchmark.py`, `train_balanced.py`, `dashboard_app.py`) are now confirmed to be functional and running correctly after the applied fixes and refactorings.