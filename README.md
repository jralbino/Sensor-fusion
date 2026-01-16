# 🚗 Sensor Fusion Studio: Object & Lane Detection

An interactive platform for experimentation and visualization of **Camera Sensor Fusion** techniques. This project allows for real-time comparison of different Object Detection and Lane Segmentation architectures.

![Sensor Fusion Demo](demo_screenshot.png)

## 🚀 Key Features

* **Object Detection:** Support for SOTA models like **YOLO11** and **RT-DETR**.
* **Lane Detection:** Visual comparison between geometric and segmentation methods:
    * **YOLOP (Panoptic Driving Perception):** Drivable area and lane line segmentation.
    * **UFLD (Ultra Fast Lane Detection):** High-speed detection based on row-anchors.
    * **PolyLaneNet:** Direct polynomial regression using deep neural networks.
    * **SegFormer (NVIDIA):** Semantic segmentation based on Transformers.
* **Interactive Interface:**
    * **Dynamic Filtering:** Filter objects by class (e.g., show only "Cars" or "Trucks") based on active detections.
    * **Selective Visualization:** Toggle layers (Vectors, Masks, Bounding Boxes).
    * **Performance Metrics:** Real-time latency calculation (ms).

## 🛠️ Installation

1.  **Clone the repository:**

2.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    
    # On Windows:
    venv\Scripts\activate
    
    # On Mac/Linux:
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 📂 Project Structure

* `Vision/app.py`: Interactive Frontend (Streamlit).
* `Vision/src/detectors/`: Inference logic for YOLO and RT-DETR.
* `Vision/src/lanes/`: Lane detector implementations (YOLOP, UFLD, etc.).
* `Vision/models/`: Directory for placing `.pt` or `.pth` model weights.
* `Vision/data/`: Directory for test images or videos.

## ▶️ Usage

1.  Ensure you have your model weights in the `Vision/models/` folder.
    * *Example: `yolo11l.pt`, `tusimple_18.pth`, etc.*
2.  Run the application from the project root:

    ```bash
    streamlit run Vision/app.py
    ```

3.  Open your browser at the address shown in the terminal (usually `http://localhost:8501`).

## 📊 Supported Models

| Task | Model | Framework |
|-------|--------|-----------|
| Objects | YOLO11 (L/X) | Ultralytics |
| Objects | RT-DETR | Ultralytics |
| Lanes | YOLOP | TorchHub |
| Lanes | UFLD | PyTorch Custom |
| Lanes | PolyLaneNet | PyTorch + EfficientNet |
| Lanes | SegFormer | HuggingFace Transformers |