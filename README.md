# 🚗 Sensor Fusion Studio: Object & Lane Detection

Una plataforma interactiva para la experimentación y visualización de técnicas de **Fusión de Sensores (Cámara)**. Este proyecto permite comparar en tiempo real diferentes arquitecturas de Detección de Objetos y Segmentación de Carriles.

![Sensor Fusion Demo](demo_screenshot.png)
*(Asegúrate de subir una captura de pantalla de tu app y nombrarla demo_screenshot.png)*

## 🚀 Características Principales

* **Detección de Objetos:** Soporte para modelos SOTA como **YOLO11** y **RT-DETR**.
* **Detección de Carriles:** Comparativa visual entre métodos geométricos y de segmentación:
    * **YOLOP (Panoptic Driving Perception):** Segmentación de área conducible y líneas.
    * **UFLD (Ultra Fast Lane Detection):** Detección de alta velocidad basada en *row-anchors*.
    * **PolyLaneNet:** Regresión polinomial directa mediante redes neuronales profundas.
    * **SegFormer (NVIDIA):** Segmentación semántica basada en Transformers.
* **Interfaz Interactiva:**
    * Filtrado dinámico de clases (ej. mostrar solo "Coches" o "Peatones").
    * Visualización selectiva de capas (Vectores, Máscaras, Bounding Boxes).
    * Cálculo de latencia en tiempo real (ms).

## 🛠️ Instalación

1.  **Clonar el repositorio:**
    ```bash
    git clone [https://github.com/TU_USUARIO/sensor-fusion.git](https://github.com/TU_USUARIO/sensor-fusion.git)
    cd sensor-fusion
    ```

2.  **Crear un entorno virtual (Recomendado):**
    ```bash
    python -m venv venv
    
    # En Windows:
    venv\Scripts\activate
    
    # En Mac/Linux:
    source venv/bin/activate
    ```

3.  **Instalar dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

## 📂 Estructura del Proyecto

* `Vision/app.py`: Frontend interactivo (Streamlit).
* `Vision/src/detectors/`: Lógica de inferencia para YOLO y RT-DETR.
* `Vision/src/lanes/`: Implementaciones de detectores de carril (YOLOP, UFLD, etc.).
* `Vision/models/`: Carpeta donde se deben colocar los pesos `.pt` o `.pth`.
* `Vision/data/`: Directorio para imágenes o videos de prueba.

## ▶️ Uso

1.  Asegúrate de tener tus modelos (pesos) en la carpeta `Vision/models/`.
    * *Ejemplo: `yolo11l.pt`, `tusimple_18.pth`, etc.*
2.  Ejecuta la aplicación desde la raíz del proyecto:

    ```bash
    streamlit run Vision/app.py
    ```

3.  Abre tu navegador en la dirección que aparece en la terminal (usualmente `http://localhost:8501`).

## 📊 Modelos Soportados

| Tarea | Modelo | Framework |
|-------|--------|-----------|
| Objetos | YOLO11 (L/X) | Ultralytics |
| Objetos | RT-DETR | Ultralytics |
| Carriles | YOLOP | TorchHub |
| Carriles | UFLD | PyTorch Custom |
| Carriles | PolyLaneNet | PyTorch + EfficientNet |