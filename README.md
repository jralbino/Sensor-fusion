Este archivo debe ir en la carpeta raíz `Sensor-fusion/` y sirve como portada del proyecto.

```markdown
# 🚗 Multi-Modal Sensor Fusion for Autonomous Driving

![Project Banner](assets/banner_demo.png)
*(Coloca aquí una imagen impactante que combine visión y datos)*

Repositorio integral para la percepción en conducción autónoma. Este proyecto implementa pipelines de **Visión Computacional**, **Procesamiento Lidar/Radar** y **Fusión de Sensores** para la detección robusta de objetos y carriles en entornos complejos (BDD100K, NuScenes).

## 🌟 Características Principales
* **Visión:** Comparativa SOTA (YOLO11, RT-DETR, YOLOP, PolyLaneNet).
* **Entrenamiento:** Scripts de *Fine-tuning* para adaptar modelos a datasets de conducción.
* **Benchmarks:** Herramientas automatizadas para medir mAP y Latencia.
* **Interfaz:** App interactiva basada en Streamlit.

## 🛠️ Instalación

1.  **Clonar el repositorio:**
    ```bash
    git clone [https://github.com/jralbino/Sensor-fusion.git](https://github.com/jralbino/Sensor-fusion.git)
    cd Sensor-fusion
    ```

2.  **Configurar entorno virtual:**
    ```bash
    python -m venv venv
    
    # Windows
    venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate
    ```

3.  **Instalar dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

## 📦 Modelos Necesarios
Para que el proyecto funcione al 100%, descarga los siguientes pesos y colócalos en `Vision/models/`:

| Modelo | Descripción | Archivo |
|--------|-------------|---------|
| **YOLO11** | Detección General | `yolo11l.pt`, `yolo11x.pt` |
| **RT-DETR** | Transformer (Original) | `rtdetr-l.pt` |
| **RT-DETR** | **Finetuned (Ours)** | `rtdetr-bdd-best.pt` |
| **UFLD** | Lane Detection Rápida | `tusimple_18.pth` |
| **PolyLaneNet** | Regresión de Carriles | `model_2305.pt` |

## 🚀 Quick Start (Visión)

Para probar el módulo de visión inmediatamente:

```bash
cd Vision
streamlit run app.py