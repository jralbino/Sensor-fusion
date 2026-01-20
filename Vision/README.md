# 👁️ Vision Module

Este módulo contiene la implementación de algoritmos de Computer Vision para la detección de objetos y segmentación de carriles utilizando modelos SOTA (State-of-the-Art).

## 📸 Demo
![Vision App Demo](../assets/vision_demo.png)


## 📂 Estructura
* `app.py`: Aplicación interactiva (Streamlit) para demos en tiempo real o pruebas con imágenes estáticas.
* `main.py`: Script para procesamiento por lotes (batch inference) y generación de videos comparativos.
* `src/`: Código fuente de los detectores (YOLO, RT-DETR, YOLOP, etc.).
* `models/`: Carpeta donde deben residir los pesos (.pt, .pth).

## 🚀 Ejecución

### 1. Interfaz Interactiva (Recomendado)
Para lanzar la interfaz gráfica y comparar modelos:
```bash
streamlit run app.py