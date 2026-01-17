from ultralytics import RTDETR
import torch
from pathlib import Path

def train():
    # 1. Configuración Básica
    # Ajusta el batch según tu VRAM. 
    # RT-DETR es pesado. Si tienes error de memoria, baja BATCH_SIZE a 4 o 2.
    BATCH_SIZE = 8 
    EPOCHS = 1           # BDD es grande, 50 épocas es un buen inicio
    IMG_SIZE = 640        # Tamaño estándar
    DEVICE = '0' if torch.cuda.is_available() else 'cpu'

    print(f"🚀 Iniciando entrenamiento en {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}...")

    # 2. Cargar Modelo Pre-entrenado
    # Usamos los pesos 'l' (Large) que ya tienes. 
    # Al cargar un .pt existente, Ultralytics hace fine-tuning automáticamente.
    model_path = "Vision/models/rtdetr-l.pt"
    
    # Si no encuentra el modelo local, descargará el oficial automáticamente
    model = RTDETR(model_path) 

    # 3. Ejecutar Entrenamiento
    results = model.train(
        data="Vision/config/bdd_det_train.yaml",
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        
        # Hiperparámetros importantes para Fine-Tuning
        lr0=0.0001,       # Learning rate inicial bajo para no romper lo pre-entrenado
        optimizer='AdamW', # Recomendado para Transformers
        
        # Guardado
        project="Vision/runs/train",
        name="rtdetr_bdd_finetune",
        exist_ok=True,    # Sobreescribir si existe la carpeta (cuidado)
        
        # Visualización
        plots=True        # Genera gráficas de pérdida y mAP
    )
    
    print("✅ Entrenamiento finalizado.")
    print(f"   Mejor modelo guardado en: Vision/runs/train/rtdetr_bdd_finetune/weights/best.pt")

if __name__ == '__main__':
    train()