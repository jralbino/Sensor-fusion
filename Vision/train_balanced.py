# En Vision/train_balanced.py
from ultralytics import RTDETR

model = RTDETR("Vision/models/rtdetr-bdd-best.pt") # <--- Tu modelo actual

model.train(
    data="Vision/config/bdd_balanced.yaml",
    epochs=30,           # Menos épocas porque ya sabe ver coches
    imgsz=1024,          # <--- CLAVE para ver objetos pequeños
    batch=4,             # Ajusta según tu VRAM
    lr0=0.00005,         # Learning rate muy bajo para no olvidar lo aprendido
    project="Vision/runs/train",
    name="rtdetr_people_signs"
)