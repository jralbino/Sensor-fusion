from ultralytics import RTDETR
import torch
from pathlib import Path

from config.utils.path_manager import path_manager # Import the consolidated path manager

def train():
    # 1. Basic Configuration
    # Adjust batch size according to your VRAM.
    # RT-DETR is heavy. If you get a memory error, lower BATCH_SIZE to 4 or 2.
    BATCH_SIZE = 8 
    EPOCHS = 1           # BDD is large, 50 epochs is a good start
    IMG_SIZE = 640        # Standard size
    DEVICE = '0' if torch.cuda.is_available() else 'cpu'

    print(f"🚀 Starting training on {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}...")

    # 2. Load Pre-trained Model
    # We use the 'l' (Large) weights you already have.
    # When loading an existing .pt, Ultralytics performs fine-tuning automatically.
    model_path = path_manager.get_model_detail("rtdetr_l_path") # Use path_manager
    
    # If the local model is not found, it will automatically download the official one
    model = RTDETR(str(model_path)) 

    # 3. Execute Training
    results = model.train(
        data=str(path_manager.get_config_path("bdd_det_train")), # Use path_manager
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        
        # Important hyperparameters for Fine-Tuning
        lr0=0.0001,       # Low initial learning rate to avoid breaking pre-trained knowledge
        optimizer='AdamW', # Recommended for Transformers
        
        # Saving
        project=str(path_manager.get("vision_rtdetr_train_output")), # Use path_manager
        name="rtdetr_bdd_finetune",
        exist_ok=True,    # Overwrite if folder exists (be careful)
        
        # Visualization
        plots=True        # Generates loss and mAP graphs
    )
    
    print("✅ Training finished.")
    print(f"   Best model saved in: {path_manager.get('vision_rtdetr_train_output')}/rtdetr_bdd_finetune/weights/best.pt")

if __name__ == '__main__':
    train()