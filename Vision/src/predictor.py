import json
import cv2
import time
from pathlib import Path
from tqdm import tqdm
from Vision.src.detectors.object_detector import ObjectDetector

class BatchPredictor:
    def __init__(self, images_dir, output_dir):
        """
        Initializes the batch prediction manager.
        
        Args:
            images_dir (str): Path to the folder of original images.
            output_dir (str): Path where the resulting JSONs will be saved.
        """
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_inference(self, model_name, model_path, conf=0.5, iou=0.45, limit=None):
        """
        Executes inference on the dataset and saves the JSON.
        
        Args:
            model_name (str): Identifier name (e.g., "YOLO11-X").
            model_path (str): Path to the .pt file (e.g., "models/yolo11x.pt").
            limit (int, optional): Maximum number of images to process (for quick tests).
        
        Returns:
            str: Path to the generated JSON file.
        """
        # 1. Define output name
        # Clean the model name for use in the file
        safe_name = model_name.lower().replace(" ", "").replace("-", "")
        json_filename = f"{safe_name}_conf{int(conf*100)}.json"
        output_json_path = self.output_dir / json_filename

        # If it already exists and we don't want to overwrite, we could return here.
        # For now, we always overwrite to have fresh data.
        
        print(f"\n🚀 Starting Inference: {model_name}")
        print(f"   Model: {model_path}")
        print(f"   Output: {output_json_path}")

        # 2. Load Detector
        # Error handling if model does not exist
        if not Path(model_path).exists():
            print(f"❌ Error: Model does not exist at: {model_path}")
            return None

        detector = ObjectDetector(model_path=model_path, conf=conf, iou=iou)
        
        # 3. List images
        image_paths = sorted(list(self.images_dir.glob("*.jpg")))
        
        # APPLY LIMIT (For quick tests)
        if limit:
            print(f"⚠️  Test mode: Processing only the first {limit} images.")
            image_paths = image_paths[:limit]
        else:
            print(f"📊 Processing full dataset ({len(image_paths)} images).")
        
        params = detector.get_parameters()
        for key, value in params.items():
            if isinstance(value, Path):
                params[key] = str(value)

        experiment_data = {
            "meta": detector.get_parameters(),
            "results": []
        }

        # 4. Inference loop
        start_global = time.time()
        
        for img_p in tqdm(image_paths, desc=f"Inferences {model_name}"):
            img = cv2.imread(str(img_p))
            if img is None: continue
            
            detections, _, stats = detector.detect(img)
            
            experiment_data["results"].append({
                "image_name": img_p.name,
                "inference_ms": stats["inference_time_ms"],
                "detections": detections
            })

        total_time = time.time() - start_global
        print(f"✅ Finished in {total_time:.2f}s. Saving JSON...")

        # 5. Save
        with open(output_json_path, 'w') as f:
            json.dump(experiment_data, f, indent=4, default=str)
            
        return str(output_json_path)