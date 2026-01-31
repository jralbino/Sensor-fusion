# -*- coding: utf-8 -*-
"""
Sensor Fusion Result Visualizer.
Generates individual and comparative videos with multiple models.
"""

import cv2
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ResultVisualizer:
    """Video generator for model comparisons."""
    
    def __init__(self, images_dir: Path, output_dir: Path):
        """
        Initialize visualizer.
        
        Args:
            images_dir: Directory with original images
            output_dir: Directory where videos will be saved
        """
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Visualizer initialized:")
        logger.info(f"  Images: {self.images_dir}")
        logger.info(f"  Output: {self.output_dir}")
    
    def generate_color(self, class_name: str) -> Tuple[int, int, int]:
        """Generate a unique color for each class."""
        hash_val = hash(class_name)
        r = (hash_val & 0xFF0000) >> 16
        g = (hash_val & 0x00FF00) >> 8
        b = hash_val & 0x0000FF
        return (b, g, r)
    
    def get_optimal_font_color(self, bg_color: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Calculate text color based on background luminance."""
        b, g, r = bg_color
        luminance = (0.299 * r + 0.587 * g + 0.114 * b)
        return (0, 0, 0) if luminance > 140 else (255, 255, 255)
    
    def draw_detections(
        self, 
        img: np.ndarray, 
        detections: List[Dict],
        model_name: str = ""
    ) -> np.ndarray:
        """
        Draw detections on the image.
        
        Args:
            img: Original image
            detections: List of detections
            model_name: Model name (for header)
            
        Returns:
            Image with drawn detections
        """
        canvas = img.copy()
        h, w = canvas.shape[:2]
        
        # Draw header with model name
        if model_name:
            cv2.rectangle(canvas, (0, 0), (w, 40), (0, 0, 0), -1)
            cv2.putText(
                canvas, model_name, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
            )
        
        # Draw each detection
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            class_name = det['class_name']
            conf = det['confidence']
            
            # Colors
            box_color = self.generate_color(class_name)
            text_color = self.get_optimal_font_color(box_color)
            
            # Bounding box
            cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, 2)
            
            # Label
            label = f"{class_name} {conf:.2f}"
            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            # Label position
            label_y = y1 - 5 if y1 - text_h - 10 > 40 else y1 + text_h + 5
            
            # Label background
            cv2.rectangle(
                canvas, 
                (x1, label_y - text_h - 5), 
                (x1 + text_w + 5, label_y + 5), 
                box_color, -1
            )
            
            # Label text
            cv2.putText(
                canvas, label, (x1 + 2, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1
            )
        
        # Detection counter in corner
        count_text = f"Objects: {len(detections)}"
        cv2.putText(
            canvas, count_text, (w - 150, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )
        
        return canvas
    
    def load_predictions(self, json_path: Path) -> Dict[str, List[Dict]]:
        """
        Load predictions from JSON (supports multiple formats).
        
        Args:
            json_path: Path to the JSON file
            
        Returns:
            Dictionary {image_name: [detections]}
        """
        logger.info(f"Loading predictions: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        predictions = {}
        
        # FORMAT WITH METADATA: {"meta": {...}, "results": [...]} 
        if isinstance(data, dict) and 'results' in data:
            logger.debug("Format with metadata detected")
            results = data['results']
            
            # Process results (list of predictions per image)
            if isinstance(results, list):
                for item in results:
                    if isinstance(item, dict):
                        # Search for image key (priority: image_name > image_path > image > file)
                        img_name = (
                            item.get('image_name') or 
                            item.get('image_path') or 
                            item.get('image') or 
                            item.get('file')
                        )
                        
                        if img_name:
                            # If it's a full path, extract only the name
                            img_name = Path(img_name).name
                            predictions[img_name] = item.get('detections', [])
                        else:
                            logger.warning(f"Item without image identifier. Keys: {list(item.keys())}")
            
            elif isinstance(results, dict):
                # Results is already a dictionary {image_name: [detections]}
                predictions = results
        
        # FORMAT 1: Direct list of dictionaries (without metadata)
        elif isinstance(data, list) and len(data) > 0:
            first_item = data[0]
            
            if isinstance(first_item, dict):
                # Detect which key contains the image name
                img_key = None
                for possible_key in ['image_name', 'image_path', 'image', 'file']:
                    if possible_key in first_item:
                        img_key = possible_key
                        break
                
                if img_key:
                    for item in data:
                        img_name = Path(item[img_key]).name
                        predictions[img_name] = item.get('detections', [])
                else:
                    logger.error(f"No image key found. Available keys: {list(first_item.keys())}")
                    raise ValueError(f"Unsupported JSON format in {json_path}")
            
            elif isinstance(first_item, str):
                logger.warning("JSON only contains image names, no detections")
                for img_name in data:
                    predictions[img_name] = []
        
        # FORMAT 2: Direct dictionary {image_name: [detections], ...}
        elif isinstance(data, dict):
            # Verify if it's the direct format (without 'meta' or 'results')
            first_key = next(iter(data.keys()))
            
            if isinstance(data[first_key], list):
                predictions = data
            else:
                logger.error(f"Unrecognized dictionary format")
                logger.error(f"Found keys: {list(data.keys())}")
                raise ValueError(f"Invalid JSON format in {json_path}")
        
        else:
            logger.error(f"JSON must be a list or dictionary, received: {type(data)}")
            raise ValueError(f"Invalid JSON format in {json_path}")
        
        logger.info(f"  ✅ Loaded {len(predictions)} images")
        
        if len(predictions) == 0:
            logger.warning(f"  ⚠️  No predictions found in {json_path}")
        
        return predictions
    
    def generate_single_video(
        self,
        model_name: str,
        predictions: Dict[str, List[Dict]],
        output_name: str,
        fps: int = 5,
        lane_detector: Optional[Any] = None,
        lane_config: Optional[Dict] = None
    ) -> Path:
        """
        Generate video for a SINGLE model.
        
        Args:
            model_name: Model name
            predictions: Dictionary of predictions
            output_name: Output file name
            fps: Frames per second
            lane_detector: Optional lane detector
            lane_config: Optional detector configuration
            
        Returns:
            Path to the generated video
        """
        logger.info(f"Generating individual video: {model_name}")
        
        output_path = self.output_dir / output_name
        
        # Sort images
        image_names = sorted(predictions.keys())
        
        if not image_names:
            logger.warning(f"No images for {model_name}")
            return None
        
        # Read first image to get dimensions
        first_img_path = self.images_dir / image_names[0]
        first_img = cv2.imread(str(first_img_path))
        
        if first_img is None:
            logger.error(f"Could not read: {first_img_path}")
            return None
        
        h, w = first_img.shape[:2]
        
        # Create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
        
        logger.info(f"  Processing {len(image_names)} frames...")
        
        # Process each image
        for img_name in tqdm(image_names, desc=f"  {model_name}"):
            img_path = self.images_dir / img_name
            img = cv2.imread(str(img_path))
            
            if img is None:
                logger.warning(f"  Skipping image not found: {img_name}")
                continue
            
            # Apply lane detector if available
            if lane_detector:
                try:
                    if lane_config:
                        img, _ = lane_detector.detect(img, **lane_config)
                    else:
                        img, _ = lane_detector.detect(img)
                except Exception as e:
                    logger.exception(f"  Error in lane detection: {e}")
            
            # Draw detections
            dets = predictions.get(img_name, [])
            frame = self.draw_detections(img, dets, model_name)
            
            # Write frame
            out.write(frame)
        
        out.release()
        
        logger.info(f"  ✅ Video saved: {output_path}")
        return output_path
    
    def generate_comparison_video_2x2(
        self,
        predictions_list: List[Tuple[str, Dict[str, List[Dict]]]],
        output_name: str,
        fps: int = 5,
        lane_detector: Optional[Any] = None,
        lane_config: Optional[Dict] = None
    ) -> Path:
        """
        Generate comparative video in 2x2 layout.
        
        Args:
            predictions_list: List of (model_name, predictions_dict)
            output_name: Output file name
            fps: Frames per second
            lane_detector: Lane detector
            lane_config: Detector configuration
            
        Returns:
            Path to the generated video
        """
        logger.info(f"Generating 2x2 comparative video with {len(predictions_list)} models")
        
        if len(predictions_list) > 4:
            logger.warning(f"Maximum 4 models supported for 2x2, using first 4")
            predictions_list = predictions_list[:4]
        
        output_path = self.output_dir / output_name
        
        # Get common list of images
        all_image_sets = [set(preds.keys()) for _, preds in predictions_list]
        common_images = sorted(set.intersection(*all_image_sets))
        
        logger.info(f"  Common images: {len(common_images)}")
        
        if not common_images:
            logger.error("No common images found between models")
            return None
        
        # Read first image for dimensions
        first_img_path = self.images_dir / common_images[0]
        first_img = cv2.imread(str(first_img_path))
        
        if first_img is None:
            logger.error(f"Could not read: {first_img_path}")
            return None
        
        h, w = first_img.shape[:2]
        
        # Calculate 2x2 grid dimensions
        grid_w = w * 2
        grid_h = h * 2
        
        # Create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (grid_w, grid_h))
        
        logger.info(f"  Processing {len(common_images)} frames...")
        
        # Process each image
        for img_name in tqdm(common_images, desc="  Comparative 2x2"):
            img_path = self.images_dir / img_name
            base_img = cv2.imread(str(img_path))
            
            if base_img is None:
                logger.warning(f"  Skipping: {img_name}")
                continue
            
            # Apply lane detector once
            if lane_detector:
                try:
                    if lane_config:
                        processed_img, _ = lane_detector.detect(base_img.copy(), **lane_config)
                    else:
                        processed_img, _ = lane_detector.detect(base_img.copy())
                except Exception as e:
                    logger.warning(f"  Error lane detection: {e}")
                    processed_img = base_img.copy()
            else:
                processed_img = base_img.copy()
            
            # Create frames for each model
            frames = []
            for model_name, predictions in predictions_list:
                dets = predictions.get(img_name, [])
                frame = self.draw_detections(processed_img.copy(), dets, model_name)
                frames.append(frame)
            
            # Fill with black frames if less than 4 models
            while len(frames) < 4:
                black_frame = np.zeros((h, w, 3), dtype=np.uint8)
                cv2.putText(
                    black_frame, "No Model", (w//2 - 80, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2
                )
                frames.append(black_frame)
            
            # Create 2x2 grid
            # Top row: frames[0] | frames[1]
            top_row = np.hstack([frames[0], frames[1]])
            
            # Bottom row: frames[2] | frames[3]
            bottom_row = np.hstack([frames[2], frames[3]])
            
            # Combine rows
            grid = np.vstack([top_row, bottom_row])
            
            # Add dividing lines
            # Vertical central line
            cv2.line(grid, (w, 0), (w, grid_h), (255, 255, 255), 2)
            
            # Horizontal central line
            cv2.line(grid, (0, h), (grid_w, h), (255, 255, 255), 2)
            
            # Write frame
            out.write(grid)
        
        out.release()
        
        logger.info(f"  ✅ Comparative video saved: {output_path}")
        return output_path
    
    def generate_video(
        self,
        predictions: List[Tuple[str, Path]],
        output_name: str,
        fps: int = 5,
        lane_detector: Optional[Any] = None,
        lane_config: Optional[Dict] = None,
        individual_videos: bool = False
    ):
        """
        Main method to generate videos.
        
        Args:
            predictions: List of (model_name, json_path)
            output_name: Output video name
            fps: Frames per second
            lane_detector: Optional lane detector
            lane_config: Detector configuration
            individual_videos: If True, generates individual videos
        """
        logger.info(f"{ '=' * 70}")
        logger.info(f"GENERATING VIDEOS")
        logger.info(f"{ '=' * 70}")
        
        # Load all predictions
        all_predictions = []
        for model_name, json_path in predictions:
            preds = self.load_predictions(json_path)
            all_predictions.append((model_name, preds))
        
        # Generate individual videos if requested
        if individual_videos:
            logger.info("\n▶️  Generating individual videos...")
            for model_name, preds in all_predictions:
                safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
                video_name = f"{safe_name}_individual.mp4"
                
                self.generate_single_video(
                    model_name=model_name,
                    predictions=preds,
                    output_name=video_name,
                    fps=fps,
                    lane_detector=lane_detector,
                    lane_config=lane_config
                )
        
        # Generate comparative video
        if len(all_predictions) > 1:
            logger.info("\n▶️  Generating 2x2 comparative video...")
            self.generate_comparison_video_2x2(
                predictions_list=all_predictions,
                output_name=output_name,
                fps=fps,
                lane_detector=lane_detector,
                lane_config=lane_config
            )
        else:
            logger.warning("Only 1 model, cannot make comparative video")
