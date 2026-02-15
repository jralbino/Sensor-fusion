# -*- coding: utf-8 -*-
"""
VERSIÓN SIN BENCHMARK - Solo genera videos
Usa esta versión si solo quieres visualizaciones sin métricas
"""

import sys
import yaml
import json
import logging
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

# Ensure project root is in sys.path for tracking import
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.utils.path_manager import path_manager
from config.logging_config import setup_logging

# Setup logging
logger = setup_logging(
    log_dir=path_manager.get("logs"),
    level=logging.INFO
)

from Vision.src.predictor import BatchPredictor
from Vision.src.visualizer import ResultVisualizer
from Vision.src.lanes.yolop_detector import YOLOPDetector
from Vision.src.lanes.polylanenet_detector import PolyLaneNetDetector
from Vision.src.lanes.ufld_detector import UFLDDetector

import torch
import gc


def load_lane_detector(detector_type: str):
    """Cargar detector de carriles."""
    logger.info(f"Cargando detector de carriles: {detector_type}")
    
    try:
        if detector_type == "YOLOP":
            return YOLOPDetector(), "YOLOP"
        elif detector_type == "UFLD":
            model_path = path_manager.get_model("ufld")
            return UFLDDetector(model_path=str(model_path)), "UFLD"
        elif detector_type == "PolyLaneNet":
            model_path = path_manager.get_model("polylanenet")
            return PolyLaneNetDetector(model_path=str(model_path)), "PolyLaneNet"
        else:
            logger.warning(f"Detector desconocido: {detector_type}, usando YOLOP")
            return YOLOPDetector(), "YOLOP"
    except Exception as e:
        logger.exception(f"Error cargando detector: {e}")
        return YOLOPDetector(), "YOLOP"


def main():
    """Pipeline SIN benchmark - solo predicciones y videos."""
    
    logger.info("=" * 70)
    logger.info("PIPELINE DE SENSOR FUSION - MODO VIDEO ONLY")
    logger.info("=" * 70)
    
    # --- CONFIGURACIÓN ---
    IMAGES_DIR = path_manager.get("bdd_images_val")
    PREDICTIONS_DIR = path_manager.get("predictions")
    VIDEOS_DIR = path_manager.get("videos")
    
    LIMIT = None  # Cantidad de imágenes para videos
    
    models_to_run = [
        ("YOLO11-X", path_manager.get_model("yolo11x")),
        ("RTDETR-L", path_manager.get_model("rtdetr_l")),
        ("RTDETR-BDD", path_manager.get_model("rtdetr_bdd")),
        ("RTDETR-people", path_manager.get_model("rtdetr_people"))
    ]
    
    LANE_DETECTOR_TYPE = "YOLOP"
    LANE_OPTIONS = {
        'show_drivable': True,
        'show_lanes': False,
        'show_lane_points': True
    }
    
    logger.info(f"Configuración:")
    logger.info(f"  - Imágenes: {IMAGES_DIR}")
    logger.info(f"  - Límite: {LIMIT}")
    logger.info(f"  - Modelos: {[name for name, _ in models_to_run]}")
    logger.info(f"  - Lane Detector: {LANE_DETECTOR_TYPE}")
    logger.info(f"  ⚠️  BENCHMARK DESACTIVADO (usar run_benchmark.py por separado)")
    
    # === FASE 1: PREDICCIONES ===
    logger.info("\n" + "=" * 70)
    logger.info(f"FASE 1: PREDICCIONES")
    logger.info("=" * 70)
    
    predictor = BatchPredictor(images_dir=IMAGES_DIR, output_dir=PREDICTIONS_DIR)
    generated_jsons: List[Tuple[str, Path]] = []
    
    for name, model_path in models_to_run:
        logger.info(f"\n▶️  Procesando: {name}")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            mem_free = torch.cuda.mem_get_info()[0] / 1e9
            logger.info(f"   GPU Memory: {mem_free:.2f}GB libre")
        
        try:
            json_path = predictor.run_inference(
                model_name=name,
                model_path=model_path,
                conf=0.50,
                limit=LIMIT
            )
            
            if json_path:
                generated_jsons.append((name, json_path))
                logger.info(f"  ✅ JSON: {json_path}")
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        except Exception as e:
            logger.exception(f"  ❌ Error: {e}")
    
    # === FASE 2: VIDEOS ===
    if generated_jsons:
        logger.info("\n" + "=" * 70)
        logger.info("FASE 2: GENERACIÓN DE VIDEOS")
        logger.info("=" * 70)
        
        lane_model, model_suffix = load_lane_detector(LANE_DETECTOR_TYPE)
        viz = ResultVisualizer(images_dir=IMAGES_DIR, output_dir=VIDEOS_DIR)
        
        # Videos individuales
        logger.info("\n▶️  Videos individuales...")
        individual_videos = []
        
        for model_name, json_path in generated_jsons:
            logger.info(f"   - {model_name}")
            
            try:
                safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
                video_name = f"{safe_name}_{model_suffix}.mp4"
                
                preds = viz.load_predictions(json_path)
                video_path = viz.generate_single_video(
                    model_name=model_name,
                    predictions=preds,
                    output_name=video_name,
                    fps=5,
                    lane_detector=lane_model,
                    lane_config=LANE_OPTIONS
                )
                
                if video_path:
                    individual_videos.append(video_path)
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            
            except Exception as e:
                logger.exception(f"  ❌ Error: {e}")
        
        # Video comparativo 2x2
        logger.info("\n▶️  Video comparativo 2x2...")
        
        try:
            all_preds = []
            for model_name, json_path in generated_jsons:
                preds = viz.load_predictions(json_path)
                all_preds.append((model_name, preds))
            
            viz.generate_comparison_video_2x2(
                predictions_list=all_preds,
                output_name=f"fusion_comparison_ALL_{model_suffix}.mp4",
                fps=5,
                lane_detector=lane_model,
                lane_config=LANE_OPTIONS
            )
        
        except Exception as e:
            logger.exception(f"  ❌ Error: {e}")
        
        # Resumen
        logger.info("\n" + "=" * 70)
        logger.info("📹 VIDEOS GENERADOS")
        logger.info("=" * 70)
        logger.info(f"  Individuales: {len(individual_videos)}")
        for v in individual_videos:
            logger.info(f"    • {v.name}")
        logger.info(f"  Comparativo: fusion_comparison_ALL_{model_suffix}.mp4")
    
    # === FINALIZACIÓN ===
    logger.info("\n" + "=" * 70)
    logger.info("✅ PIPELINE COMPLETADO")
    logger.info("=" * 70)
    logger.info(f"\n📂 RESULTADOS:")
    logger.info(f"  - JSONs: {PREDICTIONS_DIR}")
    logger.info(f"  - Videos: {VIDEOS_DIR}")
    logger.info(f"\n💡 Para benchmarks, ejecuta: python Vision/run_benchmark.py")


def run_tracking_batch():
    """Run tracking over a sequence of images using ByteTracker2D."""
    import numpy as np
    from tracking import ByteTracker2D

    parser = argparse.ArgumentParser(description="Vision 2D Tracking")
    parser.add_argument('--images-dir', default=None, help='Directory of images')
    parser.add_argument('--model', default='YOLO11-X', help='Object detector name')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--limit', type=int, default=None, help='Max images')
    parser.add_argument('--output', default=None, help='Output JSON path')
    args = parser.parse_args()

    images_dir = Path(args.images_dir) if args.images_dir else path_manager.get("bdd_images_val")
    image_files = sorted(images_dir.glob("*.jpg"))
    if args.limit:
        image_files = image_files[:args.limit]

    if not image_files:
        logger.error(f"No images found in {images_dir}")
        return

    logger.info(f"Tracking {len(image_files)} images with {args.model}")

    # Load detector
    from Vision.src.detectors.detector_factory import get_object_detector
    from Vision.config.models import OBJECT_DETECTORS

    model_info = OBJECT_DETECTORS.get(args.model)
    if not model_info:
        logger.error(f"Unknown model: {args.model}")
        return

    model_path = path_manager.get_model(model_info["key"], check_exists=True)
    detector = get_object_detector(model_info["type"], model_path=str(model_path), conf=args.conf)

    tracker = ByteTracker2D(
        high_thresh=args.conf * 0.8,
        low_thresh=args.conf * 0.3,
        match_thresh=0.3,
        max_age=10,
        min_hits=1,
    )

    all_results = {}
    for img_path in image_files:
        import cv2
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        detections, _, stats = detector.detect(img, classes=None)

        if detections:
            dets_arr = np.array([d['bbox'] for d in detections])
            scores_arr = np.array([d['confidence'] for d in detections])
            class_names = sorted(set(d['class_name'] for d in detections))
            name_to_idx = {n: i for i, n in enumerate(class_names)}
            labels_arr = np.array([name_to_idx[d['class_name']] for d in detections])
            idx_to_name = {i: n for n, i in name_to_idx.items()}

            active = tracker.update(dets_arr, scores_arr, labels_arr)

            tracked = []
            for t in active:
                state = t.get_state()
                tracked.append({
                    'bbox': state.tolist(),
                    'class_name': idx_to_name.get(t.label, 'unknown'),
                    'confidence': float(t.score),
                    'track_id': t.track_id,
                })
        else:
            tracker.update(
                np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int),
            )
            tracked = []

        all_results[img_path.name] = tracked
        logger.info(f"  {img_path.name}: {len(tracked)} tracked objects")

    output_path = Path(args.output) if args.output else path_manager.get("predictions") / f"{args.model}_tracked.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Tracking results saved to {output_path}")


if __name__ == "__main__":
    # Check for --track flag
    if '--track' in sys.argv:
        sys.argv.remove('--track')
        try:
            run_tracking_batch()
        except Exception as e:
            logger.exception(f"\nError: {e}")
            sys.exit(1)
    else:
        try:
            main()
        except KeyboardInterrupt:
            logger.warning("\nInterrupted by user")
            sys.exit(1)
        except Exception as e:
            logger.exception(f"\nFatal error: {e}")
            sys.exit(1)