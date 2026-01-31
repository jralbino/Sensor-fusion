# -*- coding: utf-8 -*-
"""
Script de benchmark masivo para modelos de detección.
Evalúa múltiples modelos en múltiples datasets y guarda métricas.
"""

import json
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List

from config.utils.path_manager import path_manager
from config.logging_config import setup_logging, get_logger

# Setup logging
logger = setup_logging(
    log_dir=path_manager.get("logs"),
    level=logging.INFO
)

# Imports de Ultralytics
from ultralytics import RTDETR, YOLO
import torch
import gc


def run_benchmark():
    """Ejecutar benchmark completo de modelos."""
    
    logger.info("=" * 70)
    logger.info("🚀 INICIANDO BENCHMARK MASIVO")
    logger.info("=" * 70)
    
    # --- 1. DEFINIR MODELOS A EVALUAR ---
    models_to_test = {
        "YOLO11-L (COCO)": path_manager.get_model("yolo11l"),
        "YOLO11-X (COCO)": path_manager.get_model("yolo11x"),
        "RTDETR-L (COCO)": path_manager.get_model("rtdetr_l"),
        "RTDETR-BDD (Finetuned)": path_manager.get_model("rtdetr_bdd"),
        "RTDETR-people": path_manager.get_model("rtdetr_people")
    }
    
    logger.info(f"Modelos a evaluar: {list(models_to_test.keys())}")
    
    # --- 2. DEFINIR DATASETS ---
    datasets = {
        "BDD100K": path_manager.get_config_path("bdd_det_train"),
        "NuScenes": path_manager.get_config_path("nuscenes")
    }
    
    logger.info(f"Datasets: {list(datasets.keys())}")
    
    # --- 3. ESTRUCTURA DE RESULTADOS ---
    results_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "datasets": {}
    }
    
    output_project = path_manager.get("benchmarks")
    output_project.mkdir(parents=True, exist_ok=True)
    
    # --- 4. EJECUTAR BENCHMARKS ---
    for dataset_name, yaml_path in datasets.items():
        logger.info(f"\n{'=' * 70}")
        logger.info(f"📂 DATASET: {dataset_name}")
        logger.info(f"   YAML: {yaml_path}")
        logger.info(f"{'=' * 70}")
        
        results_data["datasets"][dataset_name] = []
        
        for model_name, model_path in models_to_test.items():
            logger.info(f"\n  ▶️  Evaluando: {model_name}")
            logger.info(f"     Path: {model_path}")
            
            # Liberar memoria ANTES de cargar cada modelo
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                
                # Mostrar memoria disponible
                mem_free = torch.cuda.mem_get_info()[0] / 1e9
                mem_total = torch.cuda.mem_get_info()[1] / 1e9
                logger.info(f"     GPU Memory: {mem_free:.2f}GB / {mem_total:.2f}GB free")
            
            try:
                # Cargar modelo según tipo
                if "rtdetr" in str(model_path).lower():
                    model = RTDETR(str(model_path))
                    logger.info("     Tipo: RT-DETR")
                else:
                    model = YOLO(str(model_path))
                    logger.info("     Tipo: YOLO")
                
                # Ejecutar validación
                logger.info("     Ejecutando validación...")
                
                metrics = model.val(
                    data=str(yaml_path),
                    split='val',
                    device='cuda',  # Cambiar a 'cpu' si no tienes GPU
                    verbose=False,
                    plots=False,
                    project=str(output_project),
                    name=f'benchmark_{model_name.replace(" ", "_")}',
                    exist_ok=True,
                    # CRÍTICO para 6GB VRAM:
                    batch=1,  # Batch size mínimo para ahorrar memoria
                    imgsz=640  # No usar 1024, consume mucha memoria
                )
                
                # Extraer métricas principales
                entry = {
                    "model": model_name,
                    "mAP50-95": round(metrics.box.map, 4),
                    "mAP50": round(metrics.box.map50, 4),
                    "Precision": round(metrics.box.mp, 4),
                    "Recall": round(metrics.box.mr, 4),
                    "Inference_Time_ms": round(metrics.speed['inference'], 2)
                }
                
                # Métricas por clase
                class_map = {}
                for i, class_name in enumerate(metrics.names.values()):
                    if i < len(metrics.box.maps):
                        class_map[class_name] = round(metrics.box.maps[i], 4)
                
                entry["per_class"] = class_map
                
                results_data["datasets"][dataset_name].append(entry)
                
                logger.info(f"     ✅ Completado:")
                logger.info(f"        mAP50-95: {entry['mAP50-95']:.4f}")
                logger.info(f"        mAP50: {entry['mAP50']:.4f}")
                logger.info(f"        Speed: {entry['Inference_Time_ms']:.2f}ms")
                
                # CRÍTICO: Liberar modelo inmediatamente después de usarlo
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                logger.info(f"     🗑️  Modelo descargado de memoria")
            
            except FileNotFoundError as e:
                logger.error(f"     ❌ Archivo no encontrado: {e}")
            
            except RuntimeError as e:
                # Capturar errores de CUDA Out of Memory específicamente
                if "out of memory" in str(e).lower():
                    logger.error(f"     ❌ GPU Out of Memory: {e}")
                    logger.error(f"     💡 Sugerencia: Reduce el dataset o usa batch=1")
                    
                    # Intentar recuperar
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                else:
                    logger.exception(f"     ❌ Error de Runtime: {e}")
            
            except Exception as e:
                logger.exception(f"     ❌ Error: {e}")
    
    # --- 5. GUARDAR RESULTADOS ---
    output_file = path_manager.get("output") / "data" / "benchmark_results.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=4)
    
    logger.info(f"\n{'=' * 70}")
    logger.info(f"💾 RESULTADOS GUARDADOS")
    logger.info(f"   {output_file}")
    logger.info(f"{'=' * 70}\n")
    
    # --- 6. RESUMEN ---
    logger.info("📊 RESUMEN DE RESULTADOS:\n")
    
    for ds_name, results in results_data["datasets"].items():
        if results:
            logger.info(f"  {ds_name}:")
            
            # Encontrar mejor modelo
            best_model = max(results, key=lambda x: x['mAP50-95'])
            
            logger.info(f"    🏆 Mejor modelo: {best_model['model']}")
            logger.info(f"       mAP50-95: {best_model['mAP50-95']:.4f}")
            logger.info(f"       Speed: {best_model['Inference_Time_ms']:.2f}ms\n")


if __name__ == "__main__":
    import logging
    
    try:
        run_benchmark()
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Benchmark interrumpido por el usuario")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"\n❌ Error fatal en benchmark: {e}")
        sys.exit(1)