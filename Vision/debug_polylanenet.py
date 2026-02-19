#!/usr/bin/env python3
"""
Diagnóstico de PolyLaneNet vs Ground Truth BDD100K.

Genera una imagen de comparación:
  - Panel izquierdo: Ground Truth (polylines de BDD100K)
  - Panel derecho:   Predicciones de PolyLaneNet (con confianzas)

Uso:
  Vision/venv/bin/python Vision/debug_polylanenet.py --idx 0
"""

import sys
import json
import argparse
import numpy as np
import cv2
from pathlib import Path

# ── rutas del proyecto ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
VISION_DIR   = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(VISION_DIR) not in sys.path:
    sys.path.insert(0, str(VISION_DIR))

VAL_IMG_DIR  = VISION_DIR / "data" / "raw" / "bdd100k" / "images" / "100k" / "val"
VAL_LBL_DIR  = VISION_DIR / "data" / "raw" / "bdd100k" / "labels" / "100k" / "val"
MODEL_PATH   = VISION_DIR / "models" / "model_2305.pt"


# ── Ground Truth ─────────────────────────────────────────────────────────────

def draw_gt_lanes(img_bgr: np.ndarray, json_path: Path) -> np.ndarray:
    """Dibuja las lanes GT de BDD100K (poly2d) en la imagen."""
    canvas = img_bgr.copy()
    with open(json_path) as f:
        data = json.load(f)

    # BDD100K JSON tiene una lista de frames
    frames = data.get("frames", [data])  # soporte para JSON plano también
    if not frames:
        return canvas
    frame = frames[0]

    lane_colors = {
        "lane/single white":  (200, 200, 200),
        "lane/double white":  (255, 255, 255),
        "lane/single yellow": (0,   200, 200),
        "lane/double yellow": (0,   255, 255),
        "lane/road curb":     (100, 180, 100),
        "lane/crosswalk":     (180, 100, 180),
    }
    default_color = (50, 255, 50)

    for obj in frame.get("objects", []):
        cat = obj.get("category", "")
        if "lane" not in cat:
            continue
        pts = obj.get("poly2d", [])
        if len(pts) < 2:
            continue
        color = lane_colors.get(cat, default_color)
        pixel_pts = [(int(p[0]), int(p[1])) for p in pts]
        for k in range(len(pixel_pts) - 1):
            cv2.line(canvas, pixel_pts[k], pixel_pts[k + 1], color, 3, cv2.LINE_AA)
        # Marca el primer punto
        cv2.circle(canvas, pixel_pts[0], 5, color, -1)

    return canvas


def count_gt_lanes(json_path: Path) -> int:
    with open(json_path) as f:
        data = json.load(f)
    frames = data.get("frames", [data])
    if not frames:
        return 0
    return sum(1 for o in frames[0].get("objects", []) if "lane" in o.get("category", ""))


# ── PolyLaneNet ───────────────────────────────────────────────────────────────

def run_polylanenet_debug(img_bgr: np.ndarray, model_path: Path):
    """
    Corre PolyLaneNet usando el detector REAL (con todos sus filtros)
    y también extrae los raw outputs para el resumen de consola.
    Devuelve:
      - imagen con líneas dibujadas (igual que el app.py)
      - lista de dicts con info de cada lane (para el resumen)
    """
    import torch
    from Vision.src.lanes.polylanenet_detector import PolyLaneNetDetector

    det = PolyLaneNetDetector(model_path=str(model_path))
    h, w = img_bgr.shape[:2]

    # ── 1. Extraer outputs raw (solo para el resumen de consola) ──
    img_r = cv2.resize(img_bgr, (det.input_width, det.input_height))
    img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img_r).permute(2, 0, 1).float() / 255.0
    t = ((t.unsqueeze(0).to(det.device) - det.mean) / det.std)
    with torch.no_grad():
        raw_out = det.model(t)[0].cpu().numpy().reshape(det.num_lanes, 7)

    lane_infos = []
    for i in range(det.num_lanes):
        conf       = float(1 / (1 + np.exp(-raw_out[i, 0])))
        y_min_norm = float(raw_out[i, 1])
        y_max_norm = float(raw_out[i, 2])
        # una línea "dibujable" si pasa los filtros del detector real
        passes_conf   = conf >= 0.3
        passes_ymin   = y_min_norm >= 0.20
        passes_range  = (y_max_norm - y_min_norm) >= (20 / h)
        lane_infos.append({
            "lane":       i,
            "conf":       conf,
            "y_min_norm": y_min_norm,
            "y_max_norm": y_max_norm,
            "y_min_px":   y_min_norm * h,
            "y_max_px":   y_max_norm * h,
            "coeffs":     raw_out[i, 3:].tolist(),
            "drawn":      False,          # se actualiza abajo
            "filtered_by": (
                "conf<0.3" if not passes_conf else
                "y_min<0.20" if not passes_ymin else
                "rango<20px" if not passes_range else
                "OK"
            ),
        })

    # ── 2. Usar detect() REAL para obtener la imagen anotada ──
    canvas, _ = det.detect(img_bgr)

    # Marcar en lane_infos cuáles pasaron todos los filtros
    for li in lane_infos:
        li["drawn"] = li["filtered_by"] == "OK"

    return canvas, lane_infos


# ── Visualización de comparación ──────────────────────────────────────────────

def add_legend(img: np.ndarray, title: str) -> np.ndarray:
    h, w = img.shape[:2]
    bar = np.zeros((40, w, 3), dtype=np.uint8)
    cv2.putText(bar, title, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return np.vstack([bar, img])


def build_comparison(gt_img, pred_img, lane_infos) -> np.ndarray:
    """Concatena GT (izq) y predicción (der) con metadata."""
    h = max(gt_img.shape[0], pred_img.shape[0])

    def pad(img):
        dh = h - img.shape[0]
        return np.pad(img, ((0, dh), (0, 0), (0, 0)))

    combined = np.hstack([pad(gt_img), pad(pred_img)])

    # Panel de info debajo
    info_h  = 30 * (len(lane_infos) + 2)
    info_w  = combined.shape[1]
    info_panel = np.zeros((info_h, info_w, 3), dtype=np.uint8)

    cv2.putText(info_panel, "Lane | Conf  | y_min_px | y_max_px | Drawn",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    for j, li in enumerate(lane_infos):
        line = (f"  L{li['lane']}   | {li['conf']:.3f} | "
                f"{li['y_min_px']:7.1f}  | {li['y_max_px']:7.1f}  | "
                f"{'YES' if li['drawn'] else 'no '}")
        color = (50, 255, 50) if li["drawn"] else (120, 120, 120)
        cv2.putText(info_panel, line, (10, 22 + (j + 1) * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return np.vstack([combined, info_panel])


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PolyLaneNet debug vs BDD100K GT")
    parser.add_argument("--idx", type=int, default=0,
                        help="Índice del archivo en val (0, 1, 2, …)")
    parser.add_argument("--conf", type=float, default=0.3,
                        help="Umbral de confianza para dibujar (default=0.3)")
    parser.add_argument("--out", type=str, default="debug_polylanenet_comparison.jpg",
                        help="Ruta de la imagen de salida")
    args = parser.parse_args()

    # Archivos disponibles
    json_files = sorted(VAL_LBL_DIR.glob("*.json"))
    img_files  = sorted(VAL_IMG_DIR.glob("*.jpg"))

    if not json_files:
        print(f"ERROR: No se encontraron JSONs en {VAL_LBL_DIR}")
        sys.exit(1)

    idx = args.idx % len(json_files)
    json_path = json_files[idx]
    stem      = json_path.stem

    # Buscar imagen correspondiente
    img_path = VAL_IMG_DIR / f"{stem}.jpg"
    if not img_path.exists():
        # fallback
        if img_files:
            img_path = img_files[idx % len(img_files)]
        else:
            print(f"ERROR: No se encontró imagen para {stem}")
            sys.exit(1)

    print(f"Imagen : {img_path.name}  ({img_path.stat().st_size // 1024} KB)")
    print(f"Label  : {json_path.name}")
    print(f"GT lanes: {count_gt_lanes(json_path)}")

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"ERROR: No se pudo leer {img_path}")
        sys.exit(1)
    print(f"Resolución: {img_bgr.shape[1]}×{img_bgr.shape[0]}")

    # ── Ground Truth ──
    gt_canvas = draw_gt_lanes(img_bgr, json_path)
    gt_canvas = add_legend(gt_canvas, f"GT BDD100K — {stem}")

    # ── PolyLaneNet ──
    if not MODEL_PATH.exists():
        print(f"ERROR: Modelo no encontrado: {MODEL_PATH}")
        sys.exit(1)

    print("\nCorriendo PolyLaneNet…")
    pred_canvas, lane_infos = run_polylanenet_debug(img_bgr, MODEL_PATH)
    pred_canvas = add_legend(pred_canvas, f"PolyLaneNet (conf≥{args.conf}) — {stem}")

    # ── Comparación ──
    comparison = build_comparison(gt_canvas, pred_canvas, lane_infos)
    out_path = Path(args.out)
    cv2.imwrite(str(out_path), comparison)
    print(f"\nImagen guardada: {out_path.resolve()}")

    # ── Resumen en consola ──
    print("\n─── Resumen de predicciones PolyLaneNet ───")
    print(f"{'L':>2} | {'Conf':>6} | {'y_min_px':>9} | {'y_max_px':>9} | {'Filtro':>12} | Dibujada")
    print("─" * 68)
    for li in lane_infos:
        flag   = "SI" if li["drawn"] else "—"
        filtro = li.get("filtered_by", "?")
        print(f"L{li['lane']} | {li['conf']:6.3f} | {li['y_min_px']:9.1f} | "
              f"{li['y_max_px']:9.1f} | {filtro:>12} | {flag}")

    drawn = [li for li in lane_infos if li["drawn"]]
    print(f"\nTotal dibujadas: {len(drawn)}/5")

    if not drawn:
        print("\nATENCIÓN: Ninguna línea pasó los filtros.")
        print("  → El modelo no encuentra carriles válidos en esta imagen.")
        print("  → Considera usar YOLOP o UFLD para BDD100K (mejor generalización).")
    else:
        print("\nLíneas proyectadas en la imagen GT:")
        for li in drawn:
            print(f"  L{li['lane']}: y=[{li['y_min_px']:.0f}–{li['y_max_px']:.0f}]px"
                  f"  y_norm=[{li['y_min_norm']:.2f}–{li['y_max_norm']:.2f}]")


if __name__ == "__main__":
    main()
