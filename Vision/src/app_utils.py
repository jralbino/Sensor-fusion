"""
Utility functions for the Vision Streamlit app.

Extracted from app.py so they can be unit-tested without importing Streamlit.
"""

import cv2
import numpy as np
from typing import List, Dict

# Deterministic track colour palette (20 BGR colours)
TRACK_PALETTE = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
    (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
    (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255),
    (170, 110, 40), (255, 250, 200), (128, 0, 0), (170, 255, 195),
    (128, 128, 0), (255, 215, 180), (0, 0, 128), (128, 128, 128),
]


def generate_color(class_name: str) -> tuple:
    """Generate a unique BGR color based on the hash of the class name."""
    hash_val = hash(class_name)
    r = (hash_val & 0xFF0000) >> 16
    g = (hash_val & 0x00FF00) >> 8
    b = hash_val & 0x0000FF
    return (b, g, r)


def get_optimal_font_color(bg_color_bgr: tuple) -> tuple:
    """Calculate the optimal text color based on the background luminance."""
    b, g, r = bg_color_bgr
    luminance = (0.299 * r + 0.587 * g + 0.114 * b)
    return (0, 0, 0) if luminance > 140 else (255, 255, 255)


def draw_custom_boxes(img: np.ndarray, detections: List[Dict]) -> np.ndarray:
    """Draw bounding boxes with labels on the image.

    Each detection dict must have keys: 'bbox' (x1,y1,x2,y2),
    'class_name', 'confidence'. Optional: 'track_id'.

    Returns a copy of the image with boxes drawn.
    """
    canvas = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2

    for det in detections:
        x1, y1, x2, y2 = map(int, det['bbox'])
        name = det['class_name']
        conf = det['confidence']
        tid = det.get('track_id')

        if tid is not None:
            box_color = TRACK_PALETTE[int(tid) % len(TRACK_PALETTE)]
            label = f"ID:{tid} {name} {conf:.2f}"
        else:
            box_color = generate_color(name)
            label = f"{name} {conf:.2f}"
        text_color = get_optimal_font_color(box_color)

        cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, 2)

        (w, h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
        top_pos = y1 - h - 10 if y1 - h - 10 > 0 else y1 + h + 5
        text_pos_y = y1 - 5 if y1 - h - 10 > 0 else y1 + h + 5

        cv2.rectangle(canvas, (x1, top_pos), (x1 + w + 5, top_pos + h + 10), box_color, -1)
        cv2.putText(canvas, label, (x1 + 2, text_pos_y), font, font_scale,
                    text_color, font_thickness, cv2.LINE_AA)

    return canvas


def detections_to_tracker_format(detections: List[Dict]):
    """Convert detection dicts to arrays for ByteTracker2D.

    Args:
        detections: list of dicts with 'bbox', 'confidence', 'class_name'.

    Returns:
        (dets_arr, scores_arr, labels_arr, idx_to_name) or None if empty.
    """
    if not detections:
        return None

    dets_arr = np.array([d['bbox'] for d in detections])
    scores_arr = np.array([d['confidence'] for d in detections])
    class_names_list = sorted(set(d['class_name'] for d in detections))
    name_to_idx = {n: i for i, n in enumerate(class_names_list)}
    labels_arr = np.array([name_to_idx[d['class_name']] for d in detections])
    idx_to_name = {i: n for n, i in name_to_idx.items()}

    return dets_arr, scores_arr, labels_arr, idx_to_name


def tracker_output_to_detections(active_tracks, idx_to_name: dict) -> List[Dict]:
    """Convert ByteTracker2D output back to detection dicts.

    Args:
        active_tracks: list of Track2D objects from tracker.update().
        idx_to_name: mapping from label index to class name.

    Returns:
        List of detection dicts with 'track_id' field.
    """
    result = []
    for t in active_tracks:
        state = t.get_state()
        result.append({
            'bbox': state.tolist(),
            'class_name': idx_to_name.get(t.label, 'unknown'),
            'confidence': t.score,
            'track_id': t.track_id,
        })
    return result
