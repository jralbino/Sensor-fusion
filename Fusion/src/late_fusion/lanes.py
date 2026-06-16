"""Optional lane / drivable-area overlay for the fusion camera views.

The late-fusion pipeline is LiDAR-anchored: 3D boxes drive association and the
camera contributes 2D boxes for class refinement. Lane lines are a *camera-only*
semantic layer that does not participate in association — they are a perception
overlay painted under the projected 3D boxes so the fusion videos show the road
context (drivable area + lane markings) alongside the tracked objects.

We reuse the Vision YOLOP segmentation head (drivable area + lane line masks).
YOLOP is built inside the Vision module context (see ``pipeline`` phase 1); once
built it is a plain torch module and runs in any module context.
"""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


class LaneEstimator:
    """Thin wrapper around the Vision YOLOP detector exposing just the masks."""

    def __init__(self, detector):
        self._detector = detector

    @classmethod
    def build(cls) -> "LaneEstimator":
        """Construct the YOLOP detector.

        MUST be called inside ``module_loader.use_module("Vision")`` so that the
        top-level ``src`` package resolves to ``Vision/src``.

        We import ``YOLOPDetector`` directly rather than via ``lane_factory`` on
        purpose: the factory eagerly imports *every* lane detector (SegFormer,
        DeepLab, …), dragging in heavy deps like ``transformers`` that the fusion
        stack doesn't otherwise need. YOLOP only needs torch + its torch.hub repo.
        """
        from src.lanes.yolop_detector import YOLOPDetector

        return cls(YOLOPDetector())

    def masks(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """``(drivable_area, lane_line)`` uint8 masks at the image resolution."""
        return self._detector.infer_masks(img_bgr)


# BGR overlay colours (matplotlib converts BGR→RGB before display).
_DRIVABLE_BGR = (0, 150, 0)
_LANE_BGR = (0, 0, 255)


def overlay_lanes(
    img_bgr: np.ndarray,
    da_mask: np.ndarray = None,
    ll_mask: np.ndarray = None,
    da_alpha: float = 0.30,
    lane_thickness: int = 6,
) -> np.ndarray:
    """Blend a drivable-area fill and lane-line strokes onto ``img_bgr``.

    Pure NumPy/OpenCV; returns a *new* image and leaves the input untouched.
    Either mask may be ``None`` (skipped). Masks must match ``img_bgr`` in
    height and width.
    """
    out = img_bgr.copy()
    if da_mask is not None and da_mask.any():
        m = da_mask.astype(bool)
        fill = np.empty_like(out)
        fill[:] = _DRIVABLE_BGR
        out[m] = (out[m] * (1.0 - da_alpha) + fill[m] * da_alpha).astype(out.dtype)
    if ll_mask is not None and ll_mask.any():
        m = (ll_mask > 0).astype(np.uint8)
        if lane_thickness > 1:
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (lane_thickness, lane_thickness)
            )
            m = cv2.dilate(m, k)
        out[m.astype(bool)] = _LANE_BGR
    return out
