"""Unit tests for the camera lane / drivable-area overlay (`lanes.py`).

Only the pure NumPy/OpenCV blending in :func:`overlay_lanes` is exercised; the
YOLOP model itself needs torch + the Vision stack and is covered by the live
pipeline. Run inside the fusion container or any env with numpy + opencv + pytest:
    docker compose run --rm fusion python -m pytest Fusion/tests/test_lanes.py -v
"""
import os
import sys

import numpy as np

# Make `src.late_fusion` importable (Fusion/ on path → src = Fusion/src).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.late_fusion.lanes import overlay_lanes  # noqa: E402


def _blank(h=40, w=60):
    return np.full((h, w, 3), 100, dtype=np.uint8)


def test_overlay_none_is_noop():
    img = _blank()
    out = overlay_lanes(img, None, None)
    assert out is not img                      # returns a copy
    assert np.array_equal(out, img)            # nothing painted


def test_overlay_does_not_mutate_input():
    img = _blank()
    da = np.ones(img.shape[:2], dtype=np.uint8)
    before = img.copy()
    overlay_lanes(img, da, None)
    assert np.array_equal(img, before)


def test_drivable_area_blend_only_inside_mask():
    img = _blank()
    da = np.zeros(img.shape[:2], dtype=np.uint8)
    da[:20, :] = 1                             # top half is drivable
    out = overlay_lanes(img, da, None, da_alpha=0.30)
    # Top half shifted toward green (G up), bottom half untouched.
    assert out[0, 0, 1] > img[0, 0, 1]
    assert np.array_equal(out[30, 0], img[30, 0])


def test_lane_lines_painted_red_and_dilated():
    img = _blank()
    ll = np.zeros(img.shape[:2], dtype=np.uint8)
    ll[20, 30] = 1                             # single lane pixel
    out = overlay_lanes(img, None, ll, lane_thickness=6)
    # The exact pixel is solid red (BGR = 0,0,255)...
    assert tuple(out[20, 30]) == (0, 0, 255)
    # ...and dilation spreads it to neighbours.
    assert tuple(out[20, 28]) == (0, 0, 255)
    assert int((np.all(out == (0, 0, 255), axis=2)).sum()) > 1


def test_empty_masks_are_noop():
    img = _blank()
    zeros = np.zeros(img.shape[:2], dtype=np.uint8)
    out = overlay_lanes(img, zeros, zeros)
    assert np.array_equal(out, img)
