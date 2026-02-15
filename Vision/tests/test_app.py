"""Tests for Vision/app.py utility functions and tracking integration."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Vision.src.app_utils import (
    generate_color,
    get_optimal_font_color,
    draw_custom_boxes,
    detections_to_tracker_format,
    tracker_output_to_detections,
    TRACK_PALETTE,
)
from tracking.bytetrack import BaseTrack


@pytest.fixture(autouse=True)
def reset_track_ids():
    BaseTrack.reset_id_counter()
    yield
    BaseTrack.reset_id_counter()


@pytest.fixture
def blank_image():
    """480x640 black BGR image."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_detections():
    return [
        {'bbox': [100, 100, 200, 200], 'class_name': 'car', 'confidence': 0.92},
        {'bbox': [300, 150, 400, 300], 'class_name': 'person', 'confidence': 0.85},
        {'bbox': [50, 50, 120, 180], 'class_name': 'car', 'confidence': 0.70},
    ]


@pytest.fixture
def tracked_detections():
    return [
        {'bbox': [100, 100, 200, 200], 'class_name': 'car', 'confidence': 0.92, 'track_id': 1},
        {'bbox': [300, 150, 400, 300], 'class_name': 'person', 'confidence': 0.85, 'track_id': 5},
    ]


# -----------------------------------------------------------------------
# generate_color
# -----------------------------------------------------------------------

class TestGenerateColor:
    def test_returns_bgr_tuple(self):
        color = generate_color('car')
        assert isinstance(color, tuple)
        assert len(color) == 3

    def test_deterministic(self):
        assert generate_color('car') == generate_color('car')

    def test_different_classes_different_colors(self):
        c1 = generate_color('car')
        c2 = generate_color('person')
        # Very unlikely to collide
        assert c1 != c2

    def test_values_in_range(self):
        for name in ['car', 'truck', 'bus', 'pedestrian', 'bicycle']:
            b, g, r = generate_color(name)
            assert 0 <= b <= 255
            assert 0 <= g <= 255
            assert 0 <= r <= 255


# -----------------------------------------------------------------------
# get_optimal_font_color
# -----------------------------------------------------------------------

class TestGetOptimalFontColor:
    def test_dark_background_returns_white(self):
        # Pure black background → white text
        assert get_optimal_font_color((0, 0, 0)) == (255, 255, 255)

    def test_light_background_returns_black(self):
        # Pure white background → black text
        assert get_optimal_font_color((255, 255, 255)) == (0, 0, 0)

    def test_mid_dark_returns_white(self):
        # Dark blue → white text
        assert get_optimal_font_color((200, 0, 0)) == (255, 255, 255)

    def test_yellow_returns_black(self):
        # Bright yellow (high luminance) → black text
        assert get_optimal_font_color((0, 255, 255)) == (0, 0, 0)


# -----------------------------------------------------------------------
# draw_custom_boxes — without tracking
# -----------------------------------------------------------------------

class TestDrawCustomBoxes:
    def test_returns_copy(self, blank_image, sample_detections):
        result = draw_custom_boxes(blank_image, sample_detections)
        assert result is not blank_image
        assert result.shape == blank_image.shape

    def test_does_not_modify_original(self, blank_image, sample_detections):
        original = blank_image.copy()
        draw_custom_boxes(blank_image, sample_detections)
        np.testing.assert_array_equal(blank_image, original)

    def test_draws_something(self, blank_image, sample_detections):
        result = draw_custom_boxes(blank_image, sample_detections)
        # The result should have non-zero pixels (boxes drawn on black)
        assert result.sum() > 0

    def test_empty_detections(self, blank_image):
        result = draw_custom_boxes(blank_image, [])
        # Should return an identical copy
        np.testing.assert_array_equal(result, blank_image)

    def test_single_detection(self, blank_image):
        det = [{'bbox': [50, 50, 150, 150], 'class_name': 'cat', 'confidence': 0.99}]
        result = draw_custom_boxes(blank_image, det)
        assert result.sum() > 0

    def test_box_at_image_edge(self, blank_image):
        det = [{'bbox': [0, 0, 10, 10], 'class_name': 'car', 'confidence': 0.5}]
        result = draw_custom_boxes(blank_image, det)
        assert result.shape == blank_image.shape

    def test_large_bbox(self, blank_image):
        h, w = blank_image.shape[:2]
        det = [{'bbox': [0, 0, w, h], 'class_name': 'bus', 'confidence': 0.8}]
        result = draw_custom_boxes(blank_image, det)
        assert result.shape == blank_image.shape


# -----------------------------------------------------------------------
# draw_custom_boxes — with track_id
# -----------------------------------------------------------------------

class TestDrawCustomBoxesTracking:
    def test_track_id_changes_color(self, blank_image):
        det_no_track = [{'bbox': [100, 100, 200, 200], 'class_name': 'car', 'confidence': 0.9}]
        det_with_track = [{'bbox': [100, 100, 200, 200], 'class_name': 'car', 'confidence': 0.9, 'track_id': 3}]

        img_no_track = draw_custom_boxes(blank_image, det_no_track)
        img_with_track = draw_custom_boxes(blank_image, det_with_track)

        # Different coloring logic → images should differ
        assert not np.array_equal(img_no_track, img_with_track)

    def test_different_track_ids_different_colors(self, blank_image):
        det1 = [{'bbox': [100, 100, 200, 200], 'class_name': 'car', 'confidence': 0.9, 'track_id': 0}]
        det2 = [{'bbox': [100, 100, 200, 200], 'class_name': 'car', 'confidence': 0.9, 'track_id': 1}]

        img1 = draw_custom_boxes(blank_image, det1)
        img2 = draw_custom_boxes(blank_image, det2)

        assert not np.array_equal(img1, img2)

    def test_track_id_wraps_palette(self, blank_image):
        """track_id larger than palette size should wrap without error."""
        det = [{'bbox': [50, 50, 150, 150], 'class_name': 'car', 'confidence': 0.7, 'track_id': 999}]
        result = draw_custom_boxes(blank_image, det)
        assert result.sum() > 0

    def test_mixed_tracked_and_untracked(self, blank_image):
        dets = [
            {'bbox': [10, 10, 50, 50], 'class_name': 'car', 'confidence': 0.9, 'track_id': 1},
            {'bbox': [200, 200, 300, 300], 'class_name': 'person', 'confidence': 0.8},
        ]
        result = draw_custom_boxes(blank_image, dets)
        assert result.sum() > 0


# -----------------------------------------------------------------------
# detections_to_tracker_format
# -----------------------------------------------------------------------

class TestDetectionsToTrackerFormat:
    def test_basic(self, sample_detections):
        result = detections_to_tracker_format(sample_detections)
        assert result is not None
        dets_arr, scores_arr, labels_arr, idx_to_name = result

        assert dets_arr.shape == (3, 4)
        assert scores_arr.shape == (3,)
        assert labels_arr.shape == (3,)
        assert len(idx_to_name) == 2  # 'car' and 'person'

    def test_empty_list(self):
        assert detections_to_tracker_format([]) is None

    def test_class_name_mapping(self, sample_detections):
        _, _, labels_arr, idx_to_name = detections_to_tracker_format(sample_detections)
        # All labels should map back to valid class names
        for l in labels_arr:
            assert int(l) in idx_to_name

    def test_single_class(self):
        dets = [
            {'bbox': [10, 10, 50, 50], 'class_name': 'dog', 'confidence': 0.9},
            {'bbox': [60, 60, 100, 100], 'class_name': 'dog', 'confidence': 0.7},
        ]
        _, _, labels_arr, idx_to_name = detections_to_tracker_format(dets)
        assert len(idx_to_name) == 1
        assert np.all(labels_arr == labels_arr[0])

    def test_scores_match_order(self, sample_detections):
        _, scores_arr, _, _ = detections_to_tracker_format(sample_detections)
        expected = [d['confidence'] for d in sample_detections]
        np.testing.assert_allclose(scores_arr, expected)


# -----------------------------------------------------------------------
# tracker_output_to_detections
# -----------------------------------------------------------------------

class TestTrackerOutputToDetections:
    def test_round_trip(self, sample_detections):
        """detections → tracker format → tracker.update → back to detections."""
        from tracking import ByteTracker2D

        tracker = ByteTracker2D(high_thresh=0.5, min_hits=1)
        fmt = detections_to_tracker_format(sample_detections)
        dets_arr, scores_arr, labels_arr, idx_to_name = fmt

        active = tracker.update(dets_arr, scores_arr, labels_arr)
        result = tracker_output_to_detections(active, idx_to_name)

        assert isinstance(result, list)
        assert len(result) > 0

        for det in result:
            assert 'bbox' in det
            assert 'class_name' in det
            assert 'confidence' in det
            assert 'track_id' in det
            assert isinstance(det['bbox'], list)
            assert len(det['bbox']) == 4
            assert det['class_name'] in ('car', 'person')
            assert det['track_id'] >= 1

    def test_empty_tracks(self):
        result = tracker_output_to_detections([], {0: 'car'})
        assert result == []

    def test_unknown_label(self):
        """Track with label not in idx_to_name maps to 'unknown'."""
        from tracking import ByteTracker2D

        tracker = ByteTracker2D(high_thresh=0.3, min_hits=1)
        dets = np.array([[10, 10, 50, 50]])
        active = tracker.update(dets, np.array([0.9]), np.array([99]))

        result = tracker_output_to_detections(active, {0: 'car'})
        assert result[0]['class_name'] == 'unknown'


# -----------------------------------------------------------------------
# End-to-end tracking across multiple frames
# -----------------------------------------------------------------------

class TestTrackingIntegration:
    def test_consistent_ids_across_frames(self):
        from tracking import ByteTracker2D

        tracker = ByteTracker2D(high_thresh=0.5, min_hits=1)

        # Frame 1: two objects
        dets1 = [
            {'bbox': [100, 100, 200, 200], 'class_name': 'car', 'confidence': 0.9},
            {'bbox': [400, 300, 500, 400], 'class_name': 'person', 'confidence': 0.8},
        ]
        fmt1 = detections_to_tracker_format(dets1)
        active1 = tracker.update(*fmt1[:3])
        result1 = tracker_output_to_detections(active1, fmt1[3])
        ids1 = {d['track_id'] for d in result1}

        # Frame 2: same objects moved slightly
        dets2 = [
            {'bbox': [105, 105, 205, 205], 'class_name': 'car', 'confidence': 0.88},
            {'bbox': [402, 302, 502, 402], 'class_name': 'person', 'confidence': 0.82},
        ]
        fmt2 = detections_to_tracker_format(dets2)
        active2 = tracker.update(*fmt2[:3])
        result2 = tracker_output_to_detections(active2, fmt2[3])
        ids2 = {d['track_id'] for d in result2}

        # Same IDs should persist
        assert ids1 == ids2

    def test_new_object_gets_new_id(self):
        from tracking import ByteTracker2D

        tracker = ByteTracker2D(high_thresh=0.5, min_hits=1)

        # Frame 1: one object
        dets1 = [{'bbox': [100, 100, 200, 200], 'class_name': 'car', 'confidence': 0.9}]
        fmt1 = detections_to_tracker_format(dets1)
        active1 = tracker.update(*fmt1[:3])
        result1 = tracker_output_to_detections(active1, fmt1[3])
        id1 = result1[0]['track_id']

        # Frame 2: original + a new far-away object
        dets2 = [
            {'bbox': [105, 105, 205, 205], 'class_name': 'car', 'confidence': 0.88},
            {'bbox': [500, 400, 600, 480], 'class_name': 'truck', 'confidence': 0.75},
        ]
        fmt2 = detections_to_tracker_format(dets2)
        active2 = tracker.update(*fmt2[:3])
        result2 = tracker_output_to_detections(active2, fmt2[3])

        ids2 = {d['track_id'] for d in result2}
        assert id1 in ids2  # old track persists
        assert len(ids2) == 2  # new track added

    def test_draw_after_tracking(self, blank_image):
        """Full pipeline: detect → track → draw with track IDs."""
        from tracking import ByteTracker2D

        tracker = ByteTracker2D(high_thresh=0.3, min_hits=1)
        dets = [
            {'bbox': [50, 50, 150, 150], 'class_name': 'car', 'confidence': 0.9},
            {'bbox': [300, 200, 400, 350], 'class_name': 'person', 'confidence': 0.7},
        ]
        fmt = detections_to_tracker_format(dets)
        active = tracker.update(*fmt[:3])
        tracked = tracker_output_to_detections(active, fmt[3])

        result = draw_custom_boxes(blank_image, tracked)
        assert result.sum() > 0
        # All detections should have track_id
        for d in tracked:
            assert 'track_id' in d


# -----------------------------------------------------------------------
# TRACK_PALETTE
# -----------------------------------------------------------------------

class TestTrackPalette:
    def test_has_20_colors(self):
        assert len(TRACK_PALETTE) == 20

    def test_all_bgr_tuples(self):
        for color in TRACK_PALETTE:
            assert isinstance(color, tuple)
            assert len(color) == 3
            for c in color:
                assert 0 <= c <= 255
