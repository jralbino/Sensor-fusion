"""Tests for ByteTrack 2D and 3D trackers."""

import numpy as np
import pytest

from tracking.bytetrack import BaseTrack
from tracking.tracker_2d import ByteTracker2D, _iou_2d_batch
from tracking.tracker_3d import ByteTracker3D, _iou_bev_batch
from tracking.kalman_2d import KalmanFilter2D
from tracking.kalman_3d import KalmanFilter3D


@pytest.fixture(autouse=True)
def reset_track_ids():
    """Reset track ID counter between tests."""
    BaseTrack.reset_id_counter()
    yield
    BaseTrack.reset_id_counter()


# -----------------------------------------------------------------------
# 2D IoU
# -----------------------------------------------------------------------

class TestIoU2D:
    def test_identical_boxes(self):
        boxes = np.array([[10, 10, 50, 50]])
        iou = _iou_2d_batch(boxes, boxes)
        np.testing.assert_allclose(iou, [[1.0]], atol=1e-6)

    def test_no_overlap(self):
        a = np.array([[0, 0, 10, 10]])
        b = np.array([[20, 20, 30, 30]])
        iou = _iou_2d_batch(a, b)
        assert iou[0, 0] == 0.0

    def test_partial_overlap(self):
        a = np.array([[0, 0, 10, 10]])
        b = np.array([[5, 5, 15, 15]])
        iou = _iou_2d_batch(a, b)
        # Intersection = 5*5=25, union = 100+100-25=175
        np.testing.assert_allclose(iou[0, 0], 25 / 175, atol=1e-6)

    def test_batch(self):
        a = np.array([[0, 0, 10, 10], [20, 20, 30, 30]])
        b = np.array([[0, 0, 10, 10]])
        iou = _iou_2d_batch(a, b)
        assert iou.shape == (2, 1)
        np.testing.assert_allclose(iou[0, 0], 1.0, atol=1e-6)
        assert iou[1, 0] == 0.0


# -----------------------------------------------------------------------
# BEV IoU (3D)
# -----------------------------------------------------------------------

class TestIoUBEV:
    def test_identical_boxes(self):
        boxes = np.array([[0, 0, 0, 4, 2, 1.5, 0]])
        iou = _iou_bev_batch(boxes, boxes)
        np.testing.assert_allclose(iou, [[1.0]], atol=1e-6)

    def test_no_overlap(self):
        a = np.array([[0, 0, 0, 2, 2, 1, 0]])
        b = np.array([[100, 100, 0, 2, 2, 1, 0]])
        iou = _iou_bev_batch(a, b)
        assert iou[0, 0] == 0.0

    def test_rotated_box_aabb(self):
        # A box rotated 90 degrees should produce the same AABB as swapping l/w
        a = np.array([[0, 0, 0, 4, 2, 1, 0]])
        b = np.array([[0, 0, 0, 4, 2, 1, np.pi / 2]])
        iou = _iou_bev_batch(a, b)
        # The AABBs differ, so IoU < 1
        assert 0 < iou[0, 0] < 1.0


# -----------------------------------------------------------------------
# Kalman Filters
# -----------------------------------------------------------------------

class TestKalman2D:
    def test_predict_update_cycle(self):
        kf = KalmanFilter2D()
        kf.initiate(np.array([100, 100, 1.5, 50]))
        for _ in range(5):
            kf.predict()
            kf.update(np.array([100, 100, 1.5, 50]))
        pos = kf.position
        np.testing.assert_allclose(pos[:2], [100, 100], atol=1.0)

    def test_velocity_tracking(self):
        kf = KalmanFilter2D()
        kf.initiate(np.array([0, 0, 1.0, 50]))
        # Object moves right at 10 px/frame
        for i in range(1, 20):
            kf.predict()
            kf.update(np.array([10 * i, 0, 1.0, 50]))
        pos = kf.position
        assert pos[0] > 180  # should be close to 190


class TestKalman3D:
    def test_predict_update_cycle(self):
        kf = KalmanFilter3D()
        kf.initiate(np.array([10, 20, 0, 4, 2, 1.5, 0.5]))
        for _ in range(5):
            kf.predict()
            kf.update(np.array([10, 20, 0, 4, 2, 1.5, 0.5]))
        pos = kf.position
        np.testing.assert_allclose(pos[:3], [10, 20, 0], atol=0.5)

    def test_angle_normalization(self):
        kf = KalmanFilter3D()
        kf.initiate(np.array([0, 0, 0, 4, 2, 1.5, 3.0]))
        kf.predict()
        kf.update(np.array([0, 0, 0, 4, 2, 1.5, 3.2]))
        assert -np.pi <= kf.position[6] <= np.pi


# -----------------------------------------------------------------------
# 2D Tracker
# -----------------------------------------------------------------------

class TestByteTracker2D:
    def test_single_object_track(self):
        tracker = ByteTracker2D(high_thresh=0.5, min_hits=1)
        det = np.array([[100, 100, 200, 200]])
        scores = np.array([0.9])
        labels = np.array([0])

        tracks = tracker.update(det, scores, labels)
        assert len(tracks) == 1
        assert tracks[0].track_id == 1

    def test_consistent_id(self):
        tracker = ByteTracker2D(high_thresh=0.5, min_hits=1)
        for i in range(5):
            det = np.array([[100 + i * 5, 100, 200 + i * 5, 200]])
            scores = np.array([0.9])
            labels = np.array([0])
            tracks = tracker.update(det, scores, labels)

        assert len(tracks) == 1
        assert tracks[0].track_id == 1  # same ID throughout

    def test_two_objects(self):
        tracker = ByteTracker2D(high_thresh=0.5, min_hits=1)
        dets = np.array([[10, 10, 50, 50], [200, 200, 300, 300]])
        scores = np.array([0.9, 0.8])
        labels = np.array([0, 1])

        tracks = tracker.update(dets, scores, labels)
        assert len(tracks) == 2
        ids = {t.track_id for t in tracks}
        assert len(ids) == 2  # distinct IDs

    def test_lost_track_removal(self):
        tracker = ByteTracker2D(high_thresh=0.5, max_age=3, min_hits=1)

        # Create a track
        det = np.array([[100, 100, 200, 200]])
        tracker.update(det, np.array([0.9]), np.array([0]))

        # No detections for max_age + 1 frames
        for _ in range(5):
            tracks = tracker.update(
                np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)
            )

        # Track should be removed
        assert len(tracker.tracks) == 0

    def test_low_conf_association(self):
        tracker = ByteTracker2D(
            high_thresh=0.6, low_thresh=0.2, min_hits=1
        )

        # Frame 1: high-conf detection creates track
        det1 = np.array([[100, 100, 200, 200]])
        tracks = tracker.update(det1, np.array([0.9]), np.array([0]))
        assert len(tracks) == 1
        tid = tracks[0].track_id

        # Frame 2: same position but low confidence — should still match
        det2 = np.array([[102, 102, 202, 202]])
        tracks = tracker.update(det2, np.array([0.3]), np.array([0]))
        assert len(tracks) == 1
        assert tracks[0].track_id == tid  # same track

    def test_empty_input(self):
        tracker = ByteTracker2D()
        tracks = tracker.update(
            np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)
        )
        assert len(tracks) == 0


# -----------------------------------------------------------------------
# 3D Tracker
# -----------------------------------------------------------------------

class TestByteTracker3D:
    def test_single_object_track(self):
        tracker = ByteTracker3D(high_thresh=0.3, min_hits=1)
        det = np.array([[10, 20, 0, 4, 2, 1.5, 0.5]])
        scores = np.array([0.8])
        labels = np.array([0])

        tracks = tracker.update(det, scores, labels)
        assert len(tracks) == 1

    def test_consistent_id_across_frames(self):
        tracker = ByteTracker3D(high_thresh=0.3, min_hits=1)
        for i in range(5):
            det = np.array([[10 + i * 0.5, 20, 0, 4, 2, 1.5, 0.5]])
            tracks = tracker.update(det, np.array([0.8]), np.array([0]))

        assert len(tracks) == 1
        assert tracks[0].track_id == 1

    def test_track_history(self):
        tracker = ByteTracker3D(high_thresh=0.3, min_hits=1)
        for i in range(5):
            det = np.array([[i * 2, 0, 0, 4, 2, 1.5, 0]])
            tracks = tracker.update(det, np.array([0.8]), np.array([0]))

        assert len(tracks[0].history) == 5

    def test_empty_input(self):
        tracker = ByteTracker3D()
        tracks = tracker.update(
            np.empty((0, 7)), np.empty(0), np.empty(0, dtype=int)
        )
        assert len(tracks) == 0


# -----------------------------------------------------------------------
# Integration: multi-frame scenario
# -----------------------------------------------------------------------

class TestMultiFrameScenario:
    def test_crossing_objects_2d(self):
        """Two objects crossing paths should maintain separate IDs."""
        tracker = ByteTracker2D(high_thresh=0.5, min_hits=1)

        # Two objects start far apart
        ids_frame1 = set()
        for frame in range(10):
            # Object A moves right, Object B moves left
            ax1 = 10 + frame * 10
            bx1 = 500 - frame * 10
            dets = np.array([
                [ax1, 100, ax1 + 50, 150],
                [bx1, 100, bx1 + 50, 150],
            ])
            scores = np.array([0.9, 0.9])
            labels = np.array([0, 0])
            tracks = tracker.update(dets, scores, labels)

            if frame == 0:
                ids_frame1 = {t.track_id for t in tracks}

        # Should still have exactly 2 tracks
        assert len(tracks) == 2
        final_ids = {t.track_id for t in tracks}
        assert final_ids == ids_frame1  # same IDs throughout
