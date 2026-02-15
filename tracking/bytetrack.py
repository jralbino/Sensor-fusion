"""
ByteTrack Core — Two-threshold association with Kalman filtering.

Based on: ByteTrack: Multi-Object Tracking by Associating Every Detection Box
(Zhang et al., ECCV 2022)
"""

from abc import ABC, abstractmethod
from enum import IntEnum
from typing import List, Tuple, Dict, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment


class TrackState(IntEnum):
    NEW = 0
    TRACKED = 1
    LOST = 2
    REMOVED = 3


class BaseTrack(ABC):
    """Abstract base class for a single tracked object."""

    _next_id = 1

    def __init__(self, detection: np.ndarray, score: float, label: int):
        self.track_id = BaseTrack._next_id
        BaseTrack._next_id += 1
        self.state = TrackState.NEW
        self.score = score
        self.label = label
        self.age = 0
        self.hits = 1
        self.time_since_update = 0
        self.history: List[np.ndarray] = []

    @abstractmethod
    def predict(self):
        """Advance the Kalman state one step (no measurement)."""

    @abstractmethod
    def update(self, detection: np.ndarray, score: float):
        """Update the Kalman state with a new detection."""

    @abstractmethod
    def get_state(self) -> np.ndarray:
        """Return current estimated state in detection format."""

    @classmethod
    def reset_id_counter(cls):
        cls._next_id = 1


class ByteTracker(ABC):
    """Abstract ByteTrack tracker — handles the two-threshold association loop.

    Subclasses implement ``_create_track`` and ``_compute_iou_matrix``.

    Args:
        high_thresh: Confidence threshold separating high/low detections.
        low_thresh: Minimum confidence for low-confidence association.
        match_thresh: IoU threshold for association (cost = 1 - IoU).
        max_age: Frames a LOST track survives before REMOVED.
        min_hits: Minimum hits before a NEW track is promoted to TRACKED.
    """

    def __init__(
        self,
        high_thresh: float = 0.5,
        low_thresh: float = 0.1,
        match_thresh: float = 0.3,
        max_age: int = 30,
        min_hits: int = 3,
        distance_thresh: Optional[float] = None,
    ):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.distance_thresh = distance_thresh
        self.tracks: List[BaseTrack] = []
        self.frame_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        detections: np.ndarray,
        scores: np.ndarray,
        labels: np.ndarray,
    ) -> List[BaseTrack]:
        """Run one tracking step.

        Args:
            detections: (N, D) detection array (xyxy for 2D, 7-param for 3D).
            scores: (N,) confidence scores.
            labels: (N,) integer class labels.

        Returns:
            List of active tracks (state == TRACKED or NEW with enough hits).
        """
        self.frame_count += 1

        if len(detections) == 0:
            detections = np.empty((0, self._det_dim()))
            scores = np.empty(0)
            labels = np.empty(0, dtype=int)

        # --- Split detections into high / low confidence ---
        high_mask = scores >= self.high_thresh
        low_mask = (scores >= self.low_thresh) & (~high_mask)

        high_dets = detections[high_mask]
        high_scores = scores[high_mask]
        high_labels = labels[high_mask]

        low_dets = detections[low_mask]
        low_scores = scores[low_mask]
        low_labels = labels[low_mask]

        # --- Predict all existing tracks ---
        for t in self.tracks:
            t.predict()
            t.age += 1
            t.time_since_update += 1

        # --- First association: high-conf dets <-> all tracks ---
        matched_h, unmatched_tracks_h, unmatched_dets_h = self._associate(
            self.tracks, high_dets, self.match_thresh,
        )

        # Update matched tracks
        for t_idx, d_idx in matched_h:
            self.tracks[t_idx].update(high_dets[d_idx], high_scores[d_idx])
            self.tracks[t_idx].state = TrackState.TRACKED
            self.tracks[t_idx].time_since_update = 0
            self.tracks[t_idx].hits += 1
            self.tracks[t_idx].score = high_scores[d_idx]

        # --- Second association: low-conf dets <-> unmatched tracks ---
        remaining_tracks = [self.tracks[i] for i in unmatched_tracks_h]
        matched_l, unmatched_tracks_l, _ = self._associate(
            remaining_tracks, low_dets, self.match_thresh,
        )

        for rt_idx, d_idx in matched_l:
            t = remaining_tracks[rt_idx]
            t.update(low_dets[d_idx], low_scores[d_idx])
            t.state = TrackState.TRACKED
            t.time_since_update = 0
            t.hits += 1
            t.score = low_scores[d_idx]

        # Mark truly unmatched tracks as LOST
        for rt_idx in unmatched_tracks_l:
            t = remaining_tracks[rt_idx]
            if t.state != TrackState.LOST:
                t.state = TrackState.LOST

        # --- Create new tracks for unmatched high-conf detections ---
        for d_idx in unmatched_dets_h:
            new_track = self._create_track(
                high_dets[d_idx], high_scores[d_idx], int(high_labels[d_idx]),
            )
            self.tracks.append(new_track)

        # --- Remove old lost tracks ---
        kept = []
        for t in self.tracks:
            if t.state == TrackState.LOST and t.time_since_update > self.max_age:
                t.state = TrackState.REMOVED
            if t.state != TrackState.REMOVED:
                kept.append(t)
        self.tracks = kept

        # --- Return active tracks ---
        active = []
        for t in self.tracks:
            if t.state == TrackState.TRACKED or (
                t.state == TrackState.NEW and t.hits >= self.min_hits
            ):
                t.history.append(t.get_state().copy())
                active.append(t)
        return active

    # ------------------------------------------------------------------
    # Association helpers
    # ------------------------------------------------------------------

    def _associate(
        self,
        tracks: List[BaseTrack],
        detections: np.ndarray,
        thresh: float,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Hungarian matching between tracks and detections.

        Uses IoU as primary cost. If ``distance_thresh`` is set, unmatched
        tracks and detections are re-associated using center distance as a
        fallback (useful for low-framerate sequences where IoU drops to 0).

        Returns:
            (matched_pairs, unmatched_track_indices, unmatched_det_indices)
        """
        if len(tracks) == 0 or len(detections) == 0:
            return [], list(range(len(tracks))), list(range(len(detections)))

        # Get track states in detection format
        track_states = np.array([t.get_state() for t in tracks])
        iou_matrix = self._compute_iou_matrix(track_states, detections)
        cost_matrix = 1.0 - iou_matrix

        # Hungarian assignment
        row_idx, col_idx = linear_sum_assignment(cost_matrix)

        matched = []
        unmatched_tracks = set(range(len(tracks)))
        unmatched_dets = set(range(len(detections)))

        for r, c in zip(row_idx, col_idx):
            if iou_matrix[r, c] < thresh:
                continue
            matched.append((r, c))
            unmatched_tracks.discard(r)
            unmatched_dets.discard(c)

        # --- Distance fallback for remaining unmatched ---
        if self.distance_thresh is not None and unmatched_tracks and unmatched_dets:
            um_t = sorted(unmatched_tracks)
            um_d = sorted(unmatched_dets)
            dist_matrix = self._compute_distance_matrix(
                track_states[um_t], detections[um_d],
            )
            d_row, d_col = linear_sum_assignment(dist_matrix)
            for r, c in zip(d_row, d_col):
                if dist_matrix[r, c] <= self.distance_thresh:
                    matched.append((um_t[r], um_d[c]))
                    unmatched_tracks.discard(um_t[r])
                    unmatched_dets.discard(um_d[c])

        return matched, sorted(unmatched_tracks), sorted(unmatched_dets)

    # ------------------------------------------------------------------
    # Abstract interface for subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    def _create_track(
        self, detection: np.ndarray, score: float, label: int,
    ) -> BaseTrack:
        """Create a new track from a detection."""

    @abstractmethod
    def _compute_iou_matrix(
        self, tracks: np.ndarray, detections: np.ndarray,
    ) -> np.ndarray:
        """Compute IoU matrix between track states and detections.

        Args:
            tracks: (M, D) array of track states.
            detections: (N, D) array of detections.

        Returns:
            (M, N) IoU matrix.
        """

    @abstractmethod
    def _det_dim(self) -> int:
        """Return the detection dimensionality (4 for 2D, 7 for 3D)."""

    def _compute_distance_matrix(
        self, tracks: np.ndarray, detections: np.ndarray,
    ) -> np.ndarray:
        """Compute normalized center-distance matrix (fallback for IoU).

        Subclasses should override this. Default raises NotImplementedError
        so that ``distance_thresh`` is only used when explicitly supported.
        """
        raise NotImplementedError("Subclass must implement _compute_distance_matrix")
