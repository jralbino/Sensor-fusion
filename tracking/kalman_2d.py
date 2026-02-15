"""
2D Kalman Filter for bounding box tracking.

State: [cx, cy, aspect_ratio, h, vx, vy, va, vh]  (8-dim)
Measurement: [cx, cy, a, h]  (4-dim)
"""

import numpy as np


class KalmanFilter2D:
    """Constant-velocity Kalman filter for 2D bounding boxes (xyah parameterization)."""

    def __init__(self):
        ndim = 4
        dt = 1.0

        # State transition: position + velocity
        self.F = np.eye(2 * ndim)
        for i in range(ndim):
            self.F[i, ndim + i] = dt

        # Measurement matrix: observe position only
        self.H = np.eye(ndim, 2 * ndim)

        # Process noise covariance
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

        self.x = np.zeros(2 * ndim)  # state
        self.P = np.eye(2 * ndim)    # covariance

    def initiate(self, measurement: np.ndarray):
        """Initialize state from first measurement [cx, cy, a, h]."""
        self.x[:4] = measurement
        self.x[4:] = 0.0  # zero velocity

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        self.P = np.diag(np.square(std))

    def predict(self):
        """Predict next state."""
        h = self.x[3]
        std_pos = [
            self._std_weight_position * h,
            self._std_weight_position * h,
            1e-2,
            self._std_weight_position * h,
        ]
        std_vel = [
            self._std_weight_velocity * h,
            self._std_weight_velocity * h,
            1e-5,
            self._std_weight_velocity * h,
        ]
        Q = np.diag(np.square(std_pos + std_vel))

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + Q

    def update(self, measurement: np.ndarray):
        """Update state with new measurement [cx, cy, a, h]."""
        h = self.x[3]
        std = [
            self._std_weight_position * h,
            self._std_weight_position * h,
            1e-1,
            self._std_weight_position * h,
        ]
        R = np.diag(np.square(std))

        y = measurement - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        I_KH = np.eye(len(self.x)) - K @ self.H
        self.P = I_KH @ self.P

    @property
    def position(self) -> np.ndarray:
        """Return [cx, cy, a, h]."""
        return self.x[:4].copy()
