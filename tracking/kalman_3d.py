"""
3D Kalman Filter for bounding box tracking.

State: [x, y, z, l, w, h, yaw, vx, vy, vz]  (10-dim)
Measurement: [x, y, z, l, w, h, yaw]  (7-dim)
"""

import numpy as np


def _normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


class KalmanFilter3D:
    """Constant-velocity Kalman filter for 3D bounding boxes."""

    def __init__(self):
        self.ndim_meas = 7   # x, y, z, l, w, h, yaw
        self.ndim_state = 10  # + vx, vy, vz

        dt = 1.0

        # State transition matrix
        self.F = np.eye(self.ndim_state)
        # x += vx, y += vy, z += vz
        self.F[0, 7] = dt
        self.F[1, 8] = dt
        self.F[2, 9] = dt

        # Measurement matrix
        self.H = np.zeros((self.ndim_meas, self.ndim_state))
        for i in range(self.ndim_meas):
            self.H[i, i] = 1.0

        self.x = np.zeros(self.ndim_state)
        self.P = np.eye(self.ndim_state)

    def initiate(self, measurement: np.ndarray):
        """Initialize state from first measurement [x,y,z,l,w,h,yaw]."""
        self.x[:7] = measurement
        self.x[7:] = 0.0  # zero velocity

        std = [
            1.0, 1.0, 1.0,     # position uncertainty
            0.5, 0.5, 0.5,     # size uncertainty
            0.1,                # yaw uncertainty
            2.0, 2.0, 1.0,     # velocity uncertainty
        ]
        self.P = np.diag(np.square(std))

    def predict(self):
        """Predict next state."""
        std = [
            0.5, 0.5, 0.5,     # position process noise
            0.1, 0.1, 0.1,     # size process noise (small — sizes are stable)
            0.05,               # yaw process noise
            1.0, 1.0, 0.5,     # velocity process noise
        ]
        Q = np.diag(np.square(std))

        self.x = self.F @ self.x
        self.x[6] = _normalize_angle(self.x[6])
        self.P = self.F @ self.P @ self.F.T + Q

    def update(self, measurement: np.ndarray):
        """Update state with new measurement [x,y,z,l,w,h,yaw]."""
        std = [
            0.3, 0.3, 0.3,     # measurement noise
            0.2, 0.2, 0.2,     # size noise
            0.05,               # yaw noise
        ]
        R = np.diag(np.square(std))

        y = measurement - self.H @ self.x
        # Normalize yaw residual
        y[6] = _normalize_angle(y[6])

        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.x[6] = _normalize_angle(self.x[6])
        I_KH = np.eye(self.ndim_state) - K @ self.H
        self.P = I_KH @ self.P

    @property
    def position(self) -> np.ndarray:
        """Return [x, y, z, l, w, h, yaw]."""
        return self.x[:7].copy()
