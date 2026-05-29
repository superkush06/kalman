"""Standard Kalman filter for linear-Gaussian state-space models."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class KalmanFilter:
    """Linear-Gaussian state-space model:

        x_{k+1} = F x_k + B u_k + w_k,  w_k ~ N(0, Q)
        z_k    = H x_k + v_k,            v_k ~ N(0, R)

    Maintains posterior (mean, cov) over time.
    """

    F: np.ndarray  # state-transition
    H: np.ndarray  # observation
    Q: np.ndarray  # process noise covariance
    R: np.ndarray  # observation noise covariance
    x: np.ndarray  # initial state mean
    P: np.ndarray  # initial state covariance
    B: np.ndarray | None = None  # control-input model (optional)
    history: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.F = np.asarray(self.F, dtype=float)
        self.H = np.asarray(self.H, dtype=float)
        self.Q = np.asarray(self.Q, dtype=float)
        self.R = np.asarray(self.R, dtype=float)
        self.x = np.asarray(self.x, dtype=float).reshape(-1)
        self.P = np.asarray(self.P, dtype=float)
        if self.B is not None:
            self.B = np.asarray(self.B, dtype=float)
        self._validate()

    def _validate(self) -> None:
        n = self.x.shape[0]
        if self.F.shape != (n, n):
            raise ValueError(f"F must be ({n},{n})")
        if self.P.shape != (n, n):
            raise ValueError(f"P must be ({n},{n})")
        if self.Q.shape != (n, n):
            raise ValueError(f"Q must be ({n},{n})")
        if self.H.shape[1] != n:
            raise ValueError(f"H must have {n} columns")
        m = self.H.shape[0]
        if self.R.shape != (m, m):
            raise ValueError(f"R must be ({m},{m})")

    def predict(self, u: np.ndarray | None = None) -> None:
        if self.B is not None and u is not None:
            self.x = self.F @ self.x + self.B @ np.asarray(u, dtype=float)
        else:
            self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z: np.ndarray) -> None:
        z = np.asarray(z, dtype=float).reshape(-1)
        y = z - self.H @ self.x                       # innovation
        S = self.H @ self.P @ self.H.T + self.R       # innovation cov
        K = self.P @ self.H.T @ np.linalg.inv(S)      # Kalman gain
        self.x = self.x + K @ y
        eye = np.eye(self.P.shape[0])
        # Joseph form for numerical stability:
        self.P = (eye - K @ self.H) @ self.P @ (eye - K @ self.H).T + K @ self.R @ K.T
        self.history.append((self.x.copy(), self.P.copy()))

    def step(self, z: np.ndarray, u: np.ndarray | None = None) -> None:
        self.predict(u)
        self.update(z)
