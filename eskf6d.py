"""6D attitude ESKF runtime with prediction only.

The nominal state is ``[q_WB, b_g]`` and the right-multiplicative error state
ordering is ``[delta_theta, delta_b_g]``. This module intentionally contains no
observation residual, Kalman update, injection, or reset.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from config import DEFAULT_CONFIG
from initialization import ESKF6DInitialization
from rotation_utils import (
    hat,
    normalize_quaternion,
    quaternion_multiply,
    rotvec_to_quaternion,
)


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ESKF6DState:
    """Immutable copy of the mathematical filter state."""

    q_xyzw: FloatArray
    bg_rad_s: FloatArray
    P: FloatArray


@dataclass(frozen=True)
class PredictionDebug:
    """One-step prediction quantities retained for development-time review."""

    dt_s: float
    gyro_measurement_rad_s: FloatArray
    omega_hat_rad_s: FloatArray
    delta_theta_rad: FloatArray
    q_before_xyzw: FloatArray
    q_after_xyzw: FloatArray
    Fc: FloatArray
    Gc: FloatArray
    Phi: FloatArray
    Qd: FloatArray
    P_diag_before: FloatArray
    P_diag_after: FloatArray
    q_norm: float
    P_symmetry_error: float


def _finite_vector(value: ArrayLike, size: int, name: str) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.array(array, dtype=float, copy=True)


def _covariance_matrix(value: ArrayLike, name: str) -> FloatArray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (6, 6):
        raise ValueError(f"{name} must have shape (6, 6), got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-15):
        raise ValueError(f"{name} must be symmetric")
    tolerance = DEFAULT_CONFIG.numerical.covariance_negative_tolerance
    if np.any(np.diag(matrix) < -tolerance):
        raise ValueError(f"{name} has a clearly negative diagonal entry")
    return np.array(matrix, dtype=float, copy=True)


class ESKF6D:
    """Maintain and predict only the mathematical 6D attitude/bias state.

    IMU indices, timestamps, and pose matching belong to the caller/runner.
    """

    def __init__(self, initialization: ESKF6DInitialization) -> None:
        self._q_xyzw = normalize_quaternion(initialization.q0_xyzw)
        self._bg_rad_s = _finite_vector(
            initialization.bg0_rad_s, 3, "initialization.bg0_rad_s"
        )
        self._P = _covariance_matrix(initialization.P0, "initialization.P0")
        self._Qc = _covariance_matrix(initialization.Qc, "initialization.Qc")

    @classmethod
    def from_initialization(
        cls, initialization: ESKF6DInitialization
    ) -> "ESKF6D":
        """Create runtime state without recomputing any initialization quantity."""

        return cls(initialization)

    @property
    def Qc(self) -> FloatArray:
        """Return a copy of the fixed continuous-time process-noise covariance."""

        return self._Qc.copy()

    def get_state(self) -> ESKF6DState:
        """Return an immutable snapshot whose arrays do not alias internal state."""

        return ESKF6DState(
            q_xyzw=self._q_xyzw.copy(),
            bg_rad_s=self._bg_rad_s.copy(),
            P=self._P.copy(),
        )

    def predict(self, gyro_rad_s: ArrayLike, dt_s: float) -> PredictionDebug:
        """Apply one left-endpoint ZOH mathematical prediction.

        The caller owns scheduling and supplies the selected left-endpoint gyro
        measurement and interval. State mutation occurs only after the complete
        candidate state passes numerical validation.
        """

        gyro = _finite_vector(gyro_rad_s, 3, "gyro_rad_s")
        dt = float(dt_s)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt_s must be finite and strictly positive")

        q_before = self._q_xyzw.copy()
        P_before = self._P.copy()

        omega_hat = gyro - self._bg_rad_s
        delta_theta = omega_hat * dt
        delta_q = rotvec_to_quaternion(delta_theta)
        q_after = normalize_quaternion(
            quaternion_multiply(q_before, delta_q)
        )

        Fc = np.zeros((6, 6), dtype=float)
        Fc[0:3, 0:3] = -hat(omega_hat)
        Fc[0:3, 3:6] = -np.eye(3, dtype=float)

        Gc = np.zeros((6, 6), dtype=float)
        Gc[0:3, 0:3] = -np.eye(3, dtype=float)
        Gc[3:6, 3:6] = np.eye(3, dtype=float)

        Phi = np.eye(6, dtype=float) + Fc * dt
        Qd = Gc @ self._Qc @ Gc.T * dt
        P_after = Phi @ P_before @ Phi.T + Qd
        P_after = 0.5 * (P_after + P_after.T)
        P_after = _covariance_matrix(P_after, "predicted P")

        q_norm = float(np.linalg.norm(q_after))
        symmetry_error = float(np.max(np.abs(P_after - P_after.T)))

        self._q_xyzw = q_after
        self._P = P_after

        return PredictionDebug(
            dt_s=dt,
            gyro_measurement_rad_s=gyro,
            omega_hat_rad_s=omega_hat,
            delta_theta_rad=delta_theta,
            q_before_xyzw=q_before,
            q_after_xyzw=q_after.copy(),
            Fc=Fc,
            Gc=Gc,
            Phi=Phi,
            Qd=Qd,
            P_diag_before=np.diag(P_before).copy(),
            P_diag_after=np.diag(P_after).copy(),
            q_norm=q_norm,
            P_symmetry_error=symmetry_error,
        )
