"""6D attitude ESKF prediction and attitude-observation update.

The nominal state is ``[q_WB, b_g]`` and the right-multiplicative error state
ordering is ``[delta_theta, delta_b_g]``. Scheduling and pose selection remain
the caller's responsibility; this module contains no index or timestamp state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from config import DEFAULT_CONFIG, NumericalSafetyConfig
from initialization import ESKF6DInitialization
from rotation_utils import (
    hat,
    normalize_quaternion,
    quaternion_inverse,
    quaternion_multiply,
    quaternion_to_rotvec,
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


@dataclass(frozen=True)
class UpdateDebug:
    """One attitude update, injection, and reset for development review."""

    q_pred_xyzw: FloatArray
    q_obs_raw_xyzw: FloatArray
    q_obs_aligned_xyzw: FloatArray
    quaternion_sign_flipped: bool
    q_rel_xyzw: FloatArray
    r_theta_rad: FloatArray
    residual_norm: float
    R_attitude: FloatArray
    S: FloatArray
    K: FloatArray
    delta_x: FloatArray
    delta_theta_rad: FloatArray
    delta_bg_rad_s: FloatArray
    q_after_xyzw: FloatArray
    bg_before_rad_s: FloatArray
    bg_after_rad_s: FloatArray
    P_diag_before: FloatArray
    P_diag_after_joseph: FloatArray
    P_diag_after_reset: FloatArray
    G_reset: FloatArray
    raw_covariance_symmetry_error: float
    final_covariance_symmetry_error: float
    q_norm: float
    NIS: float


def _finite_vector(value: ArrayLike, size: int, name: str) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.array(array, dtype=float, copy=True)


def _symmetric_matrix(
    value: ArrayLike,
    shape: tuple[int, int],
    name: str,
    numerical: NumericalSafetyConfig,
) -> FloatArray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    symmetry_error = float(np.max(np.abs(matrix - matrix.T)))
    if symmetry_error > numerical.covariance_symmetry_tolerance:
        raise ValueError(
            f"{name} symmetry error {symmetry_error:.3e} exceeds "
            f"{numerical.covariance_symmetry_tolerance:.3e}"
        )
    return 0.5 * (matrix + matrix.T)


def _covariance_matrix(
    value: ArrayLike,
    name: str,
    numerical: NumericalSafetyConfig,
) -> FloatArray:
    matrix = _symmetric_matrix(value, (6, 6), name, numerical)
    if np.any(np.diag(matrix) < -numerical.covariance_negative_tolerance):
        raise ValueError(f"{name} has a clearly negative diagonal entry")
    return matrix


class ESKF6D:
    """Maintain and predict only the mathematical 6D attitude/bias state.

    IMU indices, timestamps, and pose matching belong to the caller/runner.
    """

    def __init__(
        self,
        initialization: ESKF6DInitialization,
        numerical: NumericalSafetyConfig = DEFAULT_CONFIG.numerical,
    ) -> None:
        self._numerical = numerical
        self._q_xyzw = normalize_quaternion(
            initialization.q0_xyzw, self._numerical
        )
        self._bg_rad_s = _finite_vector(
            initialization.bg0_rad_s, 3, "initialization.bg0_rad_s"
        )
        self._P = _covariance_matrix(
            initialization.P0, "initialization.P0", self._numerical
        )
        self._Qc = _covariance_matrix(
            initialization.Qc, "initialization.Qc", self._numerical
        )

    @classmethod
    def from_initialization(
        cls,
        initialization: ESKF6DInitialization,
        numerical: NumericalSafetyConfig = DEFAULT_CONFIG.numerical,
    ) -> "ESKF6D":
        """Create runtime state without recomputing any initialization quantity."""

        return cls(initialization, numerical)

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
        delta_q = rotvec_to_quaternion(delta_theta, self._numerical)
        q_after = normalize_quaternion(
            quaternion_multiply(q_before, delta_q), self._numerical
        )

        Fc = np.zeros((6, 6), dtype=float)
        Fc[0:3, 0:3] = -hat(omega_hat)
        Fc[0:3, 3:6] = -np.eye(3, dtype=float)

        Gc = np.zeros((6, 6), dtype=float)
        Gc[0:3, 0:3] = -np.eye(3, dtype=float)
        Gc[3:6, 3:6] = np.eye(3, dtype=float)

        Phi = np.eye(6, dtype=float) + Fc * dt
        Qd = Gc @ self._Qc @ Gc.T * dt
        P_after_raw = Phi @ P_before @ Phi.T + Qd
        P_after = _covariance_matrix(
            P_after_raw, "predicted P", self._numerical
        )

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

    def update_attitude(
        self,
        q_obs_xyzw: ArrayLike,
        R_attitude: ArrayLike,
    ) -> UpdateDebug:
        """Apply one attitude observation, injection, and error-coordinate reset.

        Observation selection, timestamps, and pose indices remain entirely
        outside this mathematical update. The operation is atomic: internal
        state is committed only after every candidate quantity is validated.
        """

        q_pred = self._q_xyzw.copy()
        bg_before = self._bg_rad_s.copy()
        P_pred = self._P.copy()

        q_obs_raw = _finite_vector(q_obs_xyzw, 4, "q_obs_xyzw")
        q_obs_normalized = normalize_quaternion(q_obs_raw, self._numerical)
        sign_flipped = bool(np.dot(q_pred, q_obs_normalized) < 0.0)
        q_obs_aligned = (
            -q_obs_normalized if sign_flipped else q_obs_normalized
        )

        R = _symmetric_matrix(
            R_attitude, (3, 3), "R_attitude", self._numerical
        )
        try:
            np.linalg.cholesky(R)
        except np.linalg.LinAlgError as exc:
            raise ValueError("R_attitude must be positive definite") from exc

        q_rel = normalize_quaternion(
            quaternion_multiply(
                quaternion_inverse(q_pred, self._numerical),
                q_obs_aligned,
            ),
            self._numerical,
        )
        r_theta = quaternion_to_rotvec(q_rel, self._numerical)
        residual_norm = float(np.linalg.norm(r_theta))

        H = np.zeros((3, 6), dtype=float)
        H[:, 0:3] = np.eye(3, dtype=float)
        PHt = P_pred @ H.T
        S = _symmetric_matrix(
            H @ PHt + R, (3, 3), "innovation covariance S", self._numerical
        )
        try:
            np.linalg.cholesky(S)
            K = np.linalg.solve(S, PHt.T).T
            innovation_solution = np.linalg.solve(S, r_theta)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "innovation covariance S must be positive definite and solvable"
            ) from exc
        if not np.all(np.isfinite(K)):
            raise ValueError("Kalman gain contains a non-finite value")

        delta_x = K @ r_theta
        if not np.all(np.isfinite(delta_x)):
            raise ValueError("delta_x contains a non-finite value")
        delta_theta = delta_x[0:3]
        delta_bg = delta_x[3:6]

        identity_6 = np.eye(6, dtype=float)
        A = identity_6 - K @ H
        P_joseph_raw = A @ P_pred @ A.T + K @ R @ K.T
        P_joseph = _covariance_matrix(
            P_joseph_raw, "Joseph-updated P", self._numerical
        )

        delta_q = rotvec_to_quaternion(delta_theta, self._numerical)
        q_after = normalize_quaternion(
            quaternion_multiply(q_pred, delta_q), self._numerical
        )
        bg_after = bg_before + delta_bg
        if not np.all(np.isfinite(bg_after)):
            raise ValueError("injected gyro bias contains a non-finite value")

        G_reset = identity_6.copy()
        G_reset[0:3, 0:3] = (
            np.eye(3, dtype=float) - 0.5 * hat(delta_theta)
        )
        P_reset_raw = G_reset @ P_joseph @ G_reset.T
        raw_symmetry_error = float(
            np.max(np.abs(P_reset_raw - P_reset_raw.T))
        )
        if raw_symmetry_error > self._numerical.covariance_symmetry_tolerance:
            raise ValueError(
                "reset covariance symmetry error "
                f"{raw_symmetry_error:.3e} exceeds "
                f"{self._numerical.covariance_symmetry_tolerance:.3e}"
            )

        # Remove only the tiny asymmetry introduced by floating-point matrix
        # arithmetic. This is not an additional Kalman update and must never
        # conceal a materially asymmetric covariance (checked immediately above).
        P_reset_symmetric = 0.5 * (P_reset_raw + P_reset_raw.T)
        P_reset = _covariance_matrix(
            P_reset_symmetric, "reset P", self._numerical
        )
        final_symmetry_error = float(
            np.max(np.abs(P_reset - P_reset.T))
        )
        q_norm = float(np.linalg.norm(q_after))
        NIS = float(r_theta @ innovation_solution)
        if not np.isfinite(NIS):
            raise ValueError("NIS is not finite")

        self._q_xyzw = q_after
        self._bg_rad_s = bg_after
        self._P = P_reset

        return UpdateDebug(
            q_pred_xyzw=q_pred,
            q_obs_raw_xyzw=q_obs_raw,
            q_obs_aligned_xyzw=q_obs_aligned.copy(),
            quaternion_sign_flipped=sign_flipped,
            q_rel_xyzw=q_rel,
            r_theta_rad=r_theta,
            residual_norm=residual_norm,
            R_attitude=R,
            S=S,
            K=K,
            delta_x=delta_x,
            delta_theta_rad=delta_theta.copy(),
            delta_bg_rad_s=delta_bg.copy(),
            q_after_xyzw=q_after.copy(),
            bg_before_rad_s=bg_before,
            bg_after_rad_s=bg_after.copy(),
            P_diag_before=np.diag(P_pred).copy(),
            P_diag_after_joseph=np.diag(P_joseph).copy(),
            P_diag_after_reset=np.diag(P_reset).copy(),
            G_reset=G_reset,
            raw_covariance_symmetry_error=raw_symmetry_error,
            final_covariance_symmetry_error=final_symmetry_error,
            q_norm=q_norm,
            NIS=NIS,
        )
