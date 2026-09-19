"""15D inertial ESKF nominal-state and covariance prediction only.

The nominal state is ``[p_W, v_W, q_WB, b_g_B, b_a_B]`` and the
right-multiplicative error-state ordering is
``[delta_p, delta_v, delta_theta, delta_b_g, delta_b_a]``. Scheduling,
timestamps, pose observations, update, injection, and reset remain outside
this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from config import DEFAULT_CONFIG, GRAVITY_WORLD_MPS2, NumericalSafetyConfig
from initialization import ESKF15DInitialization
from rotation_utils import (
    hat,
    normalize_quaternion,
    quaternion_multiply,
    quaternion_to_rotation_matrix,
    rotvec_to_quaternion,
)


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ESKF15DState:
    """Immutable copy of the mathematical 15D filter state."""

    p_W_m: FloatArray
    v_W_mps: FloatArray
    q_xyzw: FloatArray
    bg_rad_s: FloatArray
    ba_mps2: FloatArray
    P: FloatArray


@dataclass(frozen=True)
class PredictionDebug:
    """One-step 15D prediction quantities for development-time review."""

    dt_s: float
    gyro_measurement_rad_s: FloatArray
    acc_measurement_mps2: FloatArray
    omega_hat_rad_s: FloatArray
    f_hat_B_mps2: FloatArray
    a_hat_W_mps2: FloatArray
    p_before_W_m: FloatArray
    p_after_W_m: FloatArray
    v_before_W_mps: FloatArray
    v_after_W_mps: FloatArray
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


def _finite_vector(value: ArrayLike, name: str) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.array(array, dtype=float, copy=True)


def _covariance_matrix(
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
    matrix = 0.5 * (matrix + matrix.T)
    if np.any(np.diag(matrix) < -numerical.covariance_negative_tolerance):
        raise ValueError(f"{name} has a clearly negative diagonal entry")
    return matrix


class ESKF15D:
    """Maintain and predict the mathematical 15D inertial state only."""

    def __init__(
        self,
        initialization: ESKF15DInitialization,
        numerical: NumericalSafetyConfig = DEFAULT_CONFIG.numerical,
    ) -> None:
        self._numerical = numerical
        self._p_W_m = _finite_vector(initialization.p0_W_m, "initialization.p0_W_m")
        self._v_W_mps = _finite_vector(
            initialization.v0_W_mps, "initialization.v0_W_mps"
        )
        self._q_xyzw = normalize_quaternion(
            initialization.q0_xyzw, self._numerical
        )
        self._bg_rad_s = _finite_vector(
            initialization.bg0_rad_s, "initialization.bg0_rad_s"
        )
        self._ba_mps2 = _finite_vector(
            initialization.ba0_mps2, "initialization.ba0_mps2"
        )
        self._P = _covariance_matrix(
            initialization.P0, (15, 15), "initialization.P0", self._numerical
        )
        self._Qc = _covariance_matrix(
            initialization.Qc, (12, 12), "initialization.Qc", self._numerical
        )

    @classmethod
    def from_initialization(
        cls,
        initialization: ESKF15DInitialization,
        numerical: NumericalSafetyConfig = DEFAULT_CONFIG.numerical,
    ) -> "ESKF15D":
        """Create runtime state without recomputing initialization."""

        return cls(initialization, numerical)

    @property
    def Qc(self) -> FloatArray:
        """Return a copy of the fixed continuous-time process covariance."""

        return self._Qc.copy()

    def get_state(self) -> ESKF15DState:
        """Return a snapshot whose arrays do not alias internal state."""

        return ESKF15DState(
            p_W_m=self._p_W_m.copy(),
            v_W_mps=self._v_W_mps.copy(),
            q_xyzw=self._q_xyzw.copy(),
            bg_rad_s=self._bg_rad_s.copy(),
            ba_mps2=self._ba_mps2.copy(),
            P=self._P.copy(),
        )

    def predict(
        self,
        gyro_rad_s: ArrayLike,
        acc_mps2: ArrayLike,
        dt_s: float,
    ) -> PredictionDebug:
        """Apply one left-endpoint ZOH nominal and covariance prediction."""

        gyro = _finite_vector(gyro_rad_s, "gyro_rad_s")
        acc = _finite_vector(acc_mps2, "acc_mps2")
        dt = float(dt_s)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt_s must be finite and strictly positive")

        # Left-endpoint ZOH: every quantity below is derived from state[k].
        p_before = self._p_W_m.copy()
        v_before = self._v_W_mps.copy()
        q_before = self._q_xyzw.copy()
        bg_before = self._bg_rad_s.copy()
        ba_before = self._ba_mps2.copy()
        P_before = self._P.copy()

        omega_hat = gyro - bg_before
        f_hat_B = acc - ba_before
        R_before = quaternion_to_rotation_matrix(q_before)
        a_hat_W = R_before @ f_hat_B + GRAVITY_WORLD_MPS2

        p_after = _finite_vector(
            p_before + v_before * dt + 0.5 * a_hat_W * dt**2,
            "predicted position",
        )
        v_after = _finite_vector(
            v_before + a_hat_W * dt,
            "predicted velocity",
        )

        # This is the nominal IMU rotation increment, not error-state delta_theta.
        delta_theta_imu = omega_hat * dt
        delta_q = rotvec_to_quaternion(delta_theta_imu, self._numerical)
        q_after = normalize_quaternion(
            quaternion_multiply(q_before, delta_q), self._numerical
        )

        # Nominal biases are constant; random walks drive covariance only.
        bg_after = bg_before.copy()
        ba_after = ba_before.copy()

        identity_3 = np.eye(3, dtype=float)
        Fc = np.zeros((15, 15), dtype=float)
        Fc[0:3, 3:6] = identity_3
        Fc[3:6, 6:9] = -R_before @ hat(f_hat_B)
        Fc[3:6, 12:15] = -R_before
        Fc[6:9, 6:9] = -hat(omega_hat)
        Fc[6:9, 9:12] = -identity_3

        # Continuous noise ordering is fixed as [n_a, n_g, n_bg, n_ba].
        Gc = np.zeros((15, 12), dtype=float)
        Gc[3:6, 0:3] = -R_before
        Gc[6:9, 3:6] = -identity_3
        Gc[9:12, 6:9] = identity_3
        Gc[12:15, 9:12] = identity_3

        # Match the established 6D first-order discretization. In particular,
        # this O(dt) Qd has a zero position-position process-noise block.
        Phi = np.eye(15, dtype=float) + Fc * dt
        Qd = Gc @ self._Qc @ Gc.T * dt
        P_after_raw = Phi @ P_before @ Phi.T + Qd
        P_after = _covariance_matrix(
            P_after_raw, (15, 15), "predicted P", self._numerical
        )

        q_norm = float(np.linalg.norm(q_after))
        symmetry_error = float(np.max(np.abs(P_after - P_after.T)))

        # Commit only after the complete candidate state has passed validation.
        self._p_W_m = p_after
        self._v_W_mps = v_after
        self._q_xyzw = q_after
        self._bg_rad_s = bg_after
        self._ba_mps2 = ba_after
        self._P = P_after

        return PredictionDebug(
            dt_s=dt,
            gyro_measurement_rad_s=gyro,
            acc_measurement_mps2=acc,
            omega_hat_rad_s=omega_hat,
            f_hat_B_mps2=f_hat_B,
            a_hat_W_mps2=a_hat_W,
            p_before_W_m=p_before,
            p_after_W_m=p_after.copy(),
            v_before_W_mps=v_before,
            v_after_W_mps=v_after.copy(),
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
