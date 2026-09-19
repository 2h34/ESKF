"""15D inertial ESKF prediction and FAST-LIO pose update.

The nominal state is ``[p_W, v_W, q_WB, b_g_B, b_a_B]`` and the
right-multiplicative error-state ordering is
``[delta_p, delta_v, delta_theta, delta_b_g, delta_b_a]``. Scheduling,
timestamps, and pose selection remain outside this mathematical module.
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
    quaternion_inverse,
    quaternion_multiply,
    quaternion_to_rotation_matrix,
    quaternion_to_rotvec,
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


@dataclass(frozen=True)
class UpdateDebug:
    """One 15D pose update, injection, and reset for development review."""

    p_pred_W_m: FloatArray
    q_pred_xyzw: FloatArray
    q_obs_raw_xyzw: FloatArray
    q_obs_aligned_xyzw: FloatArray
    quaternion_sign_flipped: bool
    q_rel_xyzw: FloatArray
    r_position_m: FloatArray
    r_theta_rad: FloatArray
    residual: FloatArray
    R_pose: FloatArray
    H: FloatArray
    S: FloatArray
    K: FloatArray
    delta_x: FloatArray
    delta_p_m: FloatArray
    delta_v_mps: FloatArray
    delta_theta_rad: FloatArray
    delta_bg_rad_s: FloatArray
    delta_ba_mps2: FloatArray
    p_after_W_m: FloatArray
    v_after_W_mps: FloatArray
    q_after_xyzw: FloatArray
    bg_after_rad_s: FloatArray
    ba_after_mps2: FloatArray
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
    shape: tuple[int, int],
    name: str,
    numerical: NumericalSafetyConfig,
) -> FloatArray:
    matrix = _symmetric_matrix(value, shape, name, numerical)
    if np.any(np.diag(matrix) < -numerical.covariance_negative_tolerance):
        raise ValueError(f"{name} has a clearly negative diagonal entry")
    return matrix


class ESKF15D:
    """Maintain the mathematical 15D inertial state and pose update."""

    def __init__(
        self,
        initialization: ESKF15DInitialization,
        numerical: NumericalSafetyConfig = DEFAULT_CONFIG.numerical,
    ) -> None:
        self._numerical = numerical
        self._p_W_m = _finite_vector(
            initialization.p0_W_m, 3, "initialization.p0_W_m"
        )
        self._v_W_mps = _finite_vector(
            initialization.v0_W_mps, 3, "initialization.v0_W_mps"
        )
        self._q_xyzw = normalize_quaternion(
            initialization.q0_xyzw, self._numerical
        )
        self._bg_rad_s = _finite_vector(
            initialization.bg0_rad_s, 3, "initialization.bg0_rad_s"
        )
        self._ba_mps2 = _finite_vector(
            initialization.ba0_mps2, 3, "initialization.ba0_mps2"
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

        gyro = _finite_vector(gyro_rad_s, 3, "gyro_rad_s")
        acc = _finite_vector(acc_mps2, 3, "acc_mps2")
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
            3,
            "predicted position",
        )
        v_after = _finite_vector(
            v_before + a_hat_W * dt,
            3,
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

    def update_pose(
        self,
        p_obs_W_m: ArrayLike,
        q_obs_xyzw: ArrayLike,
        R_pose: ArrayLike,
    ) -> UpdateDebug:
        """Apply one FAST-LIO position/attitude update, injection, and reset."""

        p_pred = self._p_W_m.copy()
        v_pred = self._v_W_mps.copy()
        q_pred = self._q_xyzw.copy()
        bg_pred = self._bg_rad_s.copy()
        ba_pred = self._ba_mps2.copy()
        P_pred = self._P.copy()

        p_obs = _finite_vector(p_obs_W_m, 3, "p_obs_W_m")
        q_obs_raw = _finite_vector(q_obs_xyzw, 4, "q_obs_xyzw")
        q_obs_normalized = normalize_quaternion(q_obs_raw, self._numerical)

        # q and -q represent the same attitude. Align the observation to the
        # predicted quaternion before forming the relative rotation.
        sign_flipped = bool(np.dot(q_pred, q_obs_normalized) < 0.0)
        q_obs_aligned = (
            -q_obs_normalized if sign_flipped else q_obs_normalized
        )

        R = _symmetric_matrix(R_pose, (6, 6), "R_pose", self._numerical)
        try:
            np.linalg.cholesky(R)
        except np.linalg.LinAlgError as exc:
            raise ValueError("R_pose must be positive definite") from exc

        r_position = p_obs - p_pred
        # Right-error convention: this rotation points from prediction to
        # observation, q_rel = q_pred^{-1} tensor-product q_obs.
        q_rel = normalize_quaternion(
            quaternion_multiply(
                quaternion_inverse(q_pred, self._numerical),
                q_obs_aligned,
            ),
            self._numerical,
        )
        r_theta = quaternion_to_rotvec(q_rel, self._numerical)
        residual = np.concatenate((r_position, r_theta))

        H = np.zeros((6, 15), dtype=float)
        H[0:3, 0:3] = np.eye(3, dtype=float)
        H[3:6, 6:9] = np.eye(3, dtype=float)
        PHt = P_pred @ H.T
        S = _symmetric_matrix(
            H @ PHt + R,
            (6, 6),
            "innovation covariance S",
            self._numerical,
        )
        try:
            np.linalg.cholesky(S)
            K = np.linalg.solve(S, PHt.T).T
            innovation_solution = np.linalg.solve(S, residual)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "innovation covariance S must be positive definite and solvable"
            ) from exc
        if not np.all(np.isfinite(K)):
            raise ValueError("Kalman gain contains a non-finite value")

        delta_x = K @ residual
        if not np.all(np.isfinite(delta_x)):
            raise ValueError("delta_x contains a non-finite value")
        delta_p = delta_x[0:3]
        delta_v = delta_x[3:6]
        delta_theta = delta_x[6:9]
        delta_bg = delta_x[9:12]
        delta_ba = delta_x[12:15]

        identity_15 = np.eye(15, dtype=float)
        A = identity_15 - K @ H
        P_joseph_raw = A @ P_pred @ A.T + K @ R @ K.T
        P_joseph = _covariance_matrix(
            P_joseph_raw,
            (15, 15),
            "Joseph-updated P",
            self._numerical,
        )

        p_after = _finite_vector(p_pred + delta_p, 3, "injected position")
        v_after = _finite_vector(v_pred + delta_v, 3, "injected velocity")
        delta_q = rotvec_to_quaternion(delta_theta, self._numerical)
        q_after = normalize_quaternion(
            quaternion_multiply(q_pred, delta_q), self._numerical
        )
        bg_after = _finite_vector(
            bg_pred + delta_bg, 3, "injected gyro bias"
        )
        ba_after = _finite_vector(
            ba_pred + delta_ba, 3, "injected accelerometer bias"
        )

        G_reset = identity_15.copy()
        G_reset[6:9, 6:9] = (
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

        # Remove only floating-point asymmetry after first checking that it is
        # numerical in scale; this averaging is not another Kalman update.
        P_reset_symmetric = 0.5 * (P_reset_raw + P_reset_raw.T)
        P_reset = _covariance_matrix(
            P_reset_symmetric,
            (15, 15),
            "reset P",
            self._numerical,
        )
        final_symmetry_error = float(
            np.max(np.abs(P_reset - P_reset.T))
        )
        q_norm = float(np.linalg.norm(q_after))
        NIS = float(residual @ innovation_solution)
        if not np.isfinite(NIS):
            raise ValueError("NIS is not finite")

        # Atomic commit after every candidate nominal/covariance check succeeds.
        self._p_W_m = p_after
        self._v_W_mps = v_after
        self._q_xyzw = q_after
        self._bg_rad_s = bg_after
        self._ba_mps2 = ba_after
        self._P = P_reset

        return UpdateDebug(
            p_pred_W_m=p_pred,
            q_pred_xyzw=q_pred,
            q_obs_raw_xyzw=q_obs_raw,
            q_obs_aligned_xyzw=q_obs_aligned.copy(),
            quaternion_sign_flipped=sign_flipped,
            q_rel_xyzw=q_rel,
            r_position_m=r_position,
            r_theta_rad=r_theta,
            residual=residual,
            R_pose=R,
            H=H,
            S=S,
            K=K,
            delta_x=delta_x,
            delta_p_m=delta_p.copy(),
            delta_v_mps=delta_v.copy(),
            delta_theta_rad=delta_theta.copy(),
            delta_bg_rad_s=delta_bg.copy(),
            delta_ba_mps2=delta_ba.copy(),
            p_after_W_m=p_after.copy(),
            v_after_W_mps=v_after.copy(),
            q_after_xyzw=q_after.copy(),
            bg_after_rad_s=bg_after.copy(),
            ba_after_mps2=ba_after.copy(),
            P_diag_before=np.diag(P_pred).copy(),
            P_diag_after_joseph=np.diag(P_joseph).copy(),
            P_diag_after_reset=np.diag(P_reset).copy(),
            G_reset=G_reset,
            raw_covariance_symmetry_error=raw_symmetry_error,
            final_covariance_symmetry_error=final_symmetry_error,
            q_norm=q_norm,
            NIS=NIS,
        )
