"""15D ESKF: right error [dp, dv, dtheta, dbg, dba], xyzw Body-to-World attitude."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

import config as cfg
from data_types import AlignmentResult, FloatArray, InitStats, ProcessedIMUData, ProcessedPoseData
from rotation_utils import (
    hat, normalize_quaternion, quaternion_inverse, quaternion_multiply,
    quaternion_to_rotation_matrix, quaternion_to_rotvec, rotvec_to_quaternion,
)


def _finite_vector(value: ArrayLike, size: int, name: str) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite vector of shape ({size},)")
    return array.copy()


def _symmetric_matrix(value: ArrayLike, size: int, name: str) -> FloatArray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (size, size) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite ({size}, {size}) matrix")
    if np.max(np.abs(matrix - matrix.T)) > cfg.COVARIANCE_SYMMETRY_TOLERANCE:
        raise ValueError(f"{name} is not symmetric within tolerance")
    return 0.5 * (matrix + matrix.T)


def _covariance_matrix(value: ArrayLike, size: int, name: str) -> FloatArray:
    matrix = _symmetric_matrix(value, size, name)
    if np.any(np.diag(matrix) < -cfg.COVARIANCE_NEGATIVE_TOLERANCE):
        raise ValueError(f"{name} has a negative diagonal entry")
    return matrix


class ESKF15D:
    """Own p/v in World, q_WB, biases in Body, P (15x15), and fixed Qc (12x12)."""

    def __init__(self, *, p, v, q, bg, ba, P, Qc):
        self.p = _finite_vector(p, 3, "initial position")
        self.v = _finite_vector(v, 3, "initial velocity")
        self.q = normalize_quaternion(q)
        self.bg = _finite_vector(bg, 3, "initial gyro bias")
        self.ba = _finite_vector(ba, 3, "initial accelerometer bias")
        self.P = _covariance_matrix(P, 15, "initial P")
        self.Qc = _covariance_matrix(Qc, 12, "Qc")

    def predict(self, gyro: ArrayLike, acc: ArrayLike, dt: float) -> None:
        """Left-endpoint ZOH: every propagation term uses the old state."""
        gyro = _finite_vector(gyro, 3, "gyro")
        acc = _finite_vector(acc, 3, "acc")
        dt = float(dt)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and strictly positive")

        omega = gyro - self.bg
        specific_force = acc - self.ba
        R = quaternion_to_rotation_matrix(self.q)
        acceleration = R @ specific_force + cfg.GRAVITY_WORLD_MPS2
        p = _finite_vector(self.p + self.v * dt + 0.5 * acceleration * dt**2, 3, "predicted p")
        v = _finite_vector(self.v + acceleration * dt, 3, "predicted v")
        q = normalize_quaternion(quaternion_multiply(self.q, rotvec_to_quaternion(omega * dt)))

        I3 = np.eye(3)
        Fc = np.zeros((15, 15))
        Fc[0:3, 3:6] = I3
        Fc[3:6, 6:9] = -R @ hat(specific_force)
        Fc[3:6, 12:15] = -R
        Fc[6:9, 6:9] = -hat(omega)
        Fc[6:9, 9:12] = -I3

        # Continuous noise order: accelerometer, gyro, gyro bias, accel bias.
        Gc = np.zeros((15, 12))
        Gc[3:6, 0:3] = -R
        Gc[6:9, 3:6] = -I3
        Gc[9:12, 6:9] = I3
        Gc[12:15, 9:12] = I3
        Phi = np.eye(15) + Fc * dt
        Qd = Gc @ self.Qc @ Gc.T * dt
        P = _covariance_matrix(Phi @ self.P @ Phi.T + Qd, 15, "predicted P")

        # Commit only after all checks succeed. Nominal biases stay constant.
        self.p, self.v, self.q, self.P = p, v, q, P

    def update_pose(
        self, position: ArrayLike, quaternion: ArrayLike, covariance: ArrayLike,
    ) -> tuple[float, float, float]:
        """Pose update, injection, reset; return position/angle residual norms and NIS."""
        position = _finite_vector(position, 3, "observed position")
        q_obs = normalize_quaternion(quaternion)
        if np.dot(self.q, q_obs) < 0.0:
            q_obs = -q_obs
        R = _symmetric_matrix(covariance, 6, "R_pose")
        try:
            np.linalg.cholesky(R)
        except np.linalg.LinAlgError as exc:
            raise ValueError("R_pose must be positive definite") from exc

        r_position = position - self.p
        q_relative = normalize_quaternion(quaternion_multiply(quaternion_inverse(self.q), q_obs))
        r_theta = quaternion_to_rotvec(q_relative)
        residual = np.concatenate((r_position, r_theta))
        H = np.zeros((6, 15))
        H[0:3, 0:3] = np.eye(3)
        H[3:6, 6:9] = np.eye(3)
        PHt = self.P @ H.T
        S = _symmetric_matrix(H @ PHt + R, 6, "innovation covariance")
        try:
            np.linalg.cholesky(S)
            K = np.linalg.solve(S, PHt.T).T
            innovation_solution = np.linalg.solve(S, residual)
        except np.linalg.LinAlgError as exc:
            raise ValueError("innovation covariance must be positive definite and solvable") from exc
        if not np.all(np.isfinite(K)):
            raise ValueError("Kalman gain contains a non-finite value")
        dx = _finite_vector(K @ residual, 15, "delta_x")

        # Joseph covariance update, then right-multiplicative state injection.
        I15 = np.eye(15)
        A = I15 - K @ H
        P_joseph = _covariance_matrix(A @ self.P @ A.T + K @ R @ K.T, 15, "Joseph P")
        p = _finite_vector(self.p + dx[0:3], 3, "injected p")
        v = _finite_vector(self.v + dx[3:6], 3, "injected v")
        q = normalize_quaternion(quaternion_multiply(self.q, rotvec_to_quaternion(dx[6:9])))
        bg = _finite_vector(self.bg + dx[9:12], 3, "injected bg")
        ba = _finite_vector(self.ba + dx[12:15], 3, "injected ba")

        # Reset changes error coordinates, not the physical nominal state.
        G_reset = I15.copy()
        G_reset[6:9, 6:9] = np.eye(3) - 0.5 * hat(dx[6:9])
        P = _covariance_matrix(G_reset @ P_joseph @ G_reset.T, 15, "reset P")
        nis = float(residual @ innovation_solution)
        if not np.isfinite(nis):
            raise ValueError("NIS is not finite")
        self.p, self.v, self.q, self.bg, self.ba, self.P = p, v, q, bg, ba, P
        return float(np.linalg.norm(r_position)), float(np.linalg.norm(r_theta)), nis


def initialize_eskf15d(
    imu: ProcessedIMUData, pose: ProcessedPoseData,
    stats: InitStats, alignment: AlignmentResult,
) -> tuple[ESKF15D, int]:
    """Initialize at the final static sample; inputs have passed the NPZ loaders."""
    start, stop = stats.static_start_idx, stats.static_end_idx
    if not stats.is_static or not (0 <= start < stop <= imu.timestamp.size):
        raise ValueError("initialization requires an accepted non-empty static interval")
    k0 = stop - 1  # Static interval uses Python's exclusive stop index.
    if k0 >= imu.timestamp.size - 1:
        raise ValueError("an IMU sample is required after initialization")
    if not np.isclose(stats.static_end_time, imu.timestamp[k0], rtol=0.0, atol=1e-9):
        raise ValueError("static_end_time must equal the last static IMU timestamp")
    j0 = int(alignment.pose_index_for_imu[k0])
    if j0 < 0 or j0 >= pose.timestamp.size:
        raise ValueError("initialization requires a matched FAST-LIO pose")

    q0 = normalize_quaternion(pose.quaternion_xyzw[j0])
    bg0 = _finite_vector(stats.gyro_mean, 3, "gyro mean")
    gyro_std = _finite_vector(stats.gyro_std, 3, "gyro std")
    acc_mean = _finite_vector(stats.acc_mean, 3, "acc mean")
    acc_std = _finite_vector(stats.acc_std, 3, "acc std")
    cov = pose.cov_diag[j0]
    if not np.all(np.isfinite(cov)) or np.any(cov <= 0.0):
        raise ValueError("initial pose covariance must be finite and strictly positive")
    if np.any(gyro_std <= 0.0) or np.any(acc_std <= 0.0):
        raise ValueError("static standard deviations must be strictly positive")
    tuning = np.array([stats.median_dt, cfg.INITIAL_VELOCITY_STD_MPS,
                       cfg.GYRO_BIAS_RANDOM_WALK_DENSITY, cfg.ACCELEROMETER_BIAS_RANDOM_WALK_DENSITY])
    if not np.all(np.isfinite(tuning)) or np.any(tuning <= 0.0):
        raise ValueError("dt, initial velocity std, and bias noise densities must be positive")

    count = stop - start
    gravity_body = quaternion_to_rotation_matrix(q0).T @ cfg.GRAVITY_WORLD_MPS2
    ba0 = acc_mean + gravity_body
    Ptheta = np.diag(cov[3:6])
    P0 = np.zeros((15, 15))
    P0[0:3, 0:3] = np.diag(cov[0:3])
    P0[3:6, 3:6] = cfg.INITIAL_VELOCITY_STD_MPS**2 * np.eye(3)
    P0[6:9, 6:9] = Ptheta
    P0[9:12, 9:12] = np.diag(gyro_std**2 / count)
    # Initial attitude uncertainty also affects the gravity-compensated ba.
    Jg = hat(gravity_body)
    P0[6:9, 12:15] = Ptheta @ Jg.T
    P0[12:15, 6:9] = Jg @ Ptheta
    P0[12:15, 12:15] = np.diag(acc_std**2 / count) + Jg @ Ptheta @ Jg.T

    # Keep the established full-record median dt, not a static-only dt estimate.
    gyro_density = gyro_std * np.sqrt(stats.median_dt)
    acc_density = acc_std * np.sqrt(stats.median_dt)
    Qc = np.diag(np.concatenate((
        acc_density**2, gyro_density**2,
        np.full(3, cfg.GYRO_BIAS_RANDOM_WALK_DENSITY**2),
        np.full(3, cfg.ACCELEROMETER_BIAS_RANDOM_WALK_DENSITY**2),
    )))
    for name, matrix in (("P0", P0), ("Qc", Qc)):
        if not np.all(np.isfinite(matrix)) or not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-15):
            raise ValueError(f"{name} must be finite and symmetric")
        if np.any(np.diag(matrix) <= 0.0):
            raise ValueError(f"{name} diagonal must be strictly positive")
    return ESKF15D(p=pose.position[j0], v=np.zeros(3), q=q0, bg=bg0, ba=ba0, P=P0, Qc=Qc), k0
