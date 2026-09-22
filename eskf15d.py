"""15D ESKF initialization, propagation and FAST-LIO pose update."""

import numpy as np
from numpy.typing import ArrayLike

import config as cfg
from data_types import AlignmentResult, InitStats, ProcessedIMUData, ProcessedPoseData
from rotation_utils import (hat, normalize_quaternion, quaternion_inverse,
                            quaternion_multiply, quaternion_to_rotation_matrix,
                            quaternion_to_rotvec, rotvec_to_quaternion)


def _vector(value: ArrayLike, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite vector of shape ({size},)")
    return array.copy()


def _covariance(value: ArrayLike, size: int, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (size, size) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite ({size}, {size}) matrix")
    # Matrix products accumulate tiny antisymmetric round-off; this does not
    # change the covariance represented by the ESKF.
    return 0.5 * (matrix + matrix.T)


class ESKF15D:
    """Nominal p, v, q, gyro bias, accelerometer bias and 15x15 covariance."""

    def __init__(self, *, p: ArrayLike, v: ArrayLike, q: ArrayLike,
                 bg: ArrayLike, ba: ArrayLike, P: ArrayLike, Qc: ArrayLike):
        self.p = _vector(p, 3, "initial position")
        self.v = _vector(v, 3, "initial velocity")
        self.q = normalize_quaternion(q)
        self.bg = _vector(bg, 3, "initial gyro bias")
        self.ba = _vector(ba, 3, "initial accelerometer bias")
        self.P = _covariance(P, 15, "initial P")
        self.Qc = _covariance(Qc, 12, "Qc")

    def predict(self, gyro: ArrayLike, acc: ArrayLike, dt: float) -> None:
        """Left-endpoint ZOH propagation from IMU sample k to k + 1."""
        gyro, acc = _vector(gyro, 3, "gyro"), _vector(acc, 3, "acc")
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        omega = gyro - self.bg
        specific_force = acc - self.ba
        R = quaternion_to_rotation_matrix(self.q)
        acceleration = R @ specific_force + cfg.GRAVITY_WORLD_MPS2
        p = self.p + self.v * dt + 0.5 * acceleration * dt**2
        v = self.v + acceleration * dt
        q = normalize_quaternion(quaternion_multiply(self.q, rotvec_to_quaternion(omega * dt)))

        I3 = np.eye(3)
        Fc = np.zeros((15, 15))
        Fc[0:3, 3:6] = I3
        Fc[3:6, 6:9] = -R @ hat(specific_force)
        Fc[3:6, 12:15] = -R
        Fc[6:9, 6:9] = -hat(omega)
        Fc[6:9, 9:12] = -I3
        Gc = np.zeros((15, 12))
        Gc[3:6, 0:3] = -R
        Gc[6:9, 3:6] = -I3
        Gc[9:12, 6:9] = I3
        Gc[12:15, 9:12] = I3
        Phi = np.eye(15) + Fc * dt
        P = _covariance(Phi @ self.P @ Phi.T + Gc @ self.Qc @ Gc.T * dt, 15, "predicted P")
        self.p, self.v, self.q, self.P = p, v, q, P

    def update_pose(self, position: ArrayLike, quaternion: ArrayLike,
                    covariance: ArrayLike) -> tuple[float, float, float]:
        """Apply position/attitude residual, Joseph update, injection and reset."""
        position = _vector(position, 3, "observed position")
        q_obs = normalize_quaternion(quaternion)
        if np.dot(self.q, q_obs) < 0.0:
            q_obs = -q_obs
        R = _covariance(covariance, 6, "pose covariance")
        residual_position = position - self.p
        residual_attitude = quaternion_to_rotvec(
            quaternion_multiply(quaternion_inverse(self.q), q_obs))
        residual = np.r_[residual_position, residual_attitude]
        H = np.zeros((6, 15))
        H[:3, :3] = np.eye(3)
        H[3:6, 6:9] = np.eye(3)
        PHt = self.P @ H.T
        S = _covariance(H @ PHt + R, 6, "innovation covariance")
        try:
            K = np.linalg.solve(S, PHt.T).T
            nis = float(residual @ np.linalg.solve(S, residual))
        except np.linalg.LinAlgError as exc:
            raise ValueError("innovation covariance is singular") from exc
        dx = _vector(K @ residual, 15, "state correction")
        A = np.eye(15) - K @ H
        P = _covariance(A @ self.P @ A.T + K @ R @ K.T, 15, "Joseph P")
        p, v = self.p + dx[:3], self.v + dx[3:6]
        q = normalize_quaternion(quaternion_multiply(self.q, rotvec_to_quaternion(dx[6:9])))
        bg, ba = self.bg + dx[9:12], self.ba + dx[12:15]
        reset = np.eye(15)
        reset[6:9, 6:9] -= 0.5 * hat(dx[6:9])
        # Keep the reset covariance explicitly symmetric before storing it.
        P = reset @ P @ reset.T
        P = _covariance(0.5 * (P + P.T), 15, "reset P")
        self.p, self.v, self.q, self.bg, self.ba, self.P = p, v, q, bg, ba, P
        return float(np.linalg.norm(residual_position)), float(np.linalg.norm(residual_attitude)), nis


def initialize_eskf15d(imu: ProcessedIMUData, pose: ProcessedPoseData,
                       stats: InitStats, alignment: AlignmentResult) -> tuple[ESKF15D, int]:
    """Build the state at the final initial-static IMU sample."""
    start, stop = stats.static_start_idx, stats.static_end_idx
    if not stats.is_static or not 0 <= start < stop < imu.timestamp.size:
        raise ValueError("an accepted static interval followed by IMU data is required")
    k0 = stop - 1
    j0 = int(alignment.pose_index_for_imu[k0])
    if j0 < 0:
        raise ValueError("initial static endpoint has no pose match")
    q0 = normalize_quaternion(pose.quaternion_xyzw[j0])
    gravity_body = quaternion_to_rotation_matrix(q0).T @ cfg.GRAVITY_WORLD_MPS2
    bg0, ba0 = stats.gyro_mean, stats.acc_mean + gravity_body
    count = stop - start
    Ptheta = np.diag(pose.cov_diag[j0, 3:6])
    P0 = np.zeros((15, 15))
    P0[:3, :3] = np.diag(pose.cov_diag[j0, :3])
    P0[3:6, 3:6] = cfg.INITIAL_VELOCITY_STD_MPS**2 * np.eye(3)
    P0[6:9, 6:9] = Ptheta
    P0[9:12, 9:12] = np.diag(stats.gyro_std**2 / count)
    Jg = hat(gravity_body)
    P0[6:9, 12:15] = Ptheta @ Jg.T
    P0[12:15, 6:9] = Jg @ Ptheta
    P0[12:15, 12:15] = np.diag(stats.acc_std**2 / count) + Jg @ Ptheta @ Jg.T
    gyro_density = stats.gyro_std * np.sqrt(stats.median_dt)
    acc_density = stats.acc_std * np.sqrt(stats.median_dt)
    Qc = np.diag(np.r_[acc_density**2, gyro_density**2,
                       np.full(3, cfg.GYRO_BIAS_RANDOM_WALK_DENSITY**2),
                       np.full(3, cfg.ACCELEROMETER_BIAS_RANDOM_WALK_DENSITY**2)])
    return ESKF15D(p=pose.position[j0], v=np.zeros(3), q=q0, bg=bg0, ba=ba0, P=P0, Qc=Qc), k0
