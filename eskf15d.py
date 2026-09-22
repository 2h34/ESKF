"""15 维 ESKF：右乘误差 [dp, dv, dtheta, dbg, dba]，xyzw 表示的 Body 到 World 姿态。"""

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


def _covariance_matrix(value: ArrayLike, size: int, name: str) -> FloatArray:
    """校验形状与有限性，并返回数值平均 ``0.5(M + M^T)``。

    对称化是计算本身的一部分，而非防御性检查：若不做对称化，协方差在反复的
    秩 6 更新下会失去精确对称性。原先的逐次调用对称性检验与负对角元扫描已被
    移除，因为 ``predict``/``update_pose`` 中的代数运算按构造即保持这两个性质，
    且 runner 每帧会重新检查一次。
    """

    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (size, size) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite ({size}, {size}) matrix")
    return 0.5 * (matrix + matrix.T)


class ESKF15D:
    """持有 World 系下的 p/v、q_WB、Body 系下的零偏、P (15x15) 以及固定的 Qc (12x12)。"""

    def __init__(self, *, p, v, q, bg, ba, P, Qc):
        self.p = _finite_vector(p, 3, "initial position")
        self.v = _finite_vector(v, 3, "initial velocity")
        self.q = normalize_quaternion(q)
        self.bg = _finite_vector(bg, 3, "initial gyro bias")
        self.ba = _finite_vector(ba, 3, "initial accelerometer bias")
        self.P = _covariance_matrix(P, 15, "initial P")
        self.Qc = _covariance_matrix(Qc, 12, "Qc")

    def predict(self, gyro: ArrayLike, acc: ArrayLike, dt: float) -> None:
        """左端点零阶保持（ZOH）：每个传播项都使用旧状态。"""
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

        # 连续噪声顺序：加速度计、陀螺仪、陀螺零偏、加速度计零偏。
        Gc = np.zeros((15, 12))
        Gc[3:6, 0:3] = -R
        Gc[6:9, 3:6] = -I3
        Gc[9:12, 6:9] = I3
        Gc[12:15, 9:12] = I3
        Phi = np.eye(15) + Fc * dt
        Qd = Gc @ self.Qc @ Gc.T * dt
        P = _covariance_matrix(Phi @ self.P @ Phi.T + Qd, 15, "predicted P")

        # 全部检查通过后才提交。名义零偏保持不变。
        self.p, self.v, self.q, self.P = p, v, q, P

    def update_pose(
        self, position: ArrayLike, quaternion: ArrayLike, covariance: ArrayLike,
    ) -> tuple[float, float, float]:
        """位姿更新、注入、Reset；返回位置/角度残差范数与 NIS。"""
        position = _finite_vector(position, 3, "observed position")
        q_obs = normalize_quaternion(quaternion)
        if np.dot(self.q, q_obs) < 0.0:
            q_obs = -q_obs
        R = _covariance_matrix(covariance, 6, "R_pose")

        r_position = position - self.p
        q_relative = normalize_quaternion(quaternion_multiply(quaternion_inverse(self.q), q_obs))
        r_theta = quaternion_to_rotvec(q_relative)
        residual = np.concatenate((r_position, r_theta))
        H = np.zeros((6, 15))
        H[0:3, 0:3] = np.eye(3)
        H[3:6, 6:9] = np.eye(3)
        PHt = self.P @ H.T
        S = _covariance_matrix(H @ PHt + R, 6, "innovation covariance")
        try:
            np.linalg.cholesky(S)
            K = np.linalg.solve(S, PHt.T).T
            innovation_solution = np.linalg.solve(S, residual)
        except np.linalg.LinAlgError as exc:
            raise ValueError("innovation covariance must be positive definite and solvable") from exc
        if not np.all(np.isfinite(K)):
            raise ValueError("Kalman gain contains a non-finite value")
        dx = _finite_vector(K @ residual, 15, "delta_x")

        # Joseph 形式协方差更新，随后进行右乘状态注入。
        I15 = np.eye(15)
        A = I15 - K @ H
        P_joseph = _covariance_matrix(A @ self.P @ A.T + K @ R @ K.T, 15, "Joseph P")
        p = _finite_vector(self.p + dx[0:3], 3, "injected p")
        v = _finite_vector(self.v + dx[3:6], 3, "injected v")
        q = normalize_quaternion(quaternion_multiply(self.q, rotvec_to_quaternion(dx[6:9])))
        bg = _finite_vector(self.bg + dx[9:12], 3, "injected bg")
        ba = _finite_vector(self.ba + dx[12:15], 3, "injected ba")

        # Reset 改变的是误差坐标，而非物理名义状态。
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
    """在最后一个静止采样处初始化；输入均已通过 NPZ 加载器的检查。"""
    start, stop = stats.static_start_idx, stats.static_end_idx
    if not stats.is_static or not (0 <= start < stop <= imu.timestamp.size):
        raise ValueError("initialization requires an accepted non-empty static interval")
    k0 = stop - 1  # 静止区间使用 Python 的排他性 stop 索引。
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

    count = stop - start
    gravity_body = quaternion_to_rotation_matrix(q0).T @ cfg.GRAVITY_WORLD_MPS2
    ba0 = acc_mean + gravity_body
    Ptheta = np.diag(cov[3:6])
    P0 = np.zeros((15, 15))
    P0[0:3, 0:3] = np.diag(cov[0:3])
    P0[3:6, 3:6] = cfg.INITIAL_VELOCITY_STD_MPS**2 * np.eye(3)
    P0[6:9, 6:9] = Ptheta
    P0[9:12, 9:12] = np.diag(gyro_std**2 / count)
    # 初始姿态不确定性也会影响重力补偿后的 ba。
    Jg = hat(gravity_body)
    P0[6:9, 12:15] = Ptheta @ Jg.T
    P0[12:15, 6:9] = Jg @ Ptheta
    P0[12:15, 12:15] = np.diag(acc_std**2 / count) + Jg @ Ptheta @ Jg.T

    # 沿用既有的全记录中位 dt，而非仅用静止区间的 dt 估计。
    gyro_density = gyro_std * np.sqrt(stats.median_dt)
    acc_density = acc_std * np.sqrt(stats.median_dt)
    Qc = np.diag(np.concatenate((
        acc_density**2, gyro_density**2,
        np.full(3, cfg.GYRO_BIAS_RANDOM_WALK_DENSITY**2),
        np.full(3, cfg.ACCELEROMETER_BIAS_RANDOM_WALK_DENSITY**2),
    )))
    # P0/Qc 的有限、对称与正对角性由 ESKF15D.__init__ 的 _covariance_matrix 统一校验。
    return ESKF15D(p=pose.position[j0], v=np.zeros(3), q=q0, bg=bg0, ba=ba0, P=P0, Qc=Qc), k0
