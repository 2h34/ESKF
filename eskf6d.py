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

    这里的名义状态只有姿态 ``q_WB`` 和陀螺零偏 ``b_g``。``P`` 不是名义
    状态本身的数组，而是六维误差态 ``[delta_theta, delta_b_g]`` 的协方差；
    前三维描述局部右乘姿态误差，后三维描述零偏估计误差。
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

        # 陀螺仪读数同时含有真实角速度和当前估计的零偏；先减去名义零偏，
        # 得到用于传播名义姿态的 Body-frame 角速度估计。
        omega_hat = gyro - self._bg_rad_s

        # 注意：这里沿用已有变量名 delta_theta，但它表示本次 IMU 区间的
        # 名义旋转增量 Delta_theta_imu，不是滤波器长期保存的误差态姿态分量。
        delta_theta = omega_hat * dt
        delta_q = rotvec_to_quaternion(delta_theta, self._numerical)

        # q 表示 R_WB，而陀螺测得的是 Body-frame 角速度，因此增量在旧姿态
        # 右侧复合：q_new = q_old ⊗ Exp(omega_B * dt)。
        q_after = normalize_quaternion(
            quaternion_multiply(q_before, delta_q), self._numerical
        )

        # Fc 是六维误差态连续时间线性化矩阵：左上块描述当前角速度使姿态
        # 误差旋转，右上块说明零偏误差会以负号驱动姿态误差；左下块为零，
        # 表示当前模型中姿态误差不反向驱动零偏；右下块为零，对应零偏自身
        # 没有确定性动态，只由下面 Gc 引入的 bias random walk 噪声驱动。
        Fc = np.zeros((6, 6), dtype=float)
        Fc[0:3, 0:3] = -hat(omega_hat)
        Fc[0:3, 3:6] = -np.eye(3, dtype=float)

        # Gc 把连续时间噪声映射到 [delta_theta, delta_b_g]：gyro measurement
        # noise 以 -I 进入姿态误差，bias random walk noise 以 I 进入零偏误差。
        Gc = np.zeros((6, 6), dtype=float)
        Gc[0:3, 0:3] = -np.eye(3, dtype=float)
        Gc[3:6, 3:6] = np.eye(3, dtype=float)

        # 当前版本用一阶近似 Phi ≈ I + Fc*dt 离散化状态转移；Qc 是连续
        # 时间噪声强度，Qd 是该采样区间内累积得到的离散过程噪声协方差。
        Phi = np.eye(6, dtype=float) + Fc * dt
        Qd = Gc @ self._Qc @ Gc.T * dt

        # 旧的不确定性先经 Phi 传播，再叠加这一时间段新注入的过程噪声。
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

        # q 与 -q 表示同一个旋转。先让观测与预测落在同一半球，使相对四元数
        # 采用连续且与预测一致的代表元，避免纯粹的符号跳变进入残差计算。
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

        # 右乘误差约定下，相对旋转必须从 Prediction 指向 Observation：
        # q_rel = q_pred^{-1} ⊗ q_obs，对应 Log(R_pred^T R_obs)。反序会翻转残差。
        q_rel = normalize_quaternion(
            quaternion_multiply(
                quaternion_inverse(q_pred, self._numerical),
                q_obs_aligned,
            ),
            self._numerical,
        )
        r_theta = quaternion_to_rotvec(q_rel, self._numerical)
        residual_norm = float(np.linalg.norm(r_theta))

        # 姿态观测直接看到误差态前三维，所以 H=[I 0]。零偏虽未被直接观测，
        # 仍可通过 P 中姿态与零偏的交叉协方差获得间接修正。
        H = np.zeros((3, 6), dtype=float)
        H[:, 0:3] = np.eye(3, dtype=float)
        PHt = P_pred @ H.T
        S = _symmetric_matrix(
            H @ PHt + R, (3, 3), "innovation covariance S", self._numerical
        )
        # S 是残差的预测协方差；K 按各误差态与残差的相关性分配修正量。
        # 使用 solve 求解线性系统，避免显式构造数值稳定性较差的 S^{-1}。
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

        # delta_x 只是本次观测更新得到的临时误差态估计：前三维
        # delta_theta_hat 修正 attitude，后三维 delta_b_g_hat 修正 gyro bias；
        # 随后它们会注入名义状态，不会作为另一个长期状态保存在 ESKF 中。
        delta_x = K @ r_theta
        if not np.all(np.isfinite(delta_x)):
            raise ValueError("delta_x contains a non-finite value")
        delta_theta = delta_x[0:3]
        delta_bg = delta_x[3:6]

        # Joseph 形式只完成 measurement covariance update；它更新不确定性，
        # 与下面对 q、bg 的 nominal-state injection 是两个不同步骤。
        identity_6 = np.eye(6, dtype=float)
        A = identity_6 - K @ H
        P_joseph_raw = A @ P_pred @ A.T + K @ R @ K.T
        P_joseph = _covariance_matrix(
            P_joseph_raw, "Joseph-updated P", self._numerical
        )

        # 将估计出的局部右乘姿态误差注入 q，并把零偏误差加到名义零偏上。
        delta_q = rotvec_to_quaternion(delta_theta, self._numerical)
        q_after = normalize_quaternion(
            quaternion_multiply(q_pred, delta_q), self._numerical
        )
        bg_after = bg_before + delta_bg
        if not np.all(np.isfinite(bg_after)):
            raise ValueError("injected gyro bias contains a non-finite value")

        # 注入后名义状态已经改变，误差坐标的原点也随之改变。G_reset 把
        # Joseph 更新后的协方差映射到新误差坐标；reset 绝不是把 P 清零。
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
        # 中文说明：这里只清除浮点矩阵运算产生的极小非对称误差，不是新的
        # Kalman 更新，也不能用平均操作掩盖明显的 covariance 错误。
        P_reset_symmetric = 0.5 * (P_reset_raw + P_reset_raw.T)
        P_reset = _covariance_matrix(
            P_reset_symmetric, "reset P", self._numerical
        )
        final_symmetry_error = float(
            np.max(np.abs(P_reset - P_reset.T))
        )
        q_norm = float(np.linalg.norm(q_after))
        # NIS = r^T S^{-1} r，表示 residual 相对于预测 innovation uncertainty
        # 的归一化大小；这里只用于诊断，不用于 gating、拒绝观测或自动调参。
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
